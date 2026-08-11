from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from pengine.schemas import InternalStage, PersonaSnapshot, PersonaSummary

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
PERSONA_SCHEMA_V1 = "1.0.0"
PERSONA_SCHEMA_V2 = "2.0.0"
CURRENT_PERSONA_SCHEMA = PERSONA_SCHEMA_V2
V1_LOGICAL_FILES: tuple[tuple[str, str], ...] = (
    ("paradigm", "paradigm.md"),
    ("project", "project.md"),
    ("l0", "l0.md"),
    ("l1", "l1.md"),
    ("l2", "l2.md"),
    ("l3", "l3.md"),
    ("l4", "l4.md"),
    ("l5", "l5.md"),
    ("l6", "l6.md"),
)
V2_LOGICAL_FILES: tuple[tuple[str, str], ...] = (
    ("paradigm", "paradigm.md"),
    ("project", "project.md"),
    ("l0", "l0.md"),
    ("soul", "soul.md"),
    ("l3", "l3.md"),
    ("l4", "l4.md"),
    ("l5", "l5.md"),
    ("l6", "l6.md"),
)
LOGICAL_FILES_BY_SCHEMA = MappingProxyType(
    {
        PERSONA_SCHEMA_V1: V1_LOGICAL_FILES,
        PERSONA_SCHEMA_V2: V2_LOGICAL_FILES,
    }
)
# The public default describes packages selectable for new work. Historical v1
# callers must pass their schema version explicitly.
LOGICAL_FILES = V2_LOGICAL_FILES
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "persona-package.schema.json"
)
DEFAULT_STAGE_CONTEXT_CHARS = 120_000
DEFAULT_RETRIEVAL_LIMIT = 5
DEFAULT_RETRIEVAL_MAX_CHARS = 6_000
DEFAULT_RETRIEVAL_RESULT_CHARS = 1_200
MAX_RETRIEVAL_LIMIT = 20
MAX_SOUL_CHARS = 8_000
SNAPSHOT_HASH_DOMAIN = b"pengine-persona-snapshot-v1\0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s+(.+?)\s*$")
_STATUS_RE = re.compile(
    r"(?:真人已定|创作者已确认|作者已确认|AI(?:结构化)?(?:草稿)?待(?:真人)?确认|AI草稿待定)"
)
_PENDING_RE = re.compile(r"(?:AI(?:结构化)?(?:草稿)?待(?:真人)?确认|AI草稿待定)")
_OWNERSHIP_RE = re.compile(r"(?:归属\s*[:：]\s*\S+|归属(?:创作者|作者|真人))")
_WORD_RE = re.compile(r"[0-9A-Za-z_\-\u3400-\u9fff]+")
_L0_VARIANT_ID_RE = re.compile(r"\[ID:([A-Za-z0-9][A-Za-z0-9_-]{0,31})\]")
_L0_VARIANT_ID_PREFIX_RE = re.compile(r"\[\s*ID\s*:", re.IGNORECASE)


class PersonaPackageError(ValueError):
    """A safe persona-package validation error that never embeds source content."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ValidatedPersonaPackage:
    path: Path
    manifest: Mapping[str, Any]
    summary: PersonaSummary
    package_sha256: str
    _raw_entries: Mapping[str, bytes]

    @property
    def schema_version(self) -> str:
        return str(self.manifest["schema_version"])

    @property
    def logical_files(self) -> tuple[tuple[str, str], ...]:
        return _logical_files(self.schema_version)

    def text(self, logical_name: str) -> str:
        try:
            filename = dict(self.logical_files)[logical_name]
            raw = self._raw_entries[filename]
        except KeyError as exc:
            raise PersonaPackageError(
                "unknown_persona_file", f"Unknown persona file: {logical_name}"
            ) from exc
        return raw.decode("utf-8")


@dataclass(frozen=True, slots=True)
class ResolvedPersonaSnapshot:
    path: Path
    manifest: Mapping[str, Any]
    summary: PersonaSnapshot
    _package: ValidatedPersonaPackage

    def text(self, logical_name: str) -> str:
        return self._package.text(logical_name)


@dataclass(frozen=True, slots=True)
class PersonaStageContext:
    snapshot: PersonaSnapshot
    stage: InternalStage
    files: Mapping[str, str]
    total_chars: int


@dataclass(frozen=True, slots=True)
class ReferenceHit:
    source: str
    heading: str
    excerpt: str
    score: int


def _logical_files(schema_version: str) -> tuple[tuple[str, str], ...]:
    try:
        return LOGICAL_FILES_BY_SCHEMA[schema_version]
    except KeyError as exc:
        raise PersonaPackageError(
            "unsupported_persona_schema", "Persona schema version is not supported"
        ) from exc


def canonical_package_sha256(
    file_hashes: Iterable[str],
    *,
    schema_version: str = CURRENT_PERSONA_SCHEMA,
) -> str:
    hashes = tuple(file_hashes)
    logical_files = _logical_files(schema_version)
    if len(hashes) != len(logical_files) or any(
        not _SHA256_RE.fullmatch(value) for value in hashes
    ):
        raise PersonaPackageError(
            "invalid_file_hashes",
            f"Canonical package hash requires {len(logical_files)} lowercase SHA-256 values",
        )
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


def canonical_snapshot_sha256(
    manifest: Mapping[str, Any],
    package_sha256: str,
) -> str:
    if not _SHA256_RE.fullmatch(package_sha256):
        raise PersonaPackageError(
            "invalid_package_hash",
            "Snapshot hash requires a lowercase package SHA-256 value",
        )
    if manifest.get("package_sha256") != package_sha256:
        raise PersonaPackageError(
            "invalid_package_hash",
            "Snapshot manifest and package SHA-256 values must match",
        )
    canonical_manifest = json.dumps(
        _json_value(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        SNAPSHOT_HASH_DOMAIN + package_sha256.encode("ascii") + b"\0" + canonical_manifest
    ).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def validate_persona_package(
    package_path: Path,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> ValidatedPersonaPackage:
    package_path = Path(package_path)
    _validate_package_root(package_path)

    _validate_contained_regular_file(package_path, package_path / MANIFEST_NAME)
    manifest_raw = _read_entry(package_path / MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, PersonaPackageError) as exc:
        if isinstance(exc, PersonaPackageError):
            raise
        raise PersonaPackageError(
            "invalid_manifest", "Manifest must be unique-key UTF-8 JSON"
        ) from exc

    declared_schema = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    declared_files = manifest.get("files") if isinstance(manifest, Mapping) else None
    declared_file_names = set(declared_files) if isinstance(declared_files, Mapping) else set()
    if (
        declared_schema == PERSONA_SCHEMA_V2 and declared_file_names.intersection({"l1", "l2"})
    ) or (declared_schema == PERSONA_SCHEMA_V1 and "soul" in declared_file_names):
        raise PersonaPackageError(
            "mixed_persona_schema", "Persona package cannot mix Soul with L1 or L2"
        )

    schema = _load_schema(schema_path)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "manifest"
        raise PersonaPackageError(
            "manifest_schema_invalid", f"Manifest does not match schema at {location}"
        )

    schema_version = manifest["schema_version"]
    logical_files = _logical_files(schema_version)
    _validate_package_directory(package_path, schema_version, logical_files)

    raw_entries: dict[str, bytes] = {MANIFEST_NAME: manifest_raw}
    actual_hashes: list[str] = []
    for logical_name, filename in logical_files:
        declared = manifest["files"][logical_name]
        if declared["path"] != filename:
            raise PersonaPackageError("unsafe_file_path", f"Invalid fixed path for {logical_name}")
        source_file = package_path / filename
        _validate_contained_regular_file(package_path, source_file)
        raw = _read_entry(source_file)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PersonaPackageError(
                "markdown_not_utf8", f"{filename} must be valid UTF-8"
            ) from exc
        if "\x00" in text:
            raise PersonaPackageError("markdown_not_text", f"{filename} contains invalid text data")
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != declared["sha256"]:
            raise PersonaPackageError("file_hash_mismatch", f"Hash mismatch for {filename}")
        _validate_required_structure(logical_name, text, schema_version=schema_version)
        raw_entries[filename] = raw
        actual_hashes.append(actual_hash)

    package_hash = canonical_package_sha256(actual_hashes, schema_version=schema_version)
    if package_hash != manifest["package_sha256"]:
        raise PersonaPackageError("package_hash_mismatch", "Canonical package hash mismatch")
    snapshot_hash = canonical_snapshot_sha256(manifest, package_hash)

    summary = PersonaSummary(
        persona_id=manifest["persona_id"],
        display_name=manifest["display_name"],
        version=manifest["version"],
        snapshot_sha256=snapshot_hash,
    )
    return ValidatedPersonaPackage(
        path=package_path.resolve(strict=True),
        manifest=_freeze_mapping(manifest),
        summary=summary,
        package_sha256=package_hash,
        _raw_entries=MappingProxyType(raw_entries),
    )


def resolve_persona_snapshot(
    snapshot_root: Path,
    snapshot_sha256: str,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> ResolvedPersonaSnapshot:
    if not _SHA256_RE.fullmatch(snapshot_sha256):
        raise PersonaPackageError(
            "invalid_snapshot_hash", "Snapshot hash must be lowercase SHA-256"
        )
    root = Path(snapshot_root)
    if root.is_symlink():
        raise PersonaPackageError("unsafe_snapshot_root", "Snapshot root cannot be a symlink")
    target = root / snapshot_sha256
    if not target.exists():
        raise PersonaPackageError("snapshot_not_found", "Persona snapshot does not exist")
    if target.is_symlink():
        raise PersonaPackageError("unsafe_snapshot_path", "Persona snapshot cannot be a symlink")
    root_resolved = root.resolve(strict=True)
    target_resolved = target.resolve(strict=True)
    if target_resolved.parent != root_resolved:
        raise PersonaPackageError("unsafe_snapshot_path", "Persona snapshot escaped its root")
    package = validate_persona_package(target_resolved, schema_path=schema_path)
    if package.summary.snapshot_sha256 != snapshot_sha256:
        raise PersonaPackageError("snapshot_hash_mismatch", "Snapshot path and content hash differ")
    return ResolvedPersonaSnapshot(
        path=target_resolved,
        manifest=package.manifest,
        summary=PersonaSnapshot.model_validate(package.summary.model_dump()),
        _package=package,
    )


def load_stage_context(
    snapshot_root: Path,
    snapshot_sha256: str,
    stage: InternalStage | str,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    max_chars: int = DEFAULT_STAGE_CONTEXT_CHARS,
) -> PersonaStageContext:
    snapshot = resolve_persona_snapshot(snapshot_root, snapshot_sha256, schema_path=schema_path)
    context = _project_stage_context(snapshot, _resolve_stage(stage))
    _enforce_context_limit(context, max_chars)
    return context


def _resolve_stage(stage: InternalStage | str) -> InternalStage:
    try:
        return stage if isinstance(stage, InternalStage) else InternalStage(stage)
    except ValueError as exc:
        raise PersonaPackageError("unknown_stage", "Unknown workflow stage") from exc


def _project_stage_context(
    snapshot: ResolvedPersonaSnapshot,
    stage: InternalStage,
) -> PersonaStageContext:
    project = _strip_pending_blocks(snapshot.text("project"))
    l0 = _strip_pending_blocks(snapshot.text("l0"))
    files: dict[str, str] = {
        "/persona/project.md": project,
        "/persona/l0.md": l0,
        "/persona/l3-summary.md": _extract_required_section(
            snapshot.text("l3"), (("摘要", "summary"),), "l3 summary"
        ),
    }
    if snapshot.manifest["schema_version"] == PERSONA_SCHEMA_V2:
        files["/persona/soul.md"] = snapshot.text("soul")
    else:
        files["/persona/l1-summary.md"] = _extract_required_section(
            snapshot.text("l1"), (("摘要", "summary"),), "l1 summary"
        )
        files["/persona/l2-summary.md"] = _extract_required_section(
            snapshot.text("l2"), (("摘要", "summary"),), "l2 summary"
        )
    l4_context = _stage_l4_context(snapshot.text("l4"), stage)
    if l4_context:
        files["/persona/l4.md"] = l4_context

    total_chars = sum(len(value) for value in files.values())
    return PersonaStageContext(
        snapshot=snapshot.summary,
        stage=stage,
        files=MappingProxyType(files),
        total_chars=total_chars,
    )


def _enforce_context_limit(context: PersonaStageContext, max_chars: int) -> None:
    if max_chars < 1:
        raise PersonaPackageError("invalid_context_limit", "Stage context limit must be positive")
    if context.total_chars > max_chars:
        raise PersonaPackageError(
            "stage_context_too_large",
            f"Compiled stage context exceeds the configured {max_chars}-character limit",
        )


def retrieve_references(
    snapshot_root: Path,
    snapshot_sha256: str,
    query: str,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    sources: tuple[str, ...] = ("l5", "l6"),
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    max_chars: int = DEFAULT_RETRIEVAL_MAX_CHARS,
    per_result_chars: int = DEFAULT_RETRIEVAL_RESULT_CHARS,
) -> tuple[ReferenceHit, ...]:
    snapshot = resolve_persona_snapshot(snapshot_root, snapshot_sha256, schema_path=schema_path)
    return _retrieve_snapshot_references(
        snapshot,
        query,
        sources=sources,
        limit=limit,
        max_chars=max_chars,
        per_result_chars=per_result_chars,
    )


def _retrieve_snapshot_references(
    snapshot: ResolvedPersonaSnapshot,
    query: str,
    *,
    sources: tuple[str, ...],
    limit: int,
    max_chars: int,
    per_result_chars: int,
    chunks_by_source: Mapping[str, tuple[tuple[str, str], ...]] | None = None,
) -> tuple[ReferenceHit, ...]:
    query = query.strip()
    if not query:
        raise PersonaPackageError("empty_retrieval_query", "Retrieval query cannot be empty")
    if not 1 <= limit <= MAX_RETRIEVAL_LIMIT:
        raise PersonaPackageError(
            "invalid_retrieval_limit",
            f"Retrieval limit must be between 1 and {MAX_RETRIEVAL_LIMIT}",
        )
    if max_chars < 1 or per_result_chars < 1:
        raise PersonaPackageError("invalid_retrieval_budget", "Retrieval budgets must be positive")
    if not sources or any(source not in {"l5", "l6"} for source in sources):
        raise PersonaPackageError("invalid_retrieval_source", "Retrieval supports only L5 and L6")

    tokens = _query_tokens(query)
    ranked: list[tuple[int, int, str, str, str]] = []
    ordinal = 0
    for source in sources:
        chunks = (
            chunks_by_source[source]
            if chunks_by_source is not None
            else _markdown_chunks(snapshot.text(source))
        )
        for heading, chunk in chunks:
            score = _reference_score(heading, chunk, tokens)
            if score:
                ranked.append((score, -ordinal, source, heading, chunk))
            ordinal += 1
    ranked.sort(reverse=True)

    hits: list[ReferenceHit] = []
    remaining = max_chars
    for score, _, source, heading, chunk in ranked:
        if len(hits) >= limit or remaining <= 0:
            break
        excerpt_limit = min(per_result_chars, remaining)
        excerpt = chunk.strip()[:excerpt_limit].rstrip()
        if not excerpt:
            continue
        hits.append(ReferenceHit(source=source, heading=heading, excerpt=excerpt, score=score))
        remaining -= len(excerpt)
    return tuple(hits)


class PersonaCatalog:
    """Restart-scoped catalog of valid persona sources and immutable snapshots."""

    def __init__(
        self,
        source_root: Path,
        snapshot_root: Path,
        *,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.source_root = Path(source_root)
        self.snapshot_root = Path(snapshot_root)
        self.schema_path = Path(schema_path)
        self._packages: dict[str, ValidatedPersonaPackage] | None = None
        self._snapshots: dict[str, ResolvedPersonaSnapshot] = {}
        self._stage_contexts: dict[tuple[str, InternalStage], PersonaStageContext] = {}
        self._reference_chunks: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}

    def discover(self) -> list[PersonaSummary]:
        packages = self._catalog()
        return [packages[persona_id].summary for persona_id in sorted(packages)]

    def get(self, persona_id: str) -> ValidatedPersonaPackage | None:
        return self._catalog().get(persona_id)

    def create_snapshot(self, persona_id: str) -> ResolvedPersonaSnapshot:
        package = self.get(persona_id)
        if package is None:
            raise PersonaPackageError("persona_not_found", "Persona is not selectable")
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        if self.snapshot_root.is_symlink():
            raise PersonaPackageError("unsafe_snapshot_root", "Snapshot root cannot be a symlink")

        snapshot_hash = package.summary.snapshot_sha256
        target = self.snapshot_root / snapshot_hash
        if target.exists():
            return self.resolve_snapshot(snapshot_hash)

        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_hash}.", dir=self.snapshot_root))
        try:
            for name in sorted(package._raw_entries):
                destination = temporary / name
                with destination.open("xb") as stream:
                    stream.write(package._raw_entries[name])
                    stream.flush()
                    os.fsync(stream.fileno())
            copied = validate_persona_package(temporary, schema_path=self.schema_path)
            if copied.summary.snapshot_sha256 != snapshot_hash:
                raise PersonaPackageError(
                    "snapshot_hash_mismatch", "Copied snapshot does not match source hash"
                )
            try:
                temporary.rename(target)
            except OSError:
                if not target.exists():
                    raise
            return self.resolve_snapshot(snapshot_hash)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def resolve_snapshot(self, snapshot_sha256: str) -> ResolvedPersonaSnapshot:
        snapshot = self._snapshots.get(snapshot_sha256)
        if snapshot is None:
            snapshot = resolve_persona_snapshot(
                self.snapshot_root, snapshot_sha256, schema_path=self.schema_path
            )
            self._snapshots[snapshot_sha256] = snapshot
        return snapshot

    def load_stage_context(
        self,
        snapshot_sha256: str,
        stage: InternalStage | str,
        *,
        max_chars: int = DEFAULT_STAGE_CONTEXT_CHARS,
    ) -> PersonaStageContext:
        resolved_stage = _resolve_stage(stage)
        cache_key = (snapshot_sha256, resolved_stage)
        context = self._stage_contexts.get(cache_key)
        if context is None:
            context = _project_stage_context(
                self.resolve_snapshot(snapshot_sha256),
                resolved_stage,
            )
            self._stage_contexts[cache_key] = context
        _enforce_context_limit(context, max_chars)
        return context

    def retrieve_references(
        self,
        snapshot_sha256: str,
        query: str,
        *,
        sources: tuple[str, ...] = ("l5", "l6"),
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        max_chars: int = DEFAULT_RETRIEVAL_MAX_CHARS,
        per_result_chars: int = DEFAULT_RETRIEVAL_RESULT_CHARS,
    ) -> tuple[ReferenceHit, ...]:
        snapshot = self.resolve_snapshot(snapshot_sha256)
        chunks_by_source: dict[str, tuple[tuple[str, str], ...]] = {}
        for source in sources:
            cache_key = (snapshot_sha256, source)
            chunks = self._reference_chunks.get(cache_key)
            if chunks is None and source in {"l5", "l6"}:
                chunks = _markdown_chunks(snapshot.text(source))
                self._reference_chunks[cache_key] = chunks
            if chunks is not None:
                chunks_by_source[source] = chunks
        return _retrieve_snapshot_references(
            snapshot,
            query,
            sources=sources,
            limit=limit,
            max_chars=max_chars,
            per_result_chars=per_result_chars,
            chunks_by_source=chunks_by_source,
        )

    def _catalog(self) -> dict[str, ValidatedPersonaPackage]:
        if self._packages is None:
            self._packages = self._scan()
        return self._packages

    def _scan(self) -> dict[str, ValidatedPersonaPackage]:
        if not self.source_root.exists() or not self.source_root.is_dir():
            return {}
        if self.source_root.is_symlink():
            return {}

        candidates: defaultdict[str, list[ValidatedPersonaPackage]] = defaultdict(list)
        for child in sorted(self.source_root.iterdir(), key=lambda path: path.name):
            if child.is_symlink() or not child.is_dir():
                continue
            try:
                package = validate_persona_package(child, schema_path=self.schema_path)
            except PersonaPackageError as exc:
                logger.warning(
                    "persona package rejected path=%s code=%s",
                    child.name,
                    exc.code,
                )
                continue
            if package.schema_version != CURRENT_PERSONA_SCHEMA:
                logger.warning(
                    "persona package rejected path=%s code=legacy_persona_not_selectable",
                    child.name,
                )
                continue
            candidates[package.summary.persona_id].append(package)
        selectable: dict[str, ValidatedPersonaPackage] = {}
        for persona_id, packages in candidates.items():
            if len(packages) == 1:
                selectable[persona_id] = packages[0]
                continue
            for package in packages:
                logger.warning(
                    "persona package rejected path=%s code=duplicate_persona_id",
                    package.path.name,
                )
        return selectable


def _validate_package_root(package_path: Path) -> None:
    if package_path.is_symlink() or not package_path.is_dir():
        raise PersonaPackageError("unsafe_package_path", "Persona package must be a real directory")


def _validate_package_directory(
    package_path: Path,
    schema_version: str,
    logical_files: tuple[tuple[str, str], ...],
) -> None:
    try:
        entries = tuple(package_path.iterdir())
    except OSError as exc:
        raise PersonaPackageError("package_unreadable", "Persona package cannot be read") from exc
    names = {entry.name for entry in entries}
    if (schema_version == PERSONA_SCHEMA_V2 and names.intersection({"l1.md", "l2.md"})) or (
        schema_version == PERSONA_SCHEMA_V1 and "soul.md" in names
    ):
        raise PersonaPackageError(
            "mixed_persona_schema", "Persona package cannot mix Soul with L1 or L2"
        )
    expected_entry_names = frozenset({MANIFEST_NAME, *(name for _, name in logical_files)})
    if names != expected_entry_names or len(entries) != len(expected_entry_names):
        raise PersonaPackageError(
            "invalid_package_entries",
            "Persona package entries do not match its schema version",
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise PersonaPackageError(
                "unsafe_package_entry", f"Persona package entry is not a regular file: {entry.name}"
            )


def _validate_contained_regular_file(package_path: Path, source_file: Path) -> None:
    if source_file.is_symlink() or not source_file.is_file():
        raise PersonaPackageError(
            "unsafe_file_path", f"Persona file is not a regular file: {source_file.name}"
        )
    package_resolved = package_path.resolve(strict=True)
    source_resolved = source_file.resolve(strict=True)
    if source_resolved.parent != package_resolved:
        raise PersonaPackageError(
            "unsafe_file_path", f"Persona file escaped its package: {source_file.name}"
        )


def _read_entry(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PersonaPackageError(
            "package_unreadable", f"Cannot read persona entry: {path.name}"
        ) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PersonaPackageError("invalid_manifest", f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _load_schema(schema_path: Path) -> Mapping[str, Any]:
    try:
        raw = Path(schema_path).read_text(encoding="utf-8")
        schema = json.loads(raw)
        Draft202012Validator.check_schema(schema)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
        raise PersonaPackageError(
            "manifest_schema_unavailable", "Persona manifest schema is unavailable"
        ) from exc
    return schema


def _freeze_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_mapping(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_mapping(item) for item in value)
    return value


def _normalize_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\s·:：()（）\[\]【】_\-/—]+", "", normalized)


def _headings(text: str) -> tuple[tuple[int, str], ...]:
    result: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            result.append((len(match.group(1)), match.group(2).strip()))
    return tuple(result)


def _heading_matches(heading: str, aliases: tuple[str, ...]) -> bool:
    normalized = _normalize_heading(heading)
    return any(_normalize_heading(alias) in normalized for alias in aliases)


def _require_heading_groups(
    logical_name: str,
    headings: tuple[tuple[int, str], ...],
    groups: tuple[tuple[str, ...], ...],
) -> None:
    missing = [
        aliases[0]
        for aliases in groups
        if not any(_heading_matches(heading, aliases) for _, heading in headings)
    ]
    if missing:
        raise PersonaPackageError(
            "required_heading_missing",
            f"{logical_name}.md is missing required heading: {missing[0]}",
        )


def _validate_required_structure(
    logical_name: str,
    text: str,
    *,
    schema_version: str,
) -> None:
    if not text.strip():
        raise PersonaPackageError("empty_markdown", f"{logical_name}.md cannot be empty")
    headings = _headings(text)
    if not headings:
        raise PersonaPackageError(
            "required_heading_missing", f"{logical_name}.md must use Markdown headings"
        )

    paradigm_layers = (("L1",), ("L2",)) if schema_version == PERSONA_SCHEMA_V1 else (("Soul",),)
    project_layers = (("L1",), ("L2",)) if schema_version == PERSONA_SCHEMA_V1 else (("Soul",),)
    common: dict[str, tuple[tuple[str, ...], ...]] = {
        "paradigm": (
            ("L0",),
            *paradigm_layers,
            ("L3",),
            ("L4",),
            ("L5",),
            ("L6",),
            ("变体", "variant"),
            ("雷区", "红线", "redline"),
            ("温度", "temperature"),
            ("仲裁", "arbitration"),
            ("情境翻译", "translation"),
            ("反馈", "feedback"),
            ("留白", "blankspace"),
            ("验收", "闸", "gate"),
            ("纪律", "discipline"),
            ("归属", "判断权", "ownership"),
        ),
        "project": (
            ("身份", "identity"),
            ("L0",),
            ("变体", "variant"),
            ("雷区", "红线", "redline"),
            ("温度", "temperature"),
            *project_layers,
            ("L3",),
            ("L4",),
            ("L5",),
            ("L6",),
            ("铁律", "ironrule"),
            ("仲裁", "arbitration"),
            ("工作步骤", "工作流", "流程", "workflow"),
            ("验收", "闸", "gate"),
            ("反馈", "feedback"),
        ),
        "l0": (
            ("变体", "variant"),
            ("雷区", "红线", "redline"),
            ("温度", "temperature"),
        ),
        "l1": (("来源画像", "先天刻画", "原局", "profile"), ("摘要", "summary")),
        "l2": (("来源画像", "先天刻画", "星盘", "profile"), ("摘要", "summary")),
        "soul": (
            ("身份", "identity"),
            ("观察与表达", "观察", "表达", "observation", "expression"),
            ("创作能量", "能量", "creativeenergy"),
            ("生产性张力", "张力", "tension"),
            ("避免", "avoid"),
            ("权限与仲裁", "权限", "仲裁", "authority", "arbitration"),
        ),
        "l3": (
            ("手法", "method"),
            ("认知", "cognition"),
            ("短板", "不足", "shortcoming"),
            ("摘要", "summary"),
        ),
        "l4": (
            ("L4-A", "价值观", "values"),
            ("L4-B", "短剧技艺", "craft"),
            ("故事大纲", "storyoutline"),
            ("人物小传", "characterbiographies"),
            ("人物关系", "relationship"),
            ("分集大纲", "episodeoutline"),
            ("分集剧本", "episodescripts"),
            ("参数", "parameter"),
        ),
        "l5": (("作品", "works"), ("经历", "experience")),
        "l6": (("外部技法", "外部学习", "externalcraft"), ("条目", "entry")),
    }
    _require_heading_groups(logical_name, headings, common[logical_name])
    if logical_name == "l0":
        _validate_l0_markers(text)
    if logical_name == "project":
        _validate_project_statuses(text, schema_version=schema_version)
    if logical_name == "soul":
        _validate_soul(text)


def _validate_l0_markers(text: str) -> None:
    concepts = {
        "variant": ("变体", "variant"),
        "redline": ("雷区", "红线", "redline"),
        "temperature": ("温度", "temperature"),
    }
    found: dict[str, int] = {concept: 0 for concept in concepts}
    active: str | None = None
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            active = next(
                (
                    concept
                    for concept, aliases in concepts.items()
                    if _heading_matches(heading.group(2), aliases)
                ),
                None,
            )
            continue
        item = _LIST_ITEM_RE.match(line)
        if active and item:
            value = item.group(1)
            if not _STATUS_RE.search(value) or not _OWNERSHIP_RE.search(value):
                raise PersonaPackageError(
                    "l0_marker_invalid",
                    "Every L0 variant, red-line, and temperature item needs "
                    "status and ownership markers",
                )
            found[active] += 1
    if any(count == 0 for count in found.values()):
        raise PersonaPackageError(
            "l0_marker_missing",
            "L0 variants, red lines, and temperature each need at least one marked item",
        )
    extract_l0_variant_ids(text)


def extract_l0_variant_ids(text: str) -> tuple[str, ...]:
    """Return explicit IDs from confirmed L0 variant items, if the persona declares them."""

    variant_items: list[str] = []
    active = False
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            active = _heading_matches(heading.group(2), ("变体", "variant"))
            continue
        item = _LIST_ITEM_RE.match(line)
        if active and item and not _PENDING_RE.search(item.group(1)):
            variant_items.append(item.group(1))

    ids: list[str] = []
    for item in variant_items:
        markers = tuple(_L0_VARIANT_ID_RE.finditer(item))
        prefixes = tuple(_L0_VARIANT_ID_PREFIX_RE.finditer(item))
        if prefixes and (len(prefixes) != 1 or len(markers) != 1):
            raise PersonaPackageError(
                "l0_variant_id_invalid",
                "Every explicit L0 variant ID must use one [ID:<value>] marker",
            )
        if markers:
            ids.append(markers[0].group(1))

    if ids and len(ids) != len(variant_items):
        raise PersonaPackageError(
            "l0_variant_id_missing",
            "Every confirmed L0 variant needs an ID when explicit IDs are used",
        )
    if len(ids) != len(set(ids)):
        raise PersonaPackageError(
            "l0_variant_id_duplicate",
            "Explicit L0 variant IDs must be unique",
        )
    return tuple(ids)


def _validate_project_statuses(text: str, *, schema_version: str) -> None:
    layers: tuple[str, ...]
    if schema_version == PERSONA_SCHEMA_V1:
        layers = tuple(f"L{layer}" for layer in range(1, 7))
    else:
        layers = ("Soul", "L3", "L4", "L5", "L6")
    for layer in layers:
        section = _extract_required_section(text, ((layer,),), f"project {layer} summary")
        if not _STATUS_RE.search(section):
            raise PersonaPackageError(
                "project_status_missing", f"Project {layer} summary needs a status marker"
            )


def _validate_soul(text: str) -> None:
    if len(text) > MAX_SOUL_CHARS:
        raise PersonaPackageError(
            "soul_too_large", f"Soul exceeds the {MAX_SOUL_CHARS}-character limit"
        )
    if _PENDING_RE.search(text):
        raise PersonaPackageError("soul_pending", "Soul contains an unconfirmed marker")
    if not _STATUS_RE.search(text) or not _OWNERSHIP_RE.search(text):
        raise PersonaPackageError(
            "soul_marker_invalid", "Soul needs confirmed status and ownership markers"
        )


def _extract_required_section(
    text: str,
    aliases_group: tuple[tuple[str, ...], ...],
    label: str,
) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        if not all(_heading_matches(match.group(2), aliases) for aliases in aliases_group):
            continue
        level = len(match.group(1))
        collected = [line]
        for following in lines[index + 1 :]:
            following_heading = _HEADING_RE.match(following)
            if following_heading and len(following_heading.group(1)) <= level:
                break
            collected.append(following)
        section = "\n".join(collected).strip()
        if section:
            return section
    raise PersonaPackageError("required_section_missing", f"Missing required section: {label}")


def _strip_pending_blocks(text: str) -> str:
    output: list[str] = []
    skipping = False
    for line in text.splitlines():
        if _HEADING_RE.match(line):
            skipping = False
            output.append(line)
            continue
        item = _LIST_ITEM_RE.match(line)
        if item:
            skipping = bool(_PENDING_RE.search(item.group(1)))
            if not skipping:
                output.append(line)
            continue
        if not skipping:
            output.append(line)
    return "\n".join(output).strip()


_STAGE_L4_ALIASES: dict[InternalStage, tuple[tuple[str, ...], ...] | None] = {
    InternalStage.LOADING_PERSONA: None,
    InternalStage.SELECTING_L0_VARIANT: None,
    InternalStage.GENERATING_STORY_OUTLINE: (("故事大纲", "storyoutline"),),
    InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: (
        ("人物小传", "characterbiographies"),
        ("人物关系", "relationship"),
    ),
    InternalStage.GENERATING_EPISODE_OUTLINE: (("分集大纲", "episodeoutline"),),
    InternalStage.GENERATING_EPISODE_SCRIPTS: (("分集剧本", "episodescripts"),),
    InternalStage.ACCEPTING_L0: None,
    InternalStage.ACCEPTING_L4: (("L4-B", "短剧技艺", "craft"),),
    InternalStage.ASSEMBLING_DELIVERY: None,
}


def _stage_l4_context(text: str, stage: InternalStage) -> str:
    sections = [_extract_required_section(text, (("L4-A", "价值观", "values"),), "L4-A values")]
    if stage is InternalStage.ACCEPTING_L4:
        return text.strip()
    aliases = _STAGE_L4_ALIASES[stage]
    if aliases:
        for alias_group in aliases:
            label = f"L4 {stage.value} {alias_group[0]}"
            sections.append(_extract_required_section(text, (alias_group,), label))
        sections.append(
            _extract_required_section(text, (("参数", "parameter"),), "L4 numeric parameters")
        )
    return "\n\n".join(dict.fromkeys(sections))


def _query_tokens(query: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    tokens = tuple(dict.fromkeys(token for token in _WORD_RE.findall(normalized) if token))
    return tokens or (normalized,)


def _markdown_chunks(text: str, *, chunk_chars: int = 2_000) -> tuple[tuple[str, str], ...]:
    chunks: list[tuple[str, str]] = []
    heading = "未命名条目"
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if not content:
            return
        for start in range(0, len(content), chunk_chars):
            chunks.append((heading, content[start : start + chunk_chars]))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)
    flush()
    return tuple(chunks)


def _reference_score(heading: str, chunk: str, tokens: tuple[str, ...]) -> int:
    normalized_heading = unicodedata.normalize("NFKC", heading).casefold()
    normalized_chunk = unicodedata.normalize("NFKC", chunk).casefold()
    return sum(
        normalized_chunk.count(token) + (3 * normalized_heading.count(token)) for token in tokens
    )

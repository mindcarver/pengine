from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"\S")]
StableId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]

_NUMERIC_KINDS = {"duration", "count", "amount", "measurement"}
_TEMPORAL_KINDS = {"date", "time", "datetime"}
_TEMPORAL_TOKEN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|(?<!\d)\d{1,2}:\d{2}(?!\d)")
_CHINESE_DIGIT_CHARS = "零〇一二两三四五六七八九"
_CHINESE_NUMBER_CHARS = f"{_CHINESE_DIGIT_CHARS}十百千万亿点"
_NUMBER_TEXT = rf"(?:-?\d+(?:\.\d+)?|[{_CHINESE_NUMBER_CHARS}]+)"
_INTEGER_TEXT = rf"(?:\d+|[{_CHINESE_DIGIT_CHARS}十百千万亿]+)"
_CHINESE_DATE = re.compile(
    rf"(?P<year>{_INTEGER_TEXT})年"
    rf"(?P<month>{_INTEGER_TEXT})月"
    rf"(?P<day>{_INTEGER_TEXT})[日号]"
)
_CHINESE_TIME = re.compile(
    rf"(?P<hour>{_INTEGER_TEXT})点"
    rf"(?:(?P<minute>{_INTEGER_TEXT})分|整)"
)
_MEASURED_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9_.第{_CHINESE_NUMBER_CHARS}])({_NUMBER_TEXT})\s*"
    r"(分钟|分|小时|年|天|元|米|厘米|毫米|岁|人|次|件|集|%)"
)
_SPEAKER = re.compile(
    r"^\s*([A-Za-z\u3400-\u9fff][A-Za-z0-9_·\u3400-\u9fff]{0,15})"
    r"(?:\s*[（(][^）)\r\n]{0,40}[）)])?\s*[：:]"
)
_NON_CHARACTER_LABELS = {
    "画面",
    "字幕",
    "场景",
    "地点",
    "时间",
    "音效",
    "旁白",
    "同期声",
    "动作",
    "镜头",
}


class ContinuityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterSpec(ContinuityModel):
    character_id: StableId
    name: NonEmptyText
    role: NonEmptyText
    initial_known_fact_ids: list[StableId] = Field(default_factory=list)


class RelationshipSpec(ContinuityModel):
    source_character_id: StableId
    target_character_id: StableId
    relation: NonEmptyText


class StoryFact(ContinuityModel):
    fact_id: StableId
    subject: NonEmptyText
    predicate: NonEmptyText
    kind: Literal[
        "date",
        "time",
        "datetime",
        "duration",
        "count",
        "amount",
        "measurement",
        "text",
    ]
    value: NonEmptyText
    unit: NonEmptyText | None = None
    first_revealed_episode: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_typed_value(self) -> StoryFact:
        if self.kind in _NUMERIC_KINDS:
            try:
                value = Decimal(self.value)
            except InvalidOperation as exc:
                raise ValueError("Numeric facts require an exact decimal value") from exc
            if not value.is_finite() or self.unit is None:
                raise ValueError("Numeric facts require a finite value and explicit unit")
        elif self.unit is not None:
            raise ValueError("Non-numeric facts cannot declare a unit")
        if self.kind in _TEMPORAL_KINDS:
            _parse_temporal(self.kind, self.value)
        return self


class TimelineEvent(ContinuityModel):
    event_id: StableId
    order: int = Field(ge=1)
    when: NonEmptyText
    participant_ids: list[StableId] = Field(default_factory=list)
    fact_ids: list[StableId] = Field(default_factory=list)


class CharacterKnowledgeState(ContinuityModel):
    episode_number: int = Field(ge=1)
    character_id: StableId
    known_fact_ids: list[StableId] = Field(default_factory=list)


class ClueSpec(ContinuityModel):
    clue_id: StableId
    description: NonEmptyText
    introduced_episode: int = Field(ge=1)
    explained_episode: int = Field(ge=1)
    callback_episode: int | None = Field(default=None, ge=1)
    introduction_is_visible_or_audible: bool

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ClueSpec:
        if self.explained_episode < self.introduced_episode:
            raise ValueError("A clue cannot be explained before it is introduced")
        if self.callback_episode is not None and self.callback_episode < self.explained_episode:
            raise ValueError("A clue callback cannot precede its explanation")
        if not self.introduction_is_visible_or_audible:
            raise ValueError("A locked clue introduction must be visible or audible")
        return self


class EpisodeObligation(ContinuityModel):
    obligation_id: StableId
    episode_number: int = Field(ge=1)
    new_information_fact_ids: list[StableId] = Field(min_length=1)
    end_hook: NonEmptyText
    required_clue_ids: list[StableId] = Field(default_factory=list)


class StoryContract(ContinuityModel):
    version: int = Field(default=1, ge=1)
    episode_count: int = Field(ge=1)
    characters: list[CharacterSpec] = Field(min_length=1)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    facts: list[StoryFact] = Field(min_length=1)
    timeline: list[TimelineEvent] = Field(min_length=1)
    knowledge_states: list[CharacterKnowledgeState] = Field(default_factory=list)
    clues: list[ClueSpec] = Field(default_factory=list)
    prohibitions: list[NonEmptyText] = Field(default_factory=list)
    episode_obligations: list[EpisodeObligation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references_and_invariants(self) -> StoryContract:
        character_ids = _unique_ids(
            [item.character_id for item in self.characters],
            "character",
        )
        names = [item.name for item in self.characters]
        if len(names) != len(set(names)):
            raise ValueError("Character names must be unique")

        fact_ids = _unique_ids([item.fact_id for item in self.facts], "fact")
        clue_ids = _unique_ids([item.clue_id for item in self.clues], "clue")
        _unique_ids([item.event_id for item in self.timeline], "timeline event")
        _unique_ids(
            [item.obligation_id for item in self.episode_obligations],
            "episode obligation",
        )

        for character in self.characters:
            _require_subset(
                character.initial_known_fact_ids,
                fact_ids,
                f"initial knowledge for {character.character_id}",
            )
            character.initial_known_fact_ids = sorted(set(character.initial_known_fact_ids))
        for relationship in self.relationships:
            _require_subset(
                [relationship.source_character_id, relationship.target_character_id],
                character_ids,
                "relationship characters",
            )
            if relationship.source_character_id == relationship.target_character_id:
                raise ValueError("A relationship must connect two different characters")

        for fact in self.facts:
            if fact.first_revealed_episode > self.episode_count:
                raise ValueError("A fact reveal episode exceeds the contract episode count")

        expected_orders = list(range(1, len(self.timeline) + 1))
        if [event.order for event in self.timeline] != expected_orders:
            raise ValueError("Timeline events must be ordered and contiguous from 1")
        previous_temporal: date | datetime | time | None = None
        for event in self.timeline:
            _require_subset(event.participant_ids, character_ids, f"timeline {event.event_id}")
            _require_subset(event.fact_ids, fact_ids, f"timeline {event.event_id}")
            current_temporal = _parse_any_temporal(event.when)
            if (
                previous_temporal is not None
                and current_temporal is not None
                and type(previous_temporal) is type(current_temporal)
                and current_temporal < previous_temporal
            ):
                raise ValueError("Timeline timestamps contradict their declared order")
            previous_temporal = current_temporal or previous_temporal

        supplied_knowledge: dict[tuple[int, str], CharacterKnowledgeState] = {}
        for state in self.knowledge_states:
            if state.episode_number > self.episode_count:
                raise ValueError("A knowledge state episode exceeds the contract episode count")
            _require_subset([state.character_id], character_ids, "character knowledge")
            _require_subset(state.known_fact_ids, fact_ids, "character knowledge")
            pair = (state.episode_number, state.character_id)
            if pair in supplied_knowledge:
                raise ValueError("Duplicate knowledge state entries are not allowed")
            supplied_knowledge[pair] = state

        prior_knowledge = {
            character.character_id: list(character.initial_known_fact_ids)
            for character in self.characters
        }
        completed_knowledge: list[CharacterKnowledgeState] = []
        for episode in range(1, self.episode_count + 1):
            for character in self.characters:
                character_id = character.character_id
                supplied = supplied_knowledge.get((episode, character_id))
                current = (
                    sorted(set(supplied.known_fact_ids))
                    if supplied is not None
                    else list(prior_knowledge[character_id])
                )
                if not set(prior_knowledge[character_id]).issubset(current):
                    raise ValueError(
                        "Character knowledge cannot silently disappear between episodes"
                    )
                completed_knowledge.append(
                    CharacterKnowledgeState(
                        episode_number=episode,
                        character_id=character_id,
                        known_fact_ids=current,
                    )
                )
                prior_knowledge[character_id] = current
        self.knowledge_states = completed_knowledge

        for clue in self.clues:
            if (
                max(
                    clue.introduced_episode,
                    clue.explained_episode,
                    clue.callback_episode or 1,
                )
                > self.episode_count
            ):
                raise ValueError("A clue lifecycle exceeds the contract episode count")

        obligations_by_episode = {
            obligation.episode_number: obligation for obligation in self.episode_obligations
        }
        if set(obligations_by_episode) != set(range(1, self.episode_count + 1)) or len(
            obligations_by_episode
        ) != len(self.episode_obligations):
            raise ValueError("Every episode requires exactly one obligation")
        facts_by_episode: dict[int, set[str]] = {
            episode: set() for episode in range(1, self.episode_count + 1)
        }
        for fact in self.facts:
            facts_by_episode[fact.first_revealed_episode].add(fact.fact_id)
        for obligation in self.episode_obligations:
            _require_subset(
                obligation.new_information_fact_ids,
                fact_ids,
                f"obligation {obligation.obligation_id}",
            )
            if (
                set(obligation.new_information_fact_ids)
                != facts_by_episode[obligation.episode_number]
            ):
                raise ValueError(
                    "Episode new-information obligations must match fact reveal episodes"
                )
            _require_subset(
                obligation.required_clue_ids,
                clue_ids,
                f"obligation {obligation.obligation_id}",
            )
        if len(self.prohibitions) != len(set(self.prohibitions)):
            raise ValueError("Contract prohibitions must be unique")
        return self


class ReviewIssue(ContinuityModel):
    code: StableId
    message: NonEmptyText
    contract_refs: list[StableId] = Field(default_factory=list)
    script_excerpt: NonEmptyText | None = None


class SemanticReview(ContinuityModel):
    passed: bool
    evidence: NonEmptyText
    issues: list[ReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> SemanticReview:
        if self.passed and self.issues:
            raise ValueError("A passing review cannot contain issues")
        if not self.passed and not self.issues:
            raise ValueError("A failed review requires at least one issue")
        return self


class KnowledgeGain(ContinuityModel):
    character_id: StableId
    fact_ids: list[StableId] = Field(min_length=1)


class ScriptEvidence(ContinuityModel):
    target_id: StableId
    excerpt: NonEmptyText


class EpisodeStateDelta(ContinuityModel):
    episode_number: int = Field(ge=1)
    contract_sha256: Sha256
    established_fact_ids: list[StableId] = Field(default_factory=list)
    knowledge_gains: list[KnowledgeGain] = Field(default_factory=list)
    introduced_clue_ids: list[StableId] = Field(default_factory=list)
    resolved_clue_ids: list[StableId] = Field(default_factory=list)
    satisfied_obligation_ids: list[StableId] = Field(default_factory=list)
    evidence: list[ScriptEvidence] = Field(default_factory=list)
    handoff: NonEmptyText


class CharacterKnowledge(ContinuityModel):
    character_id: StableId
    known_fact_ids: list[StableId] = Field(default_factory=list)


class SeriesState(ContinuityModel):
    contract_sha256: Sha256
    locked_through_episode: int = Field(ge=0)
    established_fact_ids: list[StableId] = Field(default_factory=list)
    character_knowledge: list[CharacterKnowledge] = Field(default_factory=list)
    introduced_clue_ids: list[StableId] = Field(default_factory=list)
    resolved_clue_ids: list[StableId] = Field(default_factory=list)
    handoff: str = ""


class EpisodeLock(ContinuityModel):
    episode_number: int = Field(ge=1)
    content: NonEmptyText
    content_sha256: Sha256
    contract_sha256: Sha256
    state_delta: EpisodeStateDelta
    series_state: SeriesState
    series_state_sha256: Sha256
    semantic_review: SemanticReview
    repair_rounds: int = Field(ge=0, le=2)


class ContinuityViolation(ValueError):
    def __init__(self, issues: list[ReviewIssue]) -> None:
        super().__init__("Episode continuity validation failed")
        self.issues = issues

    @property
    def evidence(self) -> str:
        return "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)


def story_contract_sha256(contract: StoryContract) -> str:
    return canonical_model_hash(contract)


def canonical_model_hash(value: BaseModel) -> str:
    encoded = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_story_contract_markdown(contract: StoryContract, contract_sha256: str) -> str:
    lines = [
        "# Story Contract",
        "",
        f"- Version: {contract.version}",
        f"- SHA-256: `{contract_sha256}`",
        f"- Episodes: {contract.episode_count}",
        "",
        "## Characters",
        "",
    ]
    lines.extend(
        f"- `{character.character_id}` {character.name}: {character.role}"
        for character in contract.characters
    )
    lines.extend(["", "## Locked facts", ""])
    lines.extend(
        (
            f"- `{fact.fact_id}` Ep{fact.first_revealed_episode}: "
            f"{fact.subject} {fact.predicate} = {fact.value}"
            f"{f' {fact.unit}' if fact.unit else ''} ({fact.kind})"
        )
        for fact in contract.facts
    )
    lines.extend(["", "## Timeline", ""])
    lines.extend(f"{event.order}. `{event.event_id}` {event.when}" for event in contract.timeline)
    lines.extend(["", "## Clues", ""])
    lines.extend(
        f"- `{clue.clue_id}`: Ep{clue.introduced_episode} introduced; "
        f"Ep{clue.explained_episode} explained"
        + (f"; Ep{clue.callback_episode} callback" if clue.callback_episode else "")
        for clue in contract.clues
    )
    lines.extend(["", "## Episode obligations", ""])
    lines.extend(
        (
            f"- Ep{item.episode_number} `{item.obligation_id}`: "
            f"new facts {', '.join(item.new_information_fact_ids)}; hook {item.end_hook}"
        )
        for item in contract.episode_obligations
    )
    lines.extend(["", "## Prohibitions", ""])
    lines.extend(f"- {item}" for item in contract.prohibitions)
    return "\n".join(lines).rstrip() + "\n"


def initial_series_state(contract: StoryContract, contract_sha256: str) -> SeriesState:
    return SeriesState(
        contract_sha256=contract_sha256,
        locked_through_episode=0,
        character_knowledge=[
            CharacterKnowledge(
                character_id=character.character_id,
                known_fact_ids=sorted(character.initial_known_fact_ids),
            )
            for character in contract.characters
        ],
    )


def build_episode_lock(
    *,
    contract: StoryContract,
    contract_sha256: str,
    prior_state: SeriesState,
    content: str,
    delta: EpisodeStateDelta,
    semantic_review: SemanticReview,
    repair_rounds: int,
) -> EpisodeLock:
    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_sha256,
        prior_state=prior_state,
        content=content,
        delta=delta,
    )
    if issues:
        raise ContinuityViolation(issues)
    episode_number = delta.episode_number
    knowledge = {
        item.character_id: set(item.known_fact_ids) for item in prior_state.character_knowledge
    }
    for gain in delta.knowledge_gains:
        knowledge[gain.character_id].update(gain.fact_ids)
    state = SeriesState(
        contract_sha256=contract_sha256,
        locked_through_episode=episode_number,
        established_fact_ids=sorted(
            set(prior_state.established_fact_ids) | set(delta.established_fact_ids)
        ),
        character_knowledge=[
            CharacterKnowledge(
                character_id=character.character_id,
                known_fact_ids=sorted(knowledge[character.character_id]),
            )
            for character in contract.characters
        ],
        introduced_clue_ids=sorted(
            set(prior_state.introduced_clue_ids) | set(delta.introduced_clue_ids)
        ),
        resolved_clue_ids=sorted(set(prior_state.resolved_clue_ids) | set(delta.resolved_clue_ids)),
        handoff=delta.handoff,
    )
    return EpisodeLock(
        episode_number=episode_number,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        contract_sha256=contract_sha256,
        state_delta=delta,
        series_state=state,
        series_state_sha256=canonical_model_hash(state),
        semantic_review=semantic_review,
        repair_rounds=repair_rounds,
    )


def validate_episode_candidate(
    *,
    contract: StoryContract,
    contract_sha256: str,
    prior_state: SeriesState,
    content: str,
    delta: EpisodeStateDelta,
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    episode = delta.episode_number
    if delta.contract_sha256 != contract_sha256 or prior_state.contract_sha256 != contract_sha256:
        issues.append(_issue("contract_hash_mismatch", "本集未绑定已锁定的创作合同"))
    if episode != prior_state.locked_through_episode + 1 or episode > contract.episode_count:
        issues.append(_issue("episode_order_mismatch", "本集不是当前首个未锁定集"))
        return issues

    expected_facts = {
        fact.fact_id for fact in contract.facts if fact.first_revealed_episode == episode
    }
    if set(delta.established_fact_ids) != expected_facts:
        issues.append(
            _issue(
                "fact_delta_mismatch",
                "本集确立的事实与锁定的揭示计划不一致",
                sorted(expected_facts),
            )
        )

    prior_knowledge = {
        item.character_id: set(item.known_fact_ids) for item in prior_state.character_knowledge
    }
    expected_after = {
        state.character_id: set(state.known_fact_ids)
        for state in contract.knowledge_states
        if state.episode_number == episode
    }
    actual_gains = {gain.character_id: set(gain.fact_ids) for gain in delta.knowledge_gains}
    if len(actual_gains) != len(delta.knowledge_gains):
        issues.append(_issue("duplicate_knowledge_gain", "人物知识增量出现重复角色"))
    unknown_gain_characters = set(actual_gains) - set(prior_knowledge)
    if unknown_gain_characters:
        issues.append(
            _issue(
                "unknown_knowledge_character",
                "人物知识增量引用了锁定角色表之外的角色",
                sorted(unknown_gain_characters),
            )
        )
    for character in contract.characters:
        expected_gain = (
            expected_after[character.character_id] - prior_knowledge[character.character_id]
        )
        if actual_gains.get(character.character_id, set()) != expected_gain:
            issues.append(
                _issue(
                    "knowledge_state_mismatch",
                    f"人物 {character.character_id} 的知识变化与合同不一致",
                    sorted(expected_gain),
                )
            )

    expected_introduced = {
        clue.clue_id for clue in contract.clues if clue.introduced_episode == episode
    }
    expected_resolved = {
        clue.clue_id for clue in contract.clues if clue.explained_episode == episode
    }
    if set(delta.introduced_clue_ids) != expected_introduced:
        issues.append(
            _issue(
                "clue_introduction_mismatch",
                "线索引入情况与合同不一致",
                sorted(expected_introduced),
            )
        )
    if set(delta.resolved_clue_ids) != expected_resolved:
        issues.append(
            _issue(
                "clue_resolution_mismatch",
                "线索解释情况与合同不一致",
                sorted(expected_resolved),
            )
        )

    obligation = next(
        item for item in contract.episode_obligations if item.episode_number == episode
    )
    if set(delta.satisfied_obligation_ids) != {obligation.obligation_id}:
        issues.append(
            _issue(
                "obligation_mismatch",
                "本集未满足锁定的分集义务",
                [obligation.obligation_id],
            )
        )

    expected_evidence = (
        expected_facts | expected_introduced | expected_resolved | {obligation.obligation_id}
    )
    evidence = {item.target_id: item.excerpt for item in delta.evidence}
    if len(evidence) != len(delta.evidence) or set(evidence) != expected_evidence:
        issues.append(
            _issue(
                "evidence_coverage_mismatch",
                "剧本证据未覆盖全部必需事实、线索和分集义务",
                sorted(expected_evidence),
            )
        )
    for target_id, excerpt in evidence.items():
        if excerpt not in content:
            issues.append(
                _issue(
                    "evidence_not_in_script",
                    f"证据 {target_id} 未逐字出现在剧本中",
                    [target_id],
                )
            )

    allowed_temporal: set[str] = set()
    for fact in contract.facts:
        if fact.kind not in _TEMPORAL_KINDS:
            continue
        allowed_temporal.add(fact.value)
        temporal_value = _parse_temporal(fact.kind, fact.value)
        if isinstance(temporal_value, datetime):
            allowed_temporal.add(temporal_value.date().isoformat())
            allowed_temporal.add(temporal_value.time().isoformat(timespec="minutes"))
    temporal_tokens = _temporal_tokens(content)
    for token, normalized, _ in temporal_tokens:
        if normalized not in allowed_temporal:
            issues.append(
                _issue(
                    "uncontracted_time",
                    f"剧本包含合同外日期或时间 {token}（{normalized}）",
                )
            )

    allowed_measured = {
        (_normalized_decimal(fact.value), fact.unit or "")
        for fact in contract.facts
        if fact.kind in _NUMERIC_KINDS
    }
    for fact in contract.facts:
        if fact.kind not in {"date", "datetime"}:
            continue
        temporal_value = _parse_temporal(fact.kind, fact.value)
        allowed_measured.add((_normalized_decimal(str(temporal_value.year)), "年"))
    allowed_measured.add((_normalized_decimal(str(episode)), "集"))
    temporal_spans = [span for _, _, span in temporal_tokens]
    for match in _MEASURED_NUMBER.finditer(content):
        if any(_spans_overlap(match.span(), temporal_span) for temporal_span in temporal_spans):
            continue
        raw_value, unit = match.groups()
        value = _normalized_number(raw_value)
        if value is None:
            continue
        if (value, unit) not in allowed_measured:
            issues.append(
                _issue(
                    "uncontracted_number",
                    f"剧本包含合同外计量值 {raw_value}{unit}（{value}{unit}）",
                )
            )

    character_names = {character.name for character in contract.characters}
    for line in content.splitlines():
        match = _SPEAKER.match(line)
        if match and match.group(1) not in character_names | _NON_CHARACTER_LABELS:
            issues.append(
                _issue(
                    "unknown_speaker",
                    f"剧本引入了锁定角色表之外的说话人 {match.group(1)}",
                )
            )
    return issues


def _issue(code: str, message: str, refs: list[str] | None = None) -> ReviewIssue:
    return ReviewIssue(code=code, message=message, contract_refs=refs or [])


def _temporal_tokens(content: str) -> list[tuple[str, str, tuple[int, int]]]:
    tokens = [
        (match.group(0), match.group(0), match.span())
        for match in _TEMPORAL_TOKEN.finditer(content)
    ]
    for match in _CHINESE_DATE.finditer(content):
        year = _normalized_integer(match.group("year"))
        month = _normalized_integer(match.group("month"))
        day = _normalized_integer(match.group("day"))
        if year is None or month is None or day is None:
            continue
        try:
            normalized = date(year, month, day).isoformat()
        except ValueError:
            continue
        tokens.append((match.group(0), normalized, match.span()))
    for match in _CHINESE_TIME.finditer(content):
        hour = _normalized_integer(match.group("hour"))
        minute = _normalized_integer(match.group("minute")) if match.group("minute") else 0
        if hour is None or minute is None:
            continue
        try:
            normalized = time(hour, minute).isoformat(timespec="minutes")
        except ValueError:
            continue
        tokens.append((match.group(0), normalized, match.span()))
    return tokens


def _normalized_number(value: str) -> str | None:
    try:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            return _normalized_decimal(value)
        if "点" in value:
            integer, fractional = value.split("点", 1)
            integer_value = _normalized_integer(integer)
            if (
                integer_value is None
                or not fractional
                or any(character not in _CHINESE_DIGIT_CHARS for character in fractional)
            ):
                return None
            decimal_digits = "".join(str(_CHINESE_DIGITS[character]) for character in fractional)
            return _normalized_decimal(f"{integer_value}.{decimal_digits}")
        integer_value = _normalized_integer(value)
        return _normalized_decimal(str(integer_value)) if integer_value is not None else None
    except (InvalidOperation, ValueError):
        return None


def _normalized_decimal(value: str) -> str:
    return str(Decimal(value).normalize())


_CHINESE_DIGITS = {
    character: value
    for value, characters in enumerate(
        ("零〇", "一", "二两", "三", "四", "五", "六", "七", "八", "九")
    )
    for character in characters
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}


def _normalized_integer(value: str) -> int | None:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if all(character in _CHINESE_DIGITS for character in value):
        return int("".join(str(_CHINESE_DIGITS[character]) for character in value))
    if any(
        character not in _CHINESE_DIGITS | _CHINESE_SMALL_UNITS | _CHINESE_LARGE_UNITS
        for character in value
    ):
        return None

    total = 0
    section = 0
    number = 0
    for character in value:
        if character in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[character]
        elif character in _CHINESE_SMALL_UNITS:
            section += (number or 1) * _CHINESE_SMALL_UNITS[character]
            number = 0
        else:
            section += number
            total += (section or 1) * _CHINESE_LARGE_UNITS[character]
            section = 0
            number = 0
    return total + section + number


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _unique_ids(values: list[str], label: str) -> set[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} identifiers are not allowed")
    return set(values)


def _require_subset(values: list[str], allowed: set[str], label: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{label} references unknown identifiers: {sorted(unknown)}")


def _parse_temporal(kind: str, value: str) -> date | datetime | time:
    try:
        if kind == "date":
            return date.fromisoformat(value)
        if kind == "time":
            return time.fromisoformat(value)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {kind} value") from exc


def _parse_any_temporal(value: str) -> date | datetime | time | None:
    for kind in ("datetime", "date", "time"):
        try:
            return _parse_temporal(kind, value)
        except ValueError:
            continue
    return None

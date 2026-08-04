"""Standalone Pengine server for browser acceptance (Delivery #58 INT-A11).

Runs a real uvicorn server over an isolated SQLite database with a controllable
deterministic unified workflow so the workbench can be exercised in a real browser
against API + SQLite authority.

Usage:
    PYTHONPATH=tests python tests/live/browser_server.py --scenario a8_suffix --port 8765

Scenarios:
- ``a8_suffix``  : a persistent script defect exhausts the automatic suffix budget
                   and pauses at repair_authorization so the workbench shows the
                   pause evidence and the authorize-repair action.
- ``a7_context`` : an over-context request pauses at context_budget with the
                   required-versus-limit evidence.
- ``happy_path`` : a normal unified run completes to a formal delivery.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from test_unified_integration import UnifiedWorkflow  # noqa: E402

from pengine.api import create_app  # noqa: E402
from pengine.config import Settings  # noqa: E402
from pengine.personas import PersonaCatalog  # noqa: E402
from pengine.repository import Repository  # noqa: E402
from pengine.worker import Worker  # noqa: E402

# The workbench only renders the four bundled persona ids; the server must serve a
# real bundled persona so the browser can select one (Delivery #58 INT-A11).
_BUNDLED_PERSONAS = ("shouzhuo", "wuzhen", "sanfentian", "xinggui")


def _install_bundled_persona(persona_root: Path, persona_id: str) -> None:
    source = REPO_ROOT / "personas" / persona_id
    if not source.is_dir():
        raise FileNotFoundError(f"Bundled persona missing: {source}")
    shutil.copytree(source, persona_root / persona_id, dirs_exist_ok=True)


def build_app(*, scenario: str, data_dir: Path, persona_root: Path):
    _install_bundled_persona(persona_root, _BUNDLED_PERSONAS[0])
    settings = Settings(persona_root=persona_root, data_dir=data_dir)
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)

    if scenario == "a8_suffix":
        workflow = UnifiedWorkflow(
            episode_count=3,
            decisions={
                3: {
                    "category": "script_defect",
                    "evidence": "终局未满足义务闭环。",
                    "earliest_affected_episode": 3,
                }
            },
            persistent_defects=True,
        )
    elif scenario == "a7_context":
        workflow = UnifiedWorkflow(episode_count=3, preflight_block=True)
    elif scenario == "happy_path":
        workflow = UnifiedWorkflow(episode_count=3)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="browser-worker",
    )
    return (
        create_app(settings=settings, repository=repository, catalog=catalog, worker=worker),
        worker,
        repository,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="a8_suffix")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=None, help="Reuse an existing isolated data dir")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(tempfile.mkdtemp(prefix="pengine-browser-")) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    persona_root = data_dir.parent / "personas"
    persona_root.mkdir(parents=True, exist_ok=True)

    app, worker, repository = build_app(
        scenario=args.scenario,
        data_dir=data_dir,
        persona_root=persona_root,
    )
    print(f"BROWSER_SERVER data_dir={data_dir}", flush=True)
    print(f"BROWSER_SERVER scenario={args.scenario}", flush=True)
    print(f"BROWSER_SERVER port={args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

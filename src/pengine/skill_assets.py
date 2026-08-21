from importlib.resources import files

_SKILLS = (
    "canon-review",
    "continuity-repair",
    "story-repair",
)


def load_agent_skill_files() -> dict[str, str]:
    root = files("pengine.agent_skills")
    return {
        f"/skills/{skill}/SKILL.md": root.joinpath(skill, "SKILL.md").read_text(encoding="utf-8")
        for skill in _SKILLS
    }

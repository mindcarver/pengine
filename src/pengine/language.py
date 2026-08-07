from __future__ import annotations

import re
from typing import Literal

OutputLanguage = Literal["zh-CN"]

SIMPLIFIED_CHINESE: OutputLanguage = "zh-CN"

_HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_MACHINE_IDENTIFIER = re.compile(
    r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+|[A-Za-z]+\d+(?:[_-][A-Za-z0-9]+)*"
)
_LANGUAGE_NEUTRAL_NUMBER = re.compile(r"(?=.*\d)[0-9TtZ\s:./+%\-]+")
_SHORT_ENGLISH_VERDICT = re.compile(
    r"(?:^|[：:\s])(fail(?:ed)?|pass(?:ed)?|no|yes|ok|true|false)[.!。]?\s*$",
    re.IGNORECASE,
)
_TRADITIONAL_ONLY_CHARACTER = re.compile(
    r"[審證據劇過這個為與關係場戲臺後發說話時間點線錄記寫來開結應歲覺對讓誰麼還沒愛歸親導調緣聽聲視頻頭轉變壞懷實際確認誤懸資檔衝鍵謊錯醫護鄉鎮廠體優會傳倫儀價眾兒黨蘭興養農雲亞產僅從倉傾償書買亂爭於虧樂喬習龍]"
)
_LONG_ENGLISH_TEXT_LETTERS = 5
_ENGLISH_DOMINANCE_RATIO = 2


def infer_output_language(story: str, requirements: str) -> OutputLanguage | None:
    """Lock Simplified Chinese when either free-form input contains Han text."""
    if _HAN_CHARACTER.search(f"{story}\n{requirements}"):
        return SIMPLIFIED_CHINESE
    return None


def language_instruction(language: OutputLanguage | None) -> str:
    """Return the prompt contract shared by the supervisor and every specialist."""
    if language != SIMPLIFIED_CHINESE:
        return ""
    return (
        "输出语言契约：所有面向用户的标题、创作理由、大纲、人物内容、关系描述、"
        "剧本和审核证据都必须使用简体中文（zh-CN）。Schema 字段名、工具名、稳定 ID "
        "以及必须逐字保留的原文可维持原样。每个委派任务都必须重复此语言契约。"
    )


def has_obvious_language_mismatch(
    text: str,
    language: OutputLanguage | None,
    *,
    english_dominance_ratio: float = _ENGLISH_DOMINANCE_RATIO,
) -> bool:
    """Detect English-only or overwhelmingly English user-facing Chinese output.

    ``english_dominance_ratio`` controls how much Latin content is tolerated
    relative to Han characters. Story artifacts (character biographies,
    relationship logic) may contain character/place names in Latin script, so
    callers can raise the ratio to avoid false positives while still catching
    genuinely English-dominated output.
    """
    if language != SIMPLIFIED_CHINESE:
        return False

    stripped = text.strip()
    if not stripped:
        return False
    if _MACHINE_IDENTIFIER.fullmatch(stripped) and (
        any(character.isdigit() for character in stripped) or "_" in stripped
    ):
        return False
    if _LANGUAGE_NEUTRAL_NUMBER.fullmatch(stripped):
        return False
    if _SHORT_ENGLISH_VERDICT.search(stripped):
        return True

    language_sample = _MACHINE_IDENTIFIER.sub(
        lambda match: (
            ""
            if "_" in match.group(0) or any(character.isdigit() for character in match.group(0))
            else match.group(0)
        ),
        text,
    )
    han_count = len(_HAN_CHARACTER.findall(language_sample))
    if han_count == 0:
        return True

    traditional_count = len(_TRADITIONAL_ONLY_CHARACTER.findall(language_sample))
    if traditional_count >= 2 and traditional_count * 3 >= han_count:
        return True

    latin_count = len(_LATIN_LETTER.findall(language_sample))
    return (
        latin_count >= _LONG_ENGLISH_TEXT_LETTERS
        and latin_count > han_count * english_dominance_ratio
    )

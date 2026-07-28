from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

LOGICAL_FILES: tuple[tuple[str, str], ...] = (
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

NON_PRODUCTION_CONTENT: dict[str, str] = {
    "paradigm": """\
# 非生产测试范式总纲

> NON-PRODUCTION TEST FIXTURE

## L0 定义
内核母题定义。
## L1 定义
命运画像定义。
## L2 定义
能量画像定义。
## L3 定义
认知侧写定义。
## L4 定义
价值观与手艺定义。
## L5 定义
作品与经历定义。
## L6 定义
外部学习定义。
## L0 三面结构
### 变体
变体规则。
### 雷区
雷区规则。
### 温度
温度规则。
## 层间仲裁
总纲优先。
## 情境翻译
先翻译本单情境。
## 反馈回路
用户意见随任务生灭。
## 留白
保留创作者判断。
## 验收闸
先 L0 后 L4。
## 工作纪律
按阶段工作。
## 归属与判断权
创作者拥有最终判断权。
""",
    "project": """\
# 非生产测试 Project 说明

> NON-PRODUCTION TEST FIXTURE

## 身份声明
这是仅用于自动化测试的人格。
## L0 全文
### 变体
- [真人已定][归属:创作者] 困境中仍然主动选择。
- [AI草稿待真人确认][归属:创作者] 此项不得编译为已确认规则。
### 雷区
- [真人已定][归属:创作者] 不把人物写成工具。
### 温度
- [真人已定][归属:创作者] 克制但有希望。
## L1 摘要与状态
- [真人已定][归属:创作者] 表达直接。
## L2 摘要与状态
- [真人已定][归属:创作者] 张力内敛。
## L3 摘要与状态
- [真人已定][归属:创作者] 擅长因果推进。
## L4 摘要与状态
- [真人已定][归属:创作者] 先人物后技巧。
## L5 摘要与状态
- [真人已定][归属:创作者] 仅按需检索作品。
## L6 摘要与状态
- [真人已定][归属:创作者] 外部技巧不改人格。
## 最高铁律四条
总纲优先；L0 只读；写入白名单；内容与框架分离。
## 层间仲裁
冲突时遵循总纲。
## 固定工作步骤
定侧面后依次完成五阶段。
## 验收闸
先验 L0，再验 L4。
## 反馈与留白
意见仅作用于本单。
""",
    "l0": """\
# 非生产测试 L0

> NON-PRODUCTION TEST FIXTURE

## 变体
- [真人已定][归属:创作者] 在困境中主动选择。
- [AI草稿待真人确认][归属:创作者] 待定变体不得作为确认规则。
## 雷区
- [真人已定][归属:创作者] 不以巧合替代人物选择。
## 温度
- [真人已定][归属:创作者] 克制、温暖、有余味。
""",
    "l1": """\
# 非生产测试 L1

> NON-PRODUCTION TEST FIXTURE

## 来源画像
仅用于测试的稳定画像。
## 摘要
表达具有向前推动的能量。
""",
    "l2": """\
# 非生产测试 L2

> NON-PRODUCTION TEST FIXTURE

## 星盘来源画像
仅用于测试的第二套坐标。
## 摘要
冲突表现克制，但人物选择清晰。
""",
    "l3": """\
# 非生产测试 L3

> NON-PRODUCTION TEST FIXTURE

## 创作手法
用因果链推进。
## 认知路径
先确定人物选择，再组织情节。
## 明确短板
需要检查结构一致性。
## 摘要
擅长人物选择，系统补足结构校验。
""",
    "l4": """\
# 非生产测试 L4

> NON-PRODUCTION TEST FIXTURE

## L4-A 价值观
人物不是情节工具。
## L4-B 短剧技艺
### 故事大纲
必须呈现主角的主动选择。
### 人物小传
每个核心人物必须有独立欲望。
### 人物关系逻辑
关系变化必须有可见因果。
### 分集大纲
每集必须推进人物处境。
### 分集剧本
场景必须承担叙事功能。
## 分环节标准参数
测试参数：每集三场。
""",
    "l5": """\
# 非生产测试 L5

> NON-PRODUCTION TEST FIXTURE

## 作品
### 《霜桥》
霜桥反转来自人物主动承认秘密，而不是偶然发现。
### 《纸船》
纸船结尾用克制的和解收束冲突。
## 经历
测试创作者曾长期观察小城家庭关系。
""",
    "l6": """\
# 非生产测试 L6

> NON-PRODUCTION TEST FIXTURE

## 外部技法条目
### 悬念卡点
卡点必须改变观众对人物选择的理解。
### 场景压缩
合并不承担独立因果功能的场景。
""",
}


def create_persona_package(
    package_dir: Path,
    *,
    persona_id: str = "test-persona",
    display_name: str = "非生产测试人格",
    version: str = "1.0.0-test",
    content_overrides: dict[str, str] | None = None,
    manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    contents = dict(NON_PRODUCTION_CONTENT)
    contents.update(content_overrides or {})

    files: dict[str, dict[str, str]] = {}
    hashes: list[str] = []
    for logical_name, filename in LOGICAL_FILES:
        raw = contents[logical_name].encode("utf-8")
        (package_dir / filename).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        files[logical_name] = {
            "path": filename,
            "media_type": "text/markdown",
            "encoding": "utf-8",
            "sha256": digest,
        }
        hashes.append(digest)

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "persona_id": persona_id,
        "display_name": display_name,
        "version": version,
        "created_at": "2000-01-01T00:00:00Z",
        "package_sha256": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
        "files": files,
    }
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return package_dir


def package_bytes(package_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(package_dir.iterdir(), key=lambda item: item.name)
    }

"""
AI 相面推理编排
通过 AI Builders MCP (chat/completions 多模态接口) 一次调用完成：
1. 面部特征提取（图像理解）
2. 知识库规则匹配
3. 吉利判词生成
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import httpx

# 从项目根目录加载 .env（在读取 Token 之前）
_env_path = Path(__file__).resolve().parents[2].parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

logger = logging.getLogger("ai_fortune")

# ── AI Builders 配置（必须通过环境变量设置，切勿提交到 Git） ──
AI_BUILDER_BASE = os.getenv("AI_BUILDER_BASE", "https://space.ai-builders.com/backend/v1")
_tok = os.getenv("AI_BUILDER_TOKEN")
AI_BUILDER_TOKEN = (_tok or "").strip() or None  # 去除首尾空格/换行
AI_MODEL = os.getenv("AI_MODEL", "grok-4-fast")

# ── 知识库 ──
RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "face_rules.json"

def _load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_unified_prompt(rules: dict) -> str:
    """构造一次性完成「特征提取 + 知识库匹配 + 判词生成」的 system prompt"""

    # 使用全部规则（不截断），确保不同面相能匹配到差异化特征
    rules_summary = ""
    for dim in rules["dimensions"]:
        rules_summary += f"\n**{dim['name']}**（{dim['region']}）：\n"
        for rule in dim["rules"]:
            meaning = rule.get("rewritten_meaning", rule.get("meaning", ""))
            templates = "；".join(rule.get("fortune_templates", [])[:2])
            rules_summary += f"  · {rule['feature']} → {meaning}（参考：{templates}）\n"

    tech_memes_list = rules.get("tech_memes", [])
    tech_memes = " | ".join(tech_memes_list)
    forbidden = "、".join(rules["rewrite_rules"]["forbidden_words"])

    # 每次随机抽取 3 个梗作为优先推荐，强制多样化
    preferred = random.sample(tech_memes_list, min(3, len(tech_memes_list)))
    preferred_hint = "本次建议优先考虑：[" + "、".join(preferred) + "]"

    return f"""你是春节联欢晚会的 AI 相面大师。请看图完成以下三步，一次性输出结果。

## 任务
1. **观相**：从图片中识别人脸，如实分析五个维度的面部特征（印堂、眉眼神态、鼻翼鼻头、嘴角下巴、整体气色）。必须客观描述，不要泛泛而谈。
2. **断相**：根据下方知识库规则，找出 3-5 个与本次观察最匹配的规则。不同人面相不同，命中的规则应有差异。
3. **出判词**：基于本次命中的规则生成判词，可融入一个硅谷科技梗。判词必须体现本次面相的独特性。

## 面相知识库规则（必须严格按规则匹配，不可套用模板）
{rules_summary}

## 可用科技梗
{tech_memes}

{preferred_hint}

## 硬性要求
- 禁止出现负面词汇：{forbidden}
- 判词必须吉利、喜庆、正面
- 判词 20-30 字
- **特征描述必须客观、具体，基于图片中实际可见的面部细节**
- **判词必须严格基于本次命中的规则，不同人面相不同则判词应有明显差异**
- **科技梗必须从「本次建议优先考虑」中选用，或从完整列表中挑选，避免总是用同一梗**

## 输出格式（严格 JSON，不含 markdown 代码块）
{{
  "features": {{
    "印堂": "观察到的印堂特征（具体描述）",
    "眉眼神态": "观察到的眉眼特征（具体描述）",
    "鼻翼鼻头": "观察到的鼻子特征（具体描述）",
    "嘴角下巴": "观察到的嘴巴下巴特征（具体描述）",
    "整体气色": "观察到的气色特征（具体描述）"
  }},
  "matched_rules": [
    {{"dimension": "维度名", "feature": "命中特征", "meaning": "含义"}},
    ...
  ],
  "fortune": "20-30字吉利判词",
  "tech_meme_used": "使用的科技梗或null"
}}"""


async def _call_ai_builders(messages: list, trace_label: str) -> dict:
    """调用 AI Builders MCP chat/completions 接口"""
    if not AI_BUILDER_TOKEN:
        raise ValueError(
            "未设置 AI_BUILDER_TOKEN。请在 .env 中配置或执行: export AI_BUILDER_TOKEN=your_token"
        )
    url = f"{AI_BUILDER_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_BUILDER_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.9,  # 提高随机性，使不同人得出差异化判词
    }

    start = time.time()
    logger.info("[MCP调用] %s - 开始请求 model=%s", trace_label, AI_MODEL)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    elapsed = time.time() - start
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    logger.info(
        "[MCP调用] %s - 完成 耗时=%.1fs tokens=%s 返回摘要=%s",
        trace_label, elapsed,
        json.dumps(usage),
        content[:300] + "..." if len(content) > 300 else content
    )

    return {"content": content, "usage": usage, "elapsed": elapsed}


def _parse_json_response(text: str) -> dict:
    """从 AI 返回文本中提取 JSON"""
    text = text.strip()
    # 去掉可能的 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    # 尝试找到 JSON 块
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx >= 0 and end_idx > start_idx:
        text = text[start_idx:end_idx]
    return json.loads(text)


async def analyze_face(image_base64: str) -> dict:
    """
    主入口：一次 MCP 调用完成面相分析 + 判词生成。

    参数:
        image_base64: 人脸图片的 base64 data URI

    返回:
        {
            "features": {...},
            "fortune": "...",
            "matched_rules": [...],
            "trace": {...}
        }
    """
    rules = _load_rules()
    trace = {"mcp_calls": []}

    logger.info("═══ AI 相面开始（单次调用模式）═══")

    # 每次请求注入唯一标识，避免模型对相似输入返回缓存式相同结果
    request_id = random.randint(100000, 999999)
    user_text = f"请看这张照片，完成面相分析并给出吉利判词。" \
        f"（本次请求 ID: {request_id}，请务必基于本次观察给出独特判词，避免与任何其他请求重复表述）"

    messages = [
        {"role": "system", "content": _build_unified_prompt(rules)},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]}
    ]

    result = await _call_ai_builders(messages, "面相分析+判词生成(一体化)")
    trace["mcp_calls"].append({
        "step": "面相分析+判词生成(一体化)",
        "tool": "AI Builders chat/completions",
        "model": AI_MODEL,
        "input_summary": "人脸图片 + 统一prompt（特征提取+知识库匹配+判词生成）",
        "output_summary": result["content"][:400],
        "elapsed_seconds": round(result["elapsed"], 1),
        "usage": result["usage"]
    })

    data = _parse_json_response(result["content"])

    features = data.get("features", {})
    fortune = data.get("fortune", "福星高照，前途无量")
    matched_rules = data.get("matched_rules", [])
    tech_meme = data.get("tech_meme_used")

    logger.info("[特征提取] %s", json.dumps(features, ensure_ascii=False))
    logger.info("[命中规则] %s", json.dumps(matched_rules, ensure_ascii=False))
    logger.info("[判词] %s", fortune)
    logger.info("═══ AI 相面完成 ═══")

    return {
        "features": features,
        "fortune": fortune,
        "matched_rules": matched_rules,
        "tech_meme_used": tech_meme,
        "trace": trace,
    }

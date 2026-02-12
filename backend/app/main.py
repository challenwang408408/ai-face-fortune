"""
AI 相面 - 后端主服务
FastAPI 应用，提供 REST API 供前端调用。
"""
from pathlib import Path as _Path
_env_path = _Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 日志配置 ──
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ai_trace.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_fortune")

# ── FastAPI 实例 ──
app = FastAPI(title="AI 相面", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 加载知识库 ──
RULES_PATH = Path(__file__).resolve().parent / "data" / "face_rules.json"

def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_rules_cache = None

def get_rules():
    global _rules_cache
    if _rules_cache is None:
        _rules_cache = load_rules()
    return _rules_cache

# ── 数据模型 ──
class FortuneRequest(BaseModel):
    image: str  # base64 data URI

class FortuneResponse(BaseModel):
    features: dict
    fortune: str
    annotated_image: str  # base64 data URI
    trace: dict  # MCP 调用追踪信息

# ── 健康检查 ──
@app.get("/health")
def health_check():
    rules = get_rules()
    return {
        "status": "ok",
        "rules_version": rules.get("version"),
        "dimensions_count": len(rules.get("dimensions", [])),
    }

# ── 知识库查询 ──
@app.get("/api/rules")
def get_face_rules():
    """返回完整知识库规则（前端/调试用）"""
    return get_rules()

@app.get("/api/rules/dimensions")
def get_dimensions():
    """返回五维特征摘要"""
    rules = get_rules()
    return [
        {"id": d["id"], "name": d["name"], "region": d["region"], "rules_count": len(d["rules"])}
        for d in rules["dimensions"]
    ]

# ── AI 相面主接口 ──
@app.post("/api/fortune")
async def get_fortune(req: FortuneRequest):
    """
    接收人脸截图 base64，调用 AI Builders MCP 进行面相分析，
    返回特征 JSON、判词和圈注图。
    """
    from app.services.fortune_pipeline import analyze_face

    logger.info("[请求] 收到 AI 相面 图片长度=%d 前缀=%s", len(req.image), req.image[:80] + "..." if len(req.image) > 80 else req.image)

    # 图片有效性校验：过小或格式异常会导致 AI API 返回 400
    if len(req.image) < 5000:
        logger.warning("[请求] 校验失败: 图片过小 len=%d", len(req.image))
        raise HTTPException(
            status_code=400,
            detail="图片过小或无效，请确保人脸在取景框内清晰可见后再点击「AI 相面」",
        )
    if not req.image.startswith("data:image/"):
        logger.warning("[请求] 校验失败: 格式异常 前缀=%s", req.image[:50])
        raise HTTPException(status_code=400, detail="图片格式无效")

    logger.info("[请求] 校验通过，开始调用 analyze_face")
    try:
        result = await analyze_face(req.image)
    except Exception as e:
        logger.error("[请求] AI 相面失败: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 分析失败: {str(e)}")

    # ── 生成特征圈注图 ──
    annotated_image = ""
    try:
        from app.services.face_annotator import annotate_face as annotate
        annotated_image = annotate(req.image, result["features"])
        logger.info("[请求] 特征圈注图生成成功")
    except Exception as e:
        logger.error("特征圈注图生成失败（不影响主流程）: %s", str(e), exc_info=True)

    logger.info("[请求] AI 相面完成 fortune=%s", (result.get("fortune") or "")[:50])
    return {
        "features": result["features"],
        "fortune": result["fortune"],
        "matched_rules": result.get("matched_rules", []),
        "tech_meme_used": result.get("tech_meme_used"),
        "annotated_image": annotated_image,
        "trace": result["trace"],
    }


# ── 静态文件（前端） ──
# 注意：mount 在最后，且使用子路径避免覆盖 API 路由
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    """首页"""
    return FileResponse(str(FRONTEND_DIR / "index.html"))

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

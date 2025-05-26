# File: main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
import os # Import os

# Import settings and routers
from config import settings, ALLOWED_ORIGINS_LIST
from routers.chat import router as chat_router
from routers.voice_chat import router as voice_chat_router
from routers.medical import router as medical_router  # 新增：导入医疗路由
from db import initialize_db_pool # Import pool initializer

# --- 日志配置 ---
# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO, # 开发时设置为 DEBUG 以获得更详细的输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 设置 watchfiles 的日志级别为 WARNING，以隐藏频繁的变更检测消息
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
# 可选：根据需要配置特定日志记录器（如 uvicorn、httpx）
# logging.getLogger("uvicorn.error").setLevel(logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # 获取此模块的日志记录器

# --- 应用程序生命周期（用于连接池初始化/关闭）---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：初始化数据库连接池
    logger.info("应用程序启动序列已启动...")
    try:
        initialize_db_pool()
        logger.info("在生命周期启动中成功初始化数据库连接池。")
    except Exception as e:
        logger.critical(f"在数据库连接池初始化期间应用程序启动失败：{e}", exc_info=True)
        # 可选择再次引发错误以防止 FastAPI 完全启动
        raise RuntimeError("在启动期间无法初始化关键资源。") from e

    yield # 应用程序在此处运行

    # 关闭：清理（如果需要，连接池会在进程退出时自动关闭连接）
    logger.info("应用程序关闭序列已启动...")
    # 在此处添加清理逻辑（如果需要，例如关闭外部连接）
    # 通常不需要为 mysql.connector 连接池进行特定清理
    logger.info("应用程序关闭完成。")

# --- FastAPI 应用程序初始化 ---
app = FastAPI(
    title="AI 医院演示",
    description="用于 AI 医疗咨询的 FastAPI 应用程序",
    version="1.0.0",
    lifespan=lifespan # 注册生命周期上下文管理器
)

# --- 中间件 ---
logger.info(f"配置 CORS 中间件，允许的源：{ALLOWED_ORIGINS_LIST}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST, # 使用从配置加载的列表
    allow_credentials=True, # 根据您的身份验证设置进行调整
    allow_methods=["*"],    # 或限制为特定方法，如 ["GET", "POST"]
    allow_headers=["*"],    # 或限制为特定头部
)

# --- 静态文件 ---
# 确保 'static' 目录相对于 main.py 运行位置存在
static_dir = "static"
if not os.path.isdir(static_dir):
    logger.warning(f"未找到静态目录 '{static_dir}'。正在创建。")
    try:
        os.makedirs(static_dir)
        # 如果需要，同时创建应用程序所需的子目录
        os.makedirs(os.path.join(static_dir, "audio"), exist_ok=True)
    except OSError as e:
        logger.error(f"创建静态目录 '{static_dir}' 失败：{e}")

try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"已挂载静态文件服务，从目录：'{static_dir}' 到 '/static'")
except Exception as e:
    logger.error(f"无法挂载静态目录 '{static_dir}'：{e}", exc_info=True)

# 避免挂载 '.' 目录，这是安全风险。如果需要，从静态目录或专用路由提供图片。
# 如果前端需要通过 URL 访问图片（如 title_img.png），请将它们放在 'static/images' 中
# 并在 .env 中相应调整 REPORT_TITLE_IMAGE_PATH（例如，"static/images/title_img.png"）
# 并通过 "/static/images/title_img.png" URL 访问它们。
# 或者，如果需要，创建特定路由以从非静态数据文件夹提供图片。

# --- 路由 ---
app.include_router(chat_router, prefix="") # 在根级别包含聊天路由
app.include_router(voice_chat_router, prefix="") # 在根级别包含语音对话路由
app.include_router(medical_router, prefix="/api/medical") # 新增：包含医疗路由
logger.info("已包含聊天和语音对话路由。")

# --- 根端点 ---
# --- 根端点 ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    """提供前端的主 HTML 文件。"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"在 {index_path} 未找到根索引文件")
        return HTMLResponse(content="<html><body><h1>错误：未找到前端。</h1></body></html>", status_code=404)

@app.get("/voice_chat.html", response_class=HTMLResponse, include_in_schema=False)
async def read_voice_chat():
    """提供语音对话页面的 HTML 文件。"""
    voice_chat_path = os.path.join(static_dir, "voice_chat.html")
    if os.path.exists(voice_chat_path):
        return FileResponse(voice_chat_path)
    else:
        logger.error(f"在 {voice_chat_path} 未找到语音对话页面文件")
        return HTMLResponse(content="<html><body><h1>错误：未找到语音对话页面。</h1></body></html>", status_code=404)

# --- 主执行 ---
# --- 主执行 ---
if __name__ == "__main__":
    # 获取环境变量，默认为生产环境
    is_development = os.getenv("ENVIRONMENT", "production").lower() == "development"
    
    if is_development:
        logger.info("以开发模式启动 Uvicorn 服务器...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=["."],
            reload_delay=1.0
        )
    else:
        logger.info("以生产模式启动 Uvicorn 服务器...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # 生产环境禁用自动重载
            log_level="warning",  # 生产环境使用更高的日志级别
            workers=4  # 生产环境使用多个工作进程
        )
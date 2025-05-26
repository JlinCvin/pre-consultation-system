# 文件：config.py
import os
import platform
import logging
# --- 在此处添加 Optional ---
from typing import Optional  # <--- 从 typing 导入 Optional
# --------------------------
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
# 将 .env 文件中的变量加载到环境中
# 确保 .env 在同一目录下或指定路径
load_dotenv()

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # 数据库设置
    db_host: str = Field(..., alias='DB_HOST')
    db_port: int = Field(3306, alias='DB_PORT')
    db_database: str = Field(..., alias='DB_DATABASE')
    db_user: str = Field(..., alias='DB_USER')
    db_password: str = Field(..., alias='DB_PASSWORD')
    db_pool_size: int = Field(5, alias='DB_POOL_SIZE')
    db_pool_name: str = Field("ai_hospital_pool", alias='DB_POOL_NAME')

    # AI 模型设置
    dashscope_api_key: str = Field(..., alias='DASHSCOPE_API_KEY')
    dashscope_model: str = Field("qwen-plus", alias='DASHSCOPE_MODEL')
    # openai_api_key: Optional[str] = Field(None, alias='OPENAI_API_KEY') # 如果需要则取消注释

    # 服务 URL 和路径
    oss_upload_url: Optional[str] = Field(None, alias='OSS_UPLOAD_URL')  # 可选的 OSS URL
    report_title_image_path: str = Field("images/title_img.png", alias='REPORT_TITLE_IMAGE_PATH')
    font_path_linux: str = Field("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", alias='FONT_PATH_LINUX')  # Linux 常用字体示例
    font_path_windows: str = Field("msyh.ttc", alias='FONT_PATH_WINDOWS')
    font_path_darwin: str = Field("/System/Library/Fonts/PingFang.ttc", alias='FONT_PATH_DARWIN')

    # 安全设置
    allowed_origins: str = Field("*", alias='ALLOWED_ORIGINS')

    class Config:
        # 如果不使用别名，Pydantic 将期望使用大写环境变量
        case_sensitive = False  # 允许大写环境变量如 DB_HOST
        # 如果需要可以添加 extra='ignore'，但显式字段更好

settings = Settings()

# 根据操作系统确定默认字体路径并检查是否存在
def get_default_font_path() -> Optional[str]:
    system = platform.system()
    font_path: Optional[str] = None

    if system == 'Linux':
        font_path = settings.font_path_linux
    elif system == 'Windows':
        # Windows 字体查找可能很棘手，通常需要完整路径或系统字体目录
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        potential_path = os.path.join(font_dir, settings.font_path_windows)
        if os.path.exists(potential_path):
             font_path = potential_path
        else:
             # 如果需要回退逻辑，例如检查常用路径
             font_path = settings.font_path_windows  # 直接尝试名称，PIL 可能会找到它
    elif system == 'Darwin':  # macOS
        font_path = settings.font_path_darwin
    else:
        logger.warning(f"不支持默认字体路径的操作系统: {system}")
        return None

    if font_path and os.path.exists(font_path):
        logger.info(f"使用系统字体: {font_path}")
        return font_path
    elif system == 'Windows' and font_path:  # Windows 的特殊情况，名称可能有效
        logger.info(f"字体路径 {font_path} 未找到，尝试直接使用名称。")
        return font_path  # 让 PIL 尝试通过名称查找
    else:
        logger.warning(f"默认字体路径未找到或无法访问 {system}: {font_path}。回退。")
        return None  # 让 PIL 处理默认值（可能不支持 CJK）

DEFAULT_FONT_PATH = get_default_font_path()

# mysql.connector 的数据库配置字典
DB_CONFIG = {
    "host": settings.db_host,
    "port": settings.db_port,
    "database": settings.db_database,
    "user": settings.db_user,
    "password": settings.db_password,
    # 连接池特定配置 - 在创建池时单独传递
}

# 为 CORS 中间件处理允许的源
ALLOWED_ORIGINS_LIST = [origin.strip() for origin in settings.allowed_origins.split(',') if origin.strip()]
if not ALLOWED_ORIGINS_LIST:
     logger.warning("ALLOWED_ORIGINS 为空或无效，CORS 可能会阻止请求。")
     # 决定默认行为：允许所有 "*" 或不允许 []
     ALLOWED_ORIGINS_LIST = ["*"]  # 如果未设置/无效，默认为允许所有用于开发
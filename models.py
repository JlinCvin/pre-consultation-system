# 文件：app/models.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# 用户信息模型
class UserInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None

# 消息信息模型
class MessageInfo(BaseModel):
    role: str
    content: str
    conversation_id: Optional[str] = None
    timestamp: Optional[datetime] = None

# 聊天请求模型
class ChatRequest(BaseModel):
    messages: List[str]
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[UserInfo] = None    # 患者信息

# 聊天响应模型
class ChatResponse(BaseModel):
    conversation_id: str
    messages: List[MessageInfo]  # 修改为使用 MessageInfo 类型
    image_urls: Optional[List[str]] = None
    thinking_process: Optional[str] = None

# 语音合成请求与响应模型
class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"

class TTSResponse(BaseModel):
    audio_url: str
    audio_base64: Optional[str] = None
    duration: Optional[int] = None

# 历史和删除相关模型
from history_models import (
    HistoryRequest, HistoryResponse,
    ConversationInfo, ConversationRequest,
    ConversationResponse, MessageInfo
)
from delete_models import DeleteConversationRequest, DeleteConversationResponse
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

# 请求模型：历史会话查询
class HistoryRequest(BaseModel):
    user_id: Optional[str] = None  # 用户ID，可选参数
    page: int = 1  # 页码，默认第1页
    page_size: int = 10  # 每页数量，默认10条

# 会话信息模型
class ConversationInfo(BaseModel):
    conversation_id: str  # 会话ID
    created_at: Optional[str] = None  # 创建时间
    updated_at: Optional[str] = None  # 更新时间
    patient_info: Optional[dict] = None  # 完整的患者信息
    image_urls: Optional[List[str]] = None  # 病历图片URL列表

# 响应模型：历史会话查询
class HistoryResponse(BaseModel):
    conversations: List[ConversationInfo]  # 会话列表
    total: int = 0  # 总记录数
    page: int = 1  # 当前页码
    page_size: int = 10  # 每页数量
    total_pages: int = 0  # 总页数

# 请求模型：获取特定会话的消息历史
class ConversationRequest(BaseModel):
    conversation_id: str  # 会话ID

# 消息模型
class MessageInfo(BaseModel):
    content: str  # 消息内容
    role: str  # 消息角色（user/assistant/system）

# 响应模型：获取特定会话的消息历史
class ConversationResponse(BaseModel):
    conversation_id: str  # 会话ID
    messages: List[MessageInfo]  # 消息列表
    patient_info: Optional[dict] = None  # 患者信息
    image_urls: Optional[List[str]] = None  # 病历图片URL列表
    created_at: Optional[str] = None  # 创建时间
    updated_at: Optional[str] = None  # 更新时间
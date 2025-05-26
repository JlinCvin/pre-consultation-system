from pydantic import BaseModel

# 请求模型：删除病历
class DeleteConversationRequest(BaseModel):
    conversation_id: str  # 会话ID
    user_id: str  # 用户ID，用于验证权限

# 响应模型：删除病历
class DeleteConversationResponse(BaseModel):
    success: bool  # 是否成功
    message: str  # 消息
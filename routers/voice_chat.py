from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from typing import Optional
import json
import logging
import asyncio
from models import MessageInfo
from db import load_conversation as db_load_conversation, save_conversation as db_save_conversation
from agents import langchain_nurse_agent
import uuid
from voice_chat import VoiceChat

logger = logging.getLogger(__name__)
router = APIRouter()

class VoiceChatManager:
    def __init__(self):
        self.active_sessions = {}

    async def start_session(self, websocket: WebSocket, conversation_id: Optional[str] = None, user_id: Optional[str] = None):
        """启动语音对话会话"""
        try:
            await websocket.accept()
            logger.info(f"WebSocket连接已接受，会话ID: {conversation_id}")

            # 如果没有会话ID，创建一个新的
            if not conversation_id:
                conversation_id = uuid.uuid4().hex[:16]
                user_id = user_id or "default_user"

            # 加载或创建会话
            try:
                patient, history_messages, _, _, _ = await db_load_conversation(conversation_id)
            except Exception:
                patient = None
                history_messages = []

            # 创建语音对话实例，直接传入 conversation_id
            voice_chat = VoiceChat(conversation_id=conversation_id)
            voice_chat.user_id = user_id
            voice_chat.patient = patient
            voice_chat.history_messages = history_messages
            voice_chat.websocket = websocket

            # 存储会话
            self.active_sessions[conversation_id] = voice_chat

            # 启动语音对话
            await voice_chat.start()

        except Exception as e:
            logger.error(f"启动语音会话时出错: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="启动语音会话失败")

    async def stop_session(self, conversation_id: str):
        """停止语音对话会话"""
        if conversation_id in self.active_sessions:
            voice_chat = self.active_sessions[conversation_id]
            await voice_chat.stop()
            del self.active_sessions[conversation_id]

voice_manager = VoiceChatManager()

@router.websocket("/ws/voice-chat/{conversation_id}")
async def voice_chat_endpoint(websocket: WebSocket, conversation_id: str, user_id: Optional[str] = None):
    """WebSocket端点用于语音对话"""
    try:
        # 从查询参数中获取user_id
        query_params = dict(websocket.query_params)
        user_id = query_params.get('user_id', user_id)
        
        logger.info(f"收到语音对话请求 - 会话ID: {conversation_id}, 用户ID: {user_id}")
        
        await voice_manager.start_session(websocket, conversation_id, user_id)
        
        while True:
            try:
                # 接收消息
                message = await websocket.receive_text()
                voice_chat = voice_manager.active_sessions[conversation_id]
                
                # 解析 JSON 消息
                try:
                    data = json.loads(message)
                    if data['type'] in ['audio', 'audio_end']:
                        # 将数组转换为字节数据
                        int16_data = data['data']
                        # 创建字节数组（每个 Int16 值占用 2 个字节）
                        audio_bytes = bytearray(len(int16_data) * 2)
                        for i, value in enumerate(int16_data):
                            # 将 Int16 值转换为两个字节（小端序）
                            audio_bytes[i * 2] = value & 0xFF  # 低字节
                            audio_bytes[i * 2 + 1] = (value >> 8) & 0xFF  # 高字节
                        
                        # 处理音频数据
                        await voice_chat.process_audio(bytes(audio_bytes))
                        
                        # 如果是最后一段音频，停止录音
                        if data['type'] == 'audio_end':
                            await voice_chat.recognition.stop()
                    else:
                        logger.warning(f"收到未知类型的消息: {data['type']}")
                        
                except json.JSONDecodeError:
                    logger.error("无法解析 JSON 消息")
                    continue
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket连接断开: {conversation_id}")
                break
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {e}")
                # 如果是WebSocket未accept或已断开，直接退出循环
                if "accept" in str(e) or "not connected" in str(e):
                    break
                # 其它异常可适当continue
                continue
                
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        # 清理资源
        if conversation_id in voice_manager.active_sessions:
            await voice_manager.stop_session(conversation_id) 
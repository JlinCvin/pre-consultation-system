# File: routers/chat.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, status # Import status
from typing import List, Tuple, Optional, AsyncGenerator # Correctly import Tuple and AsyncGenerator
from models import (
    ChatRequest, ChatResponse,
    TTSRequest, TTSResponse,
    HistoryRequest, HistoryResponse,
    ConversationRequest, ConversationResponse,
    DeleteConversationRequest, DeleteConversationResponse,
    MessageInfo, UserInfo # Use MessageInfo consistently
)
# Import DB functions (synchronous versions)
from db import (
    get_user_conversations as db_get_user_conversations,
    load_conversation as db_load_conversation,
    save_conversation as db_save_conversation,
    delete_conversation as db_delete_conversation
)
import uuid
import json
import asyncio
import logging
import os
import base64
from gtts import gTTS # Keep gTTS for now, wrap in thread
import time # Import time
from fastapi.responses import StreamingResponse

# Import agents and services
from agents import langchain_nurse_agent # Agent now returns 3 values
# from report_service import generate_medical_record # Assuming async version exists or is wrapped

# Import report service
from report_service import generate_medical_record

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper for running sync DB functions in threads ---
async def run_db_in_thread(func, *args, **kwargs):
    """Runs a synchronous DB function in a separate thread."""
    try:
        # logger.debug(f"在线程中运行数据库函数 {func.__name__}")
        start_time = time.monotonic()
        result = await asyncio.to_thread(func, *args, **kwargs)
        duration = time.monotonic() - start_time
        # logger.debug(f"数据库函数 {func.__name__} 完成，耗时 {duration:.4f}秒")
        return result
    except RuntimeError as e:
         # 捕获特定错误，如连接池不可用
         logger.error(f"执行数据库函数 {func.__name__} 时发生运行时错误：{e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据库服务暂时不可用。")
    except Exception as e:
        logger.error(f"在线程中执行数据库函数 {func.__name__} 时出错：{e}", exc_info=True)
        # 对于其他数据库错误，引发通用 500 错误，详细信息在内部记录
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="数据库操作期间发生内部错误。")


async def stream_chat_response(agent_response: AsyncGenerator):
    async for json_line in agent_response:          # 每行已是纯 JSON
        if not json_line.strip():
            continue
        yield f"data: {json_line.strip()}\n\n"      # SSE 行


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """处理聊天请求，支持流式输出和病历生成"""
    try:
        if not request.messages:
            logger.error("消息不能为空")
            raise HTTPException(status_code=400, detail="消息不能为空")

        conversation_id = request.conversation_id
        if not conversation_id:
            logger.error("必须提供 conversation_id")
            raise HTTPException(status_code=400, detail="必须提供 conversation_id")

        logger.info(f"使用会话 ID: {conversation_id}")

        # 将字符串消息转换为 MessageInfo 对象
        message_infos = [
            MessageInfo(
                role='user',
                content=msg,
                conversation_id=conversation_id
            ) for msg in request.messages
        ]

        return StreamingResponse(
            stream_chat_response(
                langchain_nurse_agent(message_infos, conversation_id, request.user_id)
            ),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": conversation_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理聊天请求时发生错误：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误：{str(e)}")

async def generate_record(request: ChatRequest):
    """生成病历记录"""
    try:
        # 创建新的患者信息对象
        patient_info = UserInfo()
        
        # 从消息中提取患者信息
        for msg in request.messages:
            if msg.role == 'user':
                # 这里可以添加从消息中提取患者信息的逻辑
                pass
            
        # 检查必要信息是否完整
        if not patient_info.name:
            logger.error("患者姓名未提供")
            raise HTTPException(status_code=400, detail="患者姓名未提供")
            
        # 生成病历
        logger.info(f"开始为患者 {patient_info.name} 生成病历")
        result = await generate_medical_record(patient_info)
        
        if result:
            logger.info(f"成功为患者 {patient_info.name} 生成病历：{result}")
            return StreamingResponse(
                stream_chat_response(
                    f"已成功生成病历，请点击链接查看：{result}",
                    "病历生成成功"
                ),
                media_type="text/event-stream"
            )
        else:
            logger.error(f"为患者 {patient_info.name} 生成病历失败")
            raise HTTPException(status_code=500, detail="生成病历失败，请稍后重试")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成病历时发生错误：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成病历时发生错误：{str(e)}")


@router.post("/chat/history", response_model=HistoryResponse, status_code=status.HTTP_200_OK)
async def get_history(request: HistoryRequest):
    """Retrieves paginated conversation history for a user."""
    # TODO: Implement proper user authentication and authorization
    user_id = request.user_id or "default_user" # Placeholder
    logger.info(f"获取用户：{user_id} 的历史记录，页码：{request.page}，每页大小：{request.page_size}")

    # Validate pagination parameters
    if request.page < 1 or request.page_size < 1:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="页码和每页大小必须是正整数。")

    try:
        convs, total, pages = await run_db_in_thread(
            db_get_user_conversations, user_id, request.page, request.page_size
        )
        logger.info(f"用户 {user_id} 的历史查询返回 {len(convs)} 个对话，总计：{total}。")
        return HistoryResponse(
            conversations=convs,
            total=total,
            page=request.page,
            page_size=request.page_size,
            total_pages=pages
        )
    except HTTPException:
        raise # Re-raise exceptions from run_db_in_thread
    except Exception as e:
        logger.error(f"获取用户 {user_id} 的历史记录时发生意外错误：{e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="检索对话历史失败。")


@router.post("/chat/conversation", response_model=ConversationResponse, status_code=status.HTTP_200_OK)
async def get_conversation(request: ConversationRequest):
    """Retrieves the full details of a specific conversation."""
    logger.info(f"获取会话ID：{request.conversation_id} 的对话详情")
    try:
        patient, messages, image_urls, created_at, updated_at = await run_db_in_thread(
            db_load_conversation, request.conversation_id
        )

        # Check if conversation was found (db_load_conversation returns defaults if not found)
        # A more robust check might involve checking if patient has default values or timestamps are None
        if created_at is None and updated_at is None and not messages:
             logger.warning(f"未找到会话ID：{request.conversation_id} 的对话")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到对话。")


        # Convert patient UserInfo back to dict for response model
        patient_dict = patient.__dict__ if patient else None

        return ConversationResponse(
            conversation_id=request.conversation_id,
            messages=messages,
            patient_info=patient_dict,
            image_urls=image_urls,
            created_at=created_at,
            updated_at=updated_at
        )
    except HTTPException as http_exc:
        # Re-raise 404 or other HTTP exceptions from run_db_in_thread or checks
        raise http_exc
    except Exception as e:
        logger.error(f"获取对话 {request.conversation_id} 时发生意外错误：{e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="检索对话详情失败。")


@router.post("/chat/delete", response_model=DeleteConversationResponse, status_code=status.HTTP_200_OK)
async def delete_chat_conversation(request: DeleteConversationRequest):
     """Deletes a conversation after verifying ownership."""
     # TODO: Implement proper user authentication and authorization - GET user_id FROM TOKEN
     if not request.user_id: # Basic check - enhance with auth
          raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="删除需要用户ID。")
     user_id = request.user_id # Use provided ID, ideally verified against token

     logger.info(f"处理用户：{user_id} 对会话ID：{request.conversation_id} 的删除请求")

     try:
        success = await run_db_in_thread(db_delete_conversation, request.conversation_id, user_id)

        if success:
            logger.info(f"成功删除会话ID：{request.conversation_id}")
            return DeleteConversationResponse(success=True, message="删除成功")
        else:
            # Failure reason logged in db.py (not found or permission denied)
            logger.warning(f"用户：{user_id} 删除会话ID：{request.conversation_id} 失败。未找到或权限问题。")
            # Return 404 as user doesn't need to know the exact reason
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="删除失败，会话不存在或无权限")

     except HTTPException as http_exc:
         raise http_exc # Re-raise exceptions from run_db_in_thread
     except Exception as e:
        logger.error(f"用户：{user_id} 删除对话 {request.conversation_id} 时发生意外错误：{e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="删除过程中发生内部错误。")


# --- Synchronous TTS Generation Helper Function ---
def _generate_tts_sync(text: str, lang: str = 'zh-cn') -> Tuple[str, bytes, str]:
    """
    Generates TTS audio using gTTS (synchronous).
    Returns: Tuple(temporary file path, audio content bytes, generated filename)
    Raises: Exception on failure.
    """
    # Generate unique filename
    fname = f"tts_{uuid.uuid4().hex[:12]}.mp3" # Shorter UUID hex
    # Define audio directory within static path
    audio_dir = os.path.join("static", "audio")
    # Ensure the directory exists
    os.makedirs(audio_dir, exist_ok=True)
    # Full path for the temporary audio file
    temp_path = os.path.join(audio_dir, fname)

    try:
        logger.info(f"为文本片段生成 TTS 音频：'{text[:50]}...' -> {temp_path}")
        start_time = time.monotonic()
        # Initialize and save gTTS audio
        tts = gTTS(text=text, lang=lang, slow=False) # slow=False for normal speed
        tts.save(temp_path)
        duration = time.monotonic() - start_time
        logger.info(f"gTTS 音频成功保存：{temp_path}（耗时 {duration:.4f}秒）")

        # Read the generated file bytes
        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()

        # Return path, bytes, and filename for response construction
        return temp_path, audio_bytes, fname

    except Exception as e:
        logger.error(f"路径 {temp_path} 的 gTTS 生成或文件保存失败：{e}", exc_info=True)
        # Attempt to clean up failed file if it exists
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                logger.error(f"清理部分创建/失败的 TTS 文件失败：{temp_path}")
        # Re-raise the original exception to be caught by the endpoint handler
        raise


# --- TTS Endpoint ---
@router.post("/tts", response_model=TTSResponse, status_code=status.HTTP_200_OK)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Converts text to speech using gTTS, returns audio URL and Base64 data."""
    if not request.text:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文本字段不能为空。")

    logger.info(f"收到 TTS 请求，文本片段：'{request.text[:50]}...'")
    temp_path_to_delete: Optional[str] = None
    try:
        # --- Execute Sync TTS in Thread ---
        # Run the blocking gTTS generation and file I/O in a separate thread
        temp_path, audio_bytes, fname = await asyncio.to_thread(
            _generate_tts_sync, request.text, 'zh-cn' # Assuming 'zh-cn' is default or passed in request
        )
        temp_path_to_delete = temp_path # Store path for cleanup

        # --- Encode Audio ---
        # Encode the audio bytes to Base64 for embedding or direct use
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # --- Schedule Cleanup ---
        # Add a background task to delete the temporary audio file after response is sent
        background_tasks.add_task(os.unlink, temp_path_to_delete)
        logger.info(f"已安排后台任务删除临时 TTS 文件：{temp_path_to_delete}")

        # --- Calculate Duration (Estimate) ---
        # Placeholder: gTTS doesn't provide duration easily.
        # Use libraries like 'mutagen' or 'pydub' for accurate duration if needed.
        # Very rough estimate (adjust based on typical bitrate/encoding):
        # Assuming ~16 kB/s for typical MP3 quality
        duration_ms = int((len(audio_bytes) / 16000) * 1000) if len(audio_bytes) > 0 else 0

        # --- Return Response ---
        return TTSResponse(
            audio_url=f"/static/audio/{fname}", # URL for browser access (requires file to exist until fetched)
            audio_base64=audio_base64,         # Base64 data
            duration=duration_ms               # Estimated duration in ms
        )

    except Exception as e:
        # Catch errors from _generate_tts_sync or other unexpected issues
        logger.error(f"TTS 请求处理失败：{e}", exc_info=True)
        # Attempt cleanup if temp file path was assigned but deletion wasn't scheduled
        if temp_path_to_delete and os.path.exists(temp_path_to_delete):
             try:
                 # Check if deletion task was already added? Difficult here.
                 # Safest to just try unlinking again if error occurred before scheduling.
                 os.unlink(temp_path_to_delete)
                 logger.info(f"在错误处理期间清理失败的 TTS 文件：{temp_path_to_delete}")
             except OSError:
                 logger.error(f"在错误处理期间清理临时 TTS 文件失败：{temp_path_to_delete}")
        # Return a generic server error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="语音合成失败")
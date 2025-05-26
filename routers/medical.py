from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging
from patient_info import UserInfo
from agents import NurseAgentGraph
from db import load_conversation, save_conversation
from report_service import generate_medical_record

router = APIRouter()
logger = logging.getLogger(__name__)

class UpdatePatientInfoRequest(BaseModel):
    """更新患者信息的请求模型"""
    conversation_id: str
    user_id: str
    patient_info: UserInfo  # 使用完整的 UserInfo 对象

class GenerateMedicalRecordRequest(BaseModel):
    """生成病历的请求模型"""
    conversation_id: str
    user_id: str
    text: Optional[str] = None

@router.post("/update_patient_info")
async def update_patient_info(request: UpdatePatientInfoRequest):
    """更新患者信息的API端点"""
    try:
        # 直接使用传入的 patient_info
        patient_info = request.patient_info

        # 保存更新后的患者信息到数据库
        try:
            if request.conversation_id:
                # 获取当前会话的消息
                _, messages, image_urls, _, _ = load_conversation(request.conversation_id)
                # 保存更新后的会话信息
                save_conversation(
                    conversation_id=request.conversation_id,
                    patient_info=patient_info,
                    input_items=messages,
                    user_id=request.user_id,
                    image_urls=image_urls
                )
                logger.info(f"患者信息已更新到数据库，会话ID: {request.conversation_id}, 用户ID: {request.user_id}")
        except Exception as db_err:
            logger.error(f"更新患者信息到数据库时出错: {str(db_err)}")
            return f"更新患者信息成功，但保存到数据库时出错：{str(db_err)}"

        # 构建更新信息字典
        update_info = patient_info.to_dict()
        output = "患者信息已更新：" + ", ".join(f"{k}={v}" for k, v in update_info.items() if v)
        return output
    except Exception as e:
        logger.error(f"更新患者信息时发生错误：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新患者信息时发生错误：{str(e)}")

@router.post("/generate_medical_record")
async def generate_medical_record(request: GenerateMedicalRecordRequest):
    """生成病历的API端点"""
    try:
        # 获取当前会话的患者信息
        patient_info, messages, image_urls, _, _ = load_conversation(request.conversation_id)
        if not patient_info:
            raise HTTPException(status_code=400, detail="未找到患者信息，请先更新患者信息")

        # 生成病历
        result = await generate_medical_record(patient_info)
        if result:
            # 保存病历到数据库
            try:
                # 更新会话中的患者信息
                if request.conversation_id:
                    # 保存更新后的会话信息
                    save_conversation(
                        conversation_id=request.conversation_id,
                        patient_info=patient_info,
                        input_items=messages,
                        user_id=request.user_id,
                        image_urls=image_urls
                    )
                    logger.info(f"病历已保存到数据库，会话ID: {request.conversation_id}, 用户ID: {request.user_id}")
            except Exception as db_err:
                logger.error(f"保存病历到数据库时出错: {str(db_err)}")
                return f"生成病历成功，但保存到数据库时出错：{str(db_err)}"

            output = f"已成功生成病历，请点击链接查看：{result}"
            return output
        else:
            return "生成病历失败，请稍后重试"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成病历时发生错误：{str(e)}")
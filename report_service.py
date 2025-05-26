# File: report_service.py
import time
import json
import asyncio
import logging
from typing import Optional # Import Optional
from patient_info import normalize_gender, UserInfo # Import UserInfo
# Import the async wrapper from report.py
from report import generate_report_async

logger = logging.getLogger(__name__)

async def generate_medical_record(context: UserInfo, **kwargs) -> Optional[str]:
    """
    Asynchronously generates a medical record image for the given patient context.

    Calls the async report generator which handles threading for sync parts
    and asynchronous upload.

    Args:
        context: The UserInfo object containing patient data.
        **kwargs: Additional arguments passed down to the report generator (if any).

    Returns:
        The image URL (from OSS if successful) or local file path,
        or None if generation or upload fails.
    """
    # Input validation
    if not isinstance(context, UserInfo):
        logger.error(f"generate_medical_record received invalid context type: {type(context)}")
        return None
    if not context.name: # Basic check for essential info
         logger.warning("Attempting to generate report for patient with no name.")
         # Decide if this should proceed or return None/error

    # Normalization should ideally happen when data is first ingested or updated via agent tools
    # If needed here: context.gender = normalize_gender(context.gender)

    logger.info(f"开始为患者 {context.name} 调用异步报告生成服务")
    start_time = time.monotonic()
    try:
        # Call the async function which handles threading for sync parts and async upload
        image_path_or_url = await generate_report_async(context, **kwargs)

        duration = time.monotonic() - start_time
        if image_path_or_url:
            logger.info(f"为患者 {context.name} 的报告生成服务成功完成。结果：{image_path_or_url}（耗时 {duration:.4f}秒）")
        else:
            logger.error(f"为患者 {context.name} 的报告生成服务失败（耗时 {duration:.4f}秒）")

        return image_path_or_url

    except Exception as e:
        # Catch any unexpected errors during the async process
        duration = time.monotonic() - start_time
        logger.exception(f"为患者 {context.name} 执行报告服务时发生意外错误：{e}（耗时 {duration:.4f}秒）")
        return None


# Keep the old synchronous generate_report function commented out or remove if unused
# def generate_report(...): -> str:
#    ...
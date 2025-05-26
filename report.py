# File: report.py
from PIL import Image, ImageDraw, ImageFont, ImageOps
import platform
import httpx # Use httpx for async requests
import os
import tempfile
import base64
import logging
import time # Import time
import asyncio # Import asyncio for sleep
import shutil # Import shutil for file copying
import re # Import re for Chinese character detection
from typing import Optional, Tuple, Dict, List, Any # Import Tuple
from config import settings, DEFAULT_FONT_PATH # Use settings
from patient_info import UserInfo, normalize_gender # Import UserInfo and normalize_gender

logger = logging.getLogger(__name__)

# --- Async OSS Upload ---
async def upload_image_to_oss(image_path: str) -> Optional[str]:
    """
    使用 httpx 异步上传图片文件到配置的 OSS URL。

    参数：
        image_path: 要上传的图片文件的本地路径。

    返回：
        上传成功时返回 OSS URL 字符串，否则返回 None。
    """
    upload_url = settings.oss_upload_url
    if not upload_url:
        logger.warning("设置中未配置 OSS_UPLOAD_URL，跳过 OSS 上传。")
        return None
    if not os.path.exists(image_path):
         logger.error(f"无法上传图片：在 {image_path} 找不到文件")
         return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/png')}
                data = {'uploadType': 'ORDER_IMAGE_FEATURES'}

                logger.info(f"尝试上传 '{os.path.basename(image_path)}' 到 {upload_url}")
                start_time = time.monotonic()

                response = await client.post(upload_url, files=files, data=data)

                duration = time.monotonic() - start_time
                logger.info(f"上传 POST 请求完成，耗时 {duration:.4f}秒，状态码：{response.status_code}")

                response.raise_for_status()

                result = response.json()
                if result.get('code') == 200 and result.get('data'):
                    oss_url = result['data']
                    logger.info(f"OSS 上传成功。URL：{oss_url}")
                    return oss_url
                else:
                    logger.error(f"OSS 上传失败：服务器返回成功状态但 JSON 内容异常：{result}")
                    return None

    except httpx.TimeoutException:
        logger.error(f"OSS 上传超时（30秒）：{image_path}")
        return None
    except httpx.RequestError as req_err:
        logger.error(f"OSS 上传请求错误 {image_path}：{req_err}")
        return None
    except httpx.HTTPStatusError as status_err:
        logger.error(f"OSS 上传 HTTP 错误：{status_err.response.status_code} {status_err.response.reason_phrase} - 响应：{status_err.response.text}")
        return None
    except json.JSONDecodeError as json_err:
        logger.error(f"OSS 上传响应 JSON 解析失败：{json_err} - 响应文本：{response.text if 'response' in locals() else 'N/A'}")
        return None
    except IOError as io_err:
         logger.error(f"OSS 上传准备过程中文件 I/O 错误 {image_path}：{io_err}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"OSS 上传意外错误 {image_path}：{e}", exc_info=True)
        return None


# --- Sync Image Processing Helpers ---
# These remain synchronous as PIL operations are inherently blocking.
# They should be called within asyncio.to_thread.

def crop_image_height(input_path: str, output_path: str, new_height: int):
    """同步裁剪图片到指定高度（从顶部开始）。"""
    try:
        logger.debug(f"裁剪图片 '{input_path}' 到高度 {new_height}，输出：'{output_path}'")
        with Image.open(input_path) as image:
            width, _ = image.size
            crop_box = (0, 0, width, new_height)
            cropped_image = image.crop(crop_box)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped_image.save(output_path)
            logger.debug(f"图片裁剪并成功保存到 {output_path}")
    except FileNotFoundError:
        logger.error(f"裁剪失败：输入文件未找到 '{input_path}'")
        raise
    except Exception as e:
        logger.error(f"裁剪图片 '{input_path}' 时出错：{e}", exc_info=True)
        raise


def insert_image(
    canvas: Image.Image, img_path: str, y_start: int,
    center_horizontal: bool = False, display_h: Optional[int] = None,
    display_w: Optional[int] = None, margin: int = 10
) -> Tuple[Image.Image, int]:
    """
    Synchronously inserts an image onto a canvas (PIL Image).
    Handles resizing, transparency, and canvas expansion if needed.
    Returns the modified canvas and the height occupied by the inserted image.
    """
    try:
        if not os.path.exists(img_path):
             logger.error(f"无法插入图片：在 '{img_path}' 找不到文件")
             return canvas, 0

        with Image.open(img_path) as img:
            if img.mode == 'P':
                logger.debug(f"将图片 '{img_path}' 从模式 'P' 转换为 'RGBA'")
                img = img.convert('RGBA')
            elif img.mode == 'LA':
                 img = img.convert('RGBA')

            orig_w, orig_h = img.size
            if orig_w == 0 or orig_h == 0:
                 logger.warning(f"图片 '{img_path}' 宽度或高度为零。跳过插入。")
                 return canvas, 0

            aspect = orig_w / orig_h

            if display_w and display_h:
                target_size = (display_w, display_h)
            elif display_w:
                target_h = int(display_w / aspect) if aspect != 0 else orig_h
                target_size = (display_w, target_h)
            elif display_h:
                target_w = int(display_h * aspect) if aspect != 0 else orig_w
                target_size = (target_w, display_h)
            else:
                target_size = (orig_w, orig_h)

            if img.size != target_size:
                 logger.debug(f"调整图片 '{img_path}' 大小从 {img.size} 到 {target_size}")
                 img = img.resize(target_size, Image.Resampling.LANCZOS)

            if center_horizontal:
                x = max(0, (canvas.width - img.width) // 2)
            else:
                x = margin

            required_height = y_start + img.height
            new_canvas = canvas

            if required_height > canvas.height:
                logger.debug(f"扩展画布高度从 {canvas.height} 到 {required_height}")
                bg_color = (255, 255, 255, 0) if img.mode == 'RGBA' else (255, 255, 255)
                new_canvas = Image.new(canvas.mode, (canvas.width, required_height), bg_color)
                new_canvas.paste(canvas, (0, 0))

            if img.mode == 'RGBA':
                new_canvas.paste(img, (x, y_start), mask=img.split()[3])
            else:
                new_canvas.paste(img, (x, y_start))

            occupied_height = img.height
            return new_canvas, occupied_height

    except FileNotFoundError:
        logger.error(f"无法插入图片：在 '{img_path}' 找不到文件")
        return canvas, 0
    except Exception as e:
        logger.error(f"插入图片 '{img_path}' 时出错：{e}", exc_info=True)
        return canvas, 0


def is_chinese(text: str) -> bool:
    """判断字符串是否包含中文"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def draw_text_with_metrics(
    draw: ImageDraw.ImageDraw, text: str, y_start: int,
    center_horizontal: bool = False, font_color: Tuple[int, int, int] = (0, 0, 0),
    font_size: int = 20, font_path: Optional[str] = None,
    margin: int = 25, x_start: Optional[int] = None, line_spacing_ratio: float = 0.5
) -> int:
    """
    Synchronously draws multi-line text with automatic line wrapping.
    Returns the total vertical height occupied by the drawn text block.
    """
    if not text:
         return 0

    img_width, _ = draw.im.size
    font = None
    effective_font_path = font_path or DEFAULT_FONT_PATH

    if effective_font_path:
        try:
            font = ImageFont.truetype(effective_font_path, font_size)
        except IOError:
            logger.warning(f"无法加载字体 '{effective_font_path}'。尝试使用 PIL 默认字体。")
        except Exception as e:
             logger.warning(f"加载字体 '{effective_font_path}' 时出错：{e}。尝试使用 PIL 默认字体。")

    if font is None:
        try:
            font = ImageFont.load_default()
            font_size = 10
            logger.warning("使用 PIL 默认字体。文本外观/换行可能有所不同。")
        except Exception as e:
            logger.error(f"加载 PIL 默认字体失败：{e}")
            return 0

    lines = []
    text_segments = text.split('\n')

    for segment in text_segments:
        if not segment.strip():
             lines.append("")
             continue

        if is_chinese(segment):
            # 中文按字符处理
            current_line = ""
            for char in segment:
                test_line = current_line + char
                try:
                    bbox = font.getbbox(test_line)
                    line_width = bbox[2] - bbox[0]
                except AttributeError:
                    try:
                        line_width = font.getlength(test_line)
                    except AttributeError:
                        line_width = font.getsize(test_line)[0]

                max_width = img_width - 2 * margin
                if not lines and x_start is not None and not center_horizontal:
                    max_width = img_width - x_start - margin

                if line_width > max_width and current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)
        else:
            # 英文按单词处理
            words = segment.split()
            current_line = ""
            for i, word in enumerate(words):
                test_line = current_line + word + " "
                try:
                    bbox = font.getbbox(test_line.strip())
                    line_width = bbox[2] - bbox[0]
                except AttributeError:
                    try:
                        line_width = font.getlength(test_line.strip())
                    except AttributeError:
                        line_width = font.getsize(test_line.strip())[0]

                max_width = img_width - 2 * margin
                if not lines and x_start is not None and not center_horizontal:
                    max_width = img_width - x_start - margin

                if line_width > max_width and current_line:
                    lines.append(current_line.strip())
                    current_line = word + " "
                else:
                    current_line = test_line

                if i == len(words) - 1:
                    lines.append(current_line.strip())

    total_height = 0
    y_position = float(y_start)

    for i, line in enumerate(lines):
        if not line:
             y_position += font_size * (1 + line_spacing_ratio)
             total_height += font_size * (1 + line_spacing_ratio)
             continue

        try:
             bbox = font.getbbox(line)
             line_width = bbox[2] - bbox[0]
             line_height = bbox[3] - bbox[1]
        except AttributeError:
             try:
                 line_width = font.getlength(line)
             except AttributeError:
                 line_width = font.getsize(line)[0]
             try:
                 ascent, descent = font.getmetrics()
                 line_height = ascent + descent
             except AttributeError:
                  line_height = font.getsize(line)[1]

        if center_horizontal:
            x = max(0.0, (img_width - line_width) / 2.0)
        else:
            x = float(x_start) if i == 0 and x_start is not None else float(margin)

        try:
            draw.text((x, y_position), line, font=font, fill=font_color)
        except Exception as draw_err:
             logger.error(f"绘制文本行 '{line}' 时出错：{draw_err}")
             continue

        spacing = line_height * line_spacing_ratio
        y_position += line_height + spacing
        total_height += line_height + spacing

    return int(total_height - spacing) if total_height > 0 else 0


def draw_horizontal_line(
    canvas: Image.Image, y_start: int, center_horizontal: bool = True,
    line_width_percent: float = 90.0, line_height: int = 1,
    line_color: Tuple[int, int, int] = (200, 200, 200)
) -> Image.Image:
    """Synchronously draws a horizontal line on the canvas."""
    try:
        draw = ImageDraw.Draw(canvas)
        canvas_width = canvas.width

        actual_line_width = int(canvas_width * (line_width_percent / 100.0))

        if center_horizontal:
            x0 = (canvas_width - actual_line_width) // 2
        else:
            x0 = (100.0 - line_width_percent) / 2.0 * canvas_width / 100.0

        x1 = x0 + actual_line_width
        y0 = y_start
        y1 = y_start + line_height

        draw.rectangle([(x0, y0), (x1, y1)], fill=line_color)
        return canvas
    except Exception as e:
         logger.error(f"在 y={y_start} 绘制水平线时出错：{e}")
         return canvas

# --- Main Report Generation (Sync) ---
def generate_report_sync(
    patient: UserInfo, # Accept UserInfo object
    final_png_path: str # The final intended local path
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synchronously generates the medical report image using PIL.
    Accepts a UserInfo object and the final local save path.

    Returns:
        Tuple(final_local_path, temp_file_path_for_upload) on success.
        Tuple(None, None) on failure.

    NOTE: This is a blocking function (CPU/IO intensive).
          Call it using asyncio.to_thread from async code.
    """
    start_gen_time = time.monotonic()
    logger.info(f"开始为患者 {patient.name} 生成同步报告")

    gender = normalize_gender(patient.gender) if patient.gender else "未详"

    canvas_width = 800
    initial_height = 2000  # 增加初始高度以容纳更多内容
    img = Image.new('RGB', (canvas_width, initial_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y_pos = 20.0
    section_spacing = 25.0
    item_spacing = 15.0

    try:
        title_img_path = settings.report_title_image_path
        if title_img_path and os.path.exists(title_img_path):
            img, h = insert_image(img, title_img_path, int(y_pos), center_horizontal=True, display_w=600)
            y_pos += h + section_spacing
        else:
            logger.warning(f"报告标题图片未找到或未配置：{title_img_path}")
            y_pos += section_spacing
    except Exception as e:
        logger.error(f"添加标题图片时出错：{e}", exc_info=True)
        y_pos += section_spacing

    draw = ImageDraw.Draw(img)

    try:
        h = draw_text_with_metrics(draw, "预 问 诊 记 录", int(y_pos), center_horizontal=True, font_size=32, font_color=(50, 50, 50))
        y_pos += h + section_spacing
    except Exception as e:
         logger.error(f"绘制主标题时出错：{e}", exc_info=True)

    # 基本信息部分
    try:
        info_text = f'姓名：{patient.name or "未详"}    年龄：{patient.age or "未详"}    性别：{gender}'
        h = draw_text_with_metrics(draw, info_text, int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing

        info_text = f'职业：{patient.occupation or "未详"}    电话：{patient.phone or "未详"}'
        h = draw_text_with_metrics(draw, info_text, int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing

        info_text = f'就诊科室：{patient.department or "未详"}    就诊时间：{patient.visit_time or "未详"}'
        h = draw_text_with_metrics(draw, info_text, int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing

        if patient.address:
             info_text = f'住址：{patient.address}'
             h = draw_text_with_metrics(draw, info_text, int(y_pos), font_size=24, margin=40)
             y_pos += h + item_spacing

        y_pos += section_spacing / 2
    except Exception as e:
         logger.error(f"绘制患者信息部分时出错：{e}", exc_info=True)

    img = draw_horizontal_line(img, int(y_pos))
    y_pos += 5 + section_spacing

    # 主诉和症状部分
    try:
        h = draw_text_with_metrics(draw, f'主诉：{patient.chief_complaint or "无"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing

        symptoms_str = ', '.join(patient.symptoms) if patient.symptoms else "无"
        h = draw_text_with_metrics(draw, f'主要症状：{symptoms_str}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing

        h = draw_text_with_metrics(draw, f'现病史：\n{patient.current_illness or "未详"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + section_spacing/2

    except Exception as e:
         logger.error(f"绘制主诉/症状/现病史时出错：{e}", exc_info=True)

    img = draw_horizontal_line(img, int(y_pos))
    y_pos += 5 + section_spacing

    # 病史部分
    try:
        h = draw_text_with_metrics(draw, f'既往病史：{patient.medical_history or "无"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing
        h = draw_text_with_metrics(draw, f'家族史：{patient.family_history or "无"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing
        h = draw_text_with_metrics(draw, f'个人史：{patient.personal_history or "无"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing
        h = draw_text_with_metrics(draw, f'过敏史：{patient.allergy_history or "无"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + section_spacing/2
    except Exception as e:
         logger.error(f"绘制病史部分时出错：{e}", exc_info=True)

    # 生活习惯部分
    try:
        smoking_status = patient.smoking_history.get('status', '')
        if smoking_status:
            smoking_info = f"吸烟：{smoking_status}"
            if smoking_status == '是':
                smoking_info += f"，{patient.smoking_history.get('amount', '')}，{patient.smoking_history.get('years', '')}"
            h = draw_text_with_metrics(draw, smoking_info, int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing

        alcohol_status = patient.alcohol_history.get('status', '')
        if alcohol_status:
            alcohol_info = f"饮酒：{alcohol_status}"
            if alcohol_status == '是':
                alcohol_info += f"，{patient.alcohol_history.get('frequency', '')}，{patient.alcohol_history.get('type', '')}"
            h = draw_text_with_metrics(draw, alcohol_info, int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing
    except Exception as e:
        logger.error(f"绘制生活习惯部分时出错：{e}", exc_info=True)

    # 女性特有信息
    if gender == "女":
        try:
            menstrual_info = []
            if patient.menstrual_history:
                if patient.menstrual_history.get('menarche_age'):
                    menstrual_info.append(f"初潮年龄：{patient.menstrual_history['menarche_age']}")
                if patient.menstrual_history.get('last_time'):
                    menstrual_info.append(f"末次月经：{patient.menstrual_history['last_time']}")
                if patient.menstrual_history.get('cycle'):
                    menstrual_info.append(f"周期：{patient.menstrual_history['cycle']}天")
                if patient.menstrual_history.get('duration'):
                    menstrual_info.append(f"经期：{patient.menstrual_history['duration']}天")
                if patient.menstrual_history.get('amount'):
                    menstrual_info.append(f"经量：{patient.menstrual_history['amount']}")
                if patient.menstrual_history.get('regularity'):
                    menstrual_info.append(f"规律性：{patient.menstrual_history['regularity']}")
                if patient.menstrual_history.get('dysmenorrhea'):
                    menstrual_info.append(f"痛经：{patient.menstrual_history['dysmenorrhea']}")
                if patient.menstrual_history.get('menopause'):
                    menstrual_info.append(f"绝经：{patient.menstrual_history['menopause']}")

            if menstrual_info:
                h = draw_text_with_metrics(draw, f'月经史：{", ".join(menstrual_info)}', int(y_pos), font_size=24, margin=40)
                y_pos += h + item_spacing

            fertility_info = []
            if patient.fertility_history:
                if patient.fertility_history.get('pregnancy_times'):
                    fertility_info.append(f"妊娠{patient.fertility_history['pregnancy_times']}次")
                if patient.fertility_history.get('delivery_times'):
                    fertility_info.append(f"分娩{patient.fertility_history['delivery_times']}次")
                if patient.fertility_history.get('abortion_times'):
                    fertility_info.append(f"流产{patient.fertility_history['abortion_times']}次")
                if patient.fertility_history.get('children_count'):
                    fertility_info.append(f"现有子女{patient.fertility_history['children_count']}个")
                if patient.fertility_history.get('delivery_method'):
                    fertility_info.append(f"分娩方式：{patient.fertility_history['delivery_method']}")

            if fertility_info:
                h = draw_text_with_metrics(draw, f'生育史：{", ".join(fertility_info)}', int(y_pos), font_size=24, margin=40)
                y_pos += h + item_spacing

        except Exception as e:
            logger.error(f"绘制女性特有信息部分时出错：{e}", exc_info=True)

    # 婚姻史
    try:
        marriage_info = []
        if patient.marriage_history:
            if patient.marriage_history.get('status'):
                marriage_info.append(f"婚姻状况：{patient.marriage_history['status']}")
            if patient.marriage_history.get('marriage_age'):
                marriage_info.append(f"结婚年龄：{patient.marriage_history['marriage_age']}")
            if patient.marriage_history.get('spouse_health'):
                marriage_info.append(f"配偶健康状况：{patient.marriage_history['spouse_health']}")

        if marriage_info:
            h = draw_text_with_metrics(draw, f'婚姻史：{", ".join(marriage_info)}', int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing
    except Exception as e:
        logger.error(f"绘制婚姻史部分时出错：{e}", exc_info=True)

    # 体格检查部分
    try:
        physical_info = []
        if patient.physical_data:
            if patient.physical_data.get('temperature'):
                physical_info.append(f"体温：{patient.physical_data['temperature']}℃")
            if patient.physical_data.get('respiration'):
                physical_info.append(f"呼吸：{patient.physical_data['respiration']}次/分")
            if patient.physical_data.get('pulse'):
                physical_info.append(f"脉搏：{patient.physical_data['pulse']}次/分")
            if patient.physical_data.get('blood_pressure_high') and patient.physical_data.get('blood_pressure_low'):
                physical_info.append(f"血压：{patient.physical_data['blood_pressure_high']}/{patient.physical_data['blood_pressure_low']}mmHg")

        if physical_info:
            h = draw_text_with_metrics(draw, f'生命体征：{", ".join(physical_info)}', int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing

        if patient.physical_exam:
            h = draw_text_with_metrics(draw, f'体格检查：{patient.physical_exam}', int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing

        if patient.auxiliary_exam:
            h = draw_text_with_metrics(draw, f'辅助检查：{patient.auxiliary_exam}', int(y_pos), font_size=24, margin=40)
            y_pos += h + item_spacing

    except Exception as e:
        logger.error(f"绘制体格检查部分时出错：{e}", exc_info=True)

    img = draw_horizontal_line(img, int(y_pos))
    y_pos += 5 + section_spacing

    # 诊断和治疗计划
    try:
        h = draw_text_with_metrics(draw, f'AI初步诊断/评估：{patient.last_diagnosis or "待明确"}', int(y_pos), font_size=24, margin=40)
        y_pos += h + item_spacing
        if patient.treatment_plan:
             h = draw_text_with_metrics(draw, f'处理意见/建议：{patient.treatment_plan}', int(y_pos), font_size=24, margin=40)
             y_pos += h + item_spacing
    except Exception as e:
         logger.error(f"绘制诊断/治疗计划时出错：{e}", exc_info=True)

    y_pos += section_spacing
    try:
        h = draw_text_with_metrics(draw, 'AI生成内容，仅供参考，请咨询专业医师。', int(y_pos), center_horizontal=True, font_color=(128, 128, 128), font_size=18)
        y_pos += h + section_spacing
    except Exception as e:
         logger.error(f"绘制免责声明时出错：{e}", exc_info=True)

    final_height = int(y_pos)
    logger.info(f"计算最终报告高度：{final_height}像素")

    temp_file = None
    try:
        os.makedirs(os.path.dirname(final_png_path), exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        logger.debug(f"将完整高度图片临时保存到 {temp_path}")
        final_img = img.crop((0, 0, canvas_width, final_height))
        final_img.save(temp_path, quality=90)

        logger.info(f"将最终报告从 {temp_path} 复制到 {final_png_path}")
        shutil.copy2(temp_path, final_png_path)

        duration = time.monotonic() - start_gen_time
        logger.info(f"为患者 {patient.name} 的同步报告生成成功。保存到 {final_png_path}。耗时 {duration:.4f}秒。")

        return final_png_path, temp_path

    except Exception as e:
        logger.error(f"报告最终化过程中出错（裁剪/保存/复制）：{e}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"错误后清理临时文件 {temp_path}")
            except OSError as unlink_err:
                 logger.error(f"错误后删除临时文件 {temp_path} 失败：{unlink_err}")
        return None, None


# --- Async Wrapper for Generation and Upload ---
async def generate_report_async(patient: UserInfo, **kwargs) -> Optional[str]:
    """
    异步生成报告图片，通过在线程中运行同步生成器，
    然后尝试上传到 OSS。

    参数：
        patient: 包含患者数据的 UserInfo 对象
        **kwargs: 额外参数（目前未使用但保留以保持灵活性）

    返回：
        如果上传成功则返回 OSS URL，否则返回 None。
    """
    temp_path_to_upload: Optional[str] = None

    try:
        if not settings.oss_upload_url:
            return None
            
        # 创建临时文件路径
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path_to_upload = temp_file.name
        temp_file.close()
            
        generated_path, _ = await asyncio.to_thread(
            generate_report_sync, patient, temp_path_to_upload
        )

        if not generated_path:
            return None

        # 尝试上传到 OSS
        oss_url = await upload_image_to_oss(temp_path_to_upload)
        return oss_url

    except Exception as e:
        logger.error(f"报告生成失败：{e}")
        return None

    finally:
        if temp_path_to_upload and os.path.exists(temp_path_to_upload):
            try:
                os.unlink(temp_path_to_upload)
            except OSError:
                pass
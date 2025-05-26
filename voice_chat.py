import os
import signal
import sys
import time
import pyaudio
import dashscope
from dashscope.audio.asr import *
from dashscope.audio.tts import SpeechSynthesizer
import threading
import queue
import asyncio
import json
import logging
import numpy as np
from pydub import AudioSegment
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from dataclasses import dataclass
from models import MessageInfo
from agents import langchain_nurse_agent
from patient_info import UserInfo
import re

# 设置录音参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096
FORMAT = pyaudio.paInt16
NO_NEW_TEXT_TIMEOUT = 1.5  # 无新文本超时时间（秒）
MIN_AUDIO_SIZE = CHUNK_SIZE * 2  # 最小音频数据块大小

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # 设置日志级别

class VoiceChat(RecognitionCallback):
    def __init__(self, conversation_id: str = None):
        self.mic = None
        self.stream = None
        self.recognition = None
        self.tts = None
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_running = False
        self.websocket = None
        self.conversation_id = conversation_id  # 确保在初始化时设置
        self.user_id = None
        self.patient = None
        self.history_messages: List[MessageInfo] = []
        self.audio_buffer = []  # 用于存储音频数据
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_text = ""  # 用于累积当前语音识别的文本
        self.last_text_time = 0  # 上次接收到新文本的时间
        self.is_playing = False  # 是否正在播放语音
        self.play_thread = None  # 语音播放线程
        self.recognition_queue = queue.Queue()  # 用于存储识别结果
        self.is_processing = False  # 是否正在处理对话
        self.is_ai_responding = False  # 是否正在等待AI回复
        self.pending_user_text = ""  # 存储等待处理的用户文本
        self.user_speaking = False  # 用户是否正在说话
        self.ai_response_task = None  # 存储当前的AI回复任务
        self.init_dashscope_api_key()
        logger.info(f"VoiceChat 初始化完成，conversation_id: {conversation_id}")

    def init_dashscope_api_key(self):
        """初始化DashScope API密钥"""
        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            dashscope.api_key = '<your-dashscope-api-key>'

    def on_open(self):
        """语音识别开启回调"""
        logger.info('语音识别已开启')
        try:
            # 只在本地开发环境初始化 PyAudio
            if os.environ.get('ENVIRONMENT') == 'development':
                self.mic = pyaudio.PyAudio()
                
                # 检查可用的音频设备
                info = self.mic.get_host_api_info_by_index(0)
                numdevices = info.get('deviceCount')
                
                if numdevices == 0:
                    raise OSError("未找到可用的音频设备")
                
                # 查找默认输入设备
                default_input_device = self.mic.get_default_input_device_info()
                if default_input_device is None:
                    raise OSError("未找到默认输入设备")
                
                logger.info(f"使用音频输入设备: {default_input_device.get('name')}")
                
                self.stream = self.mic.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    input_device_index=default_input_device.get('index')
                )
            
            self.current_text = ""  # 重置当前文本
            self.last_text_time = time.time()  # 重置最后文本时间
            self.is_processing = False
        except Exception as e:
            logger.error(f"初始化音频设备失败: {str(e)}")
            if self.mic:
                self.mic.terminate()
            self.mic = None
            self.stream = None
            # 在服务器环境中，不抛出异常，而是继续处理
            if os.environ.get('ENVIRONMENT') != 'development':
                logger.info("服务器环境，跳过本地音频设备初始化")
            else:
                raise

    def on_close(self):
        """语音识别关闭回调"""
        logger.info('语音识别已关闭')
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()
        self.stream = None
        self.mic = None

    def on_complete(self):
        """语音识别完成回调"""
        logger.info('语音识别完成')
        # 当语音识别完成时，处理累积的文本
        if self.current_text:
            self.text_queue.put(self.current_text)
            self.current_text = ""  # 重置当前文本

    def on_error(self, message):
        """语音识别错误回调"""
        logger.error(f'语音识别错误: {message.message}')
        self.stop()

    def on_event(self, result):
        """语音识别结果回调"""
        try:
            sentence = result.get_sentence()
            print(sentence)
            if 'text' in sentence:
                text = sentence['text']
                is_final = sentence.get('sentence_end', False) # 获取 is_final 标志，默认为 False

                if text != self.current_text or is_final: # 只有当文本发生变化或结果是最终结果时才更新
                    logger.info(f'识别结果: {text} (最终结果: {is_final})')
                    self.last_text_time = time.time()
                    self.current_text = text  # 更新当前文本
                    self.user_speaking = True  # 标记用户正在说话

                    # 如果正在等待AI回复，取消当前任务
                    if self.is_ai_responding and self.ai_response_task:
                        self.ai_response_task.cancel()
                        self.is_ai_responding = False
                        logger.info("检测到用户继续说话，取消当前AI回复任务")

                        # 如果不是最终结果，暂存文本；如果是最终结果，直接处理
                        if not is_final:
                             if self.pending_user_text:
                                self.pending_user_text = self.pending_user_text + " " + text
                             else:
                                self.pending_user_text = text
                             logger.info(f"暂存未处理的文本: {self.pending_user_text}")


                    # 使用异步方式发送消息
                    if self.is_websocket_connected():
                        try:
                            # 创建新的事件循环来发送消息
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                            # 设置超时时间
                            async def send_with_timeout():
                                try:
                                    async with asyncio.timeout(1.0):  # 1秒超时
                                        await self.websocket.send_text(json.dumps({
                                            'type': 'recognition',
                                            'text': text,
                                            'is_final': is_final # 添加 is_final 标志
                                        }))
                                except asyncio.TimeoutError:
                                    logger.warning("发送识别结果超时")
                                except Exception as e:
                                    logger.error(f"发送识别结果失败: {e}")

                            loop.run_until_complete(send_with_timeout())
                            loop.close()
                        except Exception as e:
                            logger.error(f"创建事件循环失败: {e}")
        except Exception as e:
            logger.error(f"处理语音识别结果时出错: {e}")

    def is_websocket_connected(self):
        """检查 WebSocket 是否已连接"""
        return self.websocket is not None

    async def process_audio(self, audio_data):
        """处理接收到的音频数据"""
        try:
            if self.recognition and self.is_running:
                # 检查数据大小是否合适
                if len(audio_data) >= MIN_AUDIO_SIZE:
                    # 发送音频数据进行识别
                    self.recognition.send_audio_frame(audio_data)
                    
                    # 检查是否超过无新文本超时时间
                    current_time = time.time()
                    if (self.current_text and 
                        current_time - self.last_text_time >= NO_NEW_TEXT_TIMEOUT):
                        logger.info(f'检测到说话结束，更新状态')
                        self.user_speaking = False
                        
                        # 如果正在等待AI回复，取消当前任务
                        if self.is_ai_responding and self.ai_response_task:
                            self.ai_response_task.cancel()
                            self.is_ai_responding = False
                            logger.info("检测到用户继续说话，取消当前AI回复任务")
                            
                            # 保存当前文本
                            if self.pending_user_text:
                                self.pending_user_text = self.pending_user_text + " " + self.current_text
                            else:
                                self.pending_user_text = self.current_text
                            logger.info(f"保存未处理的文本: {self.pending_user_text}")
                else:
                    logger.debug(f"音频数据块大小不正确: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"处理音频数据时出错: {e}", exc_info=True)
            if self.is_websocket_connected():
                try:
                    await self.websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
                except Exception as e:
                    logger.error(f"发送错误消息失败: {e}")

    async def process_recognition_queue(self):
        """处理语音识别队列"""
        while self.is_running:
            try:
                # 从队列中获取识别结果
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.recognition_queue.get
                )
                
                if result is None:
                    continue
                    
                # 更新用户说话状态
                self.user_speaking = True
                
                # 处理识别结果
                await self.process_recognition_result(result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"处理语音识别队列时出错: {e}")
                continue

    async def process_recognition_result(self, result):
        """处理单个语音识别结果"""
        try:
            # 更新当前文本
            self.current_text = result.text
            self.last_text_time = time.time()
            
            # 将文本添加到待处理队列
            self.pending_user_text += result.text
            
            # 如果AI正在回复，取消当前的AI回复任务
            if self.is_ai_responding and self.ai_response_task:
                self.ai_response_task.cancel()
                self.is_ai_responding = False
                logger.info("检测到新的语音输入，取消当前AI回复任务")
            
            # 立即处理文本
            await self.process_user_text()
                
        except Exception as e:
            logger.error(f"处理语音识别结果时出错: {e}")

    async def process_user_text(self):
        """处理用户输入的文本"""
        if not self.pending_user_text:
            return
            
        try:
            # 设置AI回复状态
            self.is_ai_responding = True
            
            # 获取AI回复
            response = await self.get_ai_response(self.pending_user_text)
            
            # 清空待处理文本
            self.pending_user_text = ""
            
            # 发送AI回复
            if self.websocket:
                await self.websocket.send_json({
                    "type": "ai_response",
                    "text": response
                })
                
        except Exception as e:
            logger.error(f"处理用户文本时出错: {e}")
        finally:
            # 重置AI回复状态
            self.is_ai_responding = False

    async def process_recognition_results(self):
        """处理语音识别结果的协程"""
        while self.is_running:
            try:
                # 非阻塞方式检查队列
                try:
                    text = self.text_queue.get_nowait()
                    if text:
                        await self.handle_chat_response(text)
                except queue.Empty:
                    pass
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"处理识别结果时出错: {e}", exc_info=True)
            finally:
                self.is_processing = False

    async def handle_chat_response(self, text):
        """处理对话响应"""
        try:
            if not text or not text.strip():
                logger.warning("收到空文本，跳过处理")
                return

            if not self.is_websocket_connected():
                logger.warning("WebSocket未连接，跳过处理")
                return

            if not self.conversation_id:
                logger.warning("未设置 conversation_id，跳过处理")
                return

            logger.info(f"开始处理对话响应，文本内容: {text}, conversation_id: {self.conversation_id}")

            # 创建用户消息
            user_message = MessageInfo(
                role='user', 
                content=text,
                conversation_id=self.conversation_id
            )
            
            try:
                # 调用护士代理
                logger.info(f"调用护士代理处理对话，使用 conversation_id: {self.conversation_id}")
                response_text = ""
                final_response = ""  # 用于累积完整的响应文本
                
                # 使用 async for 处理异步生成器的输出
                async for chunk in langchain_nurse_agent(
                    messages=[user_message],  # 只传入当前消息
                    conversation_id=self.conversation_id,
                    user_id=self.user_id
                ):
                    try:
                        data = json.loads(chunk)
                        print(data)  # 调试输出
                        
                        # 处理不同类型的响应
                        if data["type"] == "response_token":
                            # 处理流式文本响应
                            token_text = data["text"]
                            response_text += token_text
                            final_response += token_text  # 累积到最终响应中
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'response',
                                    'text': token_text,
                                    'conversation_id': self.conversation_id
                                }))
                                
                        elif data["type"] == "tool_call_start":
                            # 处理工具调用开始
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'tool_start',
                                    'name': data["name"],
                                    'conversation_id': self.conversation_id
                                }))
                                
                        elif data["type"] == "tool_call_chunk":
                            # 处理工具调用参数
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'tool_chunk',
                                    'args': data["args_delta"],
                                    'conversation_id': self.conversation_id
                                }))
                                
                        elif data["type"] == "tool_output":
                            # 处理工具输出
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'tool_output',
                                    'text': data["text"],
                                    'conversation_id': self.conversation_id
                                }))
                                
                        elif data["type"] == "tool_progress":
                            # 处理工具进度
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'tool_progress',
                                    'data': data["data"],
                                    'conversation_id': self.conversation_id
                                }))
                                
                        elif data["type"] == "end":
                            # 处理结束标记
                            if self.is_websocket_connected():
                                await self.websocket.send_text(json.dumps({
                                    'type': 'end',
                                    'conversation_id': self.conversation_id
                                }))
                                
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析代理响应: {chunk}")
                        continue
                
                # 创建AI消息，使用累积的完整响应
                ai_message = MessageInfo(
                    role='assistant', 
                    content=final_response,
                    conversation_id=self.conversation_id
                )
                
                # 更新历史消息
                self.history_messages.extend([user_message, ai_message])
                logger.info(f"历史消息更新，当前共有 {len(self.history_messages)} 条消息")
                
                # 语音合成 - 使用累积的完整响应
                if final_response and final_response.strip():  # 确保有文本内容再进行语音合成
                    logger.info("开始语音合成...")
                    await self.speak(final_response)
                else:
                    logger.warning("响应文本为空，跳过语音合成")
                    
            except Exception as e:
                logger.error(f"调用护士代理时出错: {e}", exc_info=True)
                if self.is_websocket_connected():
                    await self.websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f'调用护士代理时出错: {str(e)}',
                        'conversation_id': self.conversation_id
                    }))
            
        except Exception as e:
            logger.error(f'处理对话响应时出错: {e}', exc_info=True)
            if self.is_websocket_connected():
                try:
                    await self.websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f'处理对话响应时出错: {str(e)}',
                        'conversation_id': self.conversation_id
                    }))
                except Exception as e:
                    logger.error(f"发送错误消息失败: {e}")

    async def _update_patient_info(self, patient_info: dict):
        """异步更新患者信息"""
        try:
            if patient_info:
                self.patient = UserInfo(**patient_info)
                logger.info(f"患者信息已更新: {self.patient}")
        except Exception as e:
            logger.error(f"更新患者信息时出错: {e}", exc_info=True)

    async def speak(self, text):
        """语音合成"""
        try:
            # 确保文本不为空且为字符串类型
            if not text or not isinstance(text, str):
                logger.warning(f"无效的文本输入: {text}")
                return
                
            # 移除文本中的特殊字符和格式
            text = text.replace('\n', '。').replace('![', '').replace('](', '').replace(')', '')
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；''""（）]', '', text)
            
            if not text.strip():
                logger.warning("清理后的文本为空")
                return
                
            # 调用语音合成
            result = SpeechSynthesizer.call(
                model='sambert-zhichu-v1',
                text=text,
                sample_rate=48000,
                format='wav'
            )
            
            if result.get_audio_data() is not None:
                # 发送音频数据给客户端
                if self.is_websocket_connected():
                    await self.websocket.send_bytes(result.get_audio_data())
                    logger.info("语音数据已发送到客户端")
            else:
                logger.warning("语音合成未返回音频数据")
                
        except Exception as e:
            logger.error(f'语音合成错误: {str(e)}')

    async def convert_webm_to_pcm(self, webm_data):
        """将 WebM 音频转换为 PCM 格式"""
        try:
            # 将字节数据转换为内存文件对象
            audio_buffer = io.BytesIO(webm_data)
            
            # 使用 pydub 加载 WebM 音频
            audio = AudioSegment.from_file(audio_buffer, format="webm")
            
            # 转换为所需格式
            audio = audio.set_frame_rate(SAMPLE_RATE)
            audio = audio.set_channels(CHANNELS)
            audio = audio.set_sample_width(2)  # 16-bit PCM
            
            # 获取原始 PCM 数据
            pcm_data = audio.raw_data
            
            return pcm_data
        except Exception as e:
            logger.error(f"音频格式转换失败: {e}")
            return None

    async def start(self):
        """启动语音对话"""
        if self.is_running:
            return

        self.is_running = True
        self.is_playing = False
        self.is_ai_responding = False  # 重置AI回复状态
        self.user_speaking = False  # 重置用户说话状态
        self.pending_user_text = ""  # 清空未处理的文本
        
        # 启动语音识别
        self.recognition = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=SAMPLE_RATE,
            semantic_punctuation_enabled=True,
            callback=self
        )
        self.recognition.start()

        # 启动识别结果处理协程
        asyncio.create_task(self.process_recognition_queue())
        asyncio.create_task(self.process_recognition_results())

        logger.info('语音对话已启动')

    async def stop(self):
        """停止语音对话"""
        if not self.is_running:
            return

        self.is_running = False
        self.is_playing = False
        try:
            if self.recognition:
                self.recognition.stop()
                self.recognition = None
        except Exception as e:
            logger.warning(f"停止语音识别时出错: {e}")
        
        # 清空队列
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

        logger.info('语音对话已停止')

    async def get_ai_response(self, text):
        """获取AI回复"""
        try:
            if not text or not text.strip():
                return

            if not self.is_websocket_connected():
                return

            # 如果用户正在说话，保存当前文本并等待
            if self.user_speaking:
                self.pending_user_text = text
                logger.info("用户正在说话，暂存当前文本")
                return

            # 标记正在等待AI回复
            self.is_ai_responding = True
            
            # 调用护士代理
            logger.info(f"调用护士代理处理对话，文本: {text}")
            
            # 创建AI回复任务
            self.ai_response_task = asyncio.create_task(self._process_ai_response(text))
            
            try:
                # 等待AI回复，但允许被取消
                await asyncio.shield(self.ai_response_task)
            except asyncio.CancelledError:
                logger.info("AI回复任务被取消")
                # 如果被取消，保存当前文本
                if self.current_text:
                    self.pending_user_text = self.current_text
                    logger.info(f"保存未处理的文本: {self.pending_user_text}")
                return
            finally:
                self.is_ai_responding = False
                self.ai_response_task = None
                
        except Exception as e:
            logger.error(f"获取AI回复时出错: {e}", exc_info=True)
            if self.is_websocket_connected():
                await self.websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'获取AI回复时出错: {str(e)}'
                }))

    async def _process_ai_response(self, text):
        """处理AI回复的内部方法"""
        response_text = ""
        try:
            # 使用 async for 处理异步生成器的输出
            async for chunk in langchain_nurse_agent(
                messages=[MessageInfo(role='user', content=text)],
                conversation_id=self.conversation_id
            ):
                try:
                    data = json.loads(chunk)
                    if data["type"] == "response_token":
                        # 处理流式文本响应
                        token_text = data["text"]
                        response_text += token_text
                        # 立即发送响应给客户端
                        if self.is_websocket_connected():
                            await self.websocket.send_text(json.dumps({
                                'type': 'response',
                                'text': token_text
                            }))
                            logger.info(f"发送AI回复到客户端: {token_text}")
                    elif data["type"] == "output":  # 处理最终输出
                        response_text = data["output"]
                        if self.is_websocket_connected():
                            await self.websocket.send_text(json.dumps({
                                'type': 'response',
                                'text': response_text
                            }))
                            logger.info(f"发送最终AI回复到客户端: {response_text}")
                except json.JSONDecodeError:
                    logger.warning(f"无法解析代理响应: {chunk}")
                except Exception as e:
                    logger.error(f"处理代理响应时出错: {e}")
            
            # 语音合成
            if response_text and response_text.strip():
                logger.info("开始语音合成...")
                await self.speak(response_text)
            else:
                logger.warning("响应文本为空，跳过语音合成")
                
        except Exception as e:
            logger.error(f"处理AI回复时出错: {e}", exc_info=True)
            raise

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print('\n正在停止语音对话...')
    voice_chat.stop()
    sys.exit(0)

if __name__ == '__main__':
    voice_chat = VoiceChat()
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(voice_chat.start()) 
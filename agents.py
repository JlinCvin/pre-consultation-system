import asyncio
import re
import uuid
import logging
import time
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent  # Use openai_tools agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from patient_info import UserInfo, update_seat, normalize_gender  # Import UserInfo
from report_service import generate_medical_record  # Import if report tool is added
from typing import List, Tuple, Optional, Dict, Any, AsyncGenerator
from models import MessageInfo  # Use MessageInfo for consistency
from config import settings  # Import settings if needed for LLM config
from pydantic import BaseModel, Field
import json
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from dataclasses import replace
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from db import save_conversation, load_conversation  # 修改为正确的函数名
from langgraph.config import get_stream_writer  # <- 新增：用于工具内部流式写入
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # 设置默认日志级别为WARNING


# -----------------------------
# 数据模型定义（保持不变）
# -----------------------------

class UpdatePatientInfoInput(BaseModel):
    """更新患者信息的输入模型"""
    # 基本信息
    name: Optional[str] = Field(None, description="患者姓名", examples=["张三"])
    age: Optional[int] = Field(None, description="患者年龄", examples=[25])
    gender: Optional[str] = Field(None, description="患者性别", examples=["男", "女"])
    occupation: Optional[str] = Field(None, description="职业", examples=["教师", "工程师"])
    phone: Optional[str] = Field(None, description="联系电话", examples=["13800138000"])
    address: Optional[str] = Field(None, description="住址", examples=["北京市海淀区中关村大街1号"])

    # 就诊信息
    department: Optional[str] = Field(None, description="就诊科室", examples=["内科", "外科"])
    visit_time: Optional[str] = Field(None, description="就诊时间", examples=["2024-03-20 14:30"])
    chief_complaint: Optional[str] = Field(None, description="主诉", examples=["发热3天，咳嗽2天"])

    # 症状和病史
    symptoms: Optional[List[str]] = Field(None, description="症状列表", examples=[["发热", "咳嗽", "头痛"]])
    medical_history: Optional[str] = Field(None, description="既往病史", examples=["高血压病史5年"])
    last_diagnosis: Optional[str] = Field(None, description="最后诊断", examples=["上呼吸道感染"])
    current_illness: Optional[str] = Field(None, description="现病史", examples=["3天前开始发热，最高体温38.5℃"])
    family_history: Optional[str] = Field(None, description="家族史", examples=["父亲有高血压病史"])
    allergy_history: Optional[str] = Field(None, description="过敏史", examples=["对青霉素过敏"])
    personal_history: Optional[str] = Field(None, description="个人史", examples=["无特殊"])

    # 体格检查
    physical_exam: Optional[str] = Field(None, description="体格检查", examples=["体温38.2℃，咽部充血"])
    auxiliary_exam: Optional[str] = Field(None, description="辅助检查", examples=["血常规：白细胞12.5×10^9/L"])
    treatment_plan: Optional[str] = Field(None, description="治疗计划", examples=["口服布洛芬退热，多饮水"])

    # 生活习惯
    smoking_history: Optional[Dict[str, str]] = Field(
        None, 
        description="吸烟史，包含status(状态)、amount(量)、years(年限)",
        examples=[{"status": "已戒烟", "amount": "20支/天", "years": "10年"}]
    )
    alcohol_history: Optional[Dict[str, str]] = Field(
        None, 
        description="饮酒史，包含status(状态)、frequency(频率)、type(类型)",
        examples=[{"status": "偶尔", "frequency": "每周1-2次", "type": "啤酒"}]
    )

    # 女性特有信息
    menstrual_history: Optional[Dict[str, str]] = Field(
        None, 
        description="月经史，包含menarche_age(初潮年龄)、last_menstrual_age(末次月经年龄)等",
        examples=[{"menarche_age": "13岁", "last_menstrual_age": "28天前"}]
    )

    # 生育史
    fertility_history: Optional[Dict[str, str]] = Field(
        None, 
        description="生育史，包含pregnancy_times(妊娠次数)、delivery_times(分娩次数)等",
        examples=[{"pregnancy_times": "2次", "delivery_times": "1次"}]
    )

    # 婚姻史
    marriage_history: Optional[Dict[str, str]] = Field(
        None, 
        description="婚姻史，包含status(状态)、marriage_age(结婚年龄)、spouse_health(配偶健康状况)",
        examples=[{"status": "已婚", "marriage_age": "25岁", "spouse_health": "健康"}]
    )

    # 体格数据
    physical_data: Optional[Dict[str, str]] = Field(
        None, 
        description="体格数据，包含temperature(体温)、respiration(呼吸)等",
        examples=[{"temperature": "37.2℃", "respiration": "20次/分", "pulse": "80次/分", "blood_pressure": "120/80mmHg"}]
    )


class GenerateMedicalRecordInput(BaseModel):
    """生成病历的输入模型"""
    text: Optional[str] = Field(None, description="可选的文本输入，用于生成病历")


# -----------------------------
# LLM 初始化
# -----------------------------
try:
    llm = ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=settings.dashscope_model,
        temperature=0.7
    )
    logger.debug(f"已初始化 LLM: {settings.dashscope_model} 通过 ChatOpenAI 指向 DashScope 端点。")
except Exception as e:
    logger.critical(f"LLM 初始化失败: {e}", exc_info=True)
    raise RuntimeError(f"LLM 初始化失败: {e}")


# -----------------------------
# 占位的患者信息抽取器（保持不变）
# -----------------------------
async def langchain_patient_info_extractor(patient_info: UserInfo, messages: List[MessageInfo]) -> UserInfo:
    logger.debug(f"为患者运行信息提取器: {patient_info.name}")
    full_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
    age_match = re.search(r"我今年(\d+)\s*岁", full_text)
    if age_match and not patient_info.age:
        try:
            extracted_age = int(age_match.group(1))
            if 0 < extracted_age < 120:
                patient_info.age = extracted_age
                logger.debug(f"提取器更新患者年龄为: {patient_info.age}")
        except (ValueError, IndexError):
            pass

    if patient_info.gender:
        patient_info.gender = normalize_gender(patient_info.gender)

    return patient_info


patient_info_store: Dict[str, UserInfo] = {}


# =============================
# NurseAgentGraph
# =============================
class NurseAgentGraph:
    def __init__(self, conversation_id: str = None):
        self.callback = AsyncIteratorCallbackHandler()
        self.conversation_id = conversation_id
        self.user_id = "default_user"  # 默认用户ID

        # 初始化对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # 加载患者信息
        try:
            if conversation_id:
                patient_info, messages, image_urls, _, _ = load_conversation(conversation_id)
                self.patient_info = patient_info if patient_info else UserInfo()
                # 加载历史消息到记忆
                for msg in messages:
                    if msg.role == "user":
                        self.memory.chat_memory.add_user_message(msg.content)
                    else:
                        self.memory.chat_memory.add_ai_message(msg.content)
            else:
                self.patient_info = UserInfo()
        except Exception:
            self.patient_info = UserInfo()

        # 初始化流式 LLM
        self.model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=settings.dashscope_model,
            temperature=0.7,
            streaming=True,
            callbacks=[self.callback]
        )

        # LangGraph workflow
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        self.memory_saver = MemorySaver()
        if conversation_id:
            self.app = self.workflow.compile(checkpointer=self.memory_saver, name=f"nurse_agent_{conversation_id}")
        else:
            self.app = self.workflow.compile(checkpointer=self.memory_saver)

        logger.debug(f"NurseAgentGraph 初始化完成，conversation_id: {conversation_id}")

    # -------------------------
    # 工具定义
    # -------------------------
    async def generate_medical_record_tool(self, text: Optional[str] = None) -> str:
        """生成病历记录（支持流式进度）"""
        try:
            writer = get_stream_writer()  # 用于实时进度
            writer({"stage": "start", "msg": "🏃 正在生成病历……"})
            result = await generate_medical_record(self.patient_info)
            if result:
                # 保存病历到数据库
                try:
                    # 更新会话中的患者信息
                    if self.conversation_id:
                        # 获取当前会话的消息
                        _, messages, image_urls, _, _ = load_conversation(self.conversation_id)
                        # 保存更新后的会话信息
                        save_conversation(
                            conversation_id=self.conversation_id,
                            patient_info=self.patient_info,
                            input_items=messages,
                            user_id=self.user_id,
                            image_urls=[result]
                        )
                        logger.info(f"病历已保存到数据库，会话ID: {self.conversation_id}, 用户ID: {self.user_id}")
                except Exception as db_err:
                    logger.error(f"保存病历到数据库时出错: {str(db_err)}")
                    return f"生成病历成功，但保存到数据库时出错：{str(db_err)}"

                output = f"已成功生成病历，请点击链接查看：{result}"
                writer({"stage": "done"})
                return output
            else:
                return "生成病历失败，请稍后重试"
        except Exception as e:
            return f"生成病历时发生错误：{str(e)}"

    async def update_patient_info_tool(self, **kwargs) -> str:
        """更新患者信息"""
        try:
            # 1. 参数验证和转换
            update_info = {k: v for k, v in kwargs.items() if v is not None}
            if not update_info:
                return "没有提供任何需要更新的信息"

            # 验证输入数据
            validation_errors = []
            for key, value in update_info.items():
                if key == "age" and not isinstance(value, int):
                    validation_errors.append(f"年龄必须是整数，当前值: {value}")
                elif key == "symptoms" and not isinstance(value, list):
                    validation_errors.append(f"症状必须是列表，当前值: {value}")
                elif key in ["smoking_history", "alcohol_history", "menstrual_history", 
                           "fertility_history", "marriage_history", "physical_data"]:
                    if not isinstance(value, dict):
                        validation_errors.append(f"{key}必须是字典类型，当前值: {value}")

            if validation_errors:
                return "输入数据验证失败：\n" + "\n".join(validation_errors)

            # 2. 创建患者信息副本并更新
            try:
                updated_info = replace(self.patient_info)
                for key, value in update_info.items():
                    if hasattr(updated_info, key):
                        setattr(updated_info, key, value)
                    else:
                        logger.warning(f"尝试更新不存在的字段: {key}")
                self.patient_info = updated_info
            except Exception as e:
                logger.error(f"更新患者信息对象时出错: {str(e)}")
                return f"更新患者信息时出错: {str(e)}"

            # 3. 保存到数据库
            try:
                if self.conversation_id:
                    # 获取当前会话的消息
                    _, messages, image_urls, _, _ = load_conversation(self.conversation_id)
                    # 保存更新后的会话信息
                    success = save_conversation(
                        conversation_id=self.conversation_id,
                        patient_info=self.patient_info,
                        input_items=messages,
                        user_id=self.user_id,
                        image_urls=image_urls
                    )
                    if not success:
                        logger.error("保存患者信息到数据库失败")
                        return "更新患者信息成功，但保存到数据库失败"
                    logger.info(f"患者信息已更新到数据库，会话ID: {self.conversation_id}, 用户ID: {self.user_id}")
            except Exception as db_err:
                logger.error(f"更新患者信息到数据库时出错: {str(db_err)}")
                return f"更新患者信息成功，但保存到数据库时出错：{str(db_err)}"

            # 4. 返回更新信息
            output = "患者信息已更新：" + ", ".join(f"{k}={v}" for k, v in update_info.items())
            return output
        except Exception as e:
            logger.error(f"更新患者信息时发生未预期的错误: {str(e)}", exc_info=True)
            return f"更新患者信息时发生错误: {str(e)}"

    # -------------------------
    # LangGraph 节点：调用模型
    # -------------------------
    async def call_model(self, state: MessagesState) -> Dict[str, Any]:
        tools = [
            StructuredTool(
                name="generate_medical_record",
                func=self.generate_medical_record_tool,
                description="生成患者的病历记录。",
                args_schema=GenerateMedicalRecordInput,
                coroutine=self.generate_medical_record_tool,
                async_callable=True,
            ),
            StructuredTool(
                name="update_patient_info",
                func=self.update_patient_info_tool,
                description="""更新患者信息。此工具用于记录和更新患者的所有相关信息，包括：
1. 基本信息（姓名、年龄、性别等）
2. 就诊信息（科室、时间、主诉等）
3. 症状和病史（症状列表、既往病史等）
4. 体格检查（检查结果、辅助检查等）
5. 生活习惯（吸烟史、饮酒史等）
6. 女性特有信息（月经史等）
7. 生育史和婚姻史
8. 体格数据（体温、呼吸等）

所有字段都是可选的，只需要提供需要更新的信息即可。""",
                args_schema=UpdatePatientInfoInput,
                coroutine=self.update_patient_info_tool,
                async_callable=True,
            ),
        ]

        system_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的AI护士助手。你的主要职责是实时更新和记录患者信息。

重要提示：
1. 每当患者提供任何新的信息时，必须立即调用 update_patient_info 工具进行更新
2. 更新患者信息是最高优先级的任务，必须在其他操作之前完成
3. 即使患者只提供了一条信息，也要立即调用 update_patient_info 工具更新
4. 只有患者提出生成病历时，再调用 generate_medical_record 工具
5. 如果发现患者信息有任何变化，必须立即调用 update_patient_info 工具更新

主动采集信息指南：
1. 首次对话时，主动询问患者的基本信息（姓名、年龄、性别等）
2. 根据患者的主诉，主动询问相关的症状和病史
3. 对于女性患者，适时询问月经史和生育史
4. 询问患者的生活习惯（吸烟、饮酒等）
5. 询问家族史和过敏史
6. 记录体格检查结果和辅助检查结果
7. 根据病情需要，主动询问其他相关信息

请严格按照以上规则操作，确保患者信息的实时性和准确性。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.model, tools, system_prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
        )

        # 用 agent_executor 直接返回 AIMessage，交给 LangGraph
        result = await agent_executor.ainvoke({
            "input": state["messages"][-1].content,
            "chat_history": state["messages"][:-1],
            "agent_scratchpad": [],
        })
        if isinstance(result, dict) and "output" in result:
            state["messages"].append(AIMessage(content=result["output"]))
        return {"messages": state["messages"]}

    # -------------------------
    # 对外入口：流式处理消息
    # -------------------------
    async def process_message(self, messages: List[MessageInfo], user_id: str):
        """边生成边 yield SSE 行：token、工具进度、最终结果"""
        # 更新用户ID
        self.user_id = user_id

        # -------- Build LC messages --------
        lc_msgs = [
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content)
            for m in messages
        ]

        # -------- Build AgentExecutor --------
        tools = [
            StructuredTool(
                name="generate_medical_record",
                func=self.generate_medical_record_tool,
                description="""生成患者的病历记录。此工具会使用当前会话中收集到的所有患者信息。""",
                args_schema=GenerateMedicalRecordInput,
                coroutine=self.generate_medical_record_tool,
                async_callable=True,
            ),
            StructuredTool(
                name="update_patient_info",
                func=self.update_patient_info_tool,
                description="""更新患者信息。此工具用于记录和更新患者的所有相关信息，包括：
1. 基本信息（姓名、年龄、性别等）
2. 就诊信息（科室、时间、主诉等）
3. 症状和病史（症状列表、既往病史等）
4. 体格检查（检查结果、辅助检查等）
5. 生活习惯（吸烟史、饮酒史等）
6. 女性特有信息（月经史等）
7. 生育史和婚姻史
8. 体格数据（体温、呼吸等）

所有字段都是可选的，只需要提供需要更新的信息即可。""",
                args_schema=UpdatePatientInfoInput,
                coroutine=self.update_patient_info_tool,
                async_callable=True,
            ),
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的AI护士助手。你的主要职责是实时更新和记录患者信息。

重要提示：
1. 每当患者提供任何新的信息时，必须立即调用 update_patient_info 工具进行更新
2. 更新患者信息是最高优先级的任务，必须在其他操作之前完成
3. 即使患者只提供了一条信息，也要立即调用 update_patient_info 工具更新
4. 只有患者提出生成病历时，再调用 generate_medical_record 工具
5. 如果发现患者信息有任何变化，必须立即调用 update_patient_info 工具更新

主动采集信息指南：
1. 首次对话时，主动询问患者的基本信息（姓名、年龄、性别等）
2. 根据患者的主诉，主动询问相关的症状和病史
3. 对于女性患者，适时询问月经史和生育史
4. 询问患者的生活习惯（吸烟、饮酒等）
5. 询问家族史和过敏史
6. 记录体格检查结果和辅助检查结果
7. 根据病情需要，主动询问其他相关信息

历史信息使用指南：
1. 仔细阅读历史对话，了解已经收集到的患者信息
2. 避免重复询问已经提供过的信息
3. 根据已有信息，有针对性地询问缺失的信息
4. 如果发现历史信息有更新，及时调用 update_patient_info 工具更新

请严格按照以上规则操作，确保患者信息的实时性和准确性。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.model, tools, prompt)
        agent_exec = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )

        # -------- Stream events --------
        final_response = ""
        async for ev in agent_exec.astream_events(
            {
                "input": lc_msgs[-1].content  # ✅ 仅此项，memory 会自动注入历史
            },
            version="v1",
        ):
            kind = ev["event"]
            payload: dict

            # --- 1) LLM token（若模型支持流式） ---
            if kind in ("on_chat_model_stream", "on_llm_stream"):
                chunk_obj = ev["data"]["chunk"]
                text = getattr(chunk_obj, "content", str(chunk_obj))
                if not text:
                    continue  # 跳过空 token
                final_response += text  # 累积完整的响应
                payload = {
                    "type": "response_token",
                    "text": text,
                    "done": False,
                    "conversation_id": self.conversation_id,
                }

            # --- 2) Tool 开始 / 参数增量 / 结束 ---
            elif kind == "on_tool_start":
                payload = {
                    "type": "tool_call_start",
                    "id": ev["data"].get("id"),
                    "name": str(ev["data"].get("name", "")),
                    "done": False,
                    "conversation_id": self.conversation_id,
                }
            elif kind == "on_tool_stream":
                payload = {
                    "type": "tool_call_chunk",
                    "id": ev["data"].get("id"),
                    "index": ev["data"]["index"],
                    "args_delta": str(ev["data"]["chunk"]),   # 转成字符串
                    "done": False,
                    "conversation_id": self.conversation_id,
                }
            elif kind == "on_tool_end":
                payload = {
                    "type": "tool_output",
                    "id": ev["data"].get("id"),
                    "text": str(ev["data"]["output"]),        # 保证可 JSON
                    "done": False,
                    "conversation_id": self.conversation_id,
                }

            # --- 3) 自定义 writer() ---
            elif kind == "on_custom_stream":
                payload = {
                    "type": "tool_progress",
                    "data": ev["data"],                      # writer() 保证数据可 JSON
                    "conversation_id": self.conversation_id,
                }

            else:
                continue  # skip unhandled kinds

            # ---- send as plain JSON line ----
            yield json.dumps(payload) + "\n"

        # -------- final end --------
        yield json.dumps({
            "type": "end",
            "text": "",
            "done": True,
            "conversation_id": self.conversation_id,
        }) + "\n"

        # 保存对话到数据库
        if self.conversation_id and final_response:
            try:
                # 获取当前会话的消息
                _, existing_messages, image_urls, _, _ = load_conversation(self.conversation_id)
                
                # 添加用户消息（如果还没有添加）
                if messages and messages[-1].role == "user":
                    existing_messages.append(messages[-1])
                
                # 添加助手响应
                new_message = MessageInfo(
                    role="assistant",
                    content=final_response
                )
                existing_messages.append(new_message)
                
                # 保存更新后的会话信息
                save_conversation(
                    conversation_id=self.conversation_id,
                    patient_info=self.patient_info,
                    input_items=existing_messages,
                    user_id=self.user_id,
                    image_urls=image_urls
                )
                logger.info(f"对话已保存到数据库，会话ID: {self.conversation_id}, 用户ID: {self.user_id}")
            except Exception as db_err:
                logger.error(f"保存对话到数据库时出错: {str(db_err)}")


# =============================
# 全局方法：langchain_nurse_agent
# =============================

def nurse_agent_instance(cid: str) -> "NurseAgentGraph":
    if cid not in nurse_agent_graphs:
        nurse_agent_graphs[cid] = NurseAgentGraph(cid)
    return nurse_agent_graphs[cid]

nurse_agent_graphs: Dict[str, NurseAgentGraph] = {}


async def langchain_nurse_agent(messages: List[MessageInfo], conversation_id: str, user_id: str = "default_user") -> AsyncGenerator[str, None]:
    if not conversation_id:
        raise ValueError("未提供 conversation_id")
    agent_graph = nurse_agent_instance(conversation_id)
    async for chunk in agent_graph.process_message(messages, user_id):
        yield chunk

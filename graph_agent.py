from typing import Dict, List, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from config import settings
import logging

logger = logging.getLogger(__name__)

class GraphAgent:
    def __init__(self):
        # 初始化 LLM
        self.model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=settings.dashscope_model,
            temperature=0.7
        )
        
        # 创建状态图
        self.workflow = StateGraph(state_schema=MessagesState)
        
        # 添加模型节点
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        
        # 添加内存持久化
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("GraphAgent 初始化完成")
    
    def call_model(self, state: MessagesState) -> Dict[str, Any]:
        """调用模型处理消息"""
        try:
            response = self.model.invoke(state["messages"])
            return {"messages": response}
        except Exception as e:
            logger.error(f"模型调用失败: {e}", exc_info=True)
            raise
    
    async def process_message(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理消息并返回响应"""
        try:
            # 创建初始状态
            state = MessagesState(messages=messages)
            
            # 执行图
            result = await self.app.ainvoke(state)
            
            return result
        except Exception as e:
            logger.error(f"消息处理失败: {e}", exc_info=True)
            raise
    
    def get_memory(self) -> MemorySaver:
        """获取内存检查点"""
        return self.memory 
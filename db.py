# 文件：db.py
import mysql.connector
from mysql.connector import pooling, Error as DBError # 导入错误
from config import DB_CONFIG, settings # 导入更新后的配置
import json
from typing import List, Tuple, Optional, Dict, Any
from models import ConversationInfo, MessageInfo # 导入 MessageInfo
from patient_info import UserInfo
from dataclasses import asdict
import logging
import datetime # 导入 datetime

logger = logging.getLogger(__name__)

# --- 数据库连接池设置 ---
db_pool: Optional[pooling.MySQLConnectionPool] = None

def initialize_db_pool():
    """初始化数据库连接池。"""
    global db_pool
    if db_pool:
        logger.warning("数据库连接池已经初始化。")
        return
    try:
        logger.info(f"正在初始化数据库连接池 '{settings.db_pool_name}'，大小为 {settings.db_pool_size}")
        # 将连接池特定参数与连接参数分开传递
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=settings.db_pool_name,
            pool_size=settings.db_pool_size,
            **DB_CONFIG # 传递主机、用户、密码、数据库、端口等
        )
        logger.info("数据库连接池初始化尝试完成。")
        # 测试连接以验证连接池是否正常工作
        logger.info("正在测试从连接池获取连接...")
        conn = db_pool.get_connection()
        logger.info(f"成功从连接池获取测试连接 {conn.connection_id}。")
        conn.close() # 将连接返回给连接池
        logger.info("数据库连接池初始化并测试成功。")
    except DBError as err:
        logger.error(f"致命错误：初始化数据库连接池时出错：{err}")
        db_pool = None # 如果初始化失败，确保连接池为 None
        raise RuntimeError(f"数据库连接池初始化失败：{err}") # 停止应用启动

def get_db_connection() -> mysql.connector.pooling.PooledMySQLConnection:
    """从连接池获取连接。如果连接池不可用，则抛出 RuntimeError。"""
    if db_pool is None:
        logger.critical("严重错误：数据库连接池未初始化或初始化失败。")
        raise RuntimeError("数据库连接池不可用。")
    try:
        # logger.debug(f"正在从连接池 '{settings.db_pool_name}' 获取连接")
        conn = db_pool.get_connection()
        # logger.debug(f"从连接池获取到连接 {conn.connection_id}。")
        return conn
    except DBError as err:
        logger.error(f"从连接池获取连接时出错：{err}")
        raise RuntimeError(f"从连接池获取连接失败：{err}") # 明确指示失败

# --- 数据库操作 ---
# 注意：这些函数保持同步，它们将被调用它们的异步函数在 routers 中使用 asyncio.to_thread 包装。

def get_user_conversations(user_id: str, page: int = 1, page_size: int = 10) -> Tuple[List[ConversationInfo], int, int]:
    """获取用户的分页会话信息。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True) # 使用字典游标

        # 计算总数
        count_query = "SELECT COUNT(*) as total FROM conversations"
        count_params = ()
        if user_id and user_id != "default_user": # 根据您处理默认/所有用户的方式调整条件
            count_query += " WHERE user_id = %s"
            count_params = (user_id,)
        elif user_id == "default_user":
             logger.warning("正在获取 'default_user' 的会话。请考虑这是否是预期的行为。")
             # 决定 default_user 是否应该看到所有会话或特定会话
             # count_query += " WHERE user_id = %s" # 示例：只获取 default_user 的会话
             # count_params = (user_id,)
             pass # 示例：允许 default_user 查看所有会话（可能不安全）

        cursor.execute(count_query, count_params)
        result = cursor.fetchone()
        total_count = result['total'] if result else 0

        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
        offset = (page - 1) * page_size

        # 获取记录
        query = """
            SELECT conversation_id, patient_info, messages, created_at, updated_at, image_urls
            FROM conversations
        """
        params = []
        # 应用与计数逻辑一致的 WHERE 子句
        if user_id and user_id != "default_user":
            query += " WHERE user_id = %s"
            params.append(user_id)
        elif user_id == "default_user":
             # query += " WHERE user_id = %s" # 示例：只获取 default_user 的会话
             # params.append(user_id,)
             pass # 示例：允许 default_user 查看所有会话

        query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
        params.extend([page_size, offset])

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        conversations = []
        for row in rows:
            try:
                # 安全加载 JSON 并解析日期
                patient_data = json.loads(row['patient_info']) if row.get('patient_info') else {}
                image_urls = json.loads(row['image_urls']) if row.get('image_urls') else None
                created_at_str = row['created_at'].isoformat() if isinstance(row.get('created_at'), datetime.datetime) else str(row.get('created_at'))
                updated_at_str = row['updated_at'].isoformat() if isinstance(row.get('updated_at'), datetime.datetime) else str(row.get('updated_at'))

                conversations.append(
                    ConversationInfo(
                        conversation_id=row['conversation_id'],
                        created_at=created_at_str,
                        updated_at=updated_at_str,
                        patient_info=patient_data,
                        image_urls=image_urls
                    )
                )
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
                logger.error(f"处理会话行 ID {row.get('conversation_id', 'N/A')} 时出错：{e}", exc_info=True)
                continue # 跳过有问题的行

        return conversations, total_count, total_pages
    except DBError as err:
        logger.error(f"在 get_user_conversations 中发生数据库错误（用户：{user_id}）：{err}", exc_info=True)
        # 出错时返回空列表以避免调用者崩溃，但记录为严重错误
        return [], 0, 0
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
             # logger.debug(f"关闭/返回连接 {conn.connection_id} 到连接池。")
             conn.close() # 重要：将连接返回给连接池


def load_conversation(conversation_id: str) -> Tuple[UserInfo, List[MessageInfo], Optional[List[str]], Optional[str], Optional[str]]:
    """加载完整的会话详情：患者信息、消息、图片、时间戳。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT patient_info, messages, image_urls, created_at, updated_at FROM conversations WHERE conversation_id = %s",
            (conversation_id,)
        )
        row = cursor.fetchone()

        # 如果会话未找到或数据损坏，则使用默认值
        patient = UserInfo()
        items: List[MessageInfo] = []
        image_urls: Optional[List[str]] = None
        created_at: Optional[str] = None
        updated_at: Optional[str] = None

        if row:
            try:
                patient_data = json.loads(row['patient_info']) if row.get('patient_info') else {}
                msgs_raw = json.loads(row['messages']) if row.get('messages') else []
                image_urls = json.loads(row['image_urls']) if row.get('image_urls') else None
                created_at = row['created_at'].isoformat() if isinstance(row.get('created_at'), datetime.datetime) else str(row.get('created_at'))
                updated_at = row['updated_at'].isoformat() if isinstance(row.get('updated_at'), datetime.datetime) else str(row.get('updated_at'))

                # 兼容旧的 'symptom' 字段
                if 'symptom' in patient_data and 'symptoms' not in patient_data:
                     symptom_val = patient_data.pop('symptom')
                     patient_data['symptoms'] = [symptom_val] if isinstance(symptom_val, str) else []

                # 使用 Pydantic 加载到 UserInfo 以进行验证/默认值
                try:
                    # Pydantic 使用 UserInfo 中定义的默认值处理缺失字段
                    patient = UserInfo(**patient_data)
                except Exception as pydantic_err:
                    logger.warning(f"加载 UserInfo 时发生 Pydantic 验证错误 {conversation_id}：{pydantic_err}。使用部分/默认数据。")
                    # 如果需要严格验证，则仅防御性地加载有效字段
                    # valid_data = {k: v for k, v in patient_data.items() if hasattr(UserInfo, k)}
                    # patient = UserInfo(**valid_data)

                # 将消息组装成 MessageInfo 对象
                for m in msgs_raw:
                    if isinstance(m, dict) and 'role' in m and 'content' in m:
                        try:
                            items.append(MessageInfo(role=m['role'], content=m['content']))
                        except Exception as msg_err:
                             logger.warning(f"由于验证错误，跳过会话 {conversation_id} 中的无效消息格式：{msg_err} - 数据：{m}")
                    else:
                        logger.warning(f"跳过会话 {conversation_id} 中的无效消息结构：{m}")

            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                 logger.error(f"解码 JSON 或处理会话 {conversation_id} 的数据时出错：{e}", exc_info=True)
                 # 如果 JSON 或核心数据损坏，则返回默认的空数据
                 patient = UserInfo()
                 items = []
                 image_urls = None
                 created_at = None
                 updated_at = None
        else:
             logger.warning(f"在数据库中未找到会话 ID {conversation_id}。")

        return patient, items, image_urls, created_at, updated_at
    except DBError as err:
        logger.error(f"加载会话 {conversation_id} 时发生数据库错误：{err}", exc_info=True)
        raise # 重新抛出数据库错误，由调用者处理（例如，返回 HTTP 500）
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
             # logger.debug(f"关闭/返回连接 {conn.connection_id} 到连接池。")
             conn.close()


def save_conversation(conversation_id: str, patient_info: UserInfo, input_items: List[MessageInfo], user_id: str, image_urls: Optional[List[str]] = None) -> bool:
    """使用 INSERT ... ON DUPLICATE KEY UPDATE 保存或更新会话。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 对数据类对象（UserInfo）使用 asdict
        patient_json = json.dumps(asdict(patient_info), ensure_ascii=False, default=str)
        # 对 Pydantic 模型（MessageInfo）使用 model_dump
        msgs_json = json.dumps([item.model_dump() for item in input_items], ensure_ascii=False)
        imgs_json = json.dumps(image_urls, ensure_ascii=False) if image_urls is not None else None

        query = """
            INSERT INTO conversations (
                conversation_id, user_id, patient_info, messages, image_urls,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                user_id = VALUES(user_id),
                patient_info = VALUES(patient_info),
                messages = VALUES(messages),
                image_urls = VALUES(image_urls),
                updated_at = NOW()
        """
        params = (conversation_id, user_id, patient_json, msgs_json, imgs_json)

        cursor.execute(query, params)
        conn.commit()
        logger.info(f"会话 {conversation_id} 为用户 {user_id} 保存/更新成功。")
        return True

    except DBError as err:
        logger.error(f"为用户 {user_id} 保存会话 {conversation_id} 时发生数据库错误：{err}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logger.error(f"为用户 {user_id} 保存会话 {conversation_id} 时发生意外错误：{e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """在验证所有权后删除会话。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        # 显式事务
        # conn.autocommit = False
        cursor = conn.cursor(dictionary=True)

        # 首先验证所有权 - 对安全很重要
        cursor.execute("SELECT user_id FROM conversations WHERE conversation_id = %s", (conversation_id,))
        row = cursor.fetchone()

        if not row:
            logger.warning(f"删除尝试：未找到会话 {conversation_id}（用户：{user_id}）。")
            return False # 会话不存在

        # 检查所有权 - 如果 user_id 可能很复杂，请使用安全比较
        # 确保 'default_user' 不能删除会话，除非业务逻辑明确允许
        if row['user_id'] != user_id:
             # 安全：记录尝试但不要给攻击者信息
             logger.error(f"安全警报：用户 {user_id} 尝试删除由 {row['user_id']} 拥有的会话 {conversation_id}。")
             return False # 权限被拒绝

        # 执行删除
        logger.info(f"正在执行由所有者 {user_id} 删除会话 {conversation_id}。")
        cursor.execute("DELETE FROM conversations WHERE conversation_id=%s AND user_id=%s", (conversation_id, user_id))
        deleted_count = cursor.rowcount # 检查有多少行受到影响
        conn.commit() # 提交删除

        if deleted_count > 0:
             logger.info(f"会话 {conversation_id} 已由用户 {user_id} 成功删除。")
             return True
        else:
             # 如果所有权检查通过且没有竞争条件，这种情况应该不会发生
             logger.warning(f"已为会话 {conversation_id}（用户：{user_id}）执行删除命令，但影响了 0 行。")
             return False

    except DBError as err:
        # 记录删除过程中的特定数据库错误
        logger.error(f"为用户 {user_id} 删除会话 {conversation_id} 时发生数据库错误：{err}", exc_info=True)
        if conn:
            conn.rollback() # 出错时回滚
        return False # 在数据库错误时指示失败
    except Exception as e:
        # 记录任何其他意外错误
        logger.error(f"为用户 {user_id} 删除会话 {conversation_id} 时发生意外错误：{e}", exc_info=True)
        if conn:
            conn.rollback()
        return False # 指示失败
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            # logger.debug(f"关闭/返回连接 {conn.connection_id} 到连接池。")
            conn.close()
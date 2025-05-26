from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
from datetime import datetime

@dataclass
class UserInfo:
    """患者信息数据类，用于存储和管理患者的基本信息和医疗记录"""
    
    # 基本信息
    name: str = field(default="", metadata={"description": "患者姓名"})
    age: int = field(default=0, metadata={"description": "患者年龄"})
    gender: str = field(default="", metadata={"description": "患者性别"})
    occupation: str = field(default="", metadata={"description": "职业"})
    phone: str = field(default="", metadata={"description": "联系电话"})
    address: str = field(default="", metadata={"description": "住址"})
    
    # 就诊信息
    department: str = field(default="", metadata={"description": "就诊科室"})
    visit_time: str = field(default="", metadata={"description": "就诊时间"})
    chief_complaint: str = field(default="", metadata={"description": "主诉"})
    
    # 症状和病史
    symptoms: List[str] = field(default_factory=list, metadata={"description": "症状列表"})
    medical_history: str = field(default="", metadata={"description": "既往病史"})
    last_diagnosis: str = field(default="", metadata={"description": "最后诊断"})
    current_illness: str = field(default="", metadata={"description": "现病史"})
    family_history: str = field(default="", metadata={"description": "家族史"})
    allergy_history: str = field(default="", metadata={"description": "过敏史"})
    personal_history: str = field(default="", metadata={"description": "个人史"})
    
    # 体格检查
    physical_exam: str = field(default="", metadata={"description": "体格检查"})
    auxiliary_exam: str = field(default="", metadata={"description": "辅助检查"})
    treatment_plan: str = field(default="", metadata={"description": "治疗计划"})
    
    # 生活习惯
    smoking_history: Dict[str, str] = field(default_factory=lambda: {
        "status": "",    # 吸烟状态（是/否）
        "amount": "",    # 吸烟量
        "years": ""      # 吸烟年限
    })
    alcohol_history: Dict[str, str] = field(default_factory=lambda: {
        "status": "",     # 饮酒状态（是/否）
        "frequency": "",  # 饮酒频率
        "type": ""        # 饮酒类型
    })
    
    # 女性特有信息
    menstrual_history: Dict[str, str] = field(default_factory=lambda: {
        "menarche_age": "",        # 初潮年龄
        "last_menstrual_age": "",  # 末次月经年龄
        "cycle": "",               # 月经周期（天）
        "duration": "",            # 经期持续时间（天）
        "last_time": "",           # 末次月经时间
        "amount": "",              # 月经量描述
        "regularity": "",          # 规律性
        "dysmenorrhea": "",        # 痛经情况
        "menopause": ""            # 是否绝经
    })
    
    # 生育史
    fertility_history: Dict[str, str] = field(default_factory=lambda: {
        "pregnancy_times": "",     # 妊娠次数
        "delivery_times": "",      # 分娩次数
        "abortion_times": "",      # 流产次数
        "abortion_history": "",    # 流产史详情
        "premature_times": "",     # 早产次数
        "premature_history": "",   # 早产史详情
        "children_count": "",      # 现有子女数
        "delivery_method": "",     # 分娩方式
        "complications": ""        # 妊娠并发症
    })
    
    # 婚姻史
    marriage_history: Dict[str, str] = field(default_factory=lambda: {
        "status": "",              # 婚姻状况
        "marriage_age": "",        # 结婚年龄
        "spouse_health": ""        # 配偶健康状况
    })
    
    # 体格数据
    physical_data: Dict[str, str] = field(default_factory=lambda: {
        "temperature": "",         # 体温
        "respiration": "",         # 呼吸
        "pulse": "",              # 脉搏
        "blood_pressure_high": "", # 收缩压
        "blood_pressure_low": "",  # 舒张压
        "development": "",         # 发育情况
        "nutrition": "",           # 营养状况
        "consciousness": "",       # 意识状态
        "facial_expression": "",   # 面容表情
        "posture": "",             # 体位
        "gait": "",               # 步态
        "skin": "",               # 皮肤
        "lymph_nodes": ""         # 淋巴结
    })

    def to_dict(self) -> Dict:
        """将患者信息转换为字典格式"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }

    def update_from_dict(self, data: Dict) -> None:
        """从字典更新患者信息"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 性别标准化
def normalize_gender(gender_str: str) -> str:
    if not gender_str:
        return ""
    g = gender_str.strip()
    if g in ["女","女性","F","f","female","Female"]:
        return "女"
    if g in ["男","男性","M","m","male","Male"]:
        return "男"
    return gender_str

# 更新现病史字段
def _update_illness_field(current_text: str, field_keyword: str, new_value: str) -> str:
    if not field_keyword.endswith('：'):
        field_keyword += '：'
    pattern = re.compile(r"(" + re.escape(field_keyword) + r")(.*?)(?=\n[^\n]+：|\Z)", re.DOTALL)
    value = new_value.strip() if new_value and new_value.strip() else "未详"
    match = pattern.search(current_text)
    if match:
        start, end = match.start(2), match.end(2)
        updated = current_text[:start] + value + current_text[end:]
    else:
        sep = '\n' if current_text and current_text.endswith('\n') else '\n'
        updated = current_text + sep + field_keyword + value
    lines = [l.strip() for l in updated.strip().split('\n') if l.strip()]
    return '\n'.join(lines)

# 更新患者信息工具函数
def update_seat(context, **kwargs) -> str:
    updates = []
    # 示例: 更新姓名、年龄、性别、症状等
    if 'name' in kwargs and kwargs['name'] is not None:
        context.name = kwargs['name']; updates.append(f"姓名={context.name}")
    if 'age' in kwargs and kwargs['age'] is not None:
        context.age = kwargs['age']; updates.append(f"年龄={context.age}")
    if 'gender' in kwargs and kwargs['gender'] is not None:
        context.gender = normalize_gender(kwargs['gender']); updates.append(f"性别={context.gender}")
    if 'symptoms' in kwargs and kwargs['symptoms'] is not None:
        context.symptoms = kwargs['symptoms']; updates.append(f"症状={','.join(context.symptoms)}")
    # 现病史字段更新示例
    ci = context.current_illness or ''
    if 'onset_time' in kwargs and kwargs['onset_time'] is not None:
        ci = _update_illness_field(ci, '起病时间', kwargs['onset_time'])
    if ci != (context.current_illness or ''):
        context.current_illness = ci; updates.append('现病史已更新')
    # 更多字段参照原实现
    return '已更新：' + '，'.join(updates) if updates else '无更新'
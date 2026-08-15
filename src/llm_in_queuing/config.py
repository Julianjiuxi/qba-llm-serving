# 导入annotations特性，允许在类型注解中使用尚未定义的类（Python 3.7+支持，向后兼容写法）
from __future__ import annotations

# 导入Path类，用于处理文件系统路径，比原生字符串路径更安全、功能更丰富
from pathlib import Path
# 导入Any类型，用于表示任意类型的注解场景
from typing import Any

# 导入第三方库yaml，用于解析YAML格式的配置文件
import yaml


# 定义加载YAML配置的函数，参数path可以是字符串或Path类型，返回值是键为字符串、值为任意类型的字典
# 语法说明：str | Path是Python 3.10+支持的联合类型写法，表示参数支持这两种类型；-> 标注函数返回值类型
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    # 将输入的路径转换为Path对象，统一后续路径处理逻辑，无论输入是字符串还是已有的Path对象都能兼容
    p = Path(path)
    # 读取路径对应的文件文本内容，指定utf-8编码避免中文乱码，用yaml.safe_load解析为Python对象
    # safe_load会限制反序列化的对象类型，避免恶意YAML文件执行任意代码，是安全的解析方式
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    # 如果YAML文件是空内容，yaml.safe_load会返回None，此时返回空字典保证后续逻辑不会报错
    if data is None:
        return {}
    # 检查解析后的数据是否是字典类型，YAML的根节点必须是键值对集合才符合配置文件的规范
    # 如果不是字典，抛出类型错误并提示实际的类型，type(data).__name__会获取对象的类名字符串，让错误信息更清晰
    if not isinstance(data, dict):
        raise TypeError(f"Config YAML must be a mapping, got {type(data).__name__}")
    # 所有校验通过，返回解析好的配置字典
    return data


# 定义统一的配置加载入口，目前内部直接调用YAML加载函数，是为了后续扩展预留的抽象层
# 比如未来需要支持JSON、TOML等其他格式的配置文件，只需要修改这个函数根据文件后缀分发即可，调用方无需修改代码
def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml_config(path)

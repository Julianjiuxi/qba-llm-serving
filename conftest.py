"""把本仓库的 ``src`` 置于 ``sys.path`` 最前，确保测试导入的是当前工作副本。

CI 通过 ``pip install -e .`` 安装本仓库；但本地环境若残留了其它同名
``llm_in_queuing`` 包（例如早期 ``LLM_in_Queuing`` 仓库的 editable 安装），
``import llm_in_queuing`` 可能解析到错误源码。这里显式固定导入来源，
使本地与 CI / 服务器的行为一致。
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
_src_str = str(_SRC)
if _src_str in sys.path:
    sys.path.remove(_src_str)
sys.path.insert(0, _src_str)

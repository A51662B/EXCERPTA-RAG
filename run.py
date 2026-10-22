# -*- coding: utf-8 -*-
"""项目根入口：不用设 PYTHONPATH 也能跑。

    python run.py structure --from-leveled data/blocks_leveled.json --dump data/blocks_structured.json

等价于 `python -m paper_rag.cli ...`，只是它自己把 src 加进了 sys.path。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

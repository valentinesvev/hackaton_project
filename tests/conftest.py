from pathlib import Path
import sys

# Позволяет запускать pytest из корня репозитория,
# хотя production-код лежит в ./src и импортируется как `core`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

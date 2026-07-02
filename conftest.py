import sys
from pathlib import Path

# Let pytest import flash_attention without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).parent))

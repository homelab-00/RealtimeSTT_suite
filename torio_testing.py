import torch
import torio

print(f"PyTorch version: {torch.__version__}")
print(f"torio imported successfully: {torio.__version__ if hasattr(torio, '__version__') else 'No version info'}")

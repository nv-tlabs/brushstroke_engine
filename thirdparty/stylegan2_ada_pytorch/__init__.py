import os
import sys
stylegan_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if stylegan_root not in sys.path:
    sys.path.insert(0, stylegan_root)

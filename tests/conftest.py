"""
Configuration for pytest.
"""

import os
import sys
from pathlib import Path

# Add package root directory to PYTHONPATH
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

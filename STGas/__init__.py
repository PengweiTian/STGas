import os
import time

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

_this_year = time.strftime("%Y")
__version__ = "1.0.0-alpha"
__author__ = "PengweiTian"
__author_email__ = "1677909891@qq.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2024-{_this_year}, {__author__}."
__homepage__ = "https://github.com/PengweiTian/ST_Gas"

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__homepage__",
    "__license__",
    "__version__",
]

"""Pytest configuration: make the flat source modules importable.

The source files live in the project root (one level up from ``tests/``), so
we add that directory to ``sys.path``. This lets tests do ``from
matching_engine import MatchingEngine`` without a package install.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

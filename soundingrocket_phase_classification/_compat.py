# -*- coding: utf-8 -*-
"""
_compat.py  — Windows UTF-8 console compatibility bootstrap.

Import this at the top of any script that uses Unicode characters
(→, ✓, █, etc.) to ensure they render correctly on Windows terminals
regardless of the system default encoding (cp1252).

Usage:
    import _compat  # must be first import (before config, utils, etc.)
"""
import sys
import io

# Reconfigure stdout/stderr to UTF-8 if the current encoding isn't already.
# On Windows with cp1252, this prevents UnicodeEncodeError for → ✓ ✗ █ etc.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
elif sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

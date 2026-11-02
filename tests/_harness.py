"""Crash-safe suite entry for ``main()``, shared by every suite here.

``run()`` turns a crash into a named ``<TOKEN> FAIL`` line and ``sys.exit(1)``,
for the reason ``run_cli``'s docstring gives (``cli/_common.py``).

Only ``main()`` is wrapped. A module-level failure (a broken core import, a
missing dependency) escapes before ``run()`` is reached and still exits 0, so
``tests/run_all.py`` trusts a suite only when the exit code is 0 AND the
``<TOKEN> OK`` line printed.
"""
import sys
import traceback


def run(main, token):
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("%s FAIL: crashed before reaching a verdict" % token)
        sys.exit(1)

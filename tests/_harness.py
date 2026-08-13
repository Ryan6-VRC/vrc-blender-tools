"""Crash-safe suite entry for ``main()``, shared by every suite here.

Blender exits 0 on an unhandled ``--python`` script exception (measured on
5.2.0): the traceback prints, ``Blender quit`` follows, and anything keyed on
the exit code reads the crash as a pass. ``run()`` converts a crash into the
failure it is — traceback kept, a named ``<TOKEN> FAIL`` line, ``sys.exit(1)``.

Only ``main()`` is wrapped. A module-level failure (a broken core import, a
missing dependency) escapes before ``run()`` is reached and still exits 0 —
there the missing ``<TOKEN> OK`` line is the failure signal, and
``run_all.py``'s token check is what catches it.

``SystemExit`` passes through untouched: a suite's own ``sys.exit(1)`` after
printing its FAIL lines is already a verdict, not a crash.

``tests/run_all.py`` is the other half of the contract: it trusts a suite only
when the exit code is 0 AND the ``<TOKEN> OK`` line printed.
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

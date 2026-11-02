import gc
import importlib
import subprocess
import sys

FROZEN_AT_IMPORT = gc.get_freeze_count()


def test_preload_freezes_inherited_objects():
    # Runs in a subprocess: freezing this test process would change later GC behaviour.
    code = (
        "import gc, importlib; before = gc.get_freeze_count(); "
        "importlib.import_module('metasl.preload'); "
        "import metasl.phase; print(before, gc.get_freeze_count())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    before, after = map(int, out.stdout.split())
    assert after - before > 10000


def test_preload_module_is_importable_name():
    assert importlib.util.find_spec("metasl.preload") is not None
    assert gc.get_freeze_count() == FROZEN_AT_IMPORT  # this process must stay unfrozen

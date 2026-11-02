"""Share frozen imports through forkserver; models and groups live in workers."""

import gc
import torch  # noqa: F401 -- preload native libraries before forking workers.
from metasl import phase  # noqa: F401 -- share the imported worker graph.

gc.collect()
gc.freeze()

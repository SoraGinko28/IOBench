# 3B · 4 ranks · shared safetensors, one rerun (Meta ring 64 MiB vs CPU)

Driver: `rerun_3b4r.py`. Plot: `rerun-3b-4r-save-load.png`. Save 38.83 / 39.34 s (CPU / Meta), writes end
17.6 / 13.9 s, cold load 37.39 / 37.33 s. CPU warm load failed the validity check twice (>64 KiB timed
block reads), so no warm-load number. Raw traces are not committed.

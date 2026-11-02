# Split index write correction

The first measurement pass completed its 120 single-file calls. Both optional
split smoke attempts failed while writing the HF index: `RecordedStream.write`
counts bytes with `memoryview`, whereas `json.dump` sent Python text strings.
The checkpoint payload writes had already completed. The first-pass logs and
`source-after-measurements.json` remain unchanged.

Only the `len(self.headers) > 1` branch in
`metasl/gdt_shared_hf_storage.py:395` changed:

```diff
-            with self.fs.create_stream(path, "w") as stream:
-                json.dump(index, stream, indent=2)
-                stream.write("\n")
+            encoded_index = (json.dumps(index, indent=2) + "\n").encode("utf-8")
+            with self.fs.create_stream(path, "wb") as stream:
+                stream.write(encoded_index)
```

The serialized index text, payload file bytes, payload IO, and existing flush /
fsync ordering are unchanged. The main single-file branch never executes this
code. Split index IO is now one recorded binary write followed by its existing
flush and fsync. Retried split smoke measurements use separate output paths.

SHA-256 of `metasl/gdt_shared_hf_storage.py`:

| State | SHA-256 |
| --- | --- |
| First pass; recorded in `source-after-measurements.json` | `3e2f2179c42d3f8e932e8f33f75b78a2110cc63ae97896ecfc725a39e950be43` |
| Split-only correction | `2cc71d5654eb168c37277d9ee2b73841d57702c5368181370b139d37c97504f9` |

The existing 40-byte split unit test now enables `RecordedFileSystem` lifecycle
and actual-open tracing with fsync enabled. It checks native safetensors byte
identity, the HF index and its rejected path traversal, and exactly one open,
write, fsync, and close for the index. Root runs this targeted test before the
split retries; no independent test or measurement was started by this agent.

SHA-256 of `tests/test_shared_hf_storage.py`: before
`ab23a4f9fc4d90bae9c441f3d70e66f92b8fc3fe68908fbf1f17fc25ec03d970`;
after `b3862c7dd1578700c91c302d554aa34c3075537f225d6417996813a7740b72e8`.

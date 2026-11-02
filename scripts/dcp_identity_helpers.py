"""Streaming BLAKE2b-256 and exact half-open byte differences for DCP audits."""

import dataclasses
import hashlib

CHUNK = 8 * 1024**2


def digest_stream(stream, length=None):
    digest = hashlib.blake2b(digest_size=32)
    remaining = length
    while remaining is None or remaining > 0:
        block = stream.read(CHUNK if remaining is None else min(CHUNK, remaining))
        if not block:
            if remaining:
                raise EOFError("Truncated range while hashing")
            break
        digest.update(block)
        if remaining is not None:
            remaining -= len(block)
    return digest.hexdigest()


def byte_diffs(left, right, offset=0):
    """Maximal contiguous differing runs, including added/deleted tails."""
    result, start = [], None
    for index in range(max(len(left), len(right)) + 1):
        differs = index < max(len(left), len(right)) and (
            index >= min(len(left), len(right)) or left[index] != right[index]
        )
        if differs and start is None:
            start = index
        if not differs and start is not None:
            result.append(
                dict(
                    start=offset + start,
                    end=offset + index,
                    left_hex=left[start:index].hex(),
                    right_hex=right[start:index].hex(),
                )
            )
            start = None
    return result


def fields(value):
    """Expose every dataclass field, dict key and sequence element as JSON data."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: fields(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): fields(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fields(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def field_diffs(left, right, path=""):
    if type(left) is not type(right):
        return [dict(field=path, left=left, right=right)]
    if isinstance(left, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append(dict(field=child, left=left.get(key), right=right.get(key)))
            else:
                result.extend(field_diffs(left[key], right[key], child))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [dict(field=path, left=left, right=right)]
        return [
            diff
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for diff in field_diffs(a, b, f"{path}[{i}]")
        ]
    return [] if left == right else [dict(field=path, left=left, right=right)]

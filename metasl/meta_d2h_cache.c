/* Opt-in Meta simulator helper; x86_64 has CLFLUSH and 64-byte cache lines.
 * Flush every intersecting line, including partial first/last lines. The fence
 * completes writeback/invalidation before the host callback may read the range.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

__attribute__((target("clflushopt")))
static void flush_opt(uintptr_t start, uintptr_t end) {
    for (uintptr_t p = start; p < end; p += 64)
        _mm_clflushopt((void *)p);
    _mm_sfence();
}

void metasl_flush(void *address, size_t size) {
    if (!size) return;
    uintptr_t start = (uintptr_t)address & ~(uintptr_t)63;
    uintptr_t end = (uintptr_t)address + size;
    if (__builtin_cpu_supports("clflushopt")) {
        flush_opt(start, end);
        return;
    }
    for (uintptr_t p = start; p < end; p += 64)
        _mm_clflush((const void *)p);
    _mm_mfence();
}

void metasl_zero_and_flush(void *address, size_t size) {
    memset(address, 0, size);
    metasl_flush(address, size);
}

void metasl_zero_non_temporal(void *address, size_t size) {
    unsigned char *p = address;
    while (size && ((uintptr_t)p & 15)) {
        *p++ = 0;
        --size;
    }
    __m128i zero = _mm_setzero_si128();
    while (size >= 16) {
        _mm_stream_si128((__m128i *)p, zero);
        p += 16;
        size -= 16;
    }
    while (size--) *p++ = 0;
    _mm_sfence();
}

uint64_t metasl_read_cachelines(const void *address, size_t size) {
    const volatile uint64_t *p = address;
    uint64_t sum = 0;
    for (size_t offset = 0; offset + 8 <= size; offset += 64)
        sum ^= p[offset / 8];
    return sum;
}

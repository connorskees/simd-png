This project explores an implementation of the PNG filters using AVX2.
Existing implementations of these filters in PNG decoders like `libpng` are only
capable of decoding up to 4 bytes at a time for 8-bit RGBA pixels.

The `up` filter is simple enough that it is trivially auto-vectorized by modern
compilers; however, the `sub`, `average`, and `paeth` filters are sufficiently
complex as to make it very unlikely for auto-vectorization to occur.

By using a modified implementation of vectorized prefix-sum, we are able to decode
the `sub` filter 32 bytes at a time for 8-bit RGBA pixels.

Based on benchmarks done on a dedicated Linux server, this implementation of the
`sub` filter is around 20-25% faster than existing SSE4.1 implementations.

Below are the timings for a buffer of size 2**20 (1,048,576 bytes):

```
RUSTFLAGS='-C target-cpu=native' cargo bench

running 3 tests
test tests::bench_baseline_memcpy               ... bench:      62,683 ns/iter (+/- 18,025)
test tests::bench_sub_sse2                      ... bench:      86,519 ns/iter (+/- 2,527)
test tests::bench_sub_sse_prefix_sum_no_extract ... bench:      69,864 ns/iter (+/- 1,696)
```

A similar performance win of 20-25% can be seen with buffers as small as 2**9
(512) and for values that are not powers of 2. 

These benchmarks live within this repository and can be reproduced by running
the above command.

The algorithm itself and the intuition behind it are described in the
[associated blog post](https://connorskees.github.io/blog/avx-png-filter/).
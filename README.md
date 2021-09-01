This project explores an implementation of the PNG filters using AVX2.
Existing implementations of these filters in PNG decoders (i.e. `libpng`) are only
capable of decoding up to 4 bytes at a time for 8-bit RGBA pixels.

The `up` filter is simple enough that it is trivially auto-vectorized by modern
compilers; however, the `sub`, `average`, and `paeth` filters are sufficiently
complex as to make it very unlikely for auto-vectorization to occur.

By using a modified implementation of vectorized prefix-sum, we are able to decode
the `sub` filter 32 bytes at a time for 8-bit RGBA pixels.

The intitial results of this implementation appear very promising -- I have so far found it possible to
achieve a speedup of _5 orders of magnitude_ when comparing a vectorized
implementation of the `sub` filter to a simple scalar implementation, given
an input of 8 million 8-bit RGBA pixels. For smaller inputs (in the range of ~500 pixels), the performance
of the scalar implementation and the vectorized implementation seem to be quite similar.

I intend to compare this to existing SSE2 implementations that operate on 4 bytes at a time.
Additionally, I hope to exercise this implementation on more realistic inputs.

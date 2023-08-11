#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::convert::TryInto;

use crate::BYTES_PER_PIXEL;

pub fn up(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    // #[cfg(target_arch = "x86_64")]
    // if is_x86_feature_detected!("avx2") {
    //     up_avx(prev, raw_row, decoded_row);
    //     return;
    // }

    if prev.is_empty() {
        decoded_row[..].copy_from_slice(raw_row);
        return;
    }

    for i in 0..decoded_row.len() {
        let prev = prev[i];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}

pub unsafe fn up_unsafe(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    if prev.is_empty() {
        decoded_row[..].copy_from_slice(raw_row);
        return;
    }

    for i in 0..decoded_row.len() {
        let prev = *prev.get_unchecked(i);

        *decoded_row.get_unchecked_mut(i) = raw_row.get_unchecked(i).wrapping_add(prev)
    }
}

#[cfg(target_arch = "x86_64")]
pub fn up_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let offset = decoded_row.len() % 32;
    let len = raw_row.len();

    let mut up: __m256i;
    let mut curr: __m256i;

    let mut i = 0;

    let f = decoded_row.len() - offset;

    while i != f {
        debug_assert!(i + 32 < prev.len());

        up = unsafe { _mm256_loadu_si256(prev.get_unchecked(i) as *const _ as *const __m256i) };
        curr =
            unsafe { _mm256_loadu_si256(raw_row.get_unchecked(i) as *const _ as *const __m256i) };

        let sum = unsafe { _mm256_add_epi8(up, curr) };

        unsafe {
            _mm256_storeu_si256(
                decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m256i,
                sum,
            )
        }

        i += 32;
    }

    for i in (decoded_row.len() - offset)..decoded_row.len() {
        let prev = unsafe { *prev.get_unchecked(i) };

        unsafe { *decoded_row.get_unchecked_mut(i) = raw_row.get_unchecked(i).wrapping_add(prev) }
    }
}

pub fn sub(raw_row: &[u8], decoded_row: &mut [u8]) {
    // #[cfg(target_arch = "x86_64")]
    // if is_x86_feature_detected!("avx2") {
    //     unsafe { sub_avx(raw_row, decoded_row) };
    //     return;
    // }

    for i in 0..BYTES_PER_PIXEL {
        decoded_row[i] = raw_row[i];
    }

    for i in BYTES_PER_PIXEL..decoded_row.len() {
        let prev = decoded_row[i - BYTES_PER_PIXEL];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}

unsafe fn load4(x: &[u8; 4]) -> __m128i {
    let tmp = i32::from_le_bytes(*x);
    _mm_cvtsi32_si128(tmp)
}

unsafe fn store4(x: &mut [u8; 4], v: __m128i) {
    let tmp = _mm_cvtsi128_si32(v);
    x.copy_from_slice(&tmp.to_le_bytes());
}

unsafe fn store4_ported(x: &mut [u8], v: __m128i) {
    let tmp = _mm_cvtsi128_si32(v);
    x.get_unchecked_mut(..4).copy_from_slice(&tmp.to_le_bytes());
}

#[allow(unused_assignments)]
#[target_feature(enable = "sse2")]
pub unsafe fn sub_sse2(raw: &[u8], current: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("sse2"));
    let (mut a, mut d) = (_mm_setzero_si128(), _mm_setzero_si128());

    for (raw, out) in raw.chunks_exact(4).zip(current.chunks_exact_mut(4)) {
        a = d;
        d = load4(raw.try_into().unwrap());
        d = _mm_add_epi8(d, a);
        store4(out.try_into().unwrap(), d);
    }
}

pub unsafe fn sub_sse2_ported(raw_row: &[u8], decoded_row: &mut [u8]) {
    let mut a: __m128i;
    let mut d = _mm_setzero_si128();

    let mut rb = raw_row.len() + 4;
    let mut idx = 0;

    while rb > 4 {
        a = d;
        d = load4(&[
            *raw_row.get_unchecked(idx),
            *raw_row.get_unchecked(idx + 1),
            *raw_row.get_unchecked(idx + 2),
            *raw_row.get_unchecked(idx + 3),
        ]);
        d = _mm_add_epi8(d, a);
        store4_ported(&mut decoded_row.get_unchecked_mut(idx..), d);

        idx += 4;
        rb -= 4;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_avx(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));
    let off = raw_row.len() % 32;

    decoded_row
        .get_unchecked_mut(0..BYTES_PER_PIXEL)
        .copy_from_slice(raw_row.get_unchecked(0..BYTES_PER_PIXEL));

    for i in BYTES_PER_PIXEL..off {
        *decoded_row.get_unchecked_mut(i) = raw_row
            .get_unchecked(i)
            .wrapping_add(*decoded_row.get_unchecked(i - BYTES_PER_PIXEL));
    }

    // todo: off > 0 && off < BYTES_PER_PIXEL
    let mut last = if off > BYTES_PER_PIXEL {
        i32::from_be_bytes([
            *decoded_row.get_unchecked(off - 1),
            *decoded_row.get_unchecked(off - 2),
            *decoded_row.get_unchecked(off - 3),
            *decoded_row.get_unchecked(off - 4),
        ])
    } else {
        0
    };

    debug_assert_eq!((raw_row.len() - off) % 32, 0);

    let mut i = off;
    let len = raw_row.len();

    while i != len {
        let mut x: __m256i;
        let mut sum: __m256i;

        // load 32 bytes from un-edited array
        x = _mm256_loadu_si256(raw_row.get_unchecked(i) as *const _ as *const __m256i);

        // do prefix sum. note that this happens on 2 16-byte sections at a time
        x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
        x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

        // extract the last 4 elements of the first 16-byte section
        // todo: experiment with impl using permute and setting lower half to zero
        let b = _mm256_extract_epi32::<3>(x);

        // load register with the last 4 byte propagated to the lower bytes
        let f = _mm256_set_epi32(b, b, b, b, 0, 0, 0, 0);
        // load register with the last 4 bytes of the previous iteration
        let f2 = _mm256_set1_epi32(last);

        // add above 2 registers
        sum = _mm256_add_epi8(f, f2);

        // add above sum and
        sum = _mm256_add_epi8(x, sum);

        _mm256_storeu_si256(
            decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m256i,
            sum,
        );

        last = _mm256_extract_epi32::<7>(sum);

        i += 32;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_avx_attempt_3(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let len = raw_row.len();

    let raw_row = &raw_row[16..];
    // let decoded_row = &mut decoded_row[16..];

    // let (start, raw_row, end) = raw_row.align_to::<u32>();
    // let raw_row = std::mem::transmute::<&[u32], &[u8]>(raw_row);
    // let (start, decoded_row, end) = decoded_row.align_to_mut::<u32>();
    // let decoded_row: &mut [u8] = std::mem::transmute::<&mut [u32], &mut [u8]>(decoded_row);
    // assert_eq!(std::mem::align_of_val(&raw_row[1..]), 32);

    // assert_eq!(raw_row.len(), decoded_row.len());

    let mut last = 0;

    let mut i = 0;
    // assert_eq!(len, raw_row.len());
    let len = raw_row.len();

    // let mut s = _mm_setzero_si128();

    // assert_eq!(len % 32, 0);
    let mut x: __m256i;

    while i < len {
        //     // let mut sum: __m256i;

        // load 32 bytes from un-edited array
        x = _mm256_loadu_si256(raw_row.get_unchecked(i) as *const _ as *const __m256i);

        // do prefix sum. note that this happens on 2 16-byte sections at a time
        x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
        x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

        // extract the last 4 elements of the first 16-byte section
        let b = _mm256_extract_epi32::<3>(x);
        // load register with the last 4 byte propagated to the lower bytes
        x = _mm256_add_epi8(x, _mm256_set_epi32(b, b, b, b, 0, 0, 0, 0));

        x = _mm256_add_epi8(_mm256_set1_epi32(last), x);

        last = _mm256_extract_epi32::<7>(x);

        _mm256_storeu_si256(
            decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m256i,
            x,
        );

        i += 32;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn baseline_memcpy(raw_row: &[u8], decoded_row: &mut [u8]) {
    decoded_row
        .get_unchecked_mut(0..raw_row.len())
        .copy_from_slice(&*raw_row);
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_avx_attempt_2(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let mut last = 0; // _mm256_setzero_si256();
    let mut x: __m256i;
    let mut sum: __m256i;

    for (raw, out) in raw_row
        .chunks_exact(32)
        .zip(decoded_row.chunks_exact_mut(32))
    {
        x = _mm256_loadu_si256(raw as *const _ as *const __m256i);

        x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
        x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

        let b = _mm256_extract_epi32::<3>(x);

        let f = _mm256_set_epi32(b, b, b, b, 0, 0, 0, 0);
        let f2 = _mm256_set1_epi32(last);

        sum = _mm256_add_epi8(f, f2);

        sum = _mm256_add_epi8(x, sum);

        _mm256_storeu_si256(out as *mut _ as *mut __m256i, sum);

        last = _mm256_extract_epi32::<7>(sum);
    }
}

pub unsafe fn sub_sse2_with_init(raw: &[u8], current: &mut [u8], init: &[u8; 4]) {
    debug_assert!(is_x86_feature_detected!("sse2"));
    let (mut a, mut d) = (_mm_setzero_si128(), load4(init));

    for (raw, out) in raw.chunks_exact(4).zip(current.chunks_exact_mut(4)) {
        a = d;
        d = load4(raw.try_into().unwrap());
        d = _mm_add_epi8(d, a);
        store4(out.try_into().unwrap(), d);
    }
}

pub unsafe fn sub_avx_attempt_5(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let mut last = 0;
    let mut x: __m256i;

    let len = raw_row.len();
    let mut i = 0;

    while len.saturating_sub(32) > i {
        x = _mm256_loadu_si256(raw_row.get_unchecked(i) as *const _ as *const __m256i);

        x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
        x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

        let b = _mm256_extract_epi32::<3>(x);
        x = _mm256_add_epi8(_mm256_set_epi32(b, b, b, b, 0, 0, 0, 0), x);

        x = _mm256_add_epi8(_mm256_set1_epi32(last), x);

        _mm256_storeu_si256(
            decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m256i,
            x,
        );

        last = _mm256_extract_epi32::<7>(x);

        i += 32;
    }

    if i != len {
        sub_sse2_with_init(&raw_row[i..], &mut decoded_row[i..], &last.to_le_bytes());
    }
}

pub unsafe fn sub_sse_prefix_sum_no_extract(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let mut last = _mm_setzero_si128();
    let mut x: __m128i;

    let len = raw_row.len();
    let mut i = 0;

    let offset = len % 16;
    if offset != 0 {
        sub_sse2(
            raw_row.get_unchecked(..offset),
            decoded_row.get_unchecked_mut(..offset),
        );
        last = _mm_castps_si128(_mm_broadcast_ss(
            &*(decoded_row.get_unchecked(offset - 4) as *const _ as *const f32),
        ));
        i = offset;
    }

    while len != i {
        // load 16 bytes from array
        x = _mm_loadu_si128(raw_row.get_unchecked(i) as *const _ as *const __m128i);

        // do prefix sum
        x = _mm_add_epi8(_mm_slli_si128::<4>(x), x);
        x = _mm_add_epi8(_mm_slli_si128::<{ 2 * 4 }>(x), x);

        // accumulate for previous chunk of 16 bytes
        x = _mm_add_epi8(x, last);

        // shift right by 12 bytes and then broadcast the lower 4 bytes
        // to the rest of the register
        last = _mm_srli_si128::<12>(x);
        last = _mm_broadcastd_epi32(last);

        _mm_storeu_si128(
            decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m128i,
            x,
        );

        i += 16;
    }
}

// void prefix(int *a, int n) {
//     v4i s = _mm_setzero_si128();

//     for (int i = 0; i < B; i += 8)
//         prefix(&a[i]);

//     for (int i = B; i < n; i += 8) {
//         prefix(&a[i]);
//         s = accumulate(&a[i - B], s);
//         s = accumulate(&a[i - B + 4], s);
//     }

//     for (int i = n - B; i < n; i += 4)
//         s = accumulate(&a[i], s);
// }

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_avx_attempt_4(raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("sse4.2"));

    let mut last = _mm_setzero_si128();
    // let mut last_i32 = 0;
    let mut x: __m128i;
    // let mut sum: __m128i;

    for (raw, out) in raw_row
        .chunks_exact(16)
        .zip(decoded_row.chunks_exact_mut(16))
    {
        x = _mm_loadu_si128(raw as *const _ as *const __m128i);

        x = _mm_add_epi8(_mm_slli_si128::<4>(x), x);
        x = _mm_add_epi8(_mm_slli_si128::<{ 2 * 4 }>(x), x);

        // let b = _mm_extract_epi32::<3>(x);

        // x = _mm_add_epi8(x, _mm_set_epi32(b, b, 0, 0));
        x = _mm_add_epi8(x, last);
        // x = _mm_add_epi8(x, _mm_set1_epi32(last_i32));

        // let f = _mm_set_epi32(b, b, 0, 0);
        // let f2 = _mm_set1_epi32(last);

        // sum = _mm_add_epi8(f, f2);

        // unsafe fn eq(a: __m128i, b: __m128i) -> bool {
        //     let a = std::mem::transmute::<__m128i, __m128>(a);
        //     let b = std::mem::transmute::<__m128i, __m128>(b);
        //     _mm_movemask_ps(_mm_cmpeq_ps(a, b)) != 0xF
        // }

        // sum = _mm_add_epi8(x, sum);
        // todo: try _mm_shuffle_epi32
        last = _mm_srli_si128::<12>(x);
        last = _mm_broadcastd_epi32(last);
        // assert!(eq(last, _mm_set1_epi32(_mm_extract_epi32::<3>(x))));
        // assert_eq!(last, _mm_set1_epi32(_mm_extract_epi32::<3>(x)));
        // last_i32 = _mm_extract_epi32::<3>(x);

        // assert_eq!(last_i32, _mm_extract_epi32::<3>(last));

        _mm_storeu_si128(out as *mut _ as *mut __m128i, x);
    }
}

/// The Average filter uses the average of the two neighboring pixels (left and
/// above) to predict the value of a pixel.
pub fn average(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    // #[cfg(target_arch = "x86_64")]
    // if is_x86_feature_detected!("avx2") {
    //     average_avx(prev, raw_row, decoded_row);
    //     return;
    // }

    for i in 0..BYTES_PER_PIXEL {
        decoded_row[i] = raw_row[i].wrapping_add(prev[i] / 2);
    }

    for i in BYTES_PER_PIXEL..decoded_row.len() {
        let up = prev[i];
        let left = decoded_row[i - BYTES_PER_PIXEL];

        let val2 = ((u16::from(up) + u16::from(left)) / 2) as u8;
        let val = (up >> 1).wrapping_add(left >> 1) + (up & left & 0b1);

        if val2 != val {
            println!("{up}:{left}:{val2}:{val}");
        }

        decoded_row[i] = raw_row[i].wrapping_add(val);
    }
}

fn half(a: &mut [u8]) {
    for el in a {
        *el /= 2;
    }
}

#[track_caller]
fn add(up: &[u8], row: &mut [u8]) {
    assert_eq!(up.len(), row.len());
    for i in 0..row.len() {
        row[i] += up[i];
    }
}

fn shift(row: &mut [u8]) {
    row.rotate_right(1);
    row[0] = 0;
    // row[1] = 0;
    // row[2] = 0;
    // row[3] = 0;
}

fn half_pf_sum(row: &mut [u8]) {
    for i in (1 * BYTES_PER_PIXEL)..row.len() {
        row[i] += row[i - 1 * BYTES_PER_PIXEL] / 2;
    }
}

// def half_pf_sum(a):
//     c = [*a]
//     for i in range(1, len(c)):
//             c[i] += c[i-1]//2
//             c[i] %= 256
//     return c

// 87, 113, 163, 113, 215, 10, 39, 50

pub fn average_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    // current algorithm is `half_pf_sum(add(row, half(up)))`
    // this runs into rounding errors, as it does not replicate the
    // 9-bit arithmetic that is required by the avg filter
    //
    // additionally, it does not seem that half prefix sum is as trivially
    // parallelizable as regular prefix sum. this algorithm is a small improvement,
    // but not sufficient for our purposes.

    let mut up = prev.to_vec();
    half(up.as_mut_slice());
    add(&up, decoded_row);
    add(raw_row, decoded_row);
    half_pf_sum(decoded_row);
}

pub fn paeth(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    // a = left, b = above, c = upper left
    fn predictor(a: i16, b: i16, c: i16) -> u8 {
        let p = a + b - c;
        let pa = (p - a).abs();
        let pb = (p - b).abs();
        let pc = (p - c).abs();

        if pa <= pb && pa <= pc {
            (a % 256) as u8
        } else if pb <= pc {
            (b % 256) as u8
        } else {
            (c % 256) as u8
        }
    }

    for i in 0..BYTES_PER_PIXEL {
        let up = prev[i];
        let left = 0;
        let upper_left = 0;

        let val = predictor(left, i16::from(up), upper_left);

        decoded_row[i] = raw_row[i].wrapping_add(val)
    }

    for i in BYTES_PER_PIXEL..decoded_row.len() {
        let up = prev[i];
        let left = decoded_row[i - BYTES_PER_PIXEL];
        let upper_left = prev[i - BYTES_PER_PIXEL];

        let val = predictor(i16::from(left), i16::from(up), i16::from(upper_left));

        decoded_row[i] = raw_row[i].wrapping_add(val)
    }
}

// d = r + d_p

// d[n] = r[n] + d[n-1]
// d[n] = r[n] + d[n-1]/2 + p[n]/2

// up = [1, 3, 5, 7, 9]
// up_pf_sum = [1, 4, 9, 16, 25]
// up_pf_sum_div_2 = [0, 2, 4, 8, 12]
// up_div_2 = [0, 1, 2, 3, 4]
// up_div_2_pf_sum = [0, 1, 3, 6, 10]

// row = [2, 4, 6, 8, 10]
// row_pf_sum = [2, 6, 12, 20, 30]
// row_pf_sum_div_2 = [1, 3, 6, 10, 15]
// row_div_2 = [1, 2, 3, 4, 5]
// row_div_2_pf_sum = [1, 3, 6, 10, 15]

// decoded_row = [2, 6, 11, 17, 23]

// [2, 4, 6, 8, 10]
// + [0, 2, 4, 6, 8]
// = [2, 6, 10, 14, 18]
//   + [0, 0, 2, 6, 10]
// = [2, 6, 12, 20, 28]
//   + [0, 0, 0, 0, 2]
// = [2, 6, 12, 20, 30]

// half_pf_row [2, 5, 8, 12, 16]
// half_pf_up = [1, 3, 6, 10, 14]

// half_pf_all
// [2, 4, 6, 8, 10]
// + [0, 0, 1, 2, 3]
// + [0, 1, 2, 3, 4]
// [2, 5, ]

// -- more complex example

// up =  [244, 132, 17, 42, 64, 64, 71, 64]
// row = [221, 4, 99, 12, 128, 127, 128, 128]

// row[0] = 221
// up[0]/2 = 122

// (row[0] + up[0]/2)/2 =

// row[1] = 4
// up[1]/2 = 66
// row[0]/2 = 110
// up[0]/4 = 61

// decoded[0] = row[0] + up[0]/2
// decoded[1] = row[1] + up[1]/2 + (row[0] + up[0]/2)/2

// decoded = [87, 113, 164, 115, 217, 11, 169, 244]

// row + shift(half(row), 1) + half_pf_sum(up)

// [221, 4, 99] row
// + [0, 43, 35] shift(half(add(row, half(up))), 1)
// + [122, 66, 8] half(up)

// row[i] + up[i]/2 + row[i-1] + up[i-1]/2

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::BYTES_PER_PIXEL;

pub fn up(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        up_avx(prev, raw_row, decoded_row);
        return;
    }

    if prev.is_empty() {
        decoded_row[..].copy_from_slice(raw_row);
        return;
    }

    for i in 0..decoded_row.len() {
        let prev = prev[i];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}

#[cfg(target_arch = "x86_64")]
fn up_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let offset = decoded_row.len() % 32;
    let len = raw_row.len();

    let mut up: __m256i;
    let mut curr: __m256i;

    let mut i = 0;

    while i < len {
        debug_assert!(i + 32 < prev.len());

        up = unsafe { _mm256_loadu_si256(&prev[i] as *const _ as *const __m256i) };
        curr = unsafe { _mm256_loadu_si256(&raw_row[i] as *const _ as *const __m256i) };

        let sum = unsafe { _mm256_add_epi8(up, curr) };

        unsafe { _mm256_storeu_si256(&mut decoded_row[i] as *mut _ as *mut __m256i, sum) }

        i += 32;
    }

    for i in (decoded_row.len() - offset)..decoded_row.len() {
        let prev = prev[i];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}

pub fn sub(raw_row: &[u8], decoded_row: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { sub_avx(raw_row, decoded_row) };
        return;
    }

    for i in 0..BYTES_PER_PIXEL {
        decoded_row[i] = raw_row[i];
    }

    for i in BYTES_PER_PIXEL..decoded_row.len() {
        let prev = decoded_row[i - BYTES_PER_PIXEL];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}


#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_avx(raw_row: &[u8], decoded_row: &mut [u8]) {
    let off = raw_row.len() % 32;

    for i in 0..BYTES_PER_PIXEL {
        decoded_row[i] = raw_row[i];
    }

    for i in BYTES_PER_PIXEL..off {
        decoded_row[i] = raw_row[i].wrapping_add(decoded_row[i - BYTES_PER_PIXEL]);
    }

    // todo: off > 0 && off < BYTES_PER_PIXEL
    let mut last = if off > BYTES_PER_PIXEL {
        i32::from_be_bytes([
            decoded_row[off - 1],
            decoded_row[off - 2],
            decoded_row[off - 3],
            decoded_row[off - 4],
        ])
    } else {
        0
    };

    debug_assert_eq!((raw_row.len() - off) % 32, 0);

    let mut i = off;
    let len = raw_row.len();

    let mut x: __m256i;
    let mut sum: __m256i;

    while i < len {
        x = _mm256_loadu_si256(&raw_row[i] as *const _ as *const __m256i);

        x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
        x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

        // todo: experiment with impl using permute and setting lower half to zero
        let b = _mm256_extract_epi32::<3>(x);

        let f = _mm256_set_epi32(b, b, b, b, 0, 0, 0, 0);
        let f2 = _mm256_set1_epi32(last);

        sum = _mm256_add_epi8(f, f2);

        sum = _mm256_add_epi8(x, sum);

        _mm256_storeu_si256(&mut decoded_row[i] as *mut _ as *mut __m256i, sum);

        last = _mm256_extract_epi32::<7>(sum);

        i += 32;
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
        decoded_row[i] = raw_row[i].wrapping_add(prev[i]/2);
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
    for i in (1*BYTES_PER_PIXEL)..row.len() {
        row[i] += row[i-1*BYTES_PER_PIXEL] / 2;
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
#![feature(stdsimd)]
#![feature(test)]
#![allow(warnings)]
#![allow(incomplete_features)]

extern crate test;

use std::{
    collections::HashMap,
    fmt::{self, Write},
    hint::black_box,
    io::Read,
    usize,
};

pub use crate::filter::{average, paeth, sub, up, up_avx, up_unsafe};
use crate::filter::{
    sub_avx, sub_avx_attempt_2, sub_avx_attempt_3, sub_avx_attempt_4, sub_avx_attempt_5, sub_sse2,
    sub_sse2_ported,
};

#[cfg(feature = "png")]
use crate::png::*;

mod deflate;
mod filter;
mod png;

pub const BYTES_PER_PIXEL: usize = 4;

// up = [0,0,0,0, 244, 132, 17, 42, 64]
// row = [0,0,0,0, 221, 4, 99, 12, 128]

fn main() {
    // let up =  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    // let row = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    // let up =  [0,0,0,0, 0,0,0,1, 0,0,0,1];
    // let row = [0,0,0,1, 0,0,0,0, 0,0,0,0];
    // let up = [244, 132, 17, 42, 64, 64, 71, 64];
    // let row = [221, 4, 99, 12, 128, 127, 128, 128];

    // let reg = decode(&up, &row);
    // let avx = decode_avx(&up, &row);

    // dbg!(reg, avx);

    // assert_eq!(reg, avx);

    // ---
    // let raw_row = Box::new([5; 32 * 32]);
    let raw_row = Box::new([3, 5, 6, 7, 54, 52, 75, 85, 99, 38, 12, 14, 7, 0, 0, 255]); //(0..16).collect::<Vec<_>>().into_boxed_slice();//Box::new([10; 32 * 32]);
    let mut decoded_row = vec![0; raw_row.len()].into_boxed_slice();

    // let start = std::time::Instant::now();
    sub(&*raw_row, &mut *decoded_row);
    // let end = start.elapsed();

    // black_box(decoded_row);

    // println!("{:?}", end);

    // let raw_row = Box::new([5; 32 * 32]);
    let raw_row = Box::new([3, 5, 6, 7, 54, 52, 75, 85, 99, 38, 12, 14, 7, 0, 0, 255]); //(0..16).collect::<Vec<_>>().into_boxed_slice();//Box::new([10; 32 * 32]);
    let mut decoded_row2 = vec![0; raw_row.len()].into_boxed_slice();

    // let start = std::time::Instant::now();
    unsafe { sub_sse2_ported(&*raw_row, &mut *decoded_row2) };
    // let end = start.elapsed();

    assert_eq!(decoded_row, decoded_row2);

    // ---
    // black_box(decoded_row);

    // println!("{:?}", end);

    // let prev = Box::new([5; 1_000_000]);
    // let raw_row = Box::new([10; 1_000_000]);
    // let mut decoded_row = Box::new([0; 1_000_000]);

    // let start = std::time::Instant::now();
    // unsafe { up_unsafe(&*prev, &*raw_row, &mut *decoded_row) };
    // let end = start.elapsed();

    // black_box(decoded_row);

    // println!("{:?}", end);

    // let prev = Box::new([5; 1_000_000]);
    // let raw_row = Box::new([10; 1_000_000]);
    // let mut decoded_row = Box::new([0; 1_000_000]);

    // let start = std::time::Instant::now();
    // up_avx(&*prev, &*raw_row, &mut *decoded_row);
    // let end = start.elapsed();

    // black_box(decoded_row);

    // println!("{:?}", end);

    // #[cfg(feature = "png")]
    // {
    //     let file = std::fs::File::open("Periodic_table_large.png").unwrap();
    //     let mmap = unsafe { memmap::MmapOptions::new().map(&file) }.unwrap();
    //     let decoder = PngReader::new(mmap);

    //     let bitmap = decoder.parse();
    // }

    // let mut window = minifb::Window::new(
    //     "Image",
    //     bitmap.width as usize,
    //     bitmap.height as usize,
    //     minifb::WindowOptions::default(),
    // )
    // .unwrap();

    // window.limit_update_rate(Some(std::time::Duration::from_millis(1000)));

    // while window.is_open() {
    //     window
    //         .update_with_buffer(
    //             &bitmap
    //                 .buffer
    //                 .chunks_exact(4)
    //                 .map(|b| u32::from_le_bytes([b[2], b[1], b[0], 0]))
    //                 .collect::<Vec<u32>>(),
    //             bitmap.width as usize,
    //             bitmap.height as usize,
    //         )
    //         .unwrap();
    // }

    // let file = std::fs::File::open("Periodic_table_large.png").unwrap();
    // let mmap = unsafe { memmap::MmapOptions::new().map(&file) }.unwrap();
    // let mut decoder = PngReader::new(mmap);

    // let bitmap = decoder.parse();

    // // let raw_row = Box::new([10; BUFFER_SIZE]);
    // // let mut decoded_row = Box::new([0; BUFFER_SIZE]);

    // std::hint::black_box(bitmap);
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::*;
    use std::convert::TryInto;
    use test::Bencher;

    use crate::filter::{sub_sse2_ported, sub_sse_prefix_sum_no_extract};

    const BYTES_PER_PIXEL: usize = 4;

    pub fn sub(raw_row: &[u8], decoded_row: &mut [u8]) {
        for i in 0..BYTES_PER_PIXEL {
            decoded_row[i] = raw_row[i];
        }

        for i in BYTES_PER_PIXEL..decoded_row.len() {
            let left = decoded_row[i - BYTES_PER_PIXEL];

            decoded_row[i] = raw_row[i].wrapping_add(left)
        }
    }

    pub unsafe fn sub_no_bound_checks(raw_row: &[u8], decoded_row: &mut [u8]) {
        for i in 0..BYTES_PER_PIXEL {
            *decoded_row.get_unchecked_mut(i) = *raw_row.get_unchecked(i);
        }

        for i in BYTES_PER_PIXEL..decoded_row.len() {
            let left = *decoded_row.get_unchecked(i - BYTES_PER_PIXEL);

            *decoded_row.get_unchecked_mut(i) = raw_row.get_unchecked(i).wrapping_add(left)
        }
    }

    pub unsafe fn baseline_memcpy(raw_row: &[u8], decoded_row: &mut [u8]) {
        decoded_row
            .get_unchecked_mut(0..raw_row.len())
            .copy_from_slice(&*raw_row);
    }

    unsafe fn load4(x: &[u8; 4]) -> __m128i {
        let tmp = i32::from_le_bytes(*x);
        _mm_cvtsi32_si128(tmp)
    }

    unsafe fn store4(x: &mut [u8; 4], v: __m128i) {
        let tmp = _mm_cvtsi128_si32(v);
        x.copy_from_slice(&tmp.to_le_bytes());
    }

    pub unsafe fn sub_sse2(raw: &[u8], current: &mut [u8]) {
        let (mut a, mut d) = (_mm_setzero_si128(), _mm_setzero_si128());

        for (raw, out) in raw.chunks_exact(4).zip(current.chunks_exact_mut(4)) {
            a = d;
            d = load4(raw.try_into().unwrap());
            d = _mm_add_epi8(d, a);
            store4(out.try_into().unwrap(), d);
        }
    }

    pub unsafe fn sub_avx(raw_row: &[u8], decoded_row: &mut [u8]) {
        debug_assert!(is_x86_feature_detected!("avx2"));

        let mut last = 0;
        let mut x: __m256i;

        let len = raw_row.len();
        let mut i = 0;

        let offset = len % 32;
        if offset != 0 {
            sub_sse2(
                raw_row.get_unchecked(..offset),
                decoded_row.get_unchecked_mut(..offset),
            );
            last = i32::from_be_bytes([
                *decoded_row.get_unchecked(offset - 1),
                *decoded_row.get_unchecked(offset - 2),
                *decoded_row.get_unchecked(offset - 3),
                *decoded_row.get_unchecked(offset - 4),
            ]);
            i = offset;
        }

        while len != i {
            // load 32 bytes from array
            x = _mm256_loadu_si256(raw_row.get_unchecked(i) as *const _ as *const __m256i);

            // do prefix sum
            x = _mm256_add_epi8(_mm256_slli_si256::<4>(x), x);
            x = _mm256_add_epi8(_mm256_slli_si256::<{ 2 * 4 }>(x), x);

            // accumulate for first 16 bytes
            let b = _mm256_extract_epi32::<3>(x);
            x = _mm256_add_epi8(_mm256_set_epi32(b, b, b, b, 0, 0, 0, 0), x);

            // accumulate for previous chunk of 16 bytes
            x = _mm256_add_epi8(_mm256_set1_epi32(last), x);

            // accumulate for last 16 bytes
            last = _mm256_extract_epi32::<7>(x);

            // write 32 bytes to out array
            _mm256_storeu_si256(
                decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m256i,
                x,
            );

            i += 32;
        }
    }

    pub unsafe fn sub_sse_prefix_sum(raw_row: &[u8], decoded_row: &mut [u8]) {
        debug_assert!(is_x86_feature_detected!("avx2"));

        let mut last = 0;
        let mut x: __m128i;

        let len = raw_row.len();
        let mut i = 0;

        let offset = len % 16;
        if offset != 0 {
            sub_sse2(
                raw_row.get_unchecked(..offset),
                decoded_row.get_unchecked_mut(..offset),
            );
            last = i32::from_be_bytes([
                *decoded_row.get_unchecked(offset - 1),
                *decoded_row.get_unchecked(offset - 2),
                *decoded_row.get_unchecked(offset - 3),
                *decoded_row.get_unchecked(offset - 4),
            ]);
            i = offset;
        }

        while len != i {
            // load 16 bytes from array
            x = _mm_loadu_si128(raw_row.get_unchecked(i) as *const _ as *const __m128i);

            // do prefix sum
            x = _mm_add_epi8(_mm_slli_si128::<4>(x), x);
            x = _mm_add_epi8(_mm_slli_si128::<{ 2 * 4 }>(x), x);

            // accumulate for previous chunk of 16 bytes
            x = _mm_add_epi8(x, _mm_set1_epi32(last));

            last = _mm_extract_epi32::<3>(x);

            // write 16 bytes to out array
            _mm_storeu_si128(
                decoded_row.get_unchecked_mut(i) as *mut _ as *mut __m128i,
                x,
            );

            i += 16;
        }
    }

    const BUFFER_SIZE: usize = 2_usize.pow(20);

    #[bench]
    fn bench_sub_naive_scalar(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| sub(&raw_row, &mut decoded_row));

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_no_bound_checks(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_no_bound_checks(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_baseline_memcpy(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { baseline_memcpy(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_sse2(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_sse2(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_avx(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_avx(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_sse_prefix_sum(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_sse_prefix_sum(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_sse_prefix_sum_no_extract(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_sse_prefix_sum_no_extract(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }

    #[bench]
    fn bench_sub_sse2_ported(b: &mut Bencher) {
        let raw_row = std::hint::black_box([10; BUFFER_SIZE]);
        let mut decoded_row = std::hint::black_box([0; BUFFER_SIZE]);

        b.iter(|| unsafe { sub_sse2_ported(&raw_row, &mut decoded_row) });

        std::hint::black_box(decoded_row);
    }
}

// notes go below here

// fn average_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {}

// [2, 2, 2, 2]
// + [0, 2, 2, 2]

// [2, 4, 4, 4]
// + [0, 0, 2, 4]

// [2, 4, 6, 8]

// [1, 2, 1, 2, 1, 2]

// + [0, 0, 1, 2, 1, 2]

// [1, 2, 2, 4, 3, 6]

// [1, 2, 1, 2, 1, 2]
// [2, 1, 2, 1, 2, 1]

// [2, 3, 4, 4, 4, 4]

// [0, 1, 0, 1, 0, 1] // up div 2
// [0, 1, 1, 2, 2, 3] // up div 2, then prefix sum
// [1, 0, 1, 0, 1, 0]

// [2, 1, 2, 1, 2, 1]

// [2, 3, 5, 6, 8, 9]

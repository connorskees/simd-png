#![feature(stdsimd)]
#![allow(incomplete_features)]

use std::{
    collections::HashMap,
    fmt::{self, Write},
    io::Read,
    usize,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

mod deflate;

#[derive(Debug)]
struct PngReader {
    buffer: memmap::Mmap,
    cursor: usize,
}

#[derive(Debug)]
struct ChunkHeader {
    name: ChunkName,
    length: u32,
}

#[derive(Debug, PartialEq, Eq)]
struct ChunkName([u8; 4]);

impl ChunkName {
    const IHDR: Self = ChunkName(*b"IHDR");
    const IDAT: Self = ChunkName(*b"IDAT");

    pub fn is_idat(&self) -> bool {
        self == &Self::IDAT
    }
}

impl fmt::Display for ChunkName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char(self.0[0] as char)?;
        f.write_char(self.0[1] as char)?;
        f.write_char(self.0[2] as char)?;
        f.write_char(self.0[3] as char)?;

        Ok(())
    }
}

/// The PNG header. In ascii, it can be represented as \x{89}PNG\r\n\x{1a}\n
pub const HEADER: [u8; 8] = [137u8, 80, 78, 71, 13, 10, 26, 10];

pub const BYTES_PER_PIXEL: usize = 4;

#[derive(Debug)]
struct HeaderChunk {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    compression_type: u8,
    filter_method: u8,
    interlacing_method: u8,
}

impl HeaderChunk {
    const NAME: ChunkName = ChunkName::IHDR;

    /// We only support a subset of PNG features. Ensure that this file only includes
    /// the features we support
    fn validate(&self) {
        debug_assert_eq!(self.bit_depth, 8);
        debug_assert_eq!(self.compression_type, 0);
        // rgba
        debug_assert_eq!(self.color_type, 6);
        // adaptive
        debug_assert_eq!(self.filter_method, 0);
        // no interlacing
        debug_assert_eq!(self.interlacing_method, 0);
    }
}

#[derive(Debug)]
struct PngDecoder {
    header_chunk: HeaderChunk,
    buffer: Vec<u8>,
}

fn sub(raw_row: &[u8], decoded_row: &mut [u8]) {
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
fn up_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    debug_assert!(is_x86_feature_detected!("avx2"));

    let offset = decoded_row.len() % 32;

    let mut up: __m256i;
    let mut curr: __m256i;

    for i in (0..(decoded_row.len() - offset)).step_by(32) {
        debug_assert!(i + 32 < prev.len());

        up = unsafe { _mm256_loadu_si256(&prev[i] as *const _ as *const __m256i) };
        curr = unsafe { _mm256_loadu_si256(&raw_row[i] as *const _ as *const __m256i) };

        let sum = unsafe { _mm256_add_epi8(up, curr) };

        unsafe { _mm256_storeu_si256(&mut decoded_row[i] as *mut _ as *mut __m256i, sum) }
    }

    for i in (decoded_row.len() - offset)..decoded_row.len() {
        let prev = prev[i];

        decoded_row[i] = raw_row[i].wrapping_add(prev)
    }
}

fn up(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
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

/// The Average filter uses the average of the two neighboring pixels (left and
/// above) to predict the value of a pixel.
fn average(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
    for i in 0..BYTES_PER_PIXEL {
        decoded_row[i] = prev[i];
    }

    for i in BYTES_PER_PIXEL..decoded_row.len() {
        let up = prev[i];
        let left = decoded_row[i - BYTES_PER_PIXEL];

        let val = (((u16::from(up) + u16::from(left)) / 2) % 256) as u8;

        decoded_row[i] = raw_row[i].wrapping_add(val)
    }
}

fn paeth(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {
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

impl PngDecoder {
    pub fn new(header_chunk: HeaderChunk, buffer: &[u8]) -> Self {
        let mut decoded_buffer = deflate::decode(buffer);

        // flate2::read::ZlibDecoder::new(buffer)
        //     .read_to_end(&mut decoded_buffer)
        //     .unwrap();

        Self {
            header_chunk,
            buffer: decoded_buffer,
        }
    }

    pub fn decode(self) -> Bitmap {
        let width = self.header_chunk.width as usize;
        let height = self.header_chunk.height as usize;

        let mut decoded_buffer = vec![0; self.buffer.len() - height as usize];

        let bytes_per_row = 1 + width * BYTES_PER_PIXEL;

        let mut counts = HashMap::new();

        for i in 0..height {
            let raw_row_start = (i * bytes_per_row) as usize;
            let decoded_row_start = (i * (bytes_per_row - 1)) as usize;
            let start = self.buffer[raw_row_start];
            let raw_row = &self.buffer[(raw_row_start + 1)..(raw_row_start + bytes_per_row)];

            let (prev, decoded_row) = decoded_buffer.split_at_mut(decoded_row_start);

            let decoded_row = &mut decoded_row[..(bytes_per_row - 1)];

            let prev = &prev[(prev.len().saturating_sub(bytes_per_row - 1))..];
            // dbg!(i, prev.len(), raw_row.len(), decoded_row.len());

            if i != 0 {
                debug_assert_eq!(prev.len(), raw_row.len());
                debug_assert_eq!(prev.len(), decoded_row.len());
            }

            *counts.entry(start).or_insert(0) += 1;

            match start {
                0 => {
                    // nop
                }
                1 => sub(raw_row, decoded_row),
                2 => up(prev, raw_row, decoded_row),
                3 => average(prev, raw_row, decoded_row),
                4 => paeth(prev, raw_row, decoded_row),
                // 0..=4 => {}
                _ => unimplemented!("{}", start),
            }
        }

        // dbg!(&decoded_buffer);

        Bitmap {
            width: self.header_chunk.width,
            height: self.header_chunk.height,
            buffer: decoded_buffer,
        }
    }
}

#[derive(Debug)]
struct Bitmap {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
}

impl PngReader {
    pub fn new(buffer: memmap::Mmap) -> Self {
        Self { buffer, cursor: 0 }
    }

    pub fn parse(mut self) -> Bitmap {
        self.read_magic();

        let header = self.read_header_chunk();
        let mut pixel_buffer = Vec::new();

        while !self.at_eof() {
            if let Some(buffer) = self.read_idat_chunk() {
                pixel_buffer.extend_from_slice(buffer);
                self.skip_chunk_end();
            }
        }

        let decoder = PngDecoder::new(header, &pixel_buffer);

        decoder.decode()
    }

    fn at_eof(&self) -> bool {
        self.cursor >= self.buffer.len()
    }

    fn read_idat_chunk(&mut self) -> Option<&[u8]> {
        let header = self.read_chunk_header();

        if !header.name.is_idat() {
            self.skip_chunk(header.length);
            return None;
        }

        let body = self.read_buffer(header.length as usize);

        Some(body)
    }

    fn skip_chunk(&mut self, len: u32) {
        self.cursor += len as usize;
        self.skip_chunk_end();
    }

    fn skip_chunk_end(&mut self) {
        self.cursor += 4;
    }

    fn read_chunk_header(&mut self) -> ChunkHeader {
        let length = self.read_u32_be().unwrap();
        let name = self.read_chunk_name().unwrap();

        ChunkHeader { name, length }
    }

    fn read_magic(&mut self) {
        let header = self.read_buffer_const::<8>();

        debug_assert_eq!(header, HEADER);
    }

    fn read_header_chunk(&mut self) -> HeaderChunk {
        let chunk_header = self.read_chunk_header();

        debug_assert_eq!(chunk_header.name, HeaderChunk::NAME);
        debug_assert_eq!(chunk_header.length, 13);

        let width = self.read_u32_be().unwrap();
        let height = self.read_u32_be().unwrap();
        let bit_depth = self.next_byte().unwrap();
        let color_type = self.next_byte().unwrap();
        let compression_type = self.next_byte().unwrap();
        let filter_method = self.next_byte().unwrap();
        let interlacing_method = self.next_byte().unwrap();

        self.skip_chunk_end();

        let header_chunk = HeaderChunk {
            width,
            height,
            bit_depth,
            color_type,
            compression_type,
            filter_method,
            interlacing_method,
        };

        header_chunk.validate();

        header_chunk
    }

    fn read_u32_be(&mut self) -> Option<u32> {
        let b1 = self.next_byte()?;
        let b2 = self.next_byte()?;
        let b3 = self.next_byte()?;
        let b4 = self.next_byte()?;

        Some(u32::from_be_bytes([b1, b2, b3, b4]))
    }

    fn read_buffer(&mut self, len: usize) -> &[u8] {
        let start = self.cursor;
        self.cursor += len;

        debug_assert!(self.cursor <= self.buffer.len());

        &self.buffer[start..self.cursor]
    }

    fn read_buffer_const<const C: usize>(&mut self) -> &[u8] {
        let start = self.cursor;
        self.cursor += C;

        debug_assert!(self.cursor <= self.buffer.len());

        &self.buffer[start..self.cursor]
    }

    fn read_chunk_name(&mut self) -> Option<ChunkName> {
        let b1 = self.next_byte()?;
        let b2 = self.next_byte()?;
        let b3 = self.next_byte()?;
        let b4 = self.next_byte()?;

        Some(ChunkName([b1, b2, b3, b4]))
    }

    fn next_byte(&mut self) -> Option<u8> {
        self.buffer.get(self.cursor).cloned().map(|b| {
            self.cursor += 1;
            b
        })
    }
}

fn main() {
    let file = std::fs::File::open("Periodic_table_large.png").unwrap();
    let mmap = unsafe { memmap::MmapOptions::new().map(&file) }.unwrap();
    let decoder = PngReader::new(mmap);

    let bitmap = decoder.parse();

    let mut window = minifb::Window::new(
        "Image",
        bitmap.width as usize,
        bitmap.height as usize,
        minifb::WindowOptions::default(),
    )
    .unwrap();

    window.limit_update_rate(Some(std::time::Duration::from_millis(1000)));

    while window.is_open() {
        window
            .update_with_buffer(
                &bitmap
                    .buffer
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[2], b[1], b[0], 0]))
                    .collect::<Vec<u32>>(),
                bitmap.width as usize,
                bitmap.height as usize,
            )
            .unwrap();
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

// notes go below here

fn average_avx(prev: &[u8], raw_row: &[u8], decoded_row: &mut [u8]) {}

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

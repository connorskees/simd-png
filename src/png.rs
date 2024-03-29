#![cfg(feature = "png")]

use std::{
    collections::HashMap,
    fmt::{self, Write},
    hint::black_box,
    io::Read,
    usize,
};

use crate::{filter::*, BYTES_PER_PIXEL};

#[derive(Debug)]
pub struct PngReader {
    pub buffer: memmap::Mmap,
    pub cursor: usize,
}

#[derive(Debug)]
pub struct ChunkHeader {
    pub name: ChunkName,
    pub length: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ChunkName([u8; 4]);

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

#[derive(Debug)]
pub struct HeaderChunk {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    pub compression_type: u8,
    pub filter_method: u8,
    pub interlacing_method: u8,
}

impl HeaderChunk {
    const NAME: ChunkName = ChunkName::IHDR;

    /// We only support a subset of PNG features. Ensure that this file only includes
    /// the features we support
    fn validate(&self) {
        assert_eq!(self.bit_depth, 8);
        assert_eq!(self.compression_type, 0);
        // rgba
        assert_eq!(self.color_type, 6);
        // adaptive
        assert_eq!(self.filter_method, 0);
        // no interlacing
        assert_eq!(self.interlacing_method, 0);
    }
}

#[derive(Debug)]
pub struct PngDecoder {
    pub header_chunk: HeaderChunk,
    pub buffer: Vec<u8>,
}

impl PngDecoder {
    pub fn new(header_chunk: HeaderChunk, buffer: &[u8]) -> Self {
        let mut decoded_buffer = Vec::new();

        flate2::read::ZlibDecoder::new(buffer)
            .read_to_end(&mut decoded_buffer)
            .unwrap();

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

            if i != 0 {
                debug_assert_eq!(prev.len(), raw_row.len());
                debug_assert_eq!(prev.len(), decoded_row.len());
            }

            *counts.entry(start).or_insert(0) += 1;

            println!("{}", start);

            match start {
                0 => {
                    // nop
                }
                // 1 => unsafe { sub_avx_attempt_4(raw_row, decoded_row) },
                1 => sub(raw_row, decoded_row),
                2 => up(prev, raw_row, decoded_row),
                3 => average(prev, raw_row, decoded_row),
                4 => paeth(prev, raw_row, decoded_row),
                _ => unimplemented!("{}", start),
            }
        }

        Bitmap {
            width: self.header_chunk.width,
            height: self.header_chunk.height,
            buffer: decoded_buffer,
        }
    }
}

#[derive(Debug)]
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    pub buffer: Vec<u8>,
}

impl PngReader {
    pub fn new(buffer: memmap::Mmap) -> Self {
        Self { buffer, cursor: 0 }
    }

    pub fn parse(&mut self) -> Bitmap {
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

fn decode<const N: usize>(up: &[u8; N], row: &[u8; N]) -> [u8; N] {
    let mut decoded_row = [0; N];

    average(up, row, &mut decoded_row);

    return decoded_row;
}

fn decode_avx<const N: usize>(up: &[u8], row: &[u8]) -> [u8; N] {
    let mut decoded_row = [0; N];

    average_avx(up, row, &mut decoded_row);

    return decoded_row;
}

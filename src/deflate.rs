pub fn decode(buffer: &[u8]) -> Vec<u8> {
    if buffer.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();

    let mut idx = 0;

    dbg!(buffer.len());

    loop {
        let first_byte = buffer[idx];

        let is_final = (first_byte & 1) != 0;
        let btype = (first_byte >> 1) & 0b011;

        dbg!(is_final, btype);

        match btype {
            0b00 => {
                idx += 1;
                let len = u16::from_be_bytes([buffer[idx], buffer[idx + 1]]);
                idx += 2;
                let nlen = u16::from_be_bytes([buffer[idx], buffer[idx + 1]]);
                dbg!(len, nlen);

                output.extend_from_slice(&buffer[idx..(idx + len as usize)]);
            }
            0b01 => todo!("compressed with fixed Huffman codes"),
            0b10 => todo!("compressed with dynamic Huffman codes"),
            0b11 => todo!("error"),
            _ => unreachable!(),
        }

        if is_final {
            break;
        }
    }

    output
}

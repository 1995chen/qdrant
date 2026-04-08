//! Generic bit pack / unpack helpers.
//!
//! TurboQuant primarily cares about 2/3/4-bit scalar levels plus 1-bit QJL
//! signs. Keeping packing generic makes the tests much easier to write.

pub fn pack_bits(values: &[u8], bit_width: u8) -> Vec<u8> {
    debug_assert!((1..=8).contains(&bit_width));

    let total_bits = values.len() * bit_width as usize;
    let mut packed = vec![0u8; total_bits.div_ceil(8)];
    let mask = if bit_width == 8 {
        u16::MAX
    } else {
        (1u16 << bit_width) - 1
    };

    let mut bit_offset = 0usize;
    for &value in values {
        let value = (value as u16) & mask;
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;

        packed[byte_index] |= (value << shift) as u8;
        if shift + bit_width as usize > 8 {
            packed[byte_index + 1] |= (value >> (8 - shift)) as u8;
        }

        bit_offset += bit_width as usize;
    }

    packed
}

pub fn unpack_bits(packed: &[u8], bit_width: u8, value_count: usize) -> Vec<u8> {
    debug_assert!((1..=8).contains(&bit_width));

    let mask = if bit_width == 8 {
        u16::MAX
    } else {
        (1u16 << bit_width) - 1
    };

    let mut output = Vec::with_capacity(value_count);
    let mut bit_offset = 0usize;
    for _ in 0..value_count {
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;
        let mut value = (packed[byte_index] as u16) >> shift;
        if shift + bit_width as usize > 8 {
            value |= (packed[byte_index + 1] as u16) << (8 - shift);
        }
        output.push((value & mask) as u8);
        bit_offset += bit_width as usize;
    }
    output
}

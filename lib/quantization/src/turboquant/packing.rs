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

#[cfg_attr(not(test), allow(dead_code))]
pub fn unpack_bits(packed: &[u8], bit_width: u8, value_count: usize) -> Vec<u8> {
    let mut output = vec![0u8; value_count];
    unpack_bits_into(packed, bit_width, &mut output);
    output
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn unpack_bits_into(packed: &[u8], bit_width: u8, output: &mut [u8]) {
    debug_assert!((1..=8).contains(&bit_width));

    let mask = if bit_width == 8 {
        u16::MAX
    } else {
        (1u16 << bit_width) - 1
    };

    let mut bit_offset = 0usize;
    for value in output.iter_mut() {
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;
        let mut unpacked = (packed[byte_index] as u16) >> shift;
        if shift + bit_width as usize > 8 {
            unpacked |= (packed[byte_index + 1] as u16) << (8 - shift);
        }
        *value = (unpacked & mask) as u8;
        bit_offset += bit_width as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_bits, unpack_bits, unpack_bits_into};

    #[test]
    fn pack_and_unpack_roundtrip_across_common_bit_widths() {
        let cases = [
            (1, vec![0, 1, 1, 0, 1, 0, 0, 1, 1]),
            (2, vec![0, 1, 2, 3, 1, 0, 3]),
            (3, vec![0, 1, 7, 3, 4, 2, 5, 6]),
            (4, vec![0, 1, 15, 7, 8, 3]),
            (8, vec![0, 17, 255, 3, 128]),
        ];

        for (bit_width, values) in cases {
            let packed = pack_bits(&values, bit_width);
            let unpacked = unpack_bits(&packed, bit_width, values.len());
            assert_eq!(
                unpacked, values,
                "roundtrip mismatch for bit_width={bit_width}"
            );
        }
    }

    #[test]
    fn pack_bits_matches_expected_layout_for_cross_byte_boundaries() {
        let values = [7, 0, 5, 3, 6];
        let packed = pack_bits(&values, 3);
        assert_eq!(packed, vec![0x47, 0x67]);

        let unpacked = unpack_bits(&packed, 3, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_bits_is_identity_for_eight_bit_values() {
        let values = [0x00, 0x12, 0x7f, 0x80, 0xff];
        let packed = pack_bits(&values, 8);
        assert_eq!(packed, values);

        let unpacked = unpack_bits(&packed, 8, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_bits_masks_off_bits_above_bit_width() {
        let values = [0b1111, 0b1001, 0b0111];
        let packed = pack_bits(&values, 3);
        let unpacked = unpack_bits(&packed, 3, values.len());
        assert_eq!(unpacked, vec![0b111, 0b001, 0b111]);
    }

    #[test]
    fn unpack_bits_into_matches_allocating_variant() {
        let values = [7, 0, 5, 3, 6];
        let packed = pack_bits(&values, 3);
        let mut unpacked = [0u8; 5];
        unpack_bits_into(&packed, 3, &mut unpacked);
        assert_eq!(unpacked, values);
    }
}

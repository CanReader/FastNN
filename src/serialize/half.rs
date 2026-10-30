//! Conversions between `f32` and the 16-bit float formats, both round to nearest even.

/// `f32` to binary16 bits.
pub fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // Force a mantissa bit so a NaN whose payload bits all shift out stays a NaN.
    if exponent == 0xFF {
        let payload = (mantissa >> 13) as u16;
        return sign | 0x7C00 | if mantissa != 0 { payload | 0x0200 } else { 0 };
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00;
    }

    if unbiased < -14 {
        // Subnormal in f16, so the implicit leading 1 becomes explicit.
        if unbiased < -25 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (-14 - unbiased + 13) as u32;
        let halfway = 1u32 << (shift - 1);
        let truncated = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
        // A carry into the smallest normal lands in the exponent field on its own.
        return sign | rounded as u16;
    }

    let halfway = 0x0000_1000u32;
    let remainder = mantissa & 0x1FFF;
    let truncated = mantissa >> 13;
    let mut rounded =
        truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
    let mut exponent16 = (unbiased + 15) as u32;
    if rounded == 0x400 {
        rounded = 0;
        exponent16 += 1;
        if exponent16 >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | ((exponent16 as u16) << 10) | rounded as u16
}

/// binary16 bits to `f32`, exactly.
pub fn f32_from_f16(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    match (exponent, mantissa) {
        (0, 0) => f32::from_bits(sign),
        (0, m) => {
            let magnitude = m as f32 * f32::from_bits(0x3380_0000); // 2^-24
            if sign != 0 { -magnitude } else { magnitude }
        }
        (0x1F, 0) => f32::from_bits(sign | 0x7F80_0000),
        (0x1F, _) => f32::NAN,
        _ => f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (mantissa << 13)),
    }
}

/// `f32` to bfloat16 bits. The addition rounds ties to even through the carry.
pub fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Truncation could turn the NaN into infinity, so pin a payload bit.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// bfloat16 bits to `f32`, exactly.
pub fn f32_from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representable_values_round_trip_exactly() {
        for value in [
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            0.15625,
            65504.0,
            6.1035156e-5,
            5.9604645e-8,
        ] {
            let widened = f32_from_f16(f16_from_f32(value));
            assert_eq!(
                widened.to_bits(),
                value.to_bits(),
                "f16 round trip of {value}"
            );
        }
        for bits in [
            0x0000_0000u32,
            0xBF80_0000,
            0x4049_0000,
            0x7E96_0000,
            0x8180_0000,
        ] {
            let value = f32::from_bits(bits);
            let widened = f32_from_bf16(bf16_from_f32(value));
            assert_eq!(widened.to_bits(), bits, "bf16 round trip of {value}");
        }
    }

    #[test]
    fn ties_round_to_even() {
        // Exact ties: the even mantissa wins.
        let tie_low = 1.0 + f32::from_bits(0x3A00_0000); // 2⁻¹¹
        assert_eq!(f32_from_f16(f16_from_f32(tie_low)), 1.0);
        let tie_high = 1.0 + 3.0 * f32::from_bits(0x3A00_0000);
        assert_eq!(
            f32_from_f16(f16_from_f32(tie_high)),
            1.0 + f32::from_bits(0x3B00_0000)
        );

        let tie = 1.0 + f32::from_bits(0x3B80_0000); // 2⁻⁸
        assert_eq!(f32_from_bf16(bf16_from_f32(tie)), 1.0);
    }

    #[test]
    fn specials_survive() {
        assert_eq!(f32_from_f16(f16_from_f32(f32::INFINITY)), f32::INFINITY);
        assert_eq!(
            f32_from_f16(f16_from_f32(f32::NEG_INFINITY)),
            f32::NEG_INFINITY
        );
        assert!(f32_from_f16(f16_from_f32(f32::NAN)).is_nan());
        assert!(f32_from_bf16(bf16_from_f32(f32::NAN)).is_nan());

        assert_eq!(f32_from_f16(f16_from_f32(1.0e6)), f32::INFINITY);
        assert_eq!(f32_from_f16(f16_from_f32(-1.0e6)), f32::NEG_INFINITY);
        assert_eq!(
            f32_from_f16(f16_from_f32(1.0e-9)).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            f32_from_f16(f16_from_f32(-1.0e-9)).to_bits(),
            (-0.0f32).to_bits()
        );
    }

    #[test]
    fn rounding_error_is_at_most_half_an_ulp() {
        for i in 0..2000u32 {
            let value = f32::from_bits(0x3F80_0000 + i * 7919); // 1.0 upward
            let rounded = f32_from_f16(f16_from_f32(value));
            let ulp = (f32_from_f16(f16_from_f32(value) + 1) - rounded).abs();
            assert!(
                (rounded - value).abs() <= ulp / 2.0 + f32::EPSILON,
                "{value} rounded to {rounded}, more than half an ulp ({ulp}) away"
            );
        }
    }
}

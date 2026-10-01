//! Bit-exact conversion between `f32` and the two 16-bit float formats.
//!
//! An IEEE 754 binary16 (`f16`) keeps 1 sign, 5 exponent, and 10 mantissa
//! bits; bfloat16 keeps 1/8/7 — the same exponent range as `f32` with less
//! precision, which is why it never overflows where `f32` did not. Both
//! conversions round to nearest, ties to even: the only rounding that is
//! unbiased under accumulation, and what hardware does.
//!
//! Widening (`f16 → f32`) is exact — every 16-bit float is representable in
//! `f32` — so a checkpoint stored in half precision loses information exactly
//! once, at save time, never again on load.

/// `f32` → binary16 bits, round-to-nearest-even.
pub fn f16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // Infinity and NaN: exponent all ones. A NaN must stay a NaN, so force a
    // mantissa bit on in case the payload's surviving bits are all zero.
    if exponent == 0xFF {
        let payload = (mantissa >> 13) as u16;
        return sign | 0x7C00 | if mantissa != 0 { payload | 0x0200 } else { 0 };
    }

    // Rebase the exponent from f32's bias (127) to f16's (15).
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflows f16's range → ±∞
    }

    if unbiased < -14 {
        // Subnormal in f16: the implicit leading 1 becomes explicit and the
        // 24-bit significand shifts right until the exponent reaches −14.
        if unbiased < -25 {
            return sign; // below half the smallest subnormal → ±0
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (-14 - unbiased + 13) as u32;
        let halfway = 1u32 << (shift - 1);
        let truncated = significand >> shift;
        // Ties to even: round up on > halfway, or on exactly halfway when the
        // kept part is odd.
        let remainder = significand & ((1 << shift) - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
        // A rounded subnormal can carry into the smallest normal; the bit
        // layout makes that carry land in the exponent field on its own.
        return sign | rounded as u16;
    }

    // Normal: drop 13 mantissa bits with the same tie-to-even rule.
    let halfway = 0x0000_1000u32;
    let remainder = mantissa & 0x1FFF;
    let truncated = mantissa >> 13;
    let mut rounded =
        truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
    let mut exponent16 = (unbiased + 15) as u32;
    if rounded == 0x400 {
        // Mantissa overflowed 10 bits: 1.111…1 rounded up to 10.0…0.
        rounded = 0;
        exponent16 += 1;
        if exponent16 >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | ((exponent16 as u16) << 10) | rounded as u16
}

/// binary16 bits → `f32`, exactly.
pub fn f32_from_f16(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    match (exponent, mantissa) {
        (0, 0) => f32::from_bits(sign),
        // Subnormal: value is mantissa · 2⁻²⁴, always exact in f32.
        (0, m) => {
            let magnitude = m as f32 * f32::from_bits(0x3380_0000); // 2⁻²⁴
            if sign != 0 {
                -magnitude
            } else {
                magnitude
            }
        }
        (0x1F, 0) => f32::from_bits(sign | 0x7F80_0000),
        (0x1F, _) => f32::NAN,
        // Normal: rebias 15 → 127 and left-align the mantissa.
        _ => f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (mantissa << 13)),
    }
}

/// `f32` → bfloat16 bits, round-to-nearest-even.
///
/// bfloat16 is the top half of an `f32`, so rounding is one addition on the
/// raw bits: add half of the dropped range, plus one more when the kept part
/// is odd — the carry implements ties-to-even for free.
pub fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Truncation could zero the payload and turn NaN into ∞; pin one bit.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// bfloat16 bits → `f32`, exactly: the bits *are* the top half.
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
        // bf16 keeps only the top 16 bits, so representable means "low half
        // zero" — constructed from bits to make that property visible.
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
        // 1 + 2⁻¹¹ sits exactly between 1.0 and 1 + 2⁻¹⁰ (f16's next value):
        // the even mantissa wins, so it rounds *down* to 1.0 —
        let tie_low = 1.0 + f32::from_bits(0x3A00_0000); // 2⁻¹¹
        assert_eq!(f32_from_f16(f16_from_f32(tie_low)), 1.0);
        // — while 1 + 3·2⁻¹¹ sits between 1 + 2⁻¹⁰ and 1 + 2⁻⁹ and rounds *up*
        // to the even 1 + 2⁻⁹.
        let tie_high = 1.0 + 3.0 * f32::from_bits(0x3A00_0000);
        assert_eq!(
            f32_from_f16(f16_from_f32(tie_high)),
            1.0 + f32::from_bits(0x3B00_0000)
        );

        // Same rule for bf16 at its own precision: 1 + 2⁻⁸ is a tie, 1.0 is even.
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

        // Values past f16's range overflow to infinity rather than wrapping.
        assert_eq!(f32_from_f16(f16_from_f32(1.0e6)), f32::INFINITY);
        assert_eq!(f32_from_f16(f16_from_f32(-1.0e6)), f32::NEG_INFINITY);
        // Values below half the smallest subnormal flush to signed zero.
        assert_eq!(
            f32_from_f16(f16_from_f32(1.0e-9)).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            f32_from_f16(f16_from_f32(-1.0e-9)).to_bits(),
            (-0.0f32).to_bits()
        );
    }

    /// Rounding must never be off by more than half an ulp: sweep a range of
    /// bit patterns and compare against the nearest of the two neighbours.
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

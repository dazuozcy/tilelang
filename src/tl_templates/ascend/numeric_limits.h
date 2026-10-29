#pragma once

#include <bit>
#include <cstdint>

namespace tl {
namespace limits {

namespace detail {

// The Ascend device toolchain (bisheng/clang) resolves <bit> from the host
// libstdc++, which may predate C++20 std::bit_cast (it appeared in GCC 11).
// Prefer the standard helper when the library advertises it and fall back to
// the compiler builtin otherwise; both are valid in constant expressions.
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
using std::bit_cast;
#else
template <typename To, typename From>
constexpr To bit_cast(const From &src) noexcept {
  return __builtin_bit_cast(To, src);
}
#endif

} // namespace detail

// --- float (fp32) ---
inline constexpr float kFloatInf = detail::bit_cast<float>(0x7F800000u);
inline constexpr float kFloatNInf = detail::bit_cast<float>(0xFF800000u);
inline constexpr float kFloatNaN = detail::bit_cast<float>(0x7FC00000u);

// --- half (fp16) ---
inline constexpr half kHalfInf = detail::bit_cast<half>(uint16_t{0x7C00u});
inline constexpr half kHalfNInf = detail::bit_cast<half>(uint16_t{0xFC00u});
inline constexpr half kHalfNaN = detail::bit_cast<half>(uint16_t{0x7E00u});

// --- bfloat16 ---
inline constexpr bfloat16_t kBf16Inf =
    detail::bit_cast<bfloat16_t>(uint16_t{0x7F80u});
inline constexpr bfloat16_t kBf16NInf =
    detail::bit_cast<bfloat16_t>(uint16_t{0xFF80u});
inline constexpr bfloat16_t kBf16NaN =
    detail::bit_cast<bfloat16_t>(uint16_t{0x7FC0u});

} // namespace limits
} // namespace tl

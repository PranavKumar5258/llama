import regex
import ctypes


class CoodepointFlags (ctypes.Structure):
    _fields_ = [  # see definition in unicode.h
        ("is_undefined",   ctypes.c_uint16, 1),
        ("is_number",      ctypes.c_uint16, 1),  # regex: \p{N}
        ("is_letter",      ctypes.c_uint16, 1),  # regex: \p{L}
        ("is_separator",   ctypes.c_uint16, 1),  # regex: \p{Z}
        ("is_accent_mark", ctypes.c_uint16, 1),  # regex: \p{M}
        ("is_punctuation", ctypes.c_uint16, 1),  # regex: \p{P}
        ("is_symbol",      ctypes.c_uint16, 1),  # regex: \p{S}
        ("is_control",     ctypes.c_uint16, 1),  # regex: \p{C}
    ]

assert(ctypes.sizeof(CoodepointFlags) == 2)


MAX_CODEPOINTS = 0x110000

regex_number      = regex.compile(r'\p{N}')
regex_letter      = regex.compile(r'\p{L}')
regex_separator   = regex.compile(r'\p{Z}')
regex_accent_mark = regex.compile(r'\p{M}')
regex_punctuation = regex.compile(r'\p{P}')
regex_symbol      = regex.compile(r'\p{S}')
regex_control     = regex.compile(r'\p{C}')
regex_whitespace  = regex.compile(r'\s')

codepoint_flags = (CoodepointFlags * MAX_CODEPOINTS)()
table_whitespace = []
table_lowercase = []
table_uppercase = []

for codepoint in range(MAX_CODEPOINTS):
    # convert codepoint to unicode character
    char = chr(codepoint)

    # regex categories
    flags = codepoint_flags[codepoint]
    flags.is_number      = bool(regex_number.match(char))
    flags.is_letter      = bool(regex_letter.match(char))
    flags.is_separator   = bool(regex_separator.match(char))
    flags.is_accent_mark = bool(regex_accent_mark.match(char))
    flags.is_punctuation = bool(regex_punctuation.match(char))
    flags.is_symbol      = bool(regex_symbol.match(char))
    flags.is_control     = bool(regex_control.match(char))
    flags.is_undefined   = bytes(flags)[0] == 0
    assert(not flags.is_undefined)

    # whitespaces
    if bool(regex_whitespace.match(char)):
        table_whitespace.append(codepoint)

    # lowercase conversion
    lower = ord(char.lower()[0])
    if codepoint != lower:
        table_lowercase.append((codepoint, lower))

    # uppercase conversion
    upper = ord(char.upper()[0])
    if codepoint != upper:
        table_uppercase.append((codepoint, upper))


ranges_flags = [(0, codepoint_flags[0])]
for codepoint, flags in enumerate(codepoint_flags):
    if bytes(flags) != bytes(ranges_flags[-1][1]):
        ranges_flags.append((codepoint, flags))
ranges_flags.append((MAX_CODEPOINTS, CoodepointFlags()))


# Generate 'unicode-data.cpp'

print("""\
// generated with scripts/gen-unicode-data.py

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
""")

print("const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1")
for codepoint, flags in ranges_flags:
    flags = int.from_bytes(bytes(flags), "little")
    print("{0x%06X, 0x%04X}," % (codepoint, flags))
print("};\n")

print("const std::unordered_set<uint32_t> unicode_set_whitespace = {")
print(", ".join("0x%06X" % cpt for cpt in table_whitespace))
print("};\n")

print("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {")
for tuple in table_lowercase:
    print("{0x%06X, 0x%06X}," % tuple)
print("};\n")

print("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {")
for tuple in table_uppercase:
    print("{0x%06X, 0x%06X}," % tuple)
print("};\n")

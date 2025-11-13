package displaywidth

import (
	"unicode/utf8"

	"github.com/clipperhouse/stringish"
	"github.com/clipperhouse/uax29/v2/graphemes"
)

// String calculates the display width of a string,
// by iterating over grapheme clusters in the string
// and summing their widths.
func String(s string) int {
	return DefaultOptions.String(s)
}

// Bytes calculates the display width of a []byte,
// by iterating over grapheme clusters in the byte slice
// and summing their widths.
func Bytes(s []byte) int {
	return DefaultOptions.Bytes(s)
}

// Rune calculates the display width of a rune. You
// should almost certainly use [String] or [Bytes] for
// most purposes.
//
// The smallest unit of display width is a grapheme
// cluster, not a rune. Iterating over runes to measure
// width is incorrect in most cases.
func Rune(r rune) int {
	return DefaultOptions.Rune(r)
}

// Options allows you to specify the treatment of ambiguous East Asian
// characters. When EastAsianWidth is false (default), ambiguous East Asian
// characters are treated as width 1. When EastAsianWidth is true, ambiguous
// East Asian characters are treated as width 2.
type Options struct {
	EastAsianWidth bool
}

// DefaultOptions is the default options for the display width
// calculation, which is EastAsianWidth: false.
var DefaultOptions = Options{EastAsianWidth: false}

// String calculates the display width of a string,
// for the given options, by iterating over grapheme clusters
// and summing their widths.
func (options Options) String(s string) int {
	if len(s) == 0 {
		return 0
	}

	total := 0
	g := graphemes.FromString(s)
	for g.Next() {
		props := lookupProperties(g.Value())
		total += props.width(options)
	}
	return total
}

// Bytes calculates the display width of a []byte,
// for the given options, by iterating over grapheme
// clusters in the byte slice and summing their widths.
func (options Options) Bytes(s []byte) int {
	if len(s) == 0 {
		return 0
	}

	total := 0
	g := graphemes.FromBytes(s)
	for g.Next() {
		props := lookupProperties(g.Value())
		total += props.width(options)
	}
	return total
}

// Rune calculates the display width of a rune,
// for the given options.
//
// The smallest unit of display width is a grapheme
// cluster, not a rune. Iterating over runes to measure
// width is incorrect in most cases.
func (options Options) Rune(r rune) int {
	// Fast path for ASCII
	if r < utf8.RuneSelf {
		if isASCIIControl(byte(r)) {
			// Control (0x00-0x1F) and DEL (0x7F)
			return 0
		}
		// ASCII printable (0x20-0x7E)
		return 1
	}

	// Surrogates (U+D800-U+DFFF) are invalid UTF-8 and have zero width
	// Other packages might turn them into the replacement character (U+FFFD)
	// in which case, we won't see it.
	if r >= 0xD800 && r <= 0xDFFF {
		return 0
	}

	// Stack-allocated to avoid heap allocation
	var buf [4]byte // UTF-8 is at most 4 bytes
	n := utf8.EncodeRune(buf[:], r)
	// Skip the grapheme iterator and directly lookup properties
	props := lookupProperties(buf[:n])
	return props.width(options)
}

func isASCIIControl(b byte) bool {
	return b < 0x20 || b == 0x7F
}

// isRIPrefix checks if the slice matches the Regional Indicator prefix
// (F0 9F 87). It assumes len(s) >= 3.
func isRIPrefix[T stringish.Interface](s T) bool {
	return s[0] == 0xF0 && s[1] == 0x9F && s[2] == 0x87
}

// isVS16 checks if the slice matches VS16 (U+FE0F) UTF-8 encoding
// (EF B8 8F). It assumes len(s) >= 3.
func isVS16[T stringish.Interface](s T) bool {
	return s[0] == 0xEF && s[1] == 0xB8 && s[2] == 0x8F
}

// lookupProperties returns the properties for the first character in a string
func lookupProperties[T stringish.Interface](s T) property {
	l := len(s)

	if l == 0 {
		return 0
	}

	b := s[0]
	if isASCIIControl(b) {
		return _Zero_Width
	}

	if b < utf8.RuneSelf {
		// Check for variation selector after ASCII (e.g., keycap sequences like 1️⃣)
		if l >= 4 {
			// Subslice may help eliminate bounds checks
			vs := s[1:4]
			if isVS16(vs) {
				// VS16 requests emoji presentation (width 2)
				return _Emoji
			}
			// VS15 (0x8E) requests text presentation but does not affect width,
			// in my reading of Unicode TR51. Falls through to _Default.
		}
		return _Default
	}

	// Regional indicator pair (flag)
	if l >= 8 {
		// Subslice may help eliminate bounds checks
		ri := s[:8]
		if isRIPrefix(ri[0:3]) {
			b3 := ri[3]
			if b3 >= 0xA6 && b3 <= 0xBF && isRIPrefix(ri[4:7]) {
				b7 := ri[7]
				if b7 >= 0xA6 && b7 <= 0xBF {
					return _Emoji
				}
			}
		}
	}

	props, size := lookup(s)
	p := property(props)

	// Variation Selectors
	if size > 0 && l >= size+3 {
		// Subslice may help eliminate bounds checks
		vs := s[size : size+3]
		if isVS16(vs) {
			// VS16 requests emoji presentation (width 2)
			return _Emoji
		}
		// VS15 (0x8E) requests text presentation but does not affect width,
		// in my reading of Unicode TR51. Falls through to return the base
		// character's property (p).
	}

	return p
}

const _Default property = 0

// a jump table of sorts, instead of a switch
var widthTable = [5]int{
	_Default:              1,
	_Zero_Width:           0,
	_East_Asian_Wide:      2,
	_East_Asian_Ambiguous: 1,
	_Emoji:                2,
}

// width determines the display width of a character based on its properties
// and configuration options
func (p property) width(options Options) int {
	if options.EastAsianWidth && p == _East_Asian_Ambiguous {
		return 2
	}

	return widthTable[p]
}

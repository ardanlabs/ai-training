# displaywidth

A high-performance Go package for measuring the monospace display width of strings, UTF-8 bytes, and runes.

[![Documentation](https://pkg.go.dev/badge/github.com/clipperhouse/displaywidth.svg)](https://pkg.go.dev/github.com/clipperhouse/displaywidth)
[![Test](https://github.com/clipperhouse/displaywidth/actions/workflows/gotest.yml/badge.svg)](https://github.com/clipperhouse/displaywidth/actions/workflows/gotest.yml)
[![Fuzz](https://github.com/clipperhouse/displaywidth/actions/workflows/gofuzz.yml/badge.svg)](https://github.com/clipperhouse/displaywidth/actions/workflows/gofuzz.yml)

## Install
```bash
go get github.com/clipperhouse/displaywidth
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/clipperhouse/displaywidth"
)

func main() {
    width := displaywidth.String("Hello, 世界!")
    fmt.Println(width)

    width = displaywidth.Bytes([]byte("🌍"))
    fmt.Println(width)

    width = displaywidth.Rune('🌍')
    fmt.Println(width)
}
```

For most purposes, you should use the `String` or `Bytes` methods.


### Options

You can specify East Asian Width settings. When false (default),
[East Asian Ambiguous characters](https://www.unicode.org/reports/tr11/#Ambiguous)
are treated as width 1. When true, East Asian Ambiguous characters are treated
as width 2.

```go
myOptions := displaywidth.Options{
    EastAsianWidth: true,
}

width := myOptions.String("Hello, 世界!")
fmt.Println(width)
```

## Technical details

This package implements the Unicode East Asian Width standard
([UAX #11](https://www.unicode.org/reports/tr11/)), and handles
[version selectors](https://en.wikipedia.org/wiki/Variation_Selectors_(Unicode_block)),
and [regional indicator pairs](https://en.wikipedia.org/wiki/Regional_indicator_symbol)
(flags). We implement [Unicode TR51](https://unicode.org/reports/tr51/).

`clipperhouse/displaywidth`, `mattn/go-runewidth`, and `rivo/uniseg` will
give the same outputs for most real-world text. See extensive details in the
[compatibility analysis](comparison/COMPATIBILITY_ANALYSIS.md).

If you wish to investigate the core logic, see the `lookupProperties` and `width`
functions in [width.go](width.go#L135). The essential trie generation logic is in
`buildPropertyBitmap` in [unicode.go](internal/gen/unicode.go#L317).

I (@clipperhouse) am keeping an eye on [emerging standards and test suites](https://www.jeffquast.com/post/state-of-terminal-emulation-2025/).

## Prior Art

[mattn/go-runewidth](https://github.com/mattn/go-runewidth)

[rivo/uniseg](https://github.com/rivo/uniseg)

[x/text/width](https://pkg.go.dev/golang.org/x/text/width)

[x/text/internal/triegen](https://pkg.go.dev/golang.org/x/text/internal/triegen)

## Benchmarks

```bash
cd comparison
go test -bench=. -benchmem
```

```
goos: darwin
goarch: arm64
pkg: github.com/clipperhouse/displaywidth/comparison
cpu: Apple M2

BenchmarkString_Mixed/clipperhouse/displaywidth-8         10929 ns/op	    154.36 MB/s	      0 B/op     0 allocs/op
BenchmarkString_Mixed/mattn/go-runewidth-8                14540 ns/op	    116.02 MB/s	      0 B/op     0 allocs/op
BenchmarkString_Mixed/rivo/uniseg-8                       19751 ns/op	     85.41 MB/s	      0 B/op     0 allocs/op

BenchmarkString_EastAsian/clipperhouse/displaywidth-8     10885 ns/op	    154.98 MB/s	      0 B/op     0 allocs/op
BenchmarkString_EastAsian/mattn/go-runewidth-8            23969 ns/op	     70.38 MB/s	      0 B/op     0 allocs/op
BenchmarkString_EastAsian/rivo/uniseg-8                   19852 ns/op	     84.98 MB/s	      0 B/op     0 allocs/op

BenchmarkString_ASCII/clipperhouse/displaywidth-8          1103 ns/op	    116.01 MB/s	      0 B/op     0 allocs/op
BenchmarkString_ASCII/mattn/go-runewidth-8                 1166 ns/op	    109.79 MB/s	      0 B/op     0 allocs/op
BenchmarkString_ASCII/rivo/uniseg-8                        1584 ns/op	     80.83 MB/s	      0 B/op     0 allocs/op

BenchmarkString_Emoji/clipperhouse/displaywidth-8          3108 ns/op	    232.93 MB/s	      0 B/op     0 allocs/op
BenchmarkString_Emoji/mattn/go-runewidth-8                 4802 ns/op	    150.76 MB/s	      0 B/op     0 allocs/op
BenchmarkString_Emoji/rivo/uniseg-8                        6607 ns/op	    109.58 MB/s	      0 B/op     0 allocs/op

BenchmarkRune_Mixed/clipperhouse/displaywidth-8            3456 ns/op	    488.20 MB/s	      0 B/op     0 allocs/op
BenchmarkRune_Mixed/mattn/go-runewidth-8                   5400 ns/op	    312.39 MB/s	      0 B/op     0 allocs/op

BenchmarkRune_EastAsian/clipperhouse/displaywidth-8        3475 ns/op	    485.41 MB/s	      0 B/op     0 allocs/op
BenchmarkRune_EastAsian/mattn/go-runewidth-8              15701 ns/op	    107.44 MB/s	      0 B/op     0 allocs/op

BenchmarkRune_ASCII/clipperhouse/displaywidth-8             257.0 ns/op	    498.13 MB/s	      0 B/op     0 allocs/op
BenchmarkRune_ASCII/mattn/go-runewidth-8                    266.4 ns/op	    480.50 MB/s	      0 B/op     0 allocs/op

BenchmarkRune_Emoji/clipperhouse/displaywidth-8            1384 ns/op	    523.02 MB/s	      0 B/op     0 allocs/op
BenchmarkRune_Emoji/mattn/go-runewidth-8                   2273 ns/op	    318.45 MB/s	      0 B/op     0 allocs/op
```

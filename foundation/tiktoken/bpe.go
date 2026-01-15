package tiktoken

import (
	"math"
	"slices"
)

func bytePairEncode(piece []byte, ranks map[string]int) []int {
	if len(piece) == 1 {
		v := ranks[string(piece)]
		return []int{v}
	}

	return bytePairMerge(piece, ranks, func(start, end int) int {
		return ranks[string(piece[start:end])]
	})
}

func bytePairMerge[T any](piece []byte, ranks map[string]int, f func(start, end int) T) []T {
	parts := make([][2]int, len(piece)+1)
	for i := range parts {
		parts[i][0], parts[i][1] = i, math.MaxInt // use max int as sentinel
	}

	getRank := func(startIdx, skip int) int {
		if startIdx+skip+2 < len(parts) {
			b := piece[parts[startIdx][0]:parts[startIdx+skip+2][0]]
			rank, ok := ranks[string(b)]
			if ok {
				return rank
			}
		}
		return -1 // use -1 to represent None
	}

	for i := range len(parts) - 2 {
		if rank := getRank(i, 0); rank >= 0 {
			parts[i][1] = rank
		}
	}

	for len(parts) > 1 {
		minRank, minIdx := math.MaxInt, -1
		for i := range len(parts) - 1 {
			if parts[i][1] < minRank {
				minRank, minIdx = parts[i][1], i
			}
		}

		if minRank < math.MaxInt {
			i := minIdx
			rank := getRank(i, 1)

			switch {
			case rank >= 0:
				parts[i][1] = rank
			default:
				parts[i][1] = math.MaxInt
			}

			if i > 0 {
				rk := getRank(i-1, 1)

				switch {
				case rk >= 0:
					parts[i-1][1] = rk
				default:
					parts[i-1][1] = math.MaxInt
				}
			}

			parts = slices.Delete(parts, i+1, i+2)

			continue
		}

		break
	}

	out := make([]T, len(parts)-1)
	for i := range out {
		out[i] = f(parts[i][0], parts[i+1][0])
	}

	return out
}

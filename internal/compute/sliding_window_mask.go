package compute

import (
	"fmt"
	"math"
)

// SlidingWindowTileKind classifies a key-value tile against the active sliding window.
type SlidingWindowTileKind string

const (
	// TileSkip: the entire tile falls outside the visible window. It is completely skipped.
	TileSkip SlidingWindowTileKind = "skip"
	// TileFull: every key in the tile is within the visible window. No masking required.
	TileFull SlidingWindowTileKind = "full"
	// TilePartial: the tile intersects one of the window boundaries. Elementwise masking required.
	TilePartial SlidingWindowTileKind = "partial"
)

// SlidingWindowTileBounds defines the position range and skip classification for one tile.
type SlidingWindowTileBounds struct {
	TileIndex int                   `json:"tile_index"`
	StartPos  int                   `json:"start_pos"`
	EndPos    int                   `json:"end_pos"`
	Kind      SlidingWindowTileKind `json:"kind"`
}

// SlidingWindowMaskReceipt records operational metrics and skip ratios.
type SlidingWindowMaskReceipt struct {
	TotalPositions int     `json:"total_positions"`
	QueryPos       int     `json:"query_pos"`
	WindowSize     int     `json:"window_size"`
	TileSize       int     `json:"tile_size"`
	TotalTiles     int     `json:"total_tiles"`
	SkippedTiles   int     `json:"skipped_tiles"`
	FullTiles      int     `json:"full_tiles"`
	PartialTiles   int     `json:"partial_tiles"`
	SkipRatio      float64 `json:"skip_ratio"`
}

// DeriveSlidingWindowTiles schedules key tiles for sliding-window attention at queryPos.
// Visible keys satisfy: max(0, queryPos - windowSize + 1) <= keyPos <= queryPos.
func DeriveSlidingWindowTiles(totalPositions, queryPos, windowSize, tileSize int) ([]SlidingWindowTileBounds, SlidingWindowMaskReceipt, error) {
	var receipt SlidingWindowMaskReceipt
	if totalPositions <= 0 || tileSize <= 0 || windowSize <= 0 {
		return nil, receipt, fmt.Errorf("dimensions must be positive: totalPositions=%d, tileSize=%d, windowSize=%d", totalPositions, tileSize, windowSize)
	}
	if queryPos < 0 || queryPos >= totalPositions {
		return nil, receipt, fmt.Errorf("queryPos %d out of range [0, %d)", queryPos, totalPositions)
	}

	winMin := queryPos - windowSize + 1
	if winMin < 0 {
		winMin = 0
	}
	winMax := queryPos

	numTiles := (totalPositions + tileSize - 1) / tileSize
	tiles := make([]SlidingWindowTileBounds, numTiles)

	var skipped, full, partial int
	for t := 0; t < numTiles; t++ {
		start := t * tileSize
		end := start + tileSize - 1
		if end >= totalPositions {
			end = totalPositions - 1
		}

		var kind SlidingWindowTileKind
		if end < winMin || start > winMax {
			kind = TileSkip
			skipped++
		} else if start >= winMin && end <= winMax {
			kind = TileFull
			full++
		} else {
			kind = TilePartial
			partial++
		}

		tiles[t] = SlidingWindowTileBounds{
			TileIndex: t,
			StartPos:  start,
			EndPos:    end,
			Kind:      kind,
		}
	}

	receipt = SlidingWindowMaskReceipt{
		TotalPositions: totalPositions,
		QueryPos:       queryPos,
		WindowSize:     windowSize,
		TileSize:       tileSize,
		TotalTiles:     numTiles,
		SkippedTiles:   skipped,
		FullTiles:      full,
		PartialTiles:   partial,
		SkipRatio:      float64(skipped) / float64(numTiles),
	}

	return tiles, receipt, nil
}

// ApplySlidingWindowTileMask evaluates attention for query Q against tiled keys/values,
// skipping fully masked tiles completely. Accessed positions invoke keyAccessHook to verify
// canary safety on skipped tiles.
func ApplySlidingWindowTileMask(
	q []float32,
	keys [][]float32,
	values [][]float32,
	headDim int,
	scale float32,
	windowSize int,
	tileSize int,
	queryPos int,
	keyAccessHook func(pos int),
) ([]float32, SlidingWindowMaskReceipt, error) {
	totalPositions := len(keys)
	tiles, receipt, err := DeriveSlidingWindowTiles(totalPositions, queryPos, windowSize, tileSize)
	if err != nil {
		return nil, receipt, err
	}

	winMin := queryPos - windowSize + 1
	if winMin < 0 {
		winMin = 0
	}
	winMax := queryPos

	// Collect logits only for processed tiles
	var activeIndices []int
	var activeScores []float32

	for _, tile := range tiles {
		if tile.Kind == TileSkip {
			// Entire tile skipped: zero DRAM reads or dot products
			continue
		}

		for j := tile.StartPos; j <= tile.EndPos; j++ {
			if tile.Kind == TilePartial && (j < winMin || j > winMax) {
				continue
			}

			if keyAccessHook != nil {
				keyAccessHook(j)
			}

			var d float32
			for e := 0; e < headDim; e++ {
				d += q[e] * keys[j][e]
			}
			activeIndices = append(activeIndices, j)
			activeScores = append(activeScores, d*scale)
		}
	}

	if len(activeScores) == 0 {
		return make([]float32, headDim), receipt, nil
	}

	// Softmax over active scores
	mx := activeScores[0]
	for _, s := range activeScores {
		if s > mx {
			mx = s
		}
	}
	var sum float64
	probs := make([]float32, len(activeScores))
	for i, s := range activeScores {
		e := math.Exp(float64(s - mx))
		probs[i] = float32(e)
		sum += e
	}
	for i := range probs {
		probs[i] = float32(float64(probs[i]) / sum)
	}

	// Weighted sum of values
	out := make([]float32, headDim)
	for i, j := range activeIndices {
		w := probs[i]
		v := values[j]
		for e := 0; e < headDim; e++ {
			out[e] += w * v[e]
		}
	}

	return out, receipt, nil
}

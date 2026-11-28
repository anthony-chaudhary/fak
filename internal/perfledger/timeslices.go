// Adapted from ai-dynamo/aiperf
// modified from the original): timeslice grid with is_complete clipping and
// CSR ragged offset index over ColumnStore records.
package perfledger

import (
	"errors"
	"fmt"
	"math"
)

// Timeslice represents one uniform time bucket and whether it completed before
// the clipping boundary.
type Timeslice struct {
	Index      int
	StartNS    float64
	EndNS      float64
	IsComplete bool
}

// TimesliceGrid holds the ordered timeslices and a CSR ragged offset index
// mapping slice k to Rows[Offsets[k]:Offsets[k+1]].
type TimesliceGrid struct {
	Slices  []Timeslice
	Offsets []int
	Rows    []int
}

// SliceRows returns the ColumnStore record indices overlapping slice k.
func (g TimesliceGrid) SliceRows(k int) []int {
	if k < 0 || k >= len(g.Slices) {
		return nil
	}
	return g.Rows[g.Offsets[k]:g.Offsets[k+1]]
}

// BuildTimeslices partitions [minStart, clipEndNS] (or the store's observed span
// when clipEndNS is NaN) into uniform sliceNS buckets, marks incomplete trailing
// slices with IsComplete=false, and builds the CSR ragged offset index.
func BuildTimeslices(s *ColumnStore, sliceNS, clipEndNS float64) (TimesliceGrid, error) {
	if s == nil {
		return TimesliceGrid{}, errors.New("perfledger: nil ColumnStore")
	}
	if math.IsNaN(sliceNS) || math.IsInf(sliceNS, 0) || sliceNS <= 0 {
		return TimesliceGrid{}, fmt.Errorf("perfledger: invalid sliceNS %v", sliceNS)
	}
	if s.Count() == 0 {
		return TimesliceGrid{Offsets: []int{0}}, nil
	}

	minStart := math.Inf(1)
	maxEnd := math.Inf(-1)
	for i := 0; i < s.Count(); i++ {
		st, en := s.startNS[i], s.endNS[i]
		if math.IsNaN(st) || math.IsNaN(en) {
			continue
		}
		if st < minStart {
			minStart = st
		}
		if en > maxEnd {
			maxEnd = en
		}
	}
	if math.IsInf(minStart, 1) || math.IsInf(maxEnd, -1) {
		return TimesliceGrid{Offsets: []int{0}}, nil
	}
	effectiveEnd := maxEnd
	if !math.IsNaN(clipEndNS) && clipEndNS < effectiveEnd {
		effectiveEnd = clipEndNS
	}
	if effectiveEnd < minStart {
		return TimesliceGrid{Offsets: []int{0}}, nil
	}

	numSlices := int(math.Ceil((effectiveEnd - minStart) / sliceNS))
	if numSlices < 1 {
		numSlices = 1
	}

	grid := TimesliceGrid{
		Slices:  make([]Timeslice, numSlices),
		Offsets: make([]int, numSlices+1),
	}
	for k := 0; k < numSlices; k++ {
		slStart := minStart + float64(k)*sliceNS
		slEnd := slStart + sliceNS
		isComplete := true
		if !math.IsNaN(clipEndNS) && slEnd > clipEndNS {
			slEnd = clipEndNS
			isComplete = false
		}
		grid.Slices[k] = Timeslice{
			Index:      k,
			StartNS:    slStart,
			EndNS:      slEnd,
			IsComplete: isComplete,
		}
		mask := s.QueryTimeRange(slStart, slEnd)
		for rowIdx, hit := range mask {
			if hit {
				grid.Rows = append(grid.Rows, rowIdx)
			}
		}
		grid.Offsets[k+1] = len(grid.Rows)
	}
	return grid, nil
}

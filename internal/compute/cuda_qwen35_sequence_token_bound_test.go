//go:build cuda && cgo

package compute

import (
	"errors"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestQwen35SequenceTokenGridBoundary(t *testing.T) {
	for _, tokens := range []int{2, 65535, 65536} {
		t.Run(strconv.Itoa(tokens), func(t *testing.T) {
			req := qwen35SequenceGeometryFixture(Qwen35DenseMainLayers)
			req.TokenIDs = make([]int, tokens)
			req.StartPos = 0
			err := validateQwen35SequenceGeometry(req)
			if tokens <= 65535 {
				if err != nil {
					t.Fatalf("representable GDN token grid refused: %v", err)
				}
				return
			}
			var contractErr *Qwen35SequenceError
			if !errors.As(err, &contractErr) || contractErr.Stage != "geometry" ||
				!strings.Contains(contractErr.Reason, "grid.y") {
				t.Fatalf("oversized GDN token grid error = %v, want typed preflight refusal", err)
			}
		})
	}
}

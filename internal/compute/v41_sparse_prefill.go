package compute

import (
	"errors"
	"fmt"
	"math"
)

// V41SparsePrefillInput owns no device state. Q is pre-RoPE BF16 in canonical
// [query,64,512] order; KV is already rotated BF16 [key,512]. Indices preserve
// slot order and duplicates. Negative-one slots are masked, not compacted.
type V41SparsePrefillInput struct {
	Queries, Keys, TopK, TableRows int
	Q, KV                          []uint16
	Indices, Lengths               []int32
	Positions                      []uint32
	Sink, CosSin                   []float32
	Scale                          float32
}

// V41SparsePrefillOutput contains owned canonical values after inverse RoPE and
// the pinned fused FP8/UE8M0 roundtrip. This is not an F32 attention oracle.
type V41SparsePrefillOutput struct{ Values, MaxLogits, LSE []float32 }

type V41SparsePrefillBackend interface {
	SupportsV41SparsePrefill() bool
	V41SparsePrefill(V41SparsePrefillInput) (V41SparsePrefillOutput, error)
}

var ErrV41SparsePrefillUnavailable = errors.New("V4.1 sparse prefill lacks physical qualification")

func validateV41SparsePrefill(x V41SparsePrefillInput) error {
	bad := func(s string) error { return fmt.Errorf("V4.1 sparse prefill: %s", s) }
	// Native strides and upstream launch geometry are signed 32-bit. Check before
	// products, allocations, uint32 casts or masked-slot interpretation.
	if x.Queries <= 0 || x.Keys <= 0 || x.TopK <= 0 || x.TopK%8 != 0 || x.TableRows <= 0 {
		return bad("invalid geometry")
	}
	for _, n := range []int{x.Queries, x.Keys, x.TopK, x.TableRows} {
		if uint64(n) > math.MaxInt32 {
			return bad("dimension overflow")
		}
	}
	product := func(a, b int) (int, bool) {
		n := uint64(a) * uint64(b)
		return int(n), n <= uint64(math.MaxInt32) && n <= uint64(^uint(0)>>1)/4
	}
	nq, ok := product(x.Queries, 64*512)
	if !ok || len(x.Q) != nq {
		return bad("Q shape")
	}
	nk, ok := product(x.Keys, 512)
	if !ok || len(x.KV) != nk {
		return bad("KV shape")
	}
	ni, ok := product(x.Queries, x.TopK)
	if !ok || len(x.Indices) != ni {
		return bad("index shape")
	}
	nt, ok := product(x.TableRows, 64)
	if !ok || len(x.CosSin) != nt {
		return bad("table shape")
	}
	if len(x.Lengths) != x.Queries || len(x.Positions) != x.Queries || len(x.Sink) != 64 {
		return bad("metadata shape")
	}
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	if !finite(x.Scale) || x.Scale <= 0 {
		return bad("scale")
	}
	for _, a := range [][]uint16{x.Q, x.KV} {
		for _, v := range a {
			if v&0x7f80 == 0x7f80 {
				return bad("nonfinite BF16 input")
			}
		}
	}
	for _, a := range [][]float32{x.Sink, x.CosSin} {
		for _, v := range a {
			if !finite(v) {
				return bad("nonfinite metadata")
			}
		}
	}
	// Validate original IDs even beyond Lengths; masking cannot hide malformed input.
	for _, id := range x.Indices {
		if id < -1 || int64(id) >= int64(x.Keys) {
			return bad("index range")
		}
	}
	for i, n := range x.Lengths {
		if n < 0 || int64(n) > int64(x.TopK) || uint64(x.Positions[i]) >= uint64(x.TableRows) {
			return bad("length/position range")
		}
	}
	return nil
}

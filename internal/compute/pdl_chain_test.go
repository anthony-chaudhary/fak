package compute

import (
	"errors"
	"testing"
)

// Closed host oracle for the PDL chain topology (pdl-megakernel-reconstruction, ADAPT).
// [SW-VERIFIED] structure and oracle only: no hardware run, no speed claim.
type pdlEdge struct {
	From, To int
	Kind     int // 0 full, 1 pdl
	HasWait  bool
}

var (
	errPDLOrder = errors.New("pdl chain: edge is not the next kernel in declared order")
	errPDLWait  = errors.New("pdl chain: PDL edge without a dependency wait before the body")
	errPDLCover = errors.New("pdl chain: edges do not cover every kernel exactly once")
)

func pdlBuild(layers, perLayer, tail int) []pdlEdge {
	n := layers*perLayer + tail
	edges := make([]pdlEdge, 0, n-1)
	for k := 1; k < n; k++ {
		edges = append(edges, pdlEdge{From: k - 1, To: k, Kind: 1, HasWait: true})
	}
	return edges
}

func pdlValidate(n int, edges []pdlEdge) error {
	if len(edges) != n-1 {
		return errPDLCover
	}
	for i, e := range edges {
		if e.From != i || e.To != i+1 {
			return errPDLOrder
		}
		if e.Kind == 1 && !e.HasWait {
			return errPDLWait
		}
	}
	return nil
}

// pdlTimeline returns, per kernel, the (bodyStart, end) on unit-cost kernels with a 0.5 prologue:
// a PDL edge lets the prologue overlap, never the body.
func pdlTimeline(edges []pdlEdge, n int) (body, end []float64) {
	body, end = make([]float64, n), make([]float64, n)
	end[0] = 1
	for _, e := range edges {
		if e.Kind == 1 {
			body[e.To] = end[e.From] // prologue overlapped; body gated on predecessor completion
		} else {
			body[e.To] = end[e.From] + 0.5
		}
		end[e.To] = body[e.To] + 1
	}
	return body, end
}

func TestPDLChain(t *testing.T) {
	const layers = 16
	per, tail := cuhDefine(t, "pdl_chain.cuh", "FAK_PDL_KERNELS_PER_LAYER"), cuhDefine(t, "pdl_chain.cuh", "FAK_PDL_TAIL_KERNELS")
	if cuhDefine(t, "pdl_chain.cuh", "FAK_PDL_EDGE_FULL") != 0 || cuhDefine(t, "pdl_chain.cuh", "FAK_PDL_EDGE_PDL") != 1 {
		t.Fatal("edge kind constants drifted")
	}
	n := layers*per + tail
	if n != 81 {
		t.Fatalf("Llama-3.2-1B chain = %d kernels, want 81", n)
	}
	edges := pdlBuild(layers, per, tail)
	t.Run("DeclaredEdgeOrderValid", func(t *testing.T) {
		if err := pdlValidate(n, edges); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("RejectsReorderedMissingAndUnwaited", func(t *testing.T) {
		swapped := append([]pdlEdge(nil), edges...)
		swapped[3], swapped[4] = swapped[4], swapped[3]
		if err := pdlValidate(n, swapped); !errors.Is(err, errPDLOrder) {
			t.Fatalf("reordered err = %v", err)
		}
		if err := pdlValidate(n, edges[:len(edges)-1]); !errors.Is(err, errPDLCover) {
			t.Fatalf("missing err = %v", err)
		}
		nowait := append([]pdlEdge(nil), edges...)
		nowait[10].HasWait = false
		if err := pdlValidate(n, nowait); !errors.Is(err, errPDLWait) {
			t.Fatalf("unwaited err = %v", err)
		}
	})
	t.Run("PDLOverlapsPrologueNeverBody", func(t *testing.T) {
		body, end := pdlTimeline(edges, n)
		full := append([]pdlEdge(nil), edges...)
		for i := range full {
			full[i].Kind = 0
		}
		_, fullEnd := pdlTimeline(full, n)
		for k := 1; k < n; k++ {
			if body[k] < end[k-1] {
				t.Fatalf("kernel %d body starts at %v before predecessor end %v", k, body[k], end[k-1])
			}
		}
		if end[n-1] >= fullEnd[n-1] {
			t.Fatalf("model: PDL chain end %v not below full-sync end %v", end[n-1], fullEnd[n-1])
		}
	})
}

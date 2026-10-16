package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41IndexQueryPublicationIndependentEvidence(t *testing.T) {
	raw := make([]float32, 8192)
	slab := make([]float32, 8196)
	words := []uint32{v41BF16CheckedStatusMagic, V41BF16PublishIndexerProjection, math.MaxUint32, 8192}
	for i, w := range words {
		slab[i] = math.Float32frombits(w)
	}
	raw[8191] = float32(math.NaN())
	slab[4] = 1
	p := &V41IndexQueryPublication{raw: raw, publication: slab}
	// Raw remains observable even when the as-yet-unconsumed envelope is invalid.
	slab[0] = 0
	if len(p.Raw()) != 8192 || !math.IsNaN(float64(p.Raw()[8191])) {
		t.Fatal("lost raw evidence")
	}
	if out, err := p.Published(); out != nil || err == nil {
		t.Fatal("malformed envelope escaped")
	}
	slab[0] = math.Float32frombits(words[0])
	out, err := p.Published()
	if err != nil || len(out) != 8192 || cap(out) != 8192 || &out[0] != &slab[4] {
		t.Fatalf("invalid output ownership: %v", err)
	}
	raw[0] = 7
	if out[0] != 1 {
		t.Fatal("output retains mutable raw backing")
	}
	slab[2] = math.Float32frombits(23)
	out, err = p.Published()
	var fault *V41BF16CheckedArithmeticError
	var backend *BackendError
	if out != nil || !errors.As(err, &fault) || fault.Index != 23 || fault.Site != V41BF16PublishIndexerProjection || !errors.As(err, &backend) || backend.Site != "V41BF16Publish" {
		t.Fatalf("wrong canonical verdict: %v", err)
	}
	var absent *V41IndexQueryPublication
	if absent.Raw() != nil {
		t.Fatal("nil raw")
	}
	if out, err := absent.Published(); out != nil || err == nil {
		t.Fatal("nil publication")
	}
}

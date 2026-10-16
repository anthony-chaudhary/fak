package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41KVNormPublicationEvidenceOrder(t *testing.T) {
	image := make([]float32, 1032)
	set := func(offset int, site, index uint32) {
		for i, v := range []uint32{v41BF16CheckedStatusMagic, site, index, 512} {
			image[offset+i] = math.Float32frombits(v)
		}
	}
	set(0, V41BF16PublishKVNormInput, math.MaxUint32)
	set(516, V41BF16PublishKVNormOutput, math.MaxUint32)
	p := &V41KVNormPublication{image: image}
	image[4+511] = float32(math.NaN())
	image[520] = 1
	if err := p.InputError(); err != nil {
		t.Fatal(err)
	}
	image[516] = 0
	if len(p.Raw()) != 512 || cap(p.Raw()) != 512 || !math.IsNaN(float64(p.Raw()[511])) {
		t.Fatal("raw evidence hidden by output envelope")
	}
	if out, err := p.Published(); out != nil || err == nil {
		t.Fatal("invalid output envelope escaped")
	}
	set(516, V41BF16PublishKVNormOutput, math.MaxUint32)
	out, err := p.Published()
	if err != nil || len(out) != 512 || cap(out) != 512 || &out[0] != &image[520] {
		t.Fatalf("output range: %v", err)
	}
	p.Raw()[0] = 7
	if out[0] != 1 {
		t.Fatal("raw/output ranges overlap")
	}
	set(0, V41BF16PublishKVNormInput, 2)
	set(516, V41BF16PublishKVNormOutput, 3)
	var fault *V41BF16CheckedArithmeticError
	var backend *BackendError
	if err = p.InputError(); !errors.As(err, &fault) || fault.Index != 2 || fault.Site != V41BF16PublishKVNormInput || !errors.As(err, &backend) {
		t.Fatalf("input attribution: %v", err)
	}
	if _, err = p.Published(); !errors.As(err, &fault) || fault.Index != 3 || fault.Site != V41BF16PublishKVNormOutput {
		t.Fatalf("output attribution: %v", err)
	}
	var absent *V41KVNormPublication
	if absent.Raw() != nil || absent.InputError() == nil {
		t.Fatal("nil image")
	}
	if out, err = absent.Published(); out != nil || err == nil {
		t.Fatal("nil publication")
	}
	for _, n := range []int{0, 1031, 1033} {
		q := &V41KVNormPublication{image: make([]float32, n)}
		if q.Raw() != nil || q.InputError() == nil {
			t.Fatalf("accepted length%d", n)
		}
	}
}

package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41KVNormPublicationEvidenceOrder(t *testing.T) {
	input := make([]float32, 512)
	output := make([]float32, len(input))
	statusWords := []uint32{v41BF16CheckedStatusMagic, 0, math.MaxUint32, uint32(len(input))}
	inputStatusOffset := 0
	inputDataOffset := len(statusWords)
	outputStatusOffset := inputDataOffset + len(input)
	outputDataOffset := outputStatusOffset + len(statusWords)
	image := make([]float32, outputDataOffset+len(output))
	set := func(offset int, site, index uint32) {
		words := append([]uint32(nil), statusWords...)
		words[1], words[2] = site, index
		for i, v := range words {
			image[offset+i] = math.Float32frombits(v)
		}
	}
	set(inputStatusOffset, V41BF16PublishKVNormInput, math.MaxUint32)
	set(outputStatusOffset, V41BF16PublishKVNormOutput, math.MaxUint32)
	p := &V41KVNormPublication{image: image}
	image[inputDataOffset+len(input)-1] = float32(math.NaN())
	image[outputDataOffset] = 1
	if err := p.InputError(); err != nil {
		t.Fatal(err)
	}
	image[outputStatusOffset] = 0
	if len(p.Raw()) != len(input) || cap(p.Raw()) != len(input) || !math.IsNaN(float64(p.Raw()[len(input)-1])) {
		t.Fatal("raw evidence hidden by output envelope")
	}
	if out, err := p.Published(); out != nil || err == nil {
		t.Fatal("invalid output envelope escaped")
	}
	set(outputStatusOffset, V41BF16PublishKVNormInput, math.MaxUint32)
	if out, err := p.Published(); out != nil || err == nil {
		t.Fatal("input publication accepted as output")
	}
	set(outputStatusOffset, V41BF16PublishKVNormOutput, math.MaxUint32)
	out, err := p.Published()
	if err != nil || len(out) != len(output) || cap(out) != len(output) || &out[0] != &image[outputDataOffset] {
		t.Fatalf("output range: %v", err)
	}
	p.Raw()[0] = 7
	if out[0] != 1 {
		t.Fatal("raw/output ranges overlap")
	}
	set(inputStatusOffset, V41BF16PublishKVNormInput, 2)
	set(outputStatusOffset, V41BF16PublishKVNormOutput, 3)
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
	for _, n := range []int{0, len(image) - 1, len(image) + 1} {
		q := &V41KVNormPublication{image: make([]float32, n)}
		if q.Raw() != nil || q.InputError() == nil {
			t.Fatalf("accepted length%d", n)
		}
	}
}

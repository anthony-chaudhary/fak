package compute

import "testing"

// Capability-only fixtures: coverage must not dispatch native work.
type coverageRingProducerOnly struct{ Backend }

func (coverageRingProducerOnly) V41CompressorRingProjectStore(Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int, bool) (Tensor, error) {
	panic("coverage must not dispatch ring producers")
}

type coverageRingProducer struct {
	coverageRingProducerOnly
	admits bool
}

func (b *coverageRingProducer) SupportsV41CompressorRing() bool { return b.admits }
func (*coverageRingProducer) V41CompressorRingAlloc(int, int) (Tensor, Tensor, error) {
	panic("coverage must not allocate a ring")
}
func (*coverageRingProducer) V41CompressorRingStore(Tensor, Tensor, Tensor, Tensor, int, int, int, bool) (Tensor, error) {
	panic("coverage must not store a ring")
}

// fak-test:runtime fast est=1ms lane=default
func TestV41OpCoverageProducerRequiresAdmittedRing(t *testing.T) {
	var probe func(Backend) (bool, bool, string)
	for _, p := range v41OpProbes {
		if p.name == "V41CompressorRingProducerBackend" {
			probe = p.probe
		}
	}
	if probe == nil {
		t.Fatal("missing compressor ring producer coverage row")
	}
	var typedNil *coverageRingProducer
	for _, tc := range []struct {
		name                   string
		be                     Backend
		implemented, available bool
	}{
		{"absent", nil, false, false},
		{"producer-only", coverageRingProducerOnly{}, true, false},
		{"ring-refused", &coverageRingProducer{}, true, false},
		{"ring-admitted", &coverageRingProducer{admits: true}, true, true},
		{"typed-nil", typedNil, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			implemented, available, kind := v41SafeProbe(probe, tc.be)
			if implemented != tc.implemented || available != tc.available || implemented && kind != V41ProbeAvailable {
				t.Fatalf("probe=(%v,%v,%q), want=(%v,%v,available when implemented)", implemented, available, kind, tc.implemented, tc.available)
			}
		})
	}
}

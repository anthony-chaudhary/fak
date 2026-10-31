package compute

import "testing"

// Test-only capability probes: no native/backend execution is simulated.
type coverageEncodingOnly struct {
	Backend
	encoding V41EngramRowEncoding
	admits   bool
}

func (b coverageEncodingOnly) SupportsV41EngramRowEncoding(e V41EngramRowEncoding) bool {
	return b.admits && e == b.encoding
}

type coverageEncodingDecode struct {
	coverageEncodingOnly
	broad bool
}

func (b coverageEncodingDecode) SupportsV41EngramRowDecode() bool { return b.broad }
func (coverageEncodingDecode) V41EngramRowDecode([]byte, V41EngramRowEncoding, int, int) (Tensor, Tensor, error) {
	panic("coverage must never dispatch decode")
}
func (coverageEncodingDecode) V41EngramRowDecodePoison(Tensor, Tensor, int, int) error {
	panic("coverage must never dispatch poison")
}

// fak-test:runtime fast est=1ms lane=default
func TestV41OpCoverageEncodingRequiresAdmittedDecode(t *testing.T) {
	var probe func(Backend) (bool, bool, string)
	for _, p := range v41OpProbes {
		if p.name == "V41EngramRowEncodingBackend" {
			probe = p.probe
		}
	}
	if probe == nil {
		t.Fatal("missing encoding coverage row")
	}
	for _, tc := range []struct {
		name                   string
		be                     Backend
		implemented, available bool
	}{
		{"absent", nil, false, false},
		{"encoding-only", coverageEncodingOnly{encoding: V41EngramRowQ2K, admits: true}, true, false},
		{"broad-refused", coverageEncodingDecode{coverageEncodingOnly{encoding: V41EngramRowQ2K, admits: true}, false}, true, false},
		{"exact-refused", coverageEncodingDecode{coverageEncodingOnly{encoding: V41EngramRowQ2K, admits: false}, true}, true, false},
		{"q2-only", coverageEncodingDecode{coverageEncodingOnly{encoding: V41EngramRowQ2K, admits: true}, true}, true, true},
		{"unknown-only", coverageEncodingDecode{coverageEncodingOnly{encoding: V41EngramRowEncoding(999), admits: true}, true}, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			implemented, available, kind := v41SafeProbe(probe, tc.be)
			if implemented != tc.implemented || available != tc.available || implemented && kind != V41ProbeAvailable {
				t.Fatalf("probe=(%v,%v,%q), want=(%v,%v,available when implemented)", implemented, available, kind, tc.implemented, tc.available)
			}
		})
	}
}

package compute

import "testing"

type v41RowEncodingLegacyWitness struct {
	enabled bool
	calls   int
}

func (b *v41RowEncodingLegacyWitness) SupportsV41EngramRowDecode() bool { return b.enabled }
func (b *v41RowEncodingLegacyWitness) V41EngramRowDecode([]byte, V41EngramRowEncoding, int, int) (Tensor, Tensor, error) {
	b.calls++
	return Tensor{}, Tensor{}, nil
}
func (*v41RowEncodingLegacyWitness) V41EngramRowDecodePoison(Tensor, Tensor, int, int) error {
	return nil
}

type v41RowEncodingExactWitness struct {
	*v41RowEncodingLegacyWitness
	encoding V41EngramRowEncoding
}

func (b *v41RowEncodingExactWitness) SupportsV41EngramRowEncoding(e V41EngramRowEncoding) bool {
	return e == b.encoding
}

// fak-test:runtime fast est=1ms lane=default
func TestV41EngramRowEncodingAdmission(t *testing.T) {
	legacy := &v41RowEncodingLegacyWitness{enabled: true}
	// A broad legacy capability must not turn native mode 3 (Vulkan poison)
	// into an admitted decoder, even if a future caller asks for that value.
	unknown := V41EngramRowEncoding(3)
	if V41EngramRowDecodeSupportsEncoding(legacy, unknown) {
		_, _, _ = legacy.V41EngramRowDecode(nil, unknown, 1, 1)
	}
	if legacy.calls != 0 {
		t.Fatal("unknown dialect reached legacy dispatch")
	}
	for _, encoding := range []V41EngramRowEncoding{V41EngramRowPacked, V41EngramRowQ2K, V41EngramRowF32} {
		if !V41EngramRowDecodeSupportsEncoding(legacy, encoding) {
			t.Fatalf("legacy dialect %d lost admission", encoding)
		}
	}
	exact := &v41RowEncodingExactWitness{legacy, V41EngramRowQ2K}
	if V41EngramRowDecodeSupportsEncoding(exact, V41EngramRowF32) || !V41EngramRowDecodeSupportsEncoding(exact, V41EngramRowQ2K) {
		t.Fatal("exact dialect capability did not narrow admission")
	}
	exact.encoding = unknown
	if V41EngramRowDecodeSupportsEncoding(exact, unknown) {
		t.Fatal("backend claim admitted an unknown stride")
	}
	legacy.enabled = false
	exact.encoding = V41EngramRowQ2K
	if V41EngramRowDecodeSupportsEncoding(exact, V41EngramRowQ2K) || V41EngramRowDecodeSupportsEncoding(nil, V41EngramRowQ2K) {
		t.Fatal("unavailable decoder admitted")
	}
}

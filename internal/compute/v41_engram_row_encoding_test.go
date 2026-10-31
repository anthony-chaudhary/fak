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
	// A known 144-byte IQ4_NL dialect must not turn native mode 3 (Vulkan
	// poison) into a legacy decoder. The refusal now exercises the legacy
	// whitelist, not the unknown-stride guard.
	if V41EngramRowIQ4NL != 3 || V41EngramRowStride(V41EngramRowIQ4NL) != 144 {
		t.Fatal("IQ4_NL dialect must name eight 18-byte blocks without renumbering legacy dialects")
	}
	if V41EngramRowDecodeSupportsEncoding(legacy, V41EngramRowIQ4NL) {
		_, _, _ = legacy.V41EngramRowDecode(nil, V41EngramRowIQ4NL, 1, 1)
	}
	if legacy.calls != 0 {
		t.Fatal("IQ4_NL dialect reached legacy dispatch")
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
	if V41EngramRowDecodeSupportsEncoding(exact, V41EngramRowIQ4NL) {
		t.Fatal("negative exact IQ4_NL capability fell back to broad support")
	}
	exact.encoding = V41EngramRowIQ4NL
	if !V41EngramRowDecodeSupportsEncoding(exact, V41EngramRowIQ4NL) {
		t.Fatal("affirmative exact IQ4_NL capability did not admit the known dialect")
	}
	unknown := V41EngramRowEncoding(4)
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

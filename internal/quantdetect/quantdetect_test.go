package quantdetect

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func TestSafetensorsConfigFixture(t *testing.T) {
	raw := `{"model_type":"demo","quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,"version":"1"},"weight_path":"do-not-open.safetensors"}`
	got := Detect(strings.NewReader(raw), FamilySafetensors, 1024)
	if got.Status != StatusDetected || got.Metadata.Scheme != "gptq" || got.Metadata.Bits != 4 || got.Metadata.GroupSize != 128 {
		t.Fatalf("got %#v", got)
	}
	if got.BytesRead != len(raw) {
		t.Fatalf("bytes=%d want %d", got.BytesRead, len(raw))
	}
}

func TestRuntimeManifestFixture(t *testing.T) {
	raw := `{"runtime":"vllm","format":"awq","quant_method":"awq","bits":4}`
	got := Detect(strings.NewReader(raw), FamilyRuntime, 512)
	if got.Status != StatusDetected || got.Metadata.Runtime != "vllm" || got.Metadata.Format != "awq" {
		t.Fatalf("got %#v", got)
	}
}

func TestGGUFFixture(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0))
	binary.Write(&b, binary.LittleEndian, uint64(1))
	writeString(&b, "general.file_type")
	binary.Write(&b, binary.LittleEndian, uint32(4))
	binary.Write(&b, binary.LittleEndian, uint32(7))
	got := Detect(bytes.NewReader(b.Bytes()), FamilyGGUF, 1024)
	if got.Status != StatusDetected || got.Metadata.Format != "gguf" || got.Metadata.Scheme != "7" {
		t.Fatalf("got %#v", got)
	}
}
func writeString(b *bytes.Buffer, s string) {
	binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func TestMalformedInputIsExplicit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		family Family
		raw    string
	}{{"json", FamilySafetensors, "{"}, {"gguf", FamilyGGUF, "GG"}} {
		t.Run(tc.name, func(t *testing.T) {
			got := Detect(strings.NewReader(tc.raw), tc.family, 64)
			if got.Status != StatusMalformed || got.Reason != ReasonMalformed {
				t.Fatalf("got %#v", got)
			}
		})
	}
}

func TestUnknownMetadataDoesNotGuess(t *testing.T) {
	got := Detect(strings.NewReader(`{"model_type":"plain"}`), FamilySafetensors, 128)
	if got.Status != StatusUnknown || got.Reason != ReasonNoMetadata || got.Metadata.Scheme != "" {
		t.Fatalf("got %#v", got)
	}
}

func TestReadLimitIsHardBound(t *testing.T) {
	raw := strings.Repeat("x", 100)
	got := Detect(strings.NewReader(raw), FamilyRuntime, 16)
	if got.Status != StatusLimitExceeded || got.Reason != ReasonReadLimit || got.BytesRead != 17 {
		t.Fatalf("got %#v", got)
	}
}

func TestUnsupportedFamilyIsExplicit(t *testing.T) {
	got := Detect(strings.NewReader(`{}`), Family("weights"), 64)
	if got.Status != StatusUnsupported || got.Reason != ReasonFamilyUnsupported {
		t.Fatalf("got %#v", got)
	}
}

package quantdetect

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func BenchmarkDetectSafetensors(b *testing.B) {
	raw := `{"model_type":"demo","quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,"version":"1"},"weight_path":"do-not-open.safetensors"}`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Detect(strings.NewReader(raw), FamilySafetensors, 1024)
		if res.Status != StatusDetected {
			b.Fatalf("unexpected status: %v", res.Status)
		}
	}
}

func BenchmarkDetectRuntimeManifest(b *testing.B) {
	raw := `{"runtime":"vllm","format":"awq","quant_method":"awq","bits":4}`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Detect(strings.NewReader(raw), FamilyRuntime, 512)
		if res.Status != StatusDetected {
			b.Fatalf("unexpected status: %v", res.Status)
		}
	}
}

func BenchmarkDetectGGUF(b *testing.B) {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(1))
	writeString(&buf, "general.file_type")
	binary.Write(&buf, binary.LittleEndian, uint32(4))
	binary.Write(&buf, binary.LittleEndian, uint32(7))
	data := buf.Bytes()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Detect(bytes.NewReader(data), FamilyGGUF, 1024)
		if res.Status != StatusDetected {
			b.Fatalf("unexpected status: %v", res.Status)
		}
	}
}

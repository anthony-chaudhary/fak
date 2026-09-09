package mtpbench

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestProductionStreamedLoaderKeepsQ4CheckpointUntilModelClose(t *testing.T) {
	if mode := os.Getenv("FAK_MTPBENCH_STREAMED_CHILD"); mode != "" {
		productionStreamedLoaderChild(t, mode)
		return
	}
	for _, mode := range []string{"readat", "mmap"} {
		t.Run(mode, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestProductionStreamedLoaderKeepsQ4CheckpointUntilModelClose$")
			cmd.Env = append(os.Environ(), "FAK_MTPBENCH_STREAMED_CHILD="+mode, "FAK_GGUF_MMAP="+map[string]string{"readat": "0", "mmap": "1"}[mode])
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%s child: %v\n%s", mode, err, out)
			}
		})
	}
}

func productionStreamedLoaderChild(t *testing.T, mode string) {
	path := filepath.Join(t.TempDir(), "tiny-q4.gguf")
	if err := os.WriteFile(path, tinyStreamedQ4GGUF(), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := loadProductionModel(context.Background(), path)
	if err != nil {
		t.Fatalf("loadProductionModel: %v", err)
	}
	defer func() { _ = m.CloseWeights() }()
	if !m.Q4KLazy("model.layers.0.mlp.down_proj.weight") {
		t.Fatal("production loader materialized the dense Q4_K payload instead of retaining checkpoint backing")
	}
	before, available := openDescriptorsFor(t, path)
	if available && before < 1 {
		t.Fatalf("%s open checkpoint descriptors after load = %d, want >= 1", mode, before)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatalf("CloseWeights: %v", err)
	}
	if got, ok := openDescriptorsFor(t, path); available && ok && got != 0 {
		t.Fatalf("%s open checkpoint descriptors after CloseWeights = %d, want 0", mode, got)
	}
}

func openDescriptorsFor(t *testing.T, path string) (int, bool) {
	t.Helper()
	want, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("lsof", "-Fn", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Logf("descriptor inventory unavailable: %v", err)
		return 0, false
	}
	n := 0
	for _, line := range strings.Split(string(out), "\n") {
		if len(line) < 2 || line[0] != 'n' {
			continue
		}
		info, err := os.Stat(line[1:])
		if err == nil && os.SameFile(want, info) {
			n++
		}
	}
	return n, true
}

func tinyStreamedQ4GGUF() []byte {
	const (
		dim      = uint64(256)
		blockLen = 144
	)
	var b bytes.Buffer
	writeU32 := func(v uint32) { _ = binary.Write(&b, binary.LittleEndian, v) }
	writeU64 := func(v uint64) { _ = binary.Write(&b, binary.LittleEndian, v) }
	writeString := func(v string) { writeU64(uint64(len(v))); b.WriteString(v) }
	writeU32(0x46554747)
	writeU32(3)
	writeU64(1)
	writeU64(8)
	writeString("general.architecture")
	writeU32(8)
	writeString("llama")
	writeString("general.alignment")
	writeU32(4)
	writeU32(32)
	writeString("llama.embedding_length")
	writeU32(4)
	writeU32(uint32(dim))
	writeString("llama.block_count")
	writeU32(4)
	writeU32(1)
	writeString("llama.attention.head_count")
	writeU32(4)
	writeU32(1)
	writeString("llama.attention.key_length")
	writeU32(4)
	writeU32(uint32(dim))
	writeString("llama.feed_forward_length")
	writeU32(4)
	writeU32(uint32(dim))
	writeString("llama.attention.layer_norm_rms_epsilon")
	writeU32(6)
	_ = binary.Write(&b, binary.LittleEndian, float32(1e-6))
	writeString("blk.0.ffn_down.weight")
	writeU32(2)
	writeU64(dim)
	writeU64(1)
	writeU32(12) // GGML_TYPE_Q4_K
	writeU64(0)
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, blockLen))
	return b.Bytes()
}

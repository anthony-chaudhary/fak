package mtpbench

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
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
	for _, count := range []int{1, 2} {
		t.Run(fmt.Sprintf("shards=%d", count), func(t *testing.T) {
			dir := t.TempDir()
			var paths []string
			for i := range count {
				path, data := filepath.Join(dir, "tiny-q4.gguf"), tinyStreamedQ4GGUF()
				if count > 1 {
					path = filepath.Join(dir, fmt.Sprintf("tiny-%05d-of-00002.gguf", i+1))
					data = tinyStreamedQ4GGUF(i)
				}
				if err := os.WriteFile(path, data, 0o600); err != nil {
					t.Fatal(err)
				}
				paths = append(paths, path)
			}
			shards, digest, err := artifactSet(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			m, err := loadProductionModel(context.Background(), shards, digest)
			if err != nil {
				t.Fatalf("loadProductionModel: %v", err)
			}
			defer func() { _ = m.CloseWeights() }()
			if m.Q4KCount() != count || !m.Q4KLazy("model.layers.0.mlp.down_proj.weight") || count > 1 && !m.Q4KLazy("model.layers.0.mlp.up_proj.weight") {
				t.Fatal("production loader lost authenticated streamed tensor ownership")
			}
			for _, path := range paths {
				if before, available := openDescriptorsFor(t, path); available && before < 1 {
					t.Fatalf("%s open checkpoint descriptors after load = %d, want >= 1", mode, before)
				}
			}
			if err := m.CloseWeights(); err != nil {
				t.Fatalf("CloseWeights: %v", err)
			}
			for _, path := range paths {
				if got, ok := openDescriptorsFor(t, path); ok && got != 0 {
					t.Fatalf("%s open checkpoint descriptors after CloseWeights = %d, want 0", mode, got)
				}
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestProductionAuthenticatedLoaderRejectsUntrustedShards(t *testing.T) {
	t.Parallel()
	for _, kind := range []string{"missing digest", "wrong digest", "missing shard", "reordered", "tampered", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			for i := range 2 {
				if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("tiny-%05d-of-00002.gguf", i+1)), tinyStreamedQ4GGUF(i), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			shards, digest, err := artifactSet(filepath.Join(dir, "tiny-00001-of-00002.gguf"))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch kind {
			case "missing digest":
				digest = ""
			case "wrong digest":
				digest = strings.Repeat("0", 64)
			case "missing shard":
				shards = shards[:1]
			case "reordered":
				slices.Reverse(shards)
			case "tampered":
				data := tinyStreamedQ4GGUF(1)
				data[len(data)-1] ^= 1
				if err := os.WriteFile(shards[1].Path, data, 0o600); err != nil {
					t.Fatal(err)
				}
			case "cancelled":
				cancel()
			}
			m, err := loadProductionModel(ctx, shards, digest)
			if err == nil || m != nil {
				if m != nil {
					_ = m.CloseWeights()
				}
				t.Fatalf("untrusted model admitted: %v", err)
			}
		})
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

func tinyStreamedQ4GGUF(shard ...int) []byte {
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
	metadataCount := uint64(8)
	if len(shard) != 0 {
		metadataCount += 2
	}
	writeU64(metadataCount)
	if len(shard) != 0 {
		writeString("split.count")
		writeU32(4)
		writeU32(2)
		writeString("split.no")
		writeU32(4)
		writeU32(uint32(shard[0]))
	}
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
	name := "blk.0.ffn_down.weight"
	if len(shard) != 0 && shard[0] == 1 {
		name = "blk.0.ffn_up.weight"
	}
	writeString(name)
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

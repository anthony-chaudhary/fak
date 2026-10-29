package ggufload

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The released GSQ-RCO header declares a separate Engram file. This is a tiny
// routing/ownership witness, not a checkpoint forward or authenticated load.
// fak-test:justify why=regression when=changed:internal/ggufload/**
// fak-test:runtime fast est=100ms lane=default
func TestDeepSeek41DeclaredEngramCompanion(t *testing.T) {
	for _, split := range []bool{false, true} {
		for _, mapped := range []bool{false, true} {
			path, want := declaredEngramFixture(t, split, "")
			open := OpenWeights
			if mapped {
				open = OpenWeightsMapped
			}
			ws, err := open(path)
			if err != nil {
				t.Fatalf("open split=%v mapped=%v: %v", split, mapped, err)
			}
			for name, expected := range want {
				got, info, err := ws.TensorBytes(name)
				if err != nil || !bytes.Equal(got, expected) {
					_ = ws.Close()
					t.Fatalf("declared companion table %s missing or routed to wrong file: %v", name, err)
				}
				if info.Offset < 32 {
					t.Fatal("fixture requires a nonzero companion-local offset")
				}
			}
			got, _, err := ws.TensorBytes("backbone.weight")
			if err != nil || !bytes.Equal(got, f32Payload(1.25)) {
				t.Fatalf("backbone routing changed: %v", err)
			}
			if split {
				got, _, err = ws.TensorBytes("tail.weight")
				if err != nil || !bytes.Equal(got, f32Payload(2.5)) {
					t.Fatalf("later shard routing changed: %v", err)
				}
			}
			arch, _ := ws.File.String("general.architecture")
			marker, _ := ws.File.String("tokenizer.ggml.model")
			if arch != "deepseek41" || marker != "backbone-only" || ws.authenticatedArtifact != nil || ws.File.readEngramAux == nil {
				t.Fatal("companion changed metadata, promoted identity, or lost source-bound auxiliary callback")
			}
			// The incomplete Config/aux fixture must open without eager Config or
			// table materialization. Every retained reader has exactly one owner.
			count := 2
			if split {
				count++
			}
			if len(ws.closers) != count {
				t.Fatalf("retained closers=%d, want%d", len(ws.closers), count)
			}
			calls := make([]int, count)
			for i, closer := range ws.closers {
				ws.closers[i] = &declaredEngramCloser{closer, &calls[i]}
			}
			if err := ws.Close(); err != nil {
				t.Fatal(err)
			}
			if err := ws.Close(); err != nil {
				t.Fatal(err)
			}
			for _, n := range calls {
				if n != 1 {
					t.Fatalf("owned reader closed %d times", n)
				}
			}
		}
	}
	for _, kind := range []string{
		"parent path", "backslash path", "absolute path", "missing file",
		"wrong architecture", "wrong source", "wrong shape", "unsupported encoding",
		"duplicate table", "extra table", "overlap", "truncated extent",
		"oversized header", "variable-width array allocation", "conflicting rows",
	} {
		t.Run(kind, func(t *testing.T) {
			path, _ := declaredEngramFixture(t, false, kind)
			ws, err := OpenWeights(path)
			if ws != nil {
				_ = ws.Close()
			}
			if err == nil {
				t.Fatal("malformed declared companion was accepted")
			}
		})
	}
	t.Run("undeclared is not discovered", func(t *testing.T) {
		path, _ := declaredEngramFixture(t, false, "undeclared")
		ws, err := OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		if _, _, err := ws.TensorBytes("blk.1.engram_embd.weight"); err == nil {
			t.Fatal("unrequested sibling discovered")
		}
	})
}

type declaredEngramCloser struct {
	io.Closer
	calls *int
}

func (c *declaredEngramCloser) Close() error {
	*c.calls++
	return c.Closer.Close()
}

func declaredEngramU32Array(b *bytes.Buffer, key string, values ...uint32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint32))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, value := range values {
		_ = binary.Write(b, binary.LittleEndian, value)
	}
}

func declaredEngramFixture(t *testing.T, split bool, kind string) (string, map[string][]byte) {
	t.Helper()
	dir := t.TempDir()
	name := "engram.gguf"
	switch kind {
	case "parent path":
		name = "../engram.gguf"
	case "backslash path":
		name = `sub\engram.gguf`
	case "absolute path":
		name = filepath.Join(dir, "engram.gguf")
	case "missing file":
		name = "missing.gguf"
	}
	kvs := uint64(8)
	if split {
		kvs += 3
	}
	if kind == "conflicting rows" {
		kvs++
	}
	if kind == "undeclared" {
		kvs--
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, 2, kvs)
	writeKVString(&b, "general.architecture", "deepseek41")
	writeKVString(&b, "tokenizer.ggml.model", "backbone-only")
	writeKVString(&b, "deepseek41.engram.source", "test-source")
	if kind != "undeclared" {
		writeKVString(&b, "deepseek41.engram.file", name)
	}
	declaredEngramU32Array(&b, "deepseek41.engram.layer_ids", 1, 14)
	declaredEngramU32Array(&b, "deepseek41.engram.num_embeddings", 2, 1)
	writeKVUint32(&b, "deepseek41.engram.key_length", 256)
	writeKVUint32(&b, "deepseek41.block_count", 15)
	if kind == "conflicting rows" {
		declaredEngramU32Array(&b, "deepseek41.engram.rows", 3, 1)
	}
	if split {
		writeKVUint32(&b, "split.no", 0)
		writeKVUint32(&b, "split.count", 2)
		writeKVUint32(&b, "split.tensors.count", 3)
	}
	writeTensorInfoForTest(&b, "backbone.weight", []uint64{1}, TensorF32, 0)
	// GGML I32=26, used only to bind the existing lazy auxiliary owner; this
	// deliberately incomplete auxiliary set must not be read during open.
	writeTensorInfoForTest(&b, "engram_token_map.weight", []uint64{1}, TensorType(26), 32)
	padToAlignment(&b, 32)
	base := b.Len()
	b.Write(f32Payload(1.25))
	padToLen(&b, base+36)
	path := filepath.Join(dir, "backbone.gguf")
	if split {
		path = filepath.Join(dir, "backbone-00001-of-00002.gguf")
	}
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if split {
		path = filepath.Join(dir, "backbone-00002-of-00002.gguf")
		data := writeSplitShard(t, 1, 2, 3, false, []splitTensor{{"tail.weight", []uint64{1}, TensorF32, f32Payload(2.5)}})
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var side bytes.Buffer
	writeMinimalHeader(&side, 2, 2)
	arch, source := "deepseek41-engram", "test-source"
	if kind == "wrong architecture" {
		arch = "other"
	}
	if kind == "wrong source" {
		source = "other"
	}
	writeKVString(&side, "general.architecture", arch)
	writeKVString(&side, "deepseek41.engram.source", source)
	first, second := "blk.1.engram_embd.weight", "blk.14.engram_embd.weight"
	if kind == "duplicate table" {
		second = first
	}
	if kind == "extra table" {
		second = "blk.2.engram_embd.weight"
	}
	dims, typ, offset := []uint64{256, 2}, TensorIQ4_NL, uint64(352)
	if kind == "wrong shape" {
		dims[0] = 128
	}
	if kind == "unsupported encoding" {
		typ = TensorF32
	}
	if kind == "overlap" {
		offset = 64
	}
	writeTensorInfoForTest(&side, first, dims, typ, 32)
	writeTensorInfoForTest(&side, second, []uint64{256, 1}, TensorIQ4_NL, offset)
	padToAlignment(&side, 32)
	payload := make([]byte, 496)
	a, z := bytes.Repeat([]byte{0x31}, 288), bytes.Repeat([]byte{0xa7}, 144)
	copy(payload[32:], a)
	copy(payload[offset:], z)
	side.Write(payload)
	data := side.Bytes()
	if kind == "truncated extent" {
		data = data[:len(data)-1]
	}
	if kind == "oversized header" {
		binary.LittleEndian.PutUint64(data[16:24], 17)
	}
	if kind == "variable-width array allocation" {
		var bad bytes.Buffer
		writeMinimalHeader(&bad, 2, 1)
		writeStringForTest(&bad, "bad")
		_ = binary.Write(&bad, binary.LittleEndian, uint32(TypeArray))
		_ = binary.Write(&bad, binary.LittleEndian, uint32(TypeString))
		_ = binary.Write(&bad, binary.LittleEndian, uint64(math.MaxInt))
		data = bad.Bytes()
	}
	if err := os.WriteFile(filepath.Join(dir, "engram.gguf"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path, map[string][]byte{first: a, second: z}
}

package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=2s lane=default
func TestDeepSeek41TensorEngramAux(t *testing.T) {
	for _, split := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "split"}[split], func(t *testing.T) {
			ws, want, raw := tensorEngramAuxFixture(t)
			if split {
				// Only the second reader contains auxiliaries. Config must wait until
				// shard routing is installed, not read the constructor's primary reader.
				ws.r, ws.size = bytes.NewReader(nil), 0
				ws.readerFor = make([]io.ReaderAt, len(ws.File.Tensors))
				ws.sizeFor = make([]int64, len(ws.File.Tensors))
				for i, info := range ws.File.Tensors {
					if strings.HasPrefix(info.Name, "engram_") {
						ws.readerFor[i], ws.sizeFor[i] = bytes.NewReader(raw), int64(len(raw))
					}
				}
			}
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatalf("tensor-carried Config: %v", err)
			}
			got := ws.File.DeepSeek41Engram
			if !reflect.DeepEqual(got.TokenMap, want.TokenMap) || !reflect.DeepEqual(got.Primes, want.Primes) ||
				!reflect.DeepEqual(got.Offsets, want.Offsets) || !reflect.DeepEqual(got.Multipliers, want.Multipliers) ||
				!reflect.DeepEqual(got.NumEmbeddings, want.NumEmbeddings) || got.PadTokenID != want.PadTokenID {
				t.Fatal("tensor-carried Engram differs from the validated metadata layout")
			}
			if _, present := ws.File.Metadata["deepseek41.engram.token_map"]; present {
				t.Fatal("Config rewrote the original metadata")
			}
			for _, info := range ws.File.Tensors {
				if strings.HasPrefix(info.Name, "engram_") {
					work := ws.computeQ4KTensorWork(info, cfg, false, nil, q4kLoadOptions{}, 1)
					if work.err != nil || len(work.pending) != 0 {
						t.Fatalf("integer auxiliary entered the weight materializer: %s: %v", info.Name, work.err)
					}
				}
			}
		})
	}
}

// fak-test:runtime medium est=3s lane=default
func TestDeepSeek41TensorEngramAuxRefuses(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		edit       func(*WeightSource, []byte)
	}{
		{"map digest", "token map digest=", func(_ *WeightSource, raw []byte) { binary.LittleEndian.PutUint32(raw, 1) }},
		{"ordinary control token", "token_type[0]=1", func(ws *WeightSource, _ []byte) {
			types, _ := ws.File.Int32Array("tokenizer.ggml.token_type")
			types[0] = 1
			ws.File.Metadata["tokenizer.ggml.token_type"] = vcruzI32Array(types)
		}},
		{"negative integer", "is negative", func(_ *WeightSource, raw []byte) { binary.LittleEndian.PutUint32(raw, ^uint32(0)) }},
		{"wrong dtype", "must be type 26", func(ws *WeightSource, _ []byte) {
			ws.File.Tensors[ws.byName["engram_token_map.weight"]].Type = TensorF32
		}},
		{"oversized shape", "dimensions", func(ws *WeightSource, _ []byte) {
			ws.File.Tensors[ws.byName["engram_token_map.weight"]].Dims = []uint64{^uint64(0)}
		}},
		{"truncated shard", "overruns its shard", func(ws *WeightSource, _ []byte) { ws.size = 1 }},
		{"metadata conflict", "tensor disagrees", func(ws *WeightSource, _ []byte) {
			ws.File.Metadata["deepseek41.engram.token_map"] = vcruzI32Array([]int32{0})
		}},
		{"offset mismatch", "cumulative table rows", func(ws *WeightSource, raw []byte) {
			binary.LittleEndian.PutUint64(raw[ws.File.Tensors[ws.byName["engram_offsets.weight"]].FileOffset:], 1)
		}},
		{"row mismatch", "rows disagree", func(ws *WeightSource, _ []byte) {
			ws.File.Metadata["deepseek41.engram.num_embeddings"] = vcruzI32Array([]int32{1, 2})
		}},
		{"closed source", "source is closed", func(ws *WeightSource, _ []byte) { _ = ws.Close() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ws, _, raw := tensorEngramAuxFixture(t)
			tc.edit(ws, raw)
			_, err := ws.File.Config()
			if !errors.Is(err, model.ErrV41ConfigAdmission) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Config error=%v, want admission refusal containing %q", err, tc.want)
			}
		})
	}
}

// tensorEngramAuxFixture repackages the existing independently pinned metadata
// fixture into the integer tensor dialect. It is not a GSQ-RCO weight fixture.
func tensorEngramAuxFixture(t *testing.T) (*WeightSource, *DeepSeek41Engram, []byte) {
	t.Helper()
	f := loadV41HeaderFixture(t)
	if _, err := f.Config(); err != nil {
		t.Fatal(err)
	}
	want := *f.DeepSeek41Engram
	layout, err := buildV41EngramLayout(&want)
	if err != nil {
		t.Fatal(err)
	}
	want.NumEmbeddings = make([]int, len(layout.Rows))
	rows := make([]Value, len(layout.Rows))
	for i, n := range layout.Rows {
		want.NumEmbeddings[i] = int(n)
		rows[i] = Value{Type: TypeUint32, Value: n}
	}
	f.Metadata = maps.Clone(f.Metadata)
	f.Tensors = append([]TensorInfo(nil), f.Tensors...)
	f.Metadata["deepseek41.engram.num_embeddings"] = Value{Type: TypeArray, Value: rows}
	delete(f.Metadata, "deepseek41.engram.rows")
	for _, pair := range [][2]string{{"max_ngram_size", "engram_max_ngram_size"}, {"head_count", "engram_n_heads"}, {"key_length", "engram_head_dim"}, {"pad_id", "engram_pad_token_id"}} {
		f.Metadata["deepseek41."+pair[1]] = f.Metadata["deepseek41.engram."+pair[0]]
		delete(f.Metadata, "deepseek41.engram."+pair[0])
	}
	var raw bytes.Buffer
	for _, key := range []string{"token_map", "primes", "offsets", "multipliers"} {
		values, ok := f.Metadata["deepseek41.engram."+key].Value.([]Value)
		if !ok {
			t.Fatalf("missing fixture array %s", key)
		}
		dims, typ := []uint64{uint64(len(values) / len(want.LayerIDs)), uint64(len(want.LayerIDs))}, TensorType(27)
		if key == "token_map" {
			dims, typ = []uint64{uint64(len(values))}, TensorType(26)
		}
		f.Tensors = append(f.Tensors, TensorInfo{Name: "engram_" + key + ".weight", Type: typ, Dims: dims, FileOffset: int64(raw.Len())})
		for _, item := range values {
			value, ok := valueUint64(item)
			if !ok {
				t.Fatalf("noninteger fixture %s", key)
			}
			if typ == TensorType(26) {
				_ = binary.Write(&raw, binary.LittleEndian, uint32(value))
			} else {
				_ = binary.Write(&raw, binary.LittleEndian, value)
			}
		}
		delete(f.Metadata, "deepseek41.engram."+key)
	}
	f.DeepSeek41Engram = nil
	ws, err := NewWeightSource(f, bytes.NewReader(raw.Bytes()), int64(raw.Len()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	return ws, &want, raw.Bytes()
}

// fak-test:runtime medium est=1s lane=optin
func TestDeepSeek41TensorEngramAuxExactArtifact(t *testing.T) {
	dir := os.Getenv("FAK_GSQ_RCO_AUX_FIXTURE_DIR")
	if dir == "" {
		t.Skip("requires bounded GSQ-RCO header/auxiliary capture; no model download")
	}
	// Captured from ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
	// at 653a9874aa38c356233c5f3cc8c96b19a43b57eb: four directories, and only
	// the first shard's 517952-byte integer region. No weight/table execution.
	var f *File
	for i, digest := range []string{
		"8d95e07b6d694f1faebf8a501e82d08449ec57d9ccce02814a89510236aa4a6d",
		"642f6d8a4766af613d6795e7e04ab836fccb7a572fdd016198e5a6eba1f50cb1",
		"c83bc205d104b1ed7822e3f34f7b0009a4c4ebd24c37e63c4ab9cd9115beef6f",
		"d14875e46ba44c443bb587930038d73a00d2772453dd439a743a1cf70f67305b",
	} {
		name := fmt.Sprintf("recommended-shard%d-header.bin", i+1)
		header, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(header)); got != digest {
			t.Fatalf("%s identity=%s, want %s", name, got, digest)
		}
		part, err := Read(bytes.NewReader(header))
		if err != nil {
			t.Fatal(err)
		}
		if f == nil {
			f = part
		} else {
			f.Tensors = append(f.Tensors, part.Tensors...)
		}
	}
	aux, err := os.ReadFile(filepath.Join(dir, "engram-auxiliary-tensors.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(aux)); got != "bac33e1bfb24da289dc4d60247a03019614856f4f8249c1e168bb3cdda2dcc90" {
		t.Fatalf("auxiliary capture identity=%s", got)
	}
	if len(aux) != 517952 { //boundarylint:ignore CHANGE_DETECTOR_TEST the auxiliary capture is pinned by the SHA-256 check above; 517952 bytes is that capture's frozen width
		t.Fatalf("unexpected auxiliary capture length %d", len(aux))
	}
	reader := &tensorEngramAuxCaptureReader{start: f.TensorDataOffset, raw: aux}
	ws, err := NewWeightSource(f, reader, f.TensorDataOffset+int64(len(aux)))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if _, err := ws.File.Config(); err != nil {
		t.Fatalf("exact GSQ-RCO header/auxiliary Config: %v", err)
	}
	if reader.bytes != len(aux) || ws.File.DeepSeek41Engram.CompressedVocabSize != 99092 ||
		!reflect.DeepEqual(ws.File.DeepSeek41Engram.NumEmbeddings, []int{384006168, 384016682}) {
		t.Fatalf("unexpected bounded reads/layout: bytes=%d", reader.bytes)
	}
}

type tensorEngramAuxCaptureReader struct {
	start int64
	raw   []byte
	bytes int
}

func (r *tensorEngramAuxCaptureReader) ReadAt(p []byte, offset int64) (int, error) {
	if offset < r.start || offset > r.start+int64(len(r.raw)) || int64(len(p)) > r.start+int64(len(r.raw))-offset {
		return 0, errors.New("read outside bounded auxiliary capture")
	}
	n := copy(p, r.raw[offset-r.start:])
	r.bytes += n
	return n, nil
}

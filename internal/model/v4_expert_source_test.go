package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestV4ExpertSourceReadsOnlyRequestedRanges(t *testing.T) {
	const experts = 8
	tensors := make(map[string]tinySTTensor, experts+2)
	wantPayload := make(map[string][]byte, experts)
	for expert := 0; expert < experts; expert++ {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		payload := []byte{byte(expert), byte(expert + 10), byte(expert + 20), byte(expert + 30)}
		wantPayload[name] = payload
		tensors[name] = tinySTTensor{dtype: "F8_E4M3", shape: []int{2, 2}, data: payload}
	}
	sharedName := "model.layers.0.ffn.shared_experts.w1.weight"
	tensors[sharedName] = tinySTTensor{dtype: "F8_E4M3", shape: []int{2, 2}, data: []byte{90, 91, 92, 93}}
	tensors["model.layers.0.ffn.gate.weight"] = tinySTTensor{dtype: "F32", shape: []int{2, 2}, data: f32TestBytes([]float32{1, 2, 3, 4})}

	buf := tinySafetensorsBytes(t, tensors)
	rr := &v4ExpertSourceReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}
	rr.dataBase = sf.dataBase
	source, err := newV4ExpertSource(sf)
	if err != nil {
		t.Fatalf("newV4ExpertSource: %v", err)
	}
	if source.len() != experts {
		t.Fatalf("indexed routed experts=%d, want %d", source.len(), experts)
	}
	if rr.tensorReads != 0 {
		t.Fatalf("constructor performed %d tensor reads, want zero", rr.tensorReads)
	}

	for _, expert := range []int{3, 7} {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		got, err := source.read(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if got.Name != name || got.Dtype != "F8_E4M3" || !v4EqualInts(got.Shape, []int{2, 2}) || string(got.Bytes) != string(wantPayload[name]) {
			t.Fatalf("read %s = name=%q dtype=%q shape=%v bytes=%v", name, got.Name, got.Dtype, got.Shape, got.Bytes)
		}
	}
	if rr.tensorReads != 2 || rr.tensorBytes != 8 {
		t.Fatalf("requested range reads/bytes=%d/%d, want 2/8", rr.tensorReads, rr.tensorBytes)
	}

	readsBeforeReject := rr.tensorReads
	if _, err := source.read(sharedName); !errors.Is(err, ErrV4ExpertNotRouted) {
		t.Fatalf("shared-expert read error=%v, want ErrV4ExpertNotRouted", err)
	}
	missing := "model.layers.0.ffn.experts.99.w1.weight"
	if _, err := source.read(missing); !errors.Is(err, ErrV4ExpertNotFound) {
		t.Fatalf("unknown expert read error=%v, want ErrV4ExpertNotFound", err)
	}
	if rr.tensorReads != readsBeforeReject {
		t.Fatalf("rejected requests performed tensor IO: reads %d -> %d", readsBeforeReject, rr.tensorReads)
	}

	for name, raw := range sf.hdr {
		var entry stEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			t.Fatalf("decode %s entry: %v", name, err)
		}
		gotReads := rr.byRange[v4ExpertRangeKey(sf.dataBase, entry)]
		wantReads := 0
		if name == "model.layers.0.ffn.experts.3.w1.weight" || name == "model.layers.0.ffn.experts.7.w1.weight" {
			wantReads = 1
		}
		if gotReads != wantReads {
			t.Fatalf("%s range reads=%d, want %d", name, gotReads, wantReads)
		}
	}
}

func TestV4ExpertSourceRejectsMalformedMetadataBeforePayloadIO(t *testing.T) {
	tests := []struct {
		name    string
		tensors map[string]tinySTTensor
	}{
		{
			name: "identity",
			tensors: map[string]tinySTTensor{
				"model.layers.bad.ffn.experts.0.w1.weight": {dtype: "F8_E4M3", shape: []int{1}, data: []byte{1}},
			},
		},
		{
			name: "shape",
			tensors: map[string]tinySTTensor{
				"model.layers.0.ffn.experts.0.w1.weight": {dtype: "F8_E4M3", shape: []int{0}, data: []byte{1}},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buf := tinySafetensorsBytes(t, tc.tensors)
			rr := &v4ExpertSourceReaderAt{data: buf}
			sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
			if err != nil {
				t.Fatalf("newSafetensorsFile: %v", err)
			}
			rr.dataBase = sf.dataBase
			if _, err := newV4ExpertSource(sf); !errors.Is(err, ErrV4ExpertMetadata) {
				t.Fatalf("newV4ExpertSource error=%v, want ErrV4ExpertMetadata", err)
			}
			if rr.tensorReads != 0 {
				t.Fatalf("malformed constructor performed %d tensor reads", rr.tensorReads)
			}
		})
	}
}

type v4ExpertSourceReaderAt struct {
	data        []byte
	dataBase    int64
	tensorReads int
	tensorBytes int64
	byRange     map[string]int
}

func (r *v4ExpertSourceReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if r.dataBase != 0 && off >= r.dataBase {
		if r.byRange == nil {
			r.byRange = make(map[string]int)
		}
		r.tensorReads++
		r.tensorBytes += int64(len(p))
		r.byRange[fmt.Sprintf("%d:%d", off, off+int64(len(p)))]++
	}
	if off < 0 || off > int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

func v4ExpertRangeKey(dataBase int64, entry stEntry) string {
	return fmt.Sprintf("%d:%d", dataBase+int64(entry.DataOffsets[0]), dataBase+int64(entry.DataOffsets[1]))
}

func v4EqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

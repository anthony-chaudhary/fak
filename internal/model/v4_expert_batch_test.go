package model

import (
	"errors"
	"fmt"
	"testing"
)

func TestV4ExpertBatchReadsExactlySelectedGroupsUnderCap(t *testing.T) {
	const (
		experts       = 8
		tensorsPerExp = 3
	)
	tensors := make(map[string]tinySTTensor, experts*tensorsPerExp+1)
	for expert := 0; expert < experts; expert++ {
		for tensorIndex, leaf := range []string{"w1.weight", "w2.weight", "w3.weight"} {
			name := fmt.Sprintf("model.layers.0.ffn.experts.%d.%s", expert, leaf)
			tensors[name] = tinySTTensor{
				dtype: "U8",
				shape: []int{2},
				data:  []byte{byte(expert), byte(tensorIndex)},
			}
		}
	}
	// A routed expert on another layer must not satisfy a layer-0 selection.
	tensors["model.layers.1.ffn.experts.0.w1.weight"] = tinySTTensor{dtype: "U8", shape: []int{2}, data: []byte{99, 1}}

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

	picks := routeTopKSoftmax([]float32{5, 2, -3, -4, 3, -2, 8, 6}, 6)
	selected := make([]int, len(picks))
	for i, pick := range picks {
		selected[i] = pick.expert
	}
	wantSelected := []int{6, 7, 0, 4, 1, 5}
	if !v4EqualInts(selected, wantSelected) {
		t.Fatalf("selected experts=%v, want %v", selected, wantSelected)
	}
	const exactBytes = int64(6 * tensorsPerExp * 2)
	batch, err := source.readV4ExpertBatch(0, selected, exactBytes)
	if err != nil {
		t.Fatalf("readV4ExpertBatch: %v", err)
	}
	if len(batch.Plan.Groups) != 6 || batch.Plan.TensorCount != 18 || batch.Plan.Bytes != exactBytes || len(batch.Tensors) != 18 {
		t.Fatalf("plan groups/tensors/bytes/read=%d/%d/%d/%d, want 6/18/%d/18", len(batch.Plan.Groups), batch.Plan.TensorCount, batch.Plan.Bytes, len(batch.Tensors), exactBytes)
	}
	for i, group := range batch.Plan.Groups {
		if group.Expert != selected[i] || group.Layer != 0 || len(group.TensorNames) != tensorsPerExp || group.Bytes != 6 {
			t.Fatalf("group[%d]=%+v", i, group)
		}
		if group.TensorNames[0] > group.TensorNames[1] || group.TensorNames[1] > group.TensorNames[2] {
			t.Fatalf("group[%d] tensor order is unstable: %v", i, group.TensorNames)
		}
	}
	if rr.tensorReads != 18 || rr.tensorBytes != exactBytes {
		t.Fatalf("actual reads/bytes=%d/%d, want 18/%d", rr.tensorReads, rr.tensorBytes, exactBytes)
	}
	selectedSet := make(map[int]bool, len(selected))
	for _, expert := range selected {
		selectedSet[expert] = true
	}
	for name, entry := range source.entries {
		layer, expert, err := parseV4ExpertIdentity(name)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		wantReads := 0
		if layer == 0 && selectedSet[expert] {
			wantReads = 1
		}
		if got := rr.byRange[v4ExpertRangeKey(sf.dataBase, entry)]; got != wantReads {
			t.Fatalf("%s range reads=%d, want %d", name, got, wantReads)
		}
	}
}

func TestV4ExpertBatchRejectsBeforePayloadIO(t *testing.T) {
	tensors := make(map[string]tinySTTensor)
	for expert := 0; expert < 8; expert++ {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		tensors[name] = tinySTTensor{dtype: "U8", shape: []int{2}, data: []byte{byte(expert), 1}}
	}
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

	tests := []struct {
		name     string
		layer    int
		selected []int
		cap      int64
		wantErr  error
	}{
		{name: "top-six-minus-one-byte", layer: 0, selected: []int{0, 1, 2, 3, 4, 5}, cap: 11, wantErr: ErrV4ExpertBatchCap},
		{name: "duplicate", layer: 0, selected: []int{0, 1, 1}, cap: 100, wantErr: ErrV4ExpertSelection},
		{name: "out-of-range", layer: 0, selected: []int{0, 8}, cap: 100, wantErr: ErrV4ExpertSelection},
		{name: "wrong-layer", layer: 1, selected: []int{0}, cap: 100, wantErr: ErrV4ExpertSelection},
		{name: "negative", layer: 0, selected: []int{-1}, cap: 100, wantErr: ErrV4ExpertSelection},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			readsBefore := rr.tensorReads
			if _, err := source.readV4ExpertBatch(tc.layer, tc.selected, tc.cap); !errors.Is(err, tc.wantErr) {
				t.Fatalf("error=%v, want %v", err, tc.wantErr)
			}
			if rr.tensorReads != readsBefore {
				t.Fatalf("rejection performed payload reads: %d -> %d", readsBefore, rr.tensorReads)
			}
		})
	}
}

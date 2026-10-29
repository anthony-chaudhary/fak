package model

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestV41PackedScaleRejectsForgedReferenceBeforeRead(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 1)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tensors := v41TestInventoryTensors(t, inv)
	path := filepath.Join(t.TempDir(), "scale.safetensors")
	writeV4RawShard(t, path, tensors)
	shard, err := OpenV41PackedShard(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer shard.Close()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	recorder := &recordingReaderAt{data: raw}
	shard.file.r, shard.file.data = recorder, nil
	ref, err := shard.Reference("layers.0.attn.wkv.weight")
	if err != nil {
		t.Fatal(err)
	}
	scaleLen := ref.ScaleEnd - ref.ScaleStart
	if scaleLen <= 1 {
		t.Fatal("fixture needs more than one scale byte")
	}
	for _, tc := range []struct {
		name   string
		change func(*V41PackedTensor)
	}{
		{"short-span", func(r *V41PackedTensor) { r.ScaleEnd = r.ScaleStart + 1 }},
		{"negative-span", func(r *V41PackedTensor) { r.ScaleEnd = r.ScaleStart - 1 }},
		{"shifted-span", func(r *V41PackedTensor) { r.ScaleStart++; r.ScaleEnd++ }},
		{"foreign-shard", func(r *V41PackedTensor) { r.Shaper += ".other" }},
		{"foreign-scale", func(r *V41PackedTensor) { r.Ref.ScaleName = "layers.0.attn.wq_a.scale" }},
		{"foreign-weight-span", func(r *V41PackedTensor) { r.Start++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			forged := ref
			tc.change(&forged)
			before := recorder.reads
			got, err := shard.ReadScale(forged, 1)
			if !errors.Is(err, ErrV41PackedUnsupported) || got != nil {
				t.Fatalf("got %d bytes, error %v", len(got), err)
			}
			if recorder.reads != before {
				t.Fatal("invalid reference reached payload reader")
			}
		})
	}
	before := recorder.reads
	if got, err := shard.ReadScale(ref, scaleLen-1); !errors.Is(err, ErrV41PackedUnsupported) || got != nil {
		t.Fatalf("bound refusal: %d bytes, %v", len(got), err)
	}
	if recorder.reads != before {
		t.Fatal("over-bound read reached payload reader")
	}
	for _, bound := range []int64{scaleLen, 0} {
		got, err := shard.ReadScale(ref, bound)
		if err != nil || !bytes.Equal(got, tensors[ref.Ref.ScaleName].data) {
			t.Fatalf("valid bound %d: %v", bound, err)
		}
	}
	entry := shard.entries[ref.Ref.ScaleName]
	entry.DataOffsets = append([]int(nil), entry.DataOffsets...)
	entry.DataOffsets[0]++
	entry.DataOffsets[1]++
	shard.entries[ref.Ref.ScaleName] = entry
	before = recorder.reads
	if got, err := shard.ReadScale(ref, scaleLen); !errors.Is(err, ErrV41PackedUnsupported) || got != nil {
		t.Fatalf("moved entry: %d bytes, %v", len(got), err)
	}
	if recorder.reads != before {
		t.Fatal("moved entry reached payload reader")
	}
}

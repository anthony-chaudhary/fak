package compute

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1s
func TestAPUSequenceRejectsOversizedRowsBeforeWrites(t *testing.T) {
	for _, isValue := range []bool{false, true} {
		name := "key"
		if isValue {
			name = "value"
		}
		t.Run(name, func(t *testing.T) {
			mgr, err := NewAPUUnifiedMemoryManager(APUTopologyInfo{IsUnifiedTopology: true, UnifiedDRAMBytes: 4096})
			if err != nil {
				t.Fatal(err)
			}
			kv, err := AllocateUnifiedKVCache(mgr, APUKVCacheConfig{NumLayers: 2, NumHeads: 1, HeadDim: 2, MaxTokens: 4, BytesPerElem: 2, BatchSize: 1})
			if err != nil {
				t.Fatal(err)
			}
			defer kv.Release()
			backing := kv.Buffer.Bytes()
			for i := range backing {
				backing[i] = byte(i + 1)
			}
			before := append([]byte(nil), backing...)
			count, _, _ := mgr.Fence().Stats()
			stored := kv.TokensStored.Load()
			valid := bytes.Repeat([]byte{0x8a}, int(kv.TokenStride))
			oversized := bytes.Repeat([]byte{0xee}, int(kv.TokenStride)+1)
			// The second row is invalid, so even the first valid row must remain unpublished.
			if err := kv.WritePromptSequenceKV(0, isValue, 1, [][]byte{valid, oversized}); err == nil {
				t.Fatal("oversized sequence row accepted")
			}
			afterCount, _, _ := mgr.Fence().Stats()
			if !bytes.Equal(backing, before) || kv.TokensStored.Load() != stored || afterCount != count {
				t.Fatal("invalid sequence changed bytes, counter, or fence")
			}
			// Same-size and shorter rows retain the single-token method's existing policy.
			short := []byte{7, 9}
			rows := [][]byte{valid, short}
			if err := kv.WritePromptSequenceKV(0, isValue, 1, rows); err != nil {
				t.Fatal(err)
			}
			got, err := kv.GPUReadSequenceKV(0, isValue, 1, len(rows))
			if err != nil {
				t.Fatal(err)
			}
			offset := int(kv.TokenStride)
			if isValue {
				offset += kv.Config.MaxTokens * int(kv.TokenStride)
			}
			want := append([]byte(nil), before[offset:offset+len(rows)*int(kv.TokenStride)]...)
			copy(want, valid)
			copy(want[int(kv.TokenStride):], short)
			if !bytes.Equal(got, want) {
				t.Fatalf("round trip=%v want=%v", got, want)
			}
			expected := append([]byte(nil), before...)
			copy(expected[offset:], want)
			if !bytes.Equal(backing, expected) {
				t.Fatal("valid sequence changed adjacent storage")
			}
			afterCount, _, _ = mgr.Fence().Stats()
			if kv.TokensStored.Load() != stored+int64(len(rows)) || afterCount != count+1 {
				t.Fatal("valid sequence publication accounting differs")
			}
		})
	}
}

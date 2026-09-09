package ctxmmu

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
)

func TestKVPoolDelayedOperationCannotMutateReusedSequenceID(t *testing.T) {
	pool, err := NewKVPool(KVPoolConfig{
		TokensPerPage: 2,
		NumLayers:     1,
		NumKVHeads:    1,
		HeadDim:       2,
		DType:         KVDTypeFP8,
		Topology:      KVTopologyGQA,
		TotalPages:    4,
	})
	if err != nil {
		t.Fatalf("NewKVPool: %v", err)
	}

	const seqID = "reused"
	oldSeq, err := pool.CreateSequence(seqID)
	if err != nil {
		t.Fatalf("CreateSequence old: %v", err)
	}
	if err := pool.AppendTokens(seqID, 1); err != nil {
		t.Fatalf("AppendTokens old: %v", err)
	}
	oldKey, oldVal := []byte{0x11, 0x12}, []byte{0x21, 0x22}
	if err := pool.WriteTokenKV(seqID, 0, 0, oldKey, oldVal); err != nil {
		t.Fatalf("WriteTokenKV old: %v", err)
	}
	if _, err := pool.ForkSequence(seqID, "fork"); err != nil {
		t.Fatalf("ForkSequence: %v", err)
	}
	if gotKey, gotVal, err := pool.ReadTokenKV("fork", 0, 0); err != nil || !bytes.Equal(gotKey, oldKey) || !bytes.Equal(gotVal, oldVal) {
		t.Fatalf("fork readback = (%x, %x, %v), want (%x, %x, nil)", gotKey, gotVal, err, oldKey, oldVal)
	}
	if err := pool.ReleaseSequence("fork"); err != nil {
		t.Fatalf("ReleaseSequence fork: %v", err)
	}
	if err := pool.SwapBlock(seqID, 0); err != nil {
		t.Fatalf("SwapBlock old: %v", err)
	}
	if err := pool.SwapInBlock(seqID, 0); err != nil {
		t.Fatalf("SwapInBlock old: %v", err)
	}
	if gotKey, gotVal, err := pool.ReadTokenKV(seqID, 0, 0); err != nil || !bytes.Equal(gotKey, oldKey) || !bytes.Equal(gotVal, oldVal) {
		t.Fatalf("old readback after swap = (%x, %x, %v), want (%x, %x, nil)", gotKey, gotVal, err, oldKey, oldVal)
	}
	oldPageID := oldSeq.Pages()[0]

	resolved := make(chan struct{})
	resume := make(chan struct{})
	pool.testHookSequenceResolved = func(seq *KVSequence) {
		if seq != oldSeq {
			return
		}
		resolved <- struct{}{}
		<-resume
	}

	appendErr := make(chan error, 1)
	swapErr := make(chan error, 1)
	go func() {
		appendErr <- pool.AppendTokens(seqID, 2)
	}()
	go func() {
		swapErr <- pool.SwapBlock(seqID, 0)
	}()
	<-resolved
	<-resolved

	if err := pool.ReleaseSequence(seqID); err != nil {
		t.Fatalf("ReleaseSequence old: %v", err)
	}
	replacement, err := pool.CreateSequence(seqID)
	if err != nil {
		t.Fatalf("CreateSequence replacement: %v", err)
	}
	if err := pool.AppendTokens(seqID, 1); err != nil {
		t.Fatalf("AppendTokens replacement: %v", err)
	}
	replacementKey, replacementVal := []byte{0xa1, 0xa2}, []byte{0xb1, 0xb2}
	if err := pool.WriteTokenKV(seqID, 0, 0, replacementKey, replacementVal); err != nil {
		t.Fatalf("WriteTokenKV replacement: %v", err)
	}

	wantTokens := replacement.Tokens()
	wantPages := replacement.Pages()
	wantAllocated := pool.PhysicalBlocksAllocated()
	close(resume)
	staleAppendErr := <-appendErr
	staleSwapErr := <-swapErr

	gotKey, gotVal, readErr := pool.ReadTokenKV(seqID, 0, 0)
	gotSeq, found := pool.GetSequence(seqID)
	if !errors.Is(staleAppendErr, ErrSequenceNotFound) || !errors.Is(staleSwapErr, ErrSequenceNotFound) ||
		!found || gotSeq != replacement ||
		replacement.Tokens() != wantTokens || !reflect.DeepEqual(replacement.Pages(), wantPages) ||
		pool.PhysicalBlocksAllocated() != wantAllocated || readErr != nil ||
		!bytes.Equal(gotKey, replacementKey) || !bytes.Equal(gotVal, replacementVal) {
		t.Fatalf("stale operations changed reused ID: appendErr=%v swapErr=%v found=%v same=%v tokens=%d/%d pages=%v/%v allocated=%d/%d read=(%x,%x,%v)",
			staleAppendErr, staleSwapErr, found, gotSeq == replacement, replacement.Tokens(), wantTokens, replacement.Pages(), wantPages,
			pool.PhysicalBlocksAllocated(), wantAllocated, gotKey, gotVal, readErr)
	}
	if err := pool.swapBlock(seqID, oldSeq, 0, oldPageID); !errors.Is(err, ErrSequenceNotFound) {
		t.Fatalf("stale eviction candidate error = %v, want ErrSequenceNotFound", err)
	}

	if err := pool.SwapBlock(seqID, 0); err != nil {
		t.Fatalf("SwapBlock replacement: %v", err)
	}
	if err := pool.SwapInBlock(seqID, 0); err != nil {
		t.Fatalf("SwapInBlock replacement: %v", err)
	}
	if gotKey, gotVal, err := pool.ReadTokenKV(seqID, 0, 0); err != nil || !bytes.Equal(gotKey, replacementKey) || !bytes.Equal(gotVal, replacementVal) {
		t.Fatalf("replacement readback after swap = (%x, %x, %v), want (%x, %x, nil)", gotKey, gotVal, err, replacementKey, replacementVal)
	}
}

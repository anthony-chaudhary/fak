package ctxmmu

import (
	"testing"
)

// TestDefaultModelShapePreservesHistoricalDefaults locks the decoupled default
// shape to the literals SaveInPlaceCheckpoint used before routing through
// ModelShape, so the abstraction change is behavior-neutral.
func TestDefaultModelShapePreservesHistoricalDefaults(t *testing.T) {
	want := ModelShape{
		AttentionHeadCount: 32,
		SlidingWindowSize:  4096,
		RoPEBaseFrequency:  1000000.0,
		MTPDraftDepth:      4,
		DeltaNetLayers:     DefaultDeltaNetLayers,
		DeltaNetValueHeads: DefaultDeltaNetValueHeads,
		DeltaNetKeyHeadDim: DefaultDeltaNetKeyHeadDim,
		DeltaNetValHeadDim: DefaultDeltaNetValHeadDim,
		DeltaNetConvKernel: DefaultDeltaNetConvKernel,
	}
	if got := DefaultModelShape; got != want {
		t.Fatalf("DefaultModelShape = %+v, want historical defaults %+v", got, want)
	}
}

func newShapeTestManager(t *testing.T, sessionID string) (*CheckpointManager, string) {
	t.Helper()
	forkMgr := NewForkManager()
	cm := NewCheckpointManager(nil, nil, forkMgr, nil)
	sess, err := forkMgr.RegisterSession(sessionID, BlockGranularity64)
	if err != nil {
		t.Fatalf("RegisterSession failed: %v", err)
	}
	if err := sess.AppendTokens([]int32{10, 20, 30, 40, 50}...); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}
	return cm, sessionID
}

// TestSetModelShapeRoutesIntoSequenceDescriptor proves per-model shape values
// injected via SetModelShape reach the saved descriptor instead of hardcoded
// literals.
func TestSetModelShapeRoutesIntoSequenceDescriptor(t *testing.T) {
	cm, sessionID := newShapeTestManager(t, "model-shape-routed")

	shape := ModelShape{
		AttentionHeadCount: 64,
		SlidingWindowSize:  8192,
		RoPEBaseFrequency:  5000000.0,
		MTPDraftDepth:      7,
		DeltaNetLayers:     32,
		DeltaNetValueHeads: 16,
		DeltaNetKeyHeadDim: 64,
		DeltaNetValHeadDim: 64,
		DeltaNetConvKernel: 2,
	}
	cm.SetModelShape(shape)

	desc, err := cm.SaveInPlaceCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint failed: %v", err)
	}

	if got := desc.AttentionMask.HeadCount; got != 64 {
		t.Errorf("AttentionMask.HeadCount = %d, want 64", got)
	}
	if got := desc.AttentionMask.SlidingWindowSize; got != 8192 {
		t.Errorf("AttentionMask.SlidingWindowSize = %d, want 8192", got)
	}
	if got := desc.RoPE.BaseFrequency; got != 5000000.0 {
		t.Errorf("RoPE.BaseFrequency = %v, want 5000000", got)
	}
	if got := desc.MTPState.DraftDepth; got != 7 {
		t.Errorf("MTPState.DraftDepth = %d, want 7", got)
	}
	dn := desc.DeltaNetState
	if dn.NumLayers != 32 || dn.ValueHeads != 16 || dn.KeyHeadDim != 64 || dn.ValueHeadDim != 64 || dn.ConvKernelDim != 2 {
		t.Errorf("DeltaNetState = {layers:%d heads:%d keyDim:%d valDim:%d conv:%d}, want {32 16 64 64 2}",
			dn.NumLayers, dn.ValueHeads, dn.KeyHeadDim, dn.ValueHeadDim, dn.ConvKernelDim)
	}
}

// TestModelShapeZeroFieldsFallBackToDefaults proves a partially specified
// shape keeps the historical defaults for every unset field.
func TestModelShapeZeroFieldsFallBackToDefaults(t *testing.T) {
	cm, sessionID := newShapeTestManager(t, "model-shape-fallback")
	cm.SetModelShape(ModelShape{AttentionHeadCount: 48})

	desc, err := cm.SaveInPlaceCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("SaveInPlaceCheckpoint failed: %v", err)
	}

	d := DefaultModelShape
	if got := desc.AttentionMask.HeadCount; got != 48 {
		t.Errorf("AttentionMask.HeadCount = %d, want overridden 48", got)
	}
	if got := desc.AttentionMask.SlidingWindowSize; got != d.SlidingWindowSize {
		t.Errorf("AttentionMask.SlidingWindowSize = %d, want default %d", got, d.SlidingWindowSize)
	}
	if got := desc.RoPE.BaseFrequency; got != d.RoPEBaseFrequency {
		t.Errorf("RoPE.BaseFrequency = %v, want default %v", got, d.RoPEBaseFrequency)
	}
	if got := desc.MTPState.DraftDepth; got != d.MTPDraftDepth {
		t.Errorf("MTPState.DraftDepth = %d, want default %d", got, d.MTPDraftDepth)
	}
	if got := desc.DeltaNetState.NumLayers; got != d.DeltaNetLayers {
		t.Errorf("DeltaNetState.NumLayers = %d, want default %d", got, d.DeltaNetLayers)
	}
}

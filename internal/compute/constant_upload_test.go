package compute

import (
	"bytes"
	"errors"
	"testing"
)

// TestConstantUploadInvariantUnderGraphCapture tests that under CUDA graph capture,
// constant uploads are unconditional and never skipped even if content matches (#10716).
func TestConstantUploadInvariantUnderGraphCapture(t *testing.T) {
	mgr := NewConstantUploadManager()

	slot := ConstantSlotID("layer0_scale")
	if err := mgr.RegisterSlot(slot, 64); err != nil {
		t.Fatalf("RegisterSlot failed: %v", err)
	}

	payload := []byte("scale-factor-1.000000000000000000000000000000")

	// Phase 1: Outside capture (Eager mode)
	// First upload: executed
	r1, err := mgr.UploadConstant(slot, payload)
	if err != nil || r1.Skipped || r1.Captured {
		t.Fatalf("r1 = %+v, want non-skipped, non-captured", r1)
	}

	// Second upload outside capture: identical payload should be skipped by eager caching
	r2, err := mgr.UploadConstant(slot, payload)
	if err != nil || !r2.Skipped || r2.Captured {
		t.Fatalf("r2 = %+v, want skipped=true in eager mode", r2)
	}

	// Phase 2: Under Graph Capture
	// Even with IDENTICAL payload, upload MUST NOT BE SKIPPED. It must record a graph copy node!
	if err := mgr.BeginCapture("graph_step_100", []ConstantSlotID{slot}); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}

	r3, err := mgr.UploadConstant(slot, payload)
	if err != nil {
		t.Fatalf("UploadConstant under capture failed: %v", err)
	}
	if r3.Skipped {
		t.Fatalf("invariant violation: upload under capture was skipped! (r3 = %+v)", r3)
	}
	if !r3.Captured {
		t.Fatalf("r3 should have Captured=true")
	}

	// Upload identical payload AGAIN under capture: MUST STILL NOT BE SKIPPED!
	r4, err := mgr.UploadConstant(slot, payload)
	if err != nil {
		t.Fatalf("second UploadConstant under capture failed: %v", err)
	}
	if r4.Skipped {
		t.Fatalf("invariant violation: second identical upload under capture was skipped!")
	}

	graph, err := mgr.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture failed: %v", err)
	}

	if graph.CapturedUploads != 2 {
		t.Fatalf("expected 2 captured upload nodes, got %d", graph.CapturedUploads)
	}
}

// TestConstantUploadTurboQuantAlternatingLayers tests the TurboQuant cross-layer pattern (#10716).
// Alternating layer parameters (L0:cfgA, L1:cfgB, L0:cfgB) must be unconditionally captured
// so replayed execution has zero cross-layer state bleeding.
func TestConstantUploadTurboQuantAlternatingLayers(t *testing.T) {
	mgr := NewConstantUploadManager()

	slot0 := ConstantSlotID("layer0_dequant")
	slot1 := ConstantSlotID("layer1_dequant")

	_ = mgr.RegisterSlot(slot0, 32)
	_ = mgr.RegisterSlot(slot1, 32)

	cfgA := []byte("quant_table_alpha_config_001")
	cfgB := []byte("quant_table_beta_config_0002")

	// Capture Graph 0 (Layer 0 with cfgA, Layer 1 with cfgB)
	if err := mgr.BeginCapture("graph_model_alpha", []ConstantSlotID{slot0, slot1}); err != nil {
		t.Fatalf("BeginCapture alpha failed: %v", err)
	}
	if _, err := mgr.UploadConstant(slot0, cfgA); err != nil {
		t.Fatalf("Upload slot0 failed: %v", err)
	}
	if _, err := mgr.UploadConstant(slot1, cfgB); err != nil {
		t.Fatalf("Upload slot1 failed: %v", err)
	}
	graphAlpha, err := mgr.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture alpha failed: %v", err)
	}
	if len(graphAlpha.RecordedNodes) != 2 {
		t.Fatalf("graphAlpha nodes = %d, want 2", len(graphAlpha.RecordedNodes))
	}

	// Capture Graph 1 (Alternating: Layer 0 now gets cfgB, Layer 1 gets cfgB)
	// Because slot1 already has cfgB, buggy conditional caching would skip slot1!
	// With unconditional invariant, both slot0 and slot1 get explicit upload nodes.
	if err := mgr.BeginCapture("graph_model_beta", []ConstantSlotID{slot0, slot1}); err != nil {
		t.Fatalf("BeginCapture beta failed: %v", err)
	}
	if _, err := mgr.UploadConstant(slot0, cfgB); err != nil {
		t.Fatalf("Upload slot0 cfgB failed: %v", err)
	}
	rSlot1, err := mgr.UploadConstant(slot1, cfgB)
	if err != nil {
		t.Fatalf("Upload slot1 cfgB failed: %v", err)
	}
	if rSlot1.Skipped {
		t.Fatalf("invariant violation: slot1 upload was skipped because cfgB matched previous state!")
	}
	graphBeta, err := mgr.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture beta failed: %v", err)
	}
	if len(graphBeta.RecordedNodes) != 2 {
		t.Fatalf("graphBeta nodes = %d, want 2", len(graphBeta.RecordedNodes))
	}

	// Now Replay Graph Alpha: Must restore slot0=cfgA, slot1=cfgB
	if _, err := mgr.ReplayGraph("graph_model_alpha"); err != nil {
		t.Fatalf("ReplayGraph alpha failed: %v", err)
	}
	data0, _, _ := mgr.ReadDeviceConstant(slot0)
	data1, _, _ := mgr.ReadDeviceConstant(slot1)
	if !bytes.Equal(data0, cfgA) {
		t.Fatalf("replay alpha: slot0 = %q, want cfgA %q", data0, cfgA)
	}
	if !bytes.Equal(data1, cfgB) {
		t.Fatalf("replay alpha: slot1 = %q, want cfgB %q", data1, cfgB)
	}

	// Now Replay Graph Beta: Must restore slot0=cfgB, slot1=cfgB
	if _, err := mgr.ReplayGraph("graph_model_beta"); err != nil {
		t.Fatalf("ReplayGraph beta failed: %v", err)
	}
	data0Beta, _, _ := mgr.ReadDeviceConstant(slot0)
	data1Beta, _, _ := mgr.ReadDeviceConstant(slot1)
	if !bytes.Equal(data0Beta, cfgB) {
		t.Fatalf("replay beta: slot0 = %q, want cfgB %q", data0Beta, cfgB)
	}
	if !bytes.Equal(data1Beta, cfgB) {
		t.Fatalf("replay beta: slot1 = %q, want cfgB %q", data1Beta, cfgB)
	}
}

// TestConstantUploadRevisionTracking tests monotonic revision updates and capture receipts.
func TestConstantUploadRevisionTracking(t *testing.T) {
	mgr := NewConstantUploadManager()
	slot := ConstantSlotID("layer_norm_weights")
	_ = mgr.RegisterSlot(slot, 16)

	r1, _ := mgr.UploadConstant(slot, []byte("weights_v1_00000"))
	r2, _ := mgr.UploadConstant(slot, []byte("weights_v2_00000"))

	if r2.Revision <= r1.Revision {
		t.Fatalf("expected monotonic revision bump: r2.Rev %d <= r1.Rev %d", r2.Revision, r1.Revision)
	}

	rev, err := mgr.SlotRevision(slot)
	if err != nil || rev != r2.Revision {
		t.Fatalf("SlotRevision = (%d, %v), want (%d, nil)", rev, err, r2.Revision)
	}
}

// TestConstantUploadIncompleteCaptureValidation tests that omitting declared slots fails capture.
func TestConstantUploadIncompleteCaptureValidation(t *testing.T) {
	mgr := NewConstantUploadManager()
	slotA := ConstantSlotID("slot_a")
	slotB := ConstantSlotID("slot_b")
	_ = mgr.RegisterSlot(slotA, 16)
	_ = mgr.RegisterSlot(slotB, 16)

	// Require both slotA and slotB
	if err := mgr.BeginCapture("incomplete_graph", []ConstantSlotID{slotA, slotB}); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}

	// Upload only slotA, omitting slotB
	_, _ = mgr.UploadConstant(slotA, []byte("data_a_000000000"))

	// EndCapture must fail closed with ErrCaptureIncomplete
	_, err := mgr.EndCapture()
	if err == nil || !errors.Is(err, ErrCaptureIncomplete) {
		t.Fatalf("EndCapture with omitted slot = %v, want ErrCaptureIncomplete", err)
	}
}

// TestConstantUploadConditionalUnderCapture verifies ErrConditionalUploadUnderCapture (#10716).
func TestConstantUploadConditionalUnderCapture(t *testing.T) {
	mgr := NewConstantUploadManager()
	slot := ConstantSlotID("layer0_bias")
	_ = mgr.RegisterSlot(slot, 32)
	payload := []byte("bias-vector-00000000000000000000")

	// Eager conditional upload succeeds
	r1, err := mgr.UploadConstantConditional(slot, payload)
	if err != nil || r1.Skipped {
		t.Fatalf("UploadConstantConditional outside capture = (%+v, %v)", r1, err)
	}

	// Under capture, conditional upload must be rejected
	_ = mgr.BeginCapture("g_capture_conditional", []ConstantSlotID{slot})
	_, err = mgr.UploadConstantConditional(slot, payload)
	if !errors.Is(err, ErrConditionalUploadUnderCapture) {
		t.Fatalf("UploadConstantConditional under capture = %v, want ErrConditionalUploadUnderCapture", err)
	}

	// Abort capture resets capturing state cleanly
	mgr.AbortCapture()
	if mgr.IsCapturing() {
		t.Fatal("IsCapturing should be false after AbortCapture")
	}

	// Can begin a new capture session immediately
	if err := mgr.BeginCapture("g_new_session", []ConstantSlotID{slot}); err != nil {
		t.Fatalf("BeginCapture after AbortCapture failed: %v", err)
	}
}

// TestConstantUploadSlotStatsAndValidation verifies slot stats, negative sizes, and unregistered slot checks.
func TestConstantUploadSlotStatsAndValidation(t *testing.T) {
	mgr := NewConstantUploadManager()

	if err := mgr.RegisterSlot("neg_size", -1); err == nil {
		t.Fatal("RegisterSlot with negative size should fail")
	}

	slot := ConstantSlotID("stats_slot")
	_ = mgr.RegisterSlot(slot, 16)

	// Unregistered slot checks
	missing := ConstantSlotID("does_not_exist")
	if _, err := mgr.UploadConstant(missing, []byte("x")); !errors.Is(err, ErrSlotNotFound) {
		t.Fatalf("UploadConstant missing = %v, want ErrSlotNotFound", err)
	}
	if _, _, err := mgr.ReadDeviceConstant(missing); !errors.Is(err, ErrSlotNotFound) {
		t.Fatalf("ReadDeviceConstant missing = %v, want ErrSlotNotFound", err)
	}
	if _, err := mgr.SlotRevision(missing); !errors.Is(err, ErrSlotNotFound) {
		t.Fatalf("SlotRevision missing = %v, want ErrSlotNotFound", err)
	}
	if _, err := mgr.SlotStats(missing); !errors.Is(err, ErrSlotNotFound) {
		t.Fatalf("SlotStats missing = %v, want ErrSlotNotFound", err)
	}

	// Upload and check stats
	payload := []byte("payload_data_001")
	_, _ = mgr.UploadConstant(slot, payload) // upload 1
	_, _ = mgr.UploadConstant(slot, payload) // upload 2 (skipped in eager)

	stats, err := mgr.SlotStats(slot)
	if err != nil {
		t.Fatalf("SlotStats failed: %v", err)
	}
	if stats.TotalUploads != 1 || stats.SkippedUploads != 1 {
		t.Fatalf("stats = %+v, want total=1, skipped=1", stats)
	}
}

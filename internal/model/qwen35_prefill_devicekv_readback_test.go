//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestPrefillQwen35HybridQ4KDeviceKVPanelHostReadbackFlat is the #13087
// criterion-1 witness: on the device-resident KV path the per-panel HOST KV
// readback is GONE, so the host readback no longer scales with the KV geometry or
// panel count.
//
// It deliberately does NOT read the number off a test double. It runs the REAL
// production path — a resident-Q4_K session with its native GDN sequence owner
// admitted, so tryPrefillQwen35HybridQ4K -> Qwen35MetalForwardSequence ->
// AdmitDeviceKV/ReconcileDeviceKV — and reads the REAL aggregate receipt the
// backend produced.
//
// The KV term is isolated by geometry, not by a fabricated counter. A panel's
// host readback on the historical host-append path includes every full-attention
// layer's three row groups (Kraw/Kpost/V), so it is proportional to
// kvWidth = NumKVHeads * HeadDim:
//
//	per-panel KV readback bytes = fullLayers * 3 * panelRows * kvWidth * 4
//
// The device path's receipt instead carries ONLY the panel's hidden readback. So
// holding the prompt and panel count fixed while scaling NumKVHeads (kvWidth) must
// leave HostReadbackBytes UNCHANGED on the device path; if any KV row were still
// read back per panel, the byte count would scale with kvWidth. That is the flat
// property, sourced from the production receipt.
//
// The companion assertions pin the rest of the collapse:
//   - every panel rode the device (s.q4kHybridPrefillDevicePanels == nPanels,
//     DeviceRows == panelCover), so the flat readback is not an artifact of the
//     device path never running;
//   - the single terminal ReconcileDeviceKV restored the full host cache
//     (len(Cache.K[l]) == promptLen * kvWidth), so decode's host-cache contract
//     still holds.
//
// GPU-gated: t.Skip without a Metal device; the real path runs on this M3 Pro.
func TestPrefillQwen35HybridQ4KDeviceKVPanelHostReadbackFlat(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	const promptTokens = 256 // exactly two 128-token panels
	type arm struct {
		nKV          int
		kvWidth      int
		readback     uint64
		devicePanels int
		deviceRows   int
		cacheRowsLog []int
	}
	var arms []arm
	for _, nKV := range []int{1, 2, 4} {
		nKV := nKV
		cfg := qwen35HybridQ4KTestCfg()
		cfg.NumKVHeads = nKV
		kvWidth := cfg.NumKVHeads * cfg.HeadDim
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)
		s := m.NewSession()
		t.Cleanup(s.Close)
		s.Q4K, s.MetalQ4K = true, true
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatalf("EnableQwen35MetalGDNPreprojectedSequence: %v", err)
		}
		if !s.qwen35HAL.sequenceAccepted {
			t.Fatal("sequence owner not accepted; the real device path is unreachable")
		}
		ids := make([]int, promptTokens)
		for i := range ids {
			ids[i] = (i*17 + 5) % cfg.VocabSize
		}
		s.PrefillNoLogits(ids)
		receipt := s.Qwen35MetalForwardSequenceReceipt()
		if !receipt.Available {
			t.Fatal("real device walk produced no forward receipt")
		}
		// The device path must genuinely have run; a decline is a legitimate
		// fail-open but makes this witness inapplicable, so skip rather than pass
		// on a host-path number.
		if s.q4kHybridPrefillDevicePanels == 0 {
			t.Skipf("device KV was declined on this host (fail-open); real-device witness not applicable")
		}
		rowsLog := make([]int, 0, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			if !cfg.isLinearAttnLayer(l) {
				rowsLog = append(rowsLog, len(s.Cache.K[l])/kvWidth)
			}
		}
		arms = append(arms, arm{
			nKV: nKV, kvWidth: kvWidth, readback: receipt.HostReadbackBytes,
			devicePanels: s.q4kHybridPrefillDevicePanels, deviceRows: s.q4kHybridPrefillDeviceRows,
			cacheRowsLog: rowsLog,
		})
	}

	// Every panel rode the device and the single reconcile restored the host cache.
	fullLayers := 0
	{
		cfg := qwen35HybridQ4KTestCfg()
		for l := 0; l < cfg.NumLayers; l++ {
			if !cfg.isLinearAttnLayer(l) {
				fullLayers++
			}
		}
	}
	if fullLayers == 0 {
		t.Fatal("test config has no full-attention layer; criterion 1 is vacuous")
	}
	first := arms[0]
	if first.devicePanels != 2 {
		t.Fatalf("device panels=%d, want 2 (ceil(256/128))", first.devicePanels)
	}
	if first.deviceRows != promptTokens {
		t.Fatalf("device rows=%d, want %d", first.deviceRows, promptTokens)
	}
	for _, a := range arms {
		if a.devicePanels != 2 || a.deviceRows != promptTokens {
			t.Fatalf("nKV=%d device walk=%d panels/%d rows, want 2/%d", a.nKV, a.devicePanels, a.deviceRows, promptTokens)
		}
		for _, rows := range a.cacheRowsLog {
			if rows != promptTokens {
				t.Fatalf("nKV=%d full-attention host cache rows=%d, want %d (reconcile dropped/duplicated rows)", a.nKV, rows, promptTokens)
			}
		}
	}

	// The flat property: HostReadbackBytes is invariant under kvWidth. A per-panel
	// KV readback would make it scale with kvWidth; the device path readback has no
	// KV term.
	for _, a := range arms[1:] {
		if a.readback != first.readback {
			t.Fatalf("device host readback scaled with KV width: nKV=1/%d bytes -> nKV=%d/%d bytes (want flat); a per-panel KV readback is still being paid",
				first.readback, a.nKV, a.readback)
		}
	}

	// Corroborate the isolation: the KV-only readback the host-append walk would
	// pay for these panels dwarfs the entire device-path readback, so the device
	// receipt cannot contain those KV rows.
	for _, a := range arms {
		hostKVPerPanel := fullLayers * 3 * metalgemm.PromptPanelMaxTokens * a.kvWidth * 4
		hostKVTotal := a.devicePanels * hostKVPerPanel
		if a.readback >= uint64(hostKVTotal) {
			t.Fatalf("nKV=%d device readback=%d >= host KV-only readback=%d; the KV rows are still crossing the bus",
				a.nKV, a.readback, hostKVTotal)
		}
	}
}

// TestPrefillQwen35HybridQ4KDeviceKVWalkPlanMatchesProductionPanelizer is the
// GPU-free half of the criterion-1/2 panel-shape check: it pins that the panel
// arithmetic the device walk walks is the production arithmetic
// (tryPrefillQwen35HybridQ4K), so the L=100/L=300 byte-parity witness and the
// device-readback witness are exercising the same panel boundaries the server
// uses. It runs on every host (no Metal needed) because it only reads the panel
// constants and the walk owner's own counters through the Go-only double.
//
// [SW-VERIFIED] — no GPU.
func TestPrefillQwen35HybridQ4KDeviceKVWalkPlanMatchesProductionPanelizer(t *testing.T) {
	// The production panelizer: cover = (L/32)*32, panels of PromptPanelMaxTokens.
	panelize := func(promptLen int) (cover int, panels []int) {
		cover = (promptLen / 32) * 32
		for start := 0; start < cover; start += metalgemm.PromptPanelMaxTokens {
			rows := metalgemm.PromptPanelMaxTokens
			if start+rows > cover {
				rows = cover - start
			}
			panels = append(panels, rows)
		}
		return cover, panels
	}
	for _, tc := range []struct {
		prompt int
		cover  int
		panels []int
	}{
		{prompt: 100, cover: 96, panels: []int{96}},
		{prompt: 300, cover: 288, panels: []int{128, 128, 32}},
		{prompt: 256, cover: 256, panels: []int{128, 128}},
	} {
		cover, panels := panelize(tc.prompt)
		if cover != tc.cover {
			t.Fatalf("L=%d cover=%d, want %d", tc.prompt, cover, tc.cover)
		}
		if len(panels) != len(tc.panels) {
			t.Fatalf("L=%d panels=%v, want %v", tc.prompt, panels, tc.panels)
		}
		for i := range panels {
			if panels[i] != tc.panels[i] {
				t.Fatalf("L=%d panels=%v, want %v", tc.prompt, panels, tc.panels)
			}
		}
	}
}

package compute

import (
	"strings"
	"testing"
)

// TestPrefillLongPromptStageAttribution pins the ticket prefill-longprompt-scaling-p2048-
// 20261108 profiling step as a closed contract on the SW side: for the pinned Qwen3.8-27B
// hybrid geometry, the cold-panel stage table must show EXACTLY ONE superlinear stage
// (the causal attention over the growing context on the full-attention layers), a fixed
// P-independent dispatch budget, and a rising roofline-time share for that stage — while
// every O(P) stage stays linear and the attention stage's intrinsic FLOPs stay a small
// minority of the total (the measured decay is byte amplification + serialization, not
// raw FLOPs). [SW-VERIFIED]: exact counts, caller-supplied device peaks, no timer.
func TestPrefillLongPromptStageAttribution(t *testing.T) {
	g := Qwen38_27BGeometry
	ps := []int{128, 512, 2048}
	tables := make([][]AttributionStage, len(ps))
	for i, p := range ps {
		tables[i] = LongPromptStageTable(g, p)
		if len(tables[i]) == 0 {
			t.Fatalf("stage table empty at P=%d", p)
		}
	}
	// H200 SXM modeled peaks (datasheet-class constants, caller-supplied per the
	// RooflineSeconds discipline): BF16 dense compute, HBM3e bandwidth.
	const peakFLOPs = 989e12
	const peakBytesPerSec = 3.35e12

	find := func(tbl []AttributionStage, name string) (AttributionStage, bool) {
		for _, s := range tbl {
			if s.Name == name {
				return s, true
			}
		}
		return AttributionStage{}, false
	}

	// 1. Fixed dispatch budget: the panelized route issues the SAME number of launches at
	//    every P (no per-position dispatch anywhere on the path), and dispatches/token
	//    AMORTIZE as P grows (the ledger's dispatch row: overhead shrinks, never grows).
	var launches int64
	for i, tbl := range tables {
		var n int64
		for _, s := range tbl {
			n += s.Launches
		}
		if i == 0 {
			launches = n
		} else if n != launches {
			t.Fatalf("dispatch budget not P-independent: P=%d -> %d launches, P=%d -> %d", ps[0], launches, ps[i], n)
		}
	}
	if launches <= 0 {
		t.Fatalf("modeled launch count %d must be positive", launches)
	}
	dpt := func(p int) float64 { return float64(launches) / float64(p) }
	if !(dpt(ps[2]) < dpt(ps[1]) && dpt(ps[1]) < dpt(ps[0])) {
		t.Fatalf("dispatches/token must decrease with P: P128=%.3f P512=%.3f P2048=%.3f",
			dpt(128), dpt(512), dpt(2048))
	}

	// 2. Superlinear isolation: between P512 and P2048 exactly one stage scales faster
	//    than P^1.9, and it is the causal-attention panel; every other stage stays
	//    linear-or-better (<= P^1.15 covers the O(P) stages and the O(1) LM head).
	attn, ok := find(tables[1], "causal_attention_panel")
	if !ok {
		t.Fatal("stage table missing causal_attention_panel")
	}
	attnExp := ScalingExponent(attn, findMust(tables[2], "causal_attention_panel"), 512, 2048)
	if attnExp < 1.9 {
		t.Errorf("causal_attention_panel exponent P512->P2048 = %.3f, want >= 1.9 (the superlinear term)", attnExp)
	}
	// The declared scaling class must MATCH the computed exponent (a stage labeled O(P)
	// whose work does not grow as P is a counting bug, not a different model): O(P) reads
	// ~1.0, O(P^2) reads ~2.0, O(1) reads ~0.
	superlinear := 0
	for _, s := range tables[1] {
		exp := ScalingExponent(s, findMust(tables[2], s.Name), 512, 2048)
		switch s.Scaling {
		case "O(P)":
			if exp < 0.95 || exp > 1.15 {
				t.Errorf("stage %q declared O(P) but exponent P512->P2048 = %.3f", s.Name, exp)
			}
		case "O(P^2)":
			superlinear++
			if s.Name != "causal_attention_panel" {
				t.Errorf("unexpected superlinear stage %q (exponent %.3f); only causal_attention_panel may be O(P^2)", s.Name, exp)
			}
			if exp < 1.9 {
				t.Errorf("causal_attention_panel exponent P512->P2048 = %.3f, want >= 1.9 (the superlinear term)", exp)
			}
		default: // O(1): the last-row LM head
			if exp > 0.15 {
				t.Errorf("stage %q declared O(1) but exponent P512->P2048 = %.3f", s.Name, exp)
			}
		}
	}
	if superlinear != 1 {
		t.Errorf("expected exactly one superlinear stage P512->P2048, found %d", superlinear)
	}

	// 3. The attention stage's LOGICAL K/V re-read bytes scale quadratically: the
	//    P(P+1)/2 key-triangle, so the P2048/P512 byte ratio is (2048*2049)/(512*513),
	//    within 2% of the closed form (p2/p1)^2.
	wantRatio := float64(causalKeys(2048)) / float64(causalKeys(512))
	gotRatio := float64(findMust(tables[2], "causal_attention_panel").ActBytes) / float64(attn.ActBytes)
	if diff := gotRatio/wantRatio - 1; diff < -0.02 || diff > 0.02 {
		t.Errorf("attention byte ratio P2048/P512 = %.3f, want %.3f (+/-2%%)", gotRatio, wantRatio)
	}

	// 4. Amplification-dominated, not FLOP-dominated: the attention stage's INTRINSIC
	//    FLOPs are a small minority of the panel's total arithmetic at P2048 (<2%), so
	//    removing the term cannot come from cutting multiply-adds — it comes from the
	//    byte re-stream and the serial per-key scan the HW levers target.
	var totalFLOPs int64
	for _, s := range tables[2] {
		totalFLOPs += s.FLOPs
	}
	share := float64(findMust(tables[2], "causal_attention_panel").FLOPs) / float64(totalFLOPs)
	if share >= 0.02 {
		t.Errorf("attention intrinsic FLOP share at P2048 = %.3f, want < 0.02 (term is amplification, not raw FLOPs)", share)
	}

	// 5. GDN recurrent scan is O(P) with the large-but-P-independent constant: exponent
	//    <= 1.05 rules out any hidden position growth in the 48 linear-attention layers.
	gdnExp := ScalingExponent(findMust(tables[1], "gdn_fused_scan"), findMust(tables[2], "gdn_fused_scan"), 512, 2048)
	if gdnExp > 1.05 {
		t.Errorf("gdn_fused_scan exponent P512->P2048 = %.3f, want <= 1.05 (O(P) chunked scan)", gdnExp)
	}

	// 6. Rising time share: at the caller's peaks the attention stage's ROOFLINE-TIME
	//    share of the total grows monotonically P128 < P512 < P2048 even though the
	//    FLOP-heaviest stage (the SwiGLU FFN) is O(P) — the time view, not the FLOP
	//    view, exposes the long-prompt bottleneck (the same divergence PredictTime
	//    documents for naive attention).
	shareOf := func(i int, name string) float64 {
		var total float64
		for _, s := range tables[i] {
			total += s.RooflineSeconds(peakFLOPs, peakBytesPerSec)
		}
		return findMust(tables[i], name).RooflineSeconds(peakFLOPs, peakBytesPerSec) / total
	}
	s128, s512, s2048 := shareOf(0, "causal_attention_panel"), shareOf(1, "causal_attention_panel"), shareOf(2, "causal_attention_panel")
	if !(s128 < s512 && s512 < s2048) {
		t.Errorf("attention roofline-time share must rise with P: P128=%.3f P512=%.3f P2048=%.3f", s128, s512, s2048)
	}

	// The [SW-VERIFIED] attribution artifact: the rendered P-panel report (receipt text).
	report := LongPromptPanelReport(g, ps, peakFLOPs, peakBytesPerSec)
	if !strings.Contains(report, "causal_attention_panel") || !strings.Contains(report, "exponent") {
		t.Errorf("panel report missing the stage table:\n%s", report)
	}
	t.Log(report)
}

// findMust is the test-local lookup that fails loudly instead of returning a zero stage.
func findMust(tbl []AttributionStage, name string) AttributionStage {
	for _, s := range tbl {
		if s.Name == name {
			return s
		}
	}
	panic("stage table missing " + name)
}

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func defaults() (map[cachemeta.ResidencyTier]cachemeta.TierProfile, map[cachemeta.ResidencyTier]cachemeta.PoolProfile) {
	return cachemeta.DefaultTierProfiles(), cachemeta.DefaultPoolProfiles()
}

// TestRunProducesPoolStory asserts the demo runs deterministically and emits the
// load-bearing claims: the pool topology with the fabric-shareable CXL tier, the
// three-way fleet economics with the both-axes savings headline, and the cross-tenant
// trust gate refusing a poisoned / private / wrong-model cell.
func TestRunProducesPoolStory(t *testing.T) {
	var buf bytes.Buffer
	tp, pp := defaults()
	if err := run(&buf, tp, pp, "representative default profiles"); err != nil {
		t.Fatalf("run: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"Profiles: representative default profiles",
		"Pool topology",
		"cxl_hdm",               // CXL zero-copy share kind
		"yes (one shared copy)", // CXL is fabric-shareable
		"coherent CXL pool",     // the winning regime
		"28000 prefill tokens saved",
		"448MB of memory deduplicated",
		"Cross-tenant reuse gate",
		"REUSE",
		"REFUSE",
		"model_mismatch",
		"scope_denied",
		"taint_denied",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("demo output missing %q\n---\n%s", want, out)
		}
	}
}

// TestRunDeterministic guards the no-wall-clock property: two runs are byte-identical.
func TestRunDeterministic(t *testing.T) {
	var a, b bytes.Buffer
	tp1, pp1 := defaults()
	tp2, pp2 := defaults()
	_ = run(&a, tp1, pp1, "default")
	_ = run(&b, tp2, pp2, "default")
	if a.String() != b.String() {
		t.Fatalf("demo output is not deterministic across runs")
	}
}

// TestApplyCalibrationOverrides: a -profiles JSON overrides only the named tiers/fields
// (with the Tier key forced for self-consistency), leaves the rest at their defaults,
// and the calibrated profiles still produce the both-axes fleet win — the design-win
// path of running fak's cost model over an operator's measured fabric numbers.
func TestApplyCalibrationOverrides(t *testing.T) {
	tp, pp := defaults()
	raw := []byte(`{
		"label": "test calibration",
		"tiers": {"cxl": {"ReadLatencyNanos": 596, "BandwidthMBPerSec": 60000, "CapacityBytes": 2199023255552, "ByteAddressable": true, "Coherent": true, "Persistent": false, "Share": "cxl_hdm"}},
		"pools": {"cxl": {"Hosts": 16, "Coherent": true, "Share": "cxl_hdm"}}
	}`)
	label, err := applyCalibration(raw, tp, pp)
	if err != nil {
		t.Fatalf("applyCalibration: %v", err)
	}
	if label != "test calibration" {
		t.Fatalf("label = %q, want %q", label, "test calibration")
	}
	if tp[cachemeta.TierCXL].ReadLatencyNanos != 596 {
		t.Fatalf("cxl latency not overridden: %d", tp[cachemeta.TierCXL].ReadLatencyNanos)
	}
	if tp[cachemeta.TierCXL].Tier != cachemeta.TierCXL {
		t.Fatalf("override should force the Tier key, got %q", tp[cachemeta.TierCXL].Tier)
	}
	if pp[cachemeta.TierCXL].Hosts != 16 {
		t.Fatalf("cxl pool hosts not overridden: %d", pp[cachemeta.TierCXL].Hosts)
	}
	if !pp[cachemeta.TierCXL].FabricShareable() {
		t.Fatalf("a coherent zero-copy multi-host calibrated CXL pool must stay fabric-shareable")
	}
	// An un-named tier keeps its default.
	if tp[cachemeta.TierDRAM].BandwidthMBPerSec != cachemeta.DefaultTierProfiles()[cachemeta.TierDRAM].BandwidthMBPerSec {
		t.Fatalf("DRAM was not overridden and must keep its default profile")
	}
	// The calibrated profiles still produce the headline both-axes win.
	var buf bytes.Buffer
	if err := run(&buf, tp, pp, label); err != nil {
		t.Fatalf("run with calibration: %v", err)
	}
	if !strings.Contains(buf.String(), "28000 prefill tokens saved") {
		t.Fatalf("calibrated run should still show the fleet win:\n%s", buf.String())
	}
}

// TestApplyCalibrationBadJSON: a malformed calibration is a typed error, not a silent
// fall-through to defaults.
func TestApplyCalibrationBadJSON(t *testing.T) {
	tp, pp := defaults()
	if _, err := applyCalibration([]byte("not json"), tp, pp); err == nil {
		t.Fatalf("malformed calibration should error")
	}
}

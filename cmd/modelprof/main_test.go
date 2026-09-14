package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestTableRendersProfile pins the human-readable roofline attribution table:
// header, one row per op-class with the documented column formatting, and the
// trailing bottleneck line. It is a deterministic formatter test with no model
// load, so it runs on every host without the smollm2 weight cache.
func TestTableRendersProfile(t *testing.T) {
	p := &model.Profile{
		Mode:          "decode",
		PerTokenMS:    12.5,
		AchievedGFLOP: 3.2,
		AchievedGBps:  42.0,
		BWUtilPct:     70.0,
		MemBWGBps:     60.0,
		Bottleneck:    "attn",
		Stats: []model.OpStat{
			{
				Class:    "attn",
				TimePct:  62.5,
				MACs:     2_000_000,
				Bytes:    4_000_000,
				IntensFB: 0.5,
				GBps:     33.3,
				Verdict:  "memory-bound",
			},
			{
				Class:    "mlp",
				TimePct:  37.5,
				MACs:     1_000_000,
				Bytes:    1_000_000,
				IntensFB: 1.0,
				GBps:     8.0,
				Verdict:  "compute-bound",
			},
		},
	}

	got := table(p)

	wantFragments := []string{
		"== decode",
		"instrumented per-token 12.5 ms",
		"achieved 3.2 GFLOP/s",
		"42.0 GB/s = 70% of 60.0 GB/s mem ceiling",
		"op-class",
		"time%",
		"flop/byte",
		"verdict",
		"attn",
		"62.5%",
		"memory-bound",
		"mlp",
		"37.5%",
		"compute-bound",
		"bottleneck: attn",
	}
	for _, want := range wantFragments {
		if !strings.Contains(got, want) {
			t.Errorf("table() output missing %q\n---\n%s", want, got)
		}
	}
}

// TestCleanDecodeTableRendersMeasurement pins the uninstrumented decode summary
// line so the clean measurement stays visibly distinct from the instrumented
// roofline table.
func TestCleanDecodeTableRendersMeasurement(t *testing.T) {
	c := &model.CleanDecode{
		Mode:         "decode",
		PerTokenMS:   8.25,
		Steps:        24,
		PromptTokens: 16,
		Summary:      "clean session path",
	}

	got := cleanDecodeTable(c)

	wantFragments := []string{
		"== clean decode",
		"uninstrumented 8.2 ms/tok",
		"24 steps",
		"16-token prompt",
		"clean session path",
	}
	for _, want := range wantFragments {
		if !strings.Contains(got, want) {
			t.Errorf("cleanDecodeTable() output missing %q\n---\n%s", want, got)
		}
	}
}

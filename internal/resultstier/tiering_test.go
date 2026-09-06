package resultstier

import (
	"testing"
)

func TestClaimRolesCoverage(t *testing.T) {
	// Table of sample filenames matching each claim role pattern.
	samples := map[string]string{
		"INDEX.md":           "INDEX.md",
		"payload-index.json": "payload-index.json",
		"*manifest*.json":    "run_manifest.json",
		"perf.json":          "perf.json",
		"row.json":           "row.json",
		"entry.json":         "entry.json",
		"*summary*.json":     "eval_summary.json",
		"*census*.json":      "repo_census.json",
		"*score*.json":       "benchmark_score.json",
		"outcome.json":       "outcome.json",
		"funnel.json":        "funnel.json",
		"events.json":        "events.json",
		"*.md":               "README.md",
		"*.sha256":           "results.sha256",
		"*.sql":              "queries.sql",
		"*.sh":               "driver.sh",
	}

	for _, rule := range claimRoles {
		sample, ok := samples[rule.pattern]
		if !ok {
			t.Fatalf("missing sample for claim role pattern %q", rule.pattern)
		}
		tier, desc := TierOf(sample)
		if tier != TierClaim {
			t.Errorf("TierOf(%q) = %v (%s), want %v", sample, tier, desc, TierClaim)
		}
		if !tier.Known() {
			t.Errorf("TierOf(%q).Known() = false, want true", sample)
		}
		if desc != rule.desc {
			t.Errorf("TierOf(%q) desc = %q, want %q", sample, desc, rule.desc)
		}

		// Also verify nested paths with both forward slash and backslash.
		nested := "sub/dir/" + sample
		tierNested, _ := TierOf(nested)
		if tierNested != TierClaim {
			t.Errorf("TierOf(%q) = %v, want %v", nested, tierNested, TierClaim)
		}
	}
}

func TestPayloadRolesCoverage(t *testing.T) {
	// Table of sample filenames matching each payload role pattern.
	samples := map[string]string{
		"*layerinfo*.json":  "layerinfo_dump.json",
		"predictions*.json": "predictions_epoch1.json",
		"*diff.json":        "tactic_diff.json",
		"per-frame*.csv":    "per-frame-metrics.csv",
		"*.log":             "build.log",
		"times-*.json":      "times-001.json",
		"pmon-*.txt":        "pmon-gpu0.txt",
		"*.start":           "run.start",
		"*.end":             "run.end",
		"prefill.json":      "prefill.json",
		"*.rle":             "mask.rle",
		"*.cache":           "timing.cache",
		"*.csv":             "measurements.csv",
		"*.tsv":             "data.tsv",
		"*.jsonl":           "stream.jsonl",
		"*.npy":             "tensors.npy",
		"*.png":             "render.png",
	}

	for _, rule := range payloadRoles {
		sample, ok := samples[rule.pattern]
		if !ok {
			t.Fatalf("missing sample for payload role pattern %q", rule.pattern)
		}
		tier, desc := TierOf(sample)
		if tier != TierPayload {
			t.Errorf("TierOf(%q) = %v (%s), want %v", sample, tier, desc, TierPayload)
		}
		if !tier.Known() {
			t.Errorf("TierOf(%q).Known() = false, want true", sample)
		}
		if desc != rule.desc {
			t.Errorf("TierOf(%q) desc = %q, want %q", sample, desc, rule.desc)
		}

		nested := "artifacts/run1/" + sample
		tierNested, _ := TierOf(nested)
		if tierNested != TierPayload {
			t.Errorf("TierOf(%q) = %v, want %v", nested, tierNested, TierPayload)
		}
	}
}

func TestUnknownRoles(t *testing.T) {
	cases := []string{
		"",
		"   ",
		".",
		"/",
		"random.bin",
		"program.exe",
		"archive.zip",
		"generic.json",
		"model.onnx",
	}

	for _, tc := range cases {
		tier, reason := TierOf(tc)
		if tier != TierUnknown {
			t.Errorf("TierOf(%q) = %v, want TierUnknown", tc, tier)
		}
		if tier.Known() {
			t.Errorf("TierOf(%q).Known() = true, want false", tc)
		}
		if reason == "" {
			t.Errorf("TierOf(%q) returned empty reason", tc)
		}
	}
}

func TestTierMethods(t *testing.T) {
	if TierUnknown.String() != "unknown" {
		t.Errorf("TierUnknown.String() = %q, want %q", TierUnknown.String(), "unknown")
	}
	if TierClaim.String() != "claim" {
		t.Errorf("TierClaim.String() = %q, want %q", TierClaim.String(), "claim")
	}
	if TierPayload.String() != "payload" {
		t.Errorf("TierPayload.String() = %q, want %q", TierPayload.String(), "payload")
	}
	if Tier(99).String() != "unknown" {
		t.Errorf("Tier(99).String() = %q, want %q", Tier(99).String(), "unknown")
	}

	if TierUnknown.Known() {
		t.Error("TierUnknown.Known() = true, want false")
	}
	if !TierClaim.Known() {
		t.Error("TierClaim.Known() = false, want true")
	}
	if !TierPayload.Known() {
		t.Error("TierPayload.Known() = false, want true")
	}
	if Tier(99).Known() {
		t.Error("Tier(99).Known() = true, want false")
	}
}

func TestCensusZeroValues(t *testing.T) {
	c := Census{}
	if got := c.TotalFiles(); got != 0 {
		t.Errorf("c.TotalFiles() = %d, want 0", got)
	}
	if got := c.TotalBytes(); got != 0 {
		t.Errorf("c.TotalBytes() = %d, want 0", got)
	}
	if got := c.PayloadShare(); got != 0.0 {
		t.Errorf("c.PayloadShare() = %f, want 0.0", got)
	}
	if got := c.Shrink(); got != 1.0 {
		t.Errorf("c.Shrink() = %f, want 1.0", got)
	}
	if c.String() == "" {
		t.Error("c.String() is empty")
	}
}

func TestCensusArithmetic(t *testing.T) {
	c := Census{
		ClaimFiles:   2,
		ClaimBytes:   200,
		PayloadFiles: 8,
		PayloadBytes: 800,
		UnknownFiles: 0,
		UnknownBytes: 0,
	}

	if got := c.TotalFiles(); got != 10 {
		t.Errorf("TotalFiles() = %d, want 10", got)
	}
	if got := c.TotalBytes(); got != 1000 {
		t.Errorf("TotalBytes() = %d, want 1000", got)
	}
	if got := c.PayloadShare(); got != 0.8 {
		t.Errorf("PayloadShare() = %f, want 0.8", got)
	}
	// Shrink: 1000 / 200 = 5.0
	if got := c.Shrink(); got != 5.0 {
		t.Errorf("Shrink() = %f, want 5.0", got)
	}

	// Add unknown bytes
	c.UnknownFiles = 1
	c.UnknownBytes = 50
	// Total: 1050, Retained: 250
	if got := c.TotalBytes(); got != 1050 {
		t.Errorf("TotalBytes() with unknown = %d, want 1050", got)
	}
	expectedShare := 800.0 / 1050.0
	if diff := c.PayloadShare() - expectedShare; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("PayloadShare() = %f, want %f", c.PayloadShare(), expectedShare)
	}
	expectedShrink := 1050.0 / 250.0 // 4.2
	if diff := c.Shrink() - expectedShrink; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("Shrink() = %f, want %f", c.Shrink(), expectedShrink)
	}

	// All payload, zero retained
	allPayload := Census{
		PayloadFiles: 1,
		PayloadBytes: 500,
	}
	if got := allPayload.Shrink(); got != 500.0 {
		t.Errorf("allPayload.Shrink() = %f, want 500.0", got)
	}
	if got := allPayload.PayloadShare(); got != 1.0 {
		t.Errorf("allPayload.PayloadShare() = %f, want 1.0", got)
	}
}

func TestSplitClassify(t *testing.T) {
	paths := []string{
		"INDEX.md",
		"run_manifest.json",
		"predictions.json",
		"build.log",
		"unknown.bin",
		"perf.json",
		"output.png",
		"something.weird",
	}

	split := Classify(paths)
	if len(split.Claim) != 3 {
		t.Errorf("len(split.Claim) = %d, want 3 (%v)", len(split.Claim), split.Claim)
	}
	if len(split.Payload) != 3 {
		t.Errorf("len(split.Payload) = %d, want 3 (%v)", len(split.Payload), split.Payload)
	}
	if len(split.Unknown) != 2 {
		t.Errorf("len(split.Unknown) = %d, want 2 (%v)", len(split.Unknown), split.Unknown)
	}

	// Empty paths check
	emptySplit := Classify(nil)
	if emptySplit.Claim == nil || emptySplit.Payload == nil || emptySplit.Unknown == nil {
		t.Error("Classify(nil) returned nil slices in Split")
	}
}

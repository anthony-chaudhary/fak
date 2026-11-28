package readmenext

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupRecordRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	for _, sub := range []string{
		"tools",
		"visuals",
		"docs/benchmarks/published/rtx5080-next",
	} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0755); err != nil {
			t.Fatal(err)
		}
	}

	receiptPath := filepath.Join(dir, "docs/benchmarks/published/rtx5080-next/RESULTS.md")
	if err := os.WriteFile(receiptPath, []byte("# Receipt\n"), 0644); err != nil {
		t.Fatal(err)
	}

	sweep := HeroSweepFile{
		About: "test sweep",
		Meta: HeroSweepMeta{
			Title:     "fak sweep",
			AsOf:      "2026-11-26",
			OutSVG:    DefaultHeroSweepSVGPath,
			Authority: "fak/BENCHMARK-AUTHORITY.md",
			Receipts:  "docs/benchmarks/README.md",
		},
		Kicker:   "BENCHMARK SWEEP · INKLING-2-MINI · 2026-11-26",
		Title:    "fak by the benchmarks",
		Subtitle: "test subtitle",
		Legend:   []HeroSweepLegend{{Label: "fak", Color: "#2e8b57"}},
		Panels: []HeroSweepPanel{
			{
				Name:   "NVFP4 W4A4 prefill",
				Pillar: "INFERENCE",
				Foot:   "RTX 5080 · pp512",
				Commit: "d3317d16fc",
				YMax:   6700,
				Bars: []HeroSweepBar{
					{Who: "fak", Badge: "f", V: 5861.0, Label: "5,861 tok/s", Subject: true},
					{Who: "fork", Badge: "F", V: 3012.4, Label: "3,012"},
				},
			},
			{
				Name:          "Served TTFT (C4, 2k)",
				Pillar:        "SERVING",
				Foot:          "RTX 5080 · lower is better",
				Commit:        "7ba76a1caf",
				LowerIsBetter: true,
				Callout:       "1.84× faster",
				YMax:          5.4,
				Bars: []HeroSweepBar{
					{Who: "fak", Badge: "f", V: 2.02, Label: "2.02 s", Subject: true},
					{Who: "fork", Badge: "F", V: 3.72, Label: "3.72 s"},
				},
			},
			{
				Name:   "Served C1 decode (2k)",
				Pillar: "FENCE",
				Foot:   "fak trails here",
				Commit: "7ba76a1caf",
				Fence:  true,
				YMax:   134,
				Bars: []HeroSweepBar{
					{Who: "fak", Badge: "f", V: 93.2, Label: "93.2 tok/s"},
					{Who: "llama.cpp fork", Badge: "F", V: 116.0, Label: "116.0 tok/s", Subject: true},
				},
			},
		},
		Footer: "footer",
		Desc:   "NVFP4 W4A4 pp512 prefill 5,861 tok/s vs 3,012; TTFT 2.02 s vs 3.72 s; fence 93.2 tok/s vs 116.0 tok/s.",
	}
	sweepBytes, err := marshalJSONPretty(&sweep)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DefaultHeroSweepDataPath), sweepBytes, 0644); err != nil {
		t.Fatal(err)
	}

	readme := `# fak

Prefill reads a 512-token prompt at 5,861 tok/s against 3,012 for the fork.

<!-- readme-verified: 2026-11-26 vs VERSION 0.60.0 + BENCHMARK-AUTHORITY -->
`
	if err := os.WriteFile(filepath.Join(dir, DefaultReadmePath), []byte(readme), 0644); err != nil {
		t.Fatal(err)
	}

	hwJSON := HardwareLatestManifest{
		Schema: "fak-hardware-latest/1",
		AsOf:   "2026-11-26",
		Platforms: map[string]*HardwarePlatformEntry{
			"NVIDIA": {
				Observed: "2026-11-26",
				Detail:   "docs/benchmarks/published/rtx5080-next/RESULTS.md",
				Row:      "| NVIDIA | old row | old status |",
			},
		},
	}
	hwBytes, err := marshalJSONPretty(&hwJSON)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DefaultHardwareJSONPath), hwBytes, 0644); err != nil {
		t.Fatal(err)
	}

	hwReadme := "## Latest hardware results — 2026-11-26\n\n| Platform | Latest | Status |\n|---|---|---|\n| NVIDIA | old row | old status |\n"
	if err := os.WriteFile(filepath.Join(dir, DefaultHardwareReadmePath), []byte(hwReadme), 0644); err != nil {
		t.Fatal(err)
	}

	return dir
}

func TestApplyRecordUpdate_HigherIsBetterMaterialBeat(t *testing.T) {
	dir := setupRecordRepo(t)

	out, err := ApplyRecordUpdate(dir, HeroRecordInput{
		Panel:       "NVFP4 W4A4 prefill",
		Value:       6500,
		Commit:      "abcdef1234",
		ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md",
		Date:        "2026-11-28",
		ReadmeOld:   "5,861 tok/s against 3,012 for the fork",
		ReadmeNew:   "6,500 tok/s against 3,012 for the fork",
		Platform:    "NVIDIA",
		HardwareRow: "| NVIDIA | new 6,500 tok/s on 2026-11-28 | [receipt](published/rtx5080-next/RESULTS.md) |",
	}, false)
	if err != nil {
		t.Fatalf("ApplyRecordUpdate failed: %v", err)
	}
	if !out.Material || !out.Beat || out.Skipped {
		t.Fatalf("unexpected outcome: %+v", out)
	}

	raw, err := os.ReadFile(filepath.Join(dir, DefaultHeroSweepDataPath))
	if err != nil {
		t.Fatal(err)
	}
	var updated HeroSweepFile
	if err := json.Unmarshal(raw, &updated); err != nil {
		t.Fatal(err)
	}
	if updated.Meta.AsOf != "2026-11-28" {
		t.Fatalf("expected as_of 2026-11-28, got %s", updated.Meta.AsOf)
	}
	if !strings.HasSuffix(updated.Kicker, "2026-11-28") {
		t.Fatalf("expected kicker date updated, got %q", updated.Kicker)
	}
	p := updated.Panels[0]
	if p.Bars[0].V != 6500 || p.Bars[0].Label != "6,500 tok/s" {
		t.Fatalf("unexpected bar update: %+v", p.Bars[0])
	}
	if p.YMax <= 6700 {
		t.Fatalf("expected YMax to bump above 6700 when 6500 > 0.9*6700, got %v", p.YMax)
	}
	if !strings.Contains(updated.Desc, "6,500 tok/s") {
		t.Fatalf("expected Desc updated with new label, got %q", updated.Desc)
	}

	readmeBytes, _ := os.ReadFile(filepath.Join(dir, DefaultReadmePath))
	readme := string(readmeBytes)
	if !strings.Contains(readme, "6,500 tok/s against 3,012 for the fork") {
		t.Fatalf("README prose not updated: %s", readme)
	}
	if !strings.Contains(readme, "<!-- readme-verified: 2026-11-28") {
		t.Fatalf("README verified stamp not updated: %s", readme)
	}

	hwJSONBytes, _ := os.ReadFile(filepath.Join(dir, DefaultHardwareJSONPath))
	var manifest HardwareLatestManifest
	if err := json.Unmarshal(hwJSONBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.AsOf != "2026-11-28" || !strings.Contains(manifest.Platforms["NVIDIA"].Row, "6,500 tok/s") {
		t.Fatalf("hardware-latest.json not updated: %+v", manifest)
	}
}

func TestApplyRecordUpdate_LowerIsBetter(t *testing.T) {
	dir := setupRecordRepo(t)

	out, err := ApplyRecordUpdate(dir, HeroRecordInput{
		Panel:       "Served TTFT (C4, 2k)",
		Value:       1.50,
		Commit:      "1122334455",
		ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md",
		Date:        "2026-11-28",
	}, false)
	if err != nil {
		t.Fatalf("ApplyRecordUpdate failed: %v", err)
	}
	if !out.LowerIsBetter || !out.Material || !out.Beat {
		t.Fatalf("expected material lower-is-better beat, got %+v", out)
	}

	raw, _ := os.ReadFile(filepath.Join(dir, DefaultHeroSweepDataPath))
	var updated HeroSweepFile
	_ = json.Unmarshal(raw, &updated)
	if updated.Panels[1].Bars[0].Label != "1.50 s" {
		t.Fatalf("expected label '1.50 s', got %q", updated.Panels[1].Bars[0].Label)
	}
}

func TestApplyRecordUpdate_NonMaterialAndRegressionSkipped(t *testing.T) {
	dir := setupRecordRepo(t)

	outSmall, err := ApplyRecordUpdate(dir, HeroRecordInput{
		Panel:       "NVFP4 W4A4 prefill",
		Value:       5900, // +0.67% < 5%
		Commit:      "abcdef1234",
		ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md",
		Date:        "2026-11-28",
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !outSmall.Beat || outSmall.Material || !outSmall.Skipped {
		t.Fatalf("expected non-material beat to be skipped: %+v", outSmall)
	}

	outRegress, err := ApplyRecordUpdate(dir, HeroRecordInput{
		Panel:       "NVFP4 W4A4 prefill",
		Value:       5000,
		Commit:      "abcdef1234",
		ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md",
		Date:        "2026-11-28",
	}, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outRegress.Beat || outRegress.Material || !outRegress.Skipped {
		t.Fatalf("expected regression to be skipped: %+v", outRegress)
	}

	raw, _ := os.ReadFile(filepath.Join(dir, DefaultHeroSweepDataPath))
	var sweep HeroSweepFile
	_ = json.Unmarshal(raw, &sweep)
	if sweep.Meta.AsOf != "2026-11-26" {
		t.Fatalf("expected hero_sweep.data.json untouched on skip, got as_of=%s", sweep.Meta.AsOf)
	}
}

func TestApplyRecordUpdate_FenceFlip(t *testing.T) {
	dir := setupRecordRepo(t)

	out, err := ApplyRecordUpdate(dir, HeroRecordInput{
		Panel:       "Served C1 decode (2k)",
		Value:       122.0, // beats fork's 116.0
		Commit:      "deadbeef01",
		ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md",
		Date:        "2026-11-28",
		Foot:        "fak now leads the fork",
	}, false)
	if err != nil {
		t.Fatalf("ApplyRecordUpdate failed: %v", err)
	}
	if !out.FlippedFence || !out.Material {
		t.Fatalf("expected FlippedFence=true and Material=true, got %+v", out)
	}

	raw, _ := os.ReadFile(filepath.Join(dir, DefaultHeroSweepDataPath))
	var sweep HeroSweepFile
	_ = json.Unmarshal(raw, &sweep)
	p := sweep.Panels[2]
	if p.Fence || p.Pillar != "SERVING" {
		t.Fatalf("expected fence cleared and pillar SERVING, got fence=%v pillar=%s", p.Fence, p.Pillar)
	}
	if !p.Bars[0].Subject || p.Bars[1].Subject {
		t.Fatalf("expected subject flipped to fak, got %+v", p.Bars)
	}
}

func TestValidateHeroRecordInput_RejectsInvalidProvenance(t *testing.T) {
	dir := setupRecordRepo(t)

	cases := []struct {
		name string
		cand HeroRecordInput
	}{
		{
			name: "missing receipt",
			cand: HeroRecordInput{Panel: "NVFP4 W4A4 prefill", Value: 6500, Commit: "abc", ReceiptPath: "docs/missing.md"},
		},
		{
			name: "path traversal",
			cand: HeroRecordInput{Panel: "NVFP4 W4A4 prefill", Value: 6500, Commit: "abc", ReceiptPath: "../secret.md"},
		},
		{
			name: "simulated claim",
			cand: HeroRecordInput{Panel: "NVFP4 W4A4 prefill", Value: 6500, Commit: "abc", ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md", Foot: "[SIMULATED] fast"},
		},
		{
			name: "naive bold lead",
			cand: HeroRecordInput{Panel: "NVFP4 W4A4 prefill", Value: 6500, Commit: "abc", ReceiptPath: "docs/benchmarks/published/rtx5080-next/RESULTS.md", ReadmeOld: "a", ReadmeNew: "**6,500 tok/s vs naive**"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateHeroRecordInput(tc.cand, dir); err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

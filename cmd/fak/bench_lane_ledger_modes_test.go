package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// launchSitesFindings returns the findings of a sites document as objects, failing if the key is absent.
func launchSitesFindings(t *testing.T, doc map[string]any) []any {
	t.Helper()
	findings, ok := doc["findings"].([]any)
	if !ok {
		t.Fatalf("findings is %T, want array", doc["findings"])
	}
	return findings
}

// siteKernel returns the kernel name of a sites finding object, or "" when it is absent.
func siteKernel(finding map[string]any) string {
	site, _ := finding["site"].(map[string]any)
	kernel, _ := site["kernel"].(string)
	return kernel
}

// findBySiteKernel returns the first finding whose site runs the named kernel.
func findBySiteKernel(t *testing.T, findings []any, kernel string) map[string]any {
	t.Helper()
	for _, item := range findings {
		if f, ok := item.(map[string]any); ok && siteKernel(f) == kernel {
			return f
		}
	}
	t.Fatalf("no finding for kernel %q", kernel)
	return nil
}

func TestBenchLaneLedgerSitesJSONKeys(t *testing.T) {
	doc := runLaneLedgerJSON(t, "sites")
	for _, key := range []string{"schema", "unit", "warp_size", "warp_provenance", "rows", "counts", "findings", "sites", "caveats"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("top-level key %q missing", key)
		}
	}
	if doc["schema"] != "fak.lane-launch-sites/1" {
		t.Errorf("schema = %v, want fak.lane-launch-sites/1", doc["schema"])
	}
	sites, _ := doc["sites"].([]any)
	if got, want := len(sites), len(compute.LaunchSites()); got != want {
		t.Errorf("sites has %d entries, want %d (one per inventory row)", got, want)
	}
	if got := laneLedgerNumber(t, doc, "rows"); got != int64(len(sites)) {
		t.Errorf("rows = %d, want %d", got, len(sites))
	}
	if prov, _ := doc["warp_provenance"].(string); !strings.HasPrefix(prov, "repo NVArch.WarpSize") {
		t.Errorf("default warp_provenance = %q, want the repo NVArch.WarpSize", prov)
	}
}

// TestBenchLaneLedgerSitesHandValues: graph_capture_once.cuh launches k_graph_once_position with one
// thread, and 1 % 32 = 1 lane in its trailing warp, so it is a partial_warp finding with partial_lanes 1.
// The k_q8_quant_act rows in graph_quant_workspace.cuh launch 32 threads, which fill a warp exactly, so they
// are not flagged as partial_warp.
// The symbolic fattn row has threads_per_block 0 and must be unjudged.
func TestBenchLaneLedgerSitesHandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, "sites", "--warp", "32")
	findings := launchSitesFindings(t, doc)

	one := findBySiteKernel(t, findings, "k_graph_once_position")
	if one["kind"] != "partial_warp" || laneLedgerNumber(t, one, "partial_lanes") != 1 {
		t.Errorf("k_graph_once_position finding = %v, want partial_warp with partial_lanes 1", one)
	}
	for _, item := range findings {
		if f, _ := item.(map[string]any); siteKernel(f) == "k_q8_quant_act" && f["kind"] == "partial_warp" {
			t.Errorf("32-thread launch k_q8_quant_act flagged as partial_warp: %v", f)
		}
	}

	unjudged := findBySiteKernel(t, findings, "fak_fattn_mma::k_qwen38_fattn_mma_hd256")
	if unjudged["kind"] != "unjudged" || laneLedgerNumber(t, laneLedgerObject(t, unjudged, "site"), "threads_per_block") != 0 {
		t.Errorf("unjudged finding = %v, want a symbolic site with threads_per_block 0", unjudged)
	}

	counts := laneLedgerObject(t, doc, "counts")
	var partial, unjudgedCount int64
	for _, item := range findings {
		f, _ := item.(map[string]any)
		if f["kind"] == "partial_warp" {
			partial++
		} else {
			unjudgedCount++
		}
	}
	if got := laneLedgerNumber(t, counts, "partial_warp"); got != partial {
		t.Errorf("counts.partial_warp = %d, want %d findings", got, partial)
	}
	if got := laneLedgerNumber(t, counts, "unjudged"); got != unjudgedCount {
		t.Errorf("counts.unjudged = %d, want %d findings", got, unjudgedCount)
	}
}

// With --warp 1 every literal thread count is a multiple of the warp, so no partial_warp finding remains
// and only the symbolic rows stay unjudged. With --warp 64 the 32-thread k_q8_quant_act launch leaves
// 32 % 64 = 32 lanes in its only warp.
func TestBenchLaneLedgerSitesWarpChangesFindings(t *testing.T) {
	one := runLaneLedgerJSON(t, "sites", "--warp", "1")
	if got := laneLedgerNumber(t, laneLedgerObject(t, one, "counts"), "partial_warp"); got != 0 {
		t.Errorf("--warp 1 partial_warp = %d, want 0", got)
	}
	if one["warp_provenance"] != "caller input: --warp" {
		t.Errorf("--warp 1 warp_provenance = %v, want caller input", one["warp_provenance"])
	}

	sixtyFour := runLaneLedgerJSON(t, "sites", "--warp", "64")
	found := false
	for _, item := range launchSitesFindings(t, sixtyFour) {
		f, _ := item.(map[string]any)
		if siteKernel(f) == "k_q8_quant_act" && f["kind"] == "partial_warp" {
			found = true
			if laneLedgerNumber(t, f, "partial_lanes") != 32 {
				t.Errorf("k_q8_quant_act at --warp 64 = %v, want partial_lanes 32", f)
			}
		}
	}
	if !found {
		t.Errorf("k_q8_quant_act (32 threads) not flagged partial_warp at --warp 64")
	}
}

func TestBenchLaneLedgerSitesRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "zero warp refused", args: []string{"sites", "--warp", "0"}, wantErr: "--warp must be a positive warp width (got 0)"},
		{name: "negative warp refused", args: []string{"sites", "--warp", "-4"}, wantErr: "--warp must be a positive warp width (got -4)"},
		{name: "positional argument refused", args: []string{"sites", "extra"}, wantErr: "unexpected argument"},
		{name: "unknown flag refused", args: []string{"sites", "--bogus"}, wantErr: "bogus"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runBenchLaneLedger(&stdout, &stderr, tc.args); code == 0 {
				t.Fatalf("exit 0, want non-zero; stdout: %s", stdout.String())
			}
			if !strings.Contains(stderr.String(), tc.wantErr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), tc.wantErr)
			}
			if stdout.Len() != 0 {
				t.Errorf("refusal wrote to stdout: %q", stdout.String())
			}
		})
	}
}

func TestBenchLaneLedgerSitesHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLaneLedger(&stdout, &stderr, []string{"sites", "--warp", "32"}); code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak bench lane-ledger sites", "warp_provenance:", "KIND", "inventory (", "caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("sites table missing %q", want)
		}
	}
}

// fillFindingNames lists the finding names of a fill document in order.
func fillFindingNames(t *testing.T, doc map[string]any) []string {
	t.Helper()
	findings, _ := doc["findings"].([]any)
	var names []string
	for _, item := range findings {
		f, _ := item.(map[string]any)
		name, _ := f["name"].(string)
		names = append(names, name)
	}
	return names
}

// fillArgs is the masked-tail launch of the hand-worked example: 1000 items in groups of 32 on 64-lane units.
func fillArgs(extra ...string) []string {
	args := []string{"fill", "--family", "wavefront", "--width", "64", "--width-source", "example caller input",
		"--group", "32", "--items", "1000", "--kernel", "k_example"}
	return append(args, extra...)
}

func TestBenchLaneLedgerFillJSONKeys(t *testing.T) {
	doc := runLaneLedgerJSON(t, fillArgs()...)
	for _, key := range []string{"schema", "unit", "provenance", "family", "width", "kernel", "tail_mode", "group_size",
		"items", "groups", "units_per_group", "capacity", "active", "partial_unit", "tail_idle", "scalar_tail",
		"conserved", "findings", "caveats"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("top-level key %q missing", key)
		}
	}
	if doc["schema"] != "fak.lane-fill/1" || doc["unit"] != "lane_slot" {
		t.Errorf("schema/unit = %v/%v, want fak.lane-fill/1 and lane_slot", doc["schema"], doc["unit"])
	}
}

// TestBenchLaneLedgerFillMaskedHandValues: groups = ceil(1000/32) = 32 and units_per_group = ceil(32/64) = 1,
// so capacity = 32*64 = 2048. Active is 1000 items. Each group leaves 64-32 = 32 lanes past its group size, so
// partial_unit = 32*32 = 1024. The last group has 32*32 - 1000 = 24 lanes with no item, so tail_idle = 24.
// 1000 + 1024 + 24 = 2048.
func TestBenchLaneLedgerFillMaskedHandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, fillArgs()...)
	wantInts := map[string]int64{
		"groups":          32,
		"units_per_group": 1,
		"capacity":        2048,
		"active":          1000,
		"partial_unit":    1024,
		"tail_idle":       24,
		"scalar_tail":     0,
		"width":           64,
		"items":           1000,
		"group_size":      32,
	}
	for key, want := range wantInts {
		if got := laneLedgerNumber(t, doc, key); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if doc["tail_mode"] != "masked" {
		t.Errorf("tail_mode = %v, want masked (the default)", doc["tail_mode"])
	}
	if doc["conserved"] != true {
		t.Errorf("conserved = %v, want true", doc["conserved"])
	}
	if doc["provenance"] != "caller input: example caller input" {
		t.Errorf("provenance = %v, want the --width-source echoed", doc["provenance"])
	}
	names := fillFindingNames(t, doc)
	if len(names) != 2 || names[0] != "partial_unit" || names[1] != "tail_idle" {
		t.Errorf("findings = %v, want [partial_unit tail_idle]", names)
	}
}

// TestBenchLaneLedgerFillScalarHandValues: 100 items in groups of 64 on 32-lane units, scalar tail. The
// one full group uses 2 units of 32 lanes, which fill its 64 slots exactly, so it has no partial lanes.
// The 36-item remainder runs one item per unit, so capacity = 64 + 36*32 = 1216 and
// scalar_tail = 36*(32-1) = 1116. Active is the 100 items, and 100 + 1116 = 1216.
func TestBenchLaneLedgerFillScalarHandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, "fill", "--family", "wavefront", "--width", "32", "--width-source", "example caller input",
		"--group", "64", "--items", "100", "--tail", "scalar")
	wantInts := map[string]int64{
		"groups":          2,
		"units_per_group": 2,
		"capacity":        1216,
		"active":          100,
		"partial_unit":    0,
		"tail_idle":       0,
		"scalar_tail":     1116,
	}
	for key, want := range wantInts {
		if got := laneLedgerNumber(t, doc, key); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if doc["tail_mode"] != "scalar" || doc["conserved"] != true {
		t.Errorf("tail_mode/conserved = %v/%v, want scalar/true", doc["tail_mode"], doc["conserved"])
	}
	if names := fillFindingNames(t, doc); len(names) != 1 || names[0] != "scalar_tail" {
		t.Errorf("findings = %v, want [scalar_tail]", names)
	}
}

func TestBenchLaneLedgerFillRefusals(t *testing.T) {
	base := []string{"fill", "--family", "wavefront", "--width", "64", "--width-source", "example caller input",
		"--group", "32", "--items", "1000"}
	without := func(name string) []string {
		var out []string
		for i := 0; i < len(base); i++ {
			if base[i] == name {
				i++
				continue
			}
			out = append(out, base[i])
		}
		return out
	}
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "missing width-source refused", args: without("--width-source"), wantErr: "--width-source is required"},
		{
			name:    "blank width-source refused",
			args:    append(without("--width-source"), "--width-source", "   "),
			wantErr: "--width-source is required",
		},
		{name: "missing family refused", args: without("--family"), wantErr: "--family is required"},
		{name: "missing width refused", args: without("--width"), wantErr: "--width is required and must be positive"},
		{name: "missing group refused", args: without("--group"), wantErr: "--group is required and must be positive"},
		{name: "missing items refused", args: without("--items"), wantErr: "--items is required and must be positive"},
		{
			name:    "unknown tail refused",
			args:    append(base, "--tail", "bogus"),
			wantErr: `--tail must be "masked" or "scalar" (got "bogus")`,
		},
		{
			name:    "scalar tail with group not a multiple of width refused",
			args:    []string{"fill", "--family", "wavefront", "--width", "32", "--width-source", "example caller input", "--group", "48", "--items", "100", "--tail", "scalar"},
			wantErr: "TailScalar needs GroupSize to be a multiple of Width",
		},
		{name: "positional argument refused", args: append(base, "extra"), wantErr: "unexpected argument"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runBenchLaneLedger(&stdout, &stderr, tc.args); code == 0 {
				t.Fatalf("exit 0, want non-zero; stdout: %s", stdout.String())
			}
			if !strings.Contains(stderr.String(), tc.wantErr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), tc.wantErr)
			}
			if stdout.Len() != 0 {
				t.Errorf("refusal wrote to stdout: %q", stdout.String())
			}
		})
	}
}

// The groups count of a scalar tail with a remainder does not reconcile to capacity, so the report says so; a
// launch with no remainder has nothing to explain.
func TestBenchLaneLedgerFillScalarGroupsCaveat(t *testing.T) {
	hasCaveat := func(doc map[string]any) bool {
		caveats, _ := doc["caveats"].([]any)
		for _, c := range caveats {
			if c == laneFillScalarGroupsCaveat {
				return true
			}
		}
		return false
	}
	base := []string{"fill", "--family", "wavefront", "--width", "32", "--width-source", "example caller input",
		"--group", "64", "--tail", "scalar"}
	if got := hasCaveat(runLaneLedgerJSON(t, append(base, "--items", "100")...)); !got {
		t.Errorf("scalar tail with a remainder has no groups caveat")
	}
	if got := hasCaveat(runLaneLedgerJSON(t, append(base, "--items", "128")...)); got {
		t.Errorf("scalar tail with no remainder carries the groups caveat")
	}
}

func TestBenchLaneLedgerFillHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLaneLedger(&stdout, &stderr, fillArgs()); code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak bench lane-ledger fill", "provenance: caller input: example caller input",
		"partial_unit", "tail_idle", "caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("fill table missing %q:\n%s", want, out)
		}
	}
}

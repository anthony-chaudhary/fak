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

	unjudged := findBySiteKernel(t, findings, "k_qwen35_gdn_qk_norm")
	if unjudged["kind"] != "unjudged" || laneLedgerNumber(t, laneLedgerObject(t, unjudged, "site"), "threads_per_block") != 0 {
		t.Errorf("unjudged finding = %v, want a runtime site with threads_per_block 0", unjudged)
	}
	for _, item := range findings {
		if siteKernel(item.(map[string]any)) == "fak_fattn_mma::k_qwen38_fattn_mma_hd256" {
			t.Errorf("k_qwen38_fattn_mma_hd256 resolves to 128 threads (a warp multiple), so it is no finding: %v", item)
		}
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

// ---- sites resolution ----

func TestBenchLaneLedgerSitesResolutionCounts(t *testing.T) {
	doc := runLaneLedgerJSON(t, "sites")
	counts := laneLedgerObject(t, doc, "resolution_counts")
	want := map[string]int{}
	for _, s := range compute.LaunchSites() {
		want[s.Resolution]++
	}
	total := 0
	for class, n := range want {
		if got := laneLedgerNumber(t, counts, class); got != int64(n) {
			t.Errorf("resolution_counts[%q] = %d, want %d", class, got, n)
		}
		total += n
	}
	if total != len(compute.LaunchSites()) {
		t.Errorf("classes sum to %d, want %d rows", total, len(compute.LaunchSites()))
	}
	for _, item := range doc["sites"].([]any) {
		site := item.(map[string]any)
		if _, ok := site["threads_resolution"].(string); !ok {
			t.Fatalf("site %v has no threads_resolution", site["kernel"])
		}
	}
}

func TestBenchLaneLedgerSitesTableResolutionColumn(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLaneLedger(&stdout, &stderr, []string{"sites"}); code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"RES", "resolution: literal=", "resolved-chain=", "runtime=", "template="} {
		if !strings.Contains(out, want) {
			t.Errorf("sites table missing %q", want)
		}
	}
}

// ---- sweep ----

const benchLaneExcerpt = "testdata/bench_lane_ledger/ptxas_sm120_excerpt.txt"

// benchLaneSweepDocOf runs the sweep verb on the excerpt with the given extra flags and returns its document.
func benchLaneSweepDocOf(t *testing.T, extra ...string) map[string]any {
	t.Helper()
	args := append([]string{"sweep", "--ptxas-log", benchLaneExcerpt}, extra...)
	return runLaneLedgerJSON(t, args...)
}

func TestBenchLaneSweepJSONKeys(t *testing.T) {
	doc := benchLaneSweepDocOf(t, "--arch", "sm_120", "--sms", "84")
	for _, key := range []string{
		"schema", "unit", "arch", "device_sms", "grid_blocks", "grid_source", "provenance", "parse",
		"archs_seen", "join", "limiter_histogram", "rows_total", "rows_shown", "rows", "caveats",
	} {
		if _, ok := doc[key]; !ok {
			t.Errorf("top-level key %q missing", key)
		}
	}
	if doc["schema"] != "fak.lane-ptxas-sweep/1" {
		t.Errorf("schema = %v, want fak.lane-ptxas-sweep/1", doc["schema"])
	}
	if _, ok := doc["device_props"]; ok {
		t.Error("device_props present without --device-props")
	}
	join := laneLedgerObject(t, doc, "join")
	for _, key := range []string{"site_count", "joined_site_count", "join_rate", "unjoined_site_count", "unjoined_entry_count"} {
		if _, ok := join[key]; !ok {
			t.Errorf("join.%s missing", key)
		}
	}
}

// TestBenchLaneSweepHandValues pins three rows of the excerpt against values worked out by hand on sm_120.
// k_ink2d_fuse: 256 threads at 42 registers is 1536 registers per warp, 12288 per block, 65536/12288 = 5 blocks
// per SM, 40 warps, register-limited. k_copyrow: 10 registers round to 512 per warp, so the 1536-thread limit
// gives 6 blocks, 48 warps, warp-limited. k_rope_panel: 128 threads gives 12 blocks, warp-limited.
// The capacity is 84 SMs times 1536 threads, 129024 lane-slot-waves.
func TestBenchLaneSweepHandValues(t *testing.T) {
	doc := benchLaneSweepDocOf(t, "--device-props", benchLaneProps, "--top", "50")
	rows := doc["rows"].([]any)
	cases := []struct {
		kernel  string
		blocks  int64
		warps   int64
		limiter string
		threads int64
		regs    int64
	}{
		{"k_ink2d_fuse", 5, 40, "register", 256, 42},
		{"k_copyrow", 6, 48, "warp", 256, 10},
		{"k_rope_panel", 12, 48, "warp", 128, 32},
	}
	for _, c := range cases {
		row := findByString(t, rows, "kernel", c.kernel)
		if got := laneLedgerNumber(t, row, "blocks_per_sm"); got != c.blocks {
			t.Errorf("%s blocks_per_sm = %d, want %d", c.kernel, got, c.blocks)
		}
		if got := laneLedgerNumber(t, row, "warps_per_sm"); got != c.warps {
			t.Errorf("%s warps_per_sm = %d, want %d", c.kernel, got, c.warps)
		}
		if row["limiter"] != c.limiter {
			t.Errorf("%s limiter = %v, want %s", c.kernel, row["limiter"], c.limiter)
		}
		if got := laneLedgerNumber(t, row, "threads_per_block"); got != c.threads {
			t.Errorf("%s threads_per_block = %d, want %d", c.kernel, got, c.threads)
		}
		if got := laneLedgerNumber(t, row, "regs_per_thread"); got != c.regs {
			t.Errorf("%s regs_per_thread = %d, want %d", c.kernel, got, c.regs)
		}
		if got := laneLedgerNumber(t, row, "capacity_lane_slot_waves"); got != 129024 {
			t.Errorf("%s capacity = %d, want 129024", c.kernel, got)
		}
	}
	if got := laneLedgerNumber(t, doc, "grid_blocks"); got != 84 {
		t.Errorf("grid_blocks = %d, want the SM count 84 by default", got)
	}
	if !strings.HasPrefix(doc["grid_source"].(string), "default:") {
		t.Errorf("grid_source = %v, want the default stated", doc["grid_source"])
	}
	if _, ok := doc["device_props"]; !ok {
		t.Error("device_props missing with --device-props")
	}
}

func TestBenchLaneSweepCallerGridIsStated(t *testing.T) {
	doc := benchLaneSweepDocOf(t, "--arch", "sm_120", "--sms", "84", "--grid", "168")
	if got := laneLedgerNumber(t, doc, "grid_blocks"); got != 168 {
		t.Errorf("grid_blocks = %d, want 168", got)
	}
	if doc["grid_source"] != "caller input: --grid" {
		t.Errorf("grid_source = %v", doc["grid_source"])
	}
}

func TestBenchLaneSweepTopBoundsRows(t *testing.T) {
	doc := benchLaneSweepDocOf(t, "--arch", "sm_120", "--sms", "84", "--top", "2")
	if got := laneLedgerNumber(t, doc, "rows_shown"); got != 2 {
		t.Errorf("rows_shown = %d, want 2", got)
	}
	if got := len(doc["rows"].([]any)); got != 2 {
		t.Errorf("rows has %d entries, want 2", got)
	}
	if rt := laneLedgerNumber(t, doc, "rows_total"); rt < 2 {
		t.Errorf("rows_total = %d, want the full joined pair count", rt)
	}
}

func TestBenchLaneSweepRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "no hardware refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt}, wantErr: "--arch is missing"},
		{name: "missing ptxas log refused", args: []string{"sweep", "--arch", "sm_120", "--sms", "84"}, wantErr: "--ptxas-log is required"},
		{name: "zero top refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt, "--arch", "sm_120", "--sms", "84", "--top", "0"}, wantErr: "--top must be positive"},
		{name: "zero grid refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt, "--arch", "sm_120", "--sms", "84", "--grid", "0"}, wantErr: "--grid must be positive"},
		{name: "log for another arch refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt, "--arch", "sm_89", "--sms", "84"}, wantErr: "log has sm_120, NVArch is sm_89"},
		{name: "arch disagreeing with props refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt, "--arch", "sm_89", "--device-props", benchLaneProps}, wantErr: "--arch disagrees"},
		{name: "refuted props refused", args: []string{"sweep", "--ptxas-log", benchLaneExcerpt, "--device-props", "testdata/bench_lane_ledger/rtx5080_props_refuted.json"}, wantErr: "refutes the NVArch row"},
		{name: "missing log file refused", args: []string{"sweep", "--ptxas-log", "testdata/bench_lane_ledger/none.log", "--arch", "sm_120", "--sms", "84"}, wantErr: "--ptxas-log"},
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

func TestBenchLaneSweepHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr, []string{
		"sweep", "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps, "--top", "3",
	})
	if code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak bench lane-ledger sweep", "device props check: arch=sm_120", "join: sites=", "limiter histogram", "ranked rows", "caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("sweep table missing %q", want)
		}
	}
}

// ---- observed ----

const benchLaneProfile = "testdata/bench_lane_ledger/observed_profile.json"

// benchLaneObservedDocOf runs the observed verb on the excerpt and fixture profile with the device props.
func benchLaneObservedDocOf(t *testing.T, extra ...string) map[string]any {
	t.Helper()
	args := append([]string{"observed", "--profile", benchLaneProfile, "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps}, extra...)
	return runLaneLedgerJSON(t, args...)
}

func TestBenchLaneObservedJSONKeys(t *testing.T) {
	doc := benchLaneObservedDocOf(t)
	for _, key := range []string{"schema", "unit", "arch", "device_sms", "device_props", "inputs", "totals", "rows_total", "top_modelled", "top_unmodelled", "caveats"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("top-level key %q missing", key)
		}
	}
	if doc["schema"] != "fak.lane-observed/1" {
		t.Errorf("schema = %v, want fak.lane-observed/1", doc["schema"])
	}
	totals := laneLedgerObject(t, doc, "totals")
	for _, key := range []string{"total_ns", "modelled_ns", "unmodelled_ns", "modelled_share", "unmodelled_ns_by_reason", "time_weighted_idle_ns", "cell_time_ns"} {
		if _, ok := totals[key]; !ok {
			t.Errorf("totals.%s missing", key)
		}
	}
	caveats, _ := doc["caveats"].([]any)
	joined := ""
	for _, c := range caveats {
		joined += c.(string) + "\n"
	}
	for _, want := range []string{"prioritisation heuristic", "measured input", "model-only"} {
		if !strings.Contains(joined, want) {
			t.Errorf("caveats do not state %q", want)
		}
	}
}

// TestBenchLaneObservedHandValues: k_copyrow is 84 blocks on sm_120 at 256 threads, 6 resident blocks per SM,
// so its active share is 256*84 over 129024, which is 1/6. Its 1000 ns splits as 167 active and 833 idle: the
// largest-remainder rule gives the .67 remainder of the active cell to it. k_ink2d_fuse is 500 ns with the same
// active share; its three cells tie at .33, so the lower-numbered active cell takes the spare nanosecond:
// 84 active and 416 idle. k_not_in_log has no registers in the ptxas log, so it stays unmodelled.
func TestBenchLaneObservedHandValues(t *testing.T) {
	doc := benchLaneObservedDocOf(t, "--top", "5")
	modelled := doc["top_modelled"].([]any)
	copyrow := findByString(t, modelled, "kernel", "k_copyrow")
	if got := laneLedgerNumber(t, copyrow, "total_ns"); got != 1000 {
		t.Errorf("k_copyrow total_ns = %d, want 1000", got)
	}
	model := laneLedgerObject(t, copyrow, "model")
	if got := laneLedgerNumber(t, model, "time_weighted_idle_ns"); got != 833 {
		t.Errorf("k_copyrow time_weighted_idle_ns = %d, want 833", got)
	}
	if got := model["active_share"].(float64); got < 0.16666 || got > 0.16667 {
		t.Errorf("k_copyrow active_share = %v, want 1/6", got)
	}
	if got := laneLedgerNumber(t, model, "blocks_per_sm"); got != 6 {
		t.Errorf("k_copyrow blocks_per_sm = %d, want 6", got)
	}
	ink := findByString(t, modelled, "kernel", "k_ink2d_fuse")
	if got := laneLedgerNumber(t, laneLedgerObject(t, ink, "model"), "time_weighted_idle_ns"); got != 416 {
		t.Errorf("k_ink2d_fuse time_weighted_idle_ns = %d, want 416", got)
	}

	unmodelled := doc["top_unmodelled"].([]any)
	missing := findByString(t, unmodelled, "kernel", "k_not_in_log")
	if missing["unmodelled_reason"] != "regs_unmeasured" {
		t.Errorf("k_not_in_log reason = %v, want regs_unmeasured", missing["unmodelled_reason"])
	}

	totals := laneLedgerObject(t, doc, "totals")
	if got := laneLedgerNumber(t, totals, "total_ns"); got != 1750 {
		t.Errorf("total_ns = %d, want 1750", got)
	}
	if got := laneLedgerNumber(t, totals, "unmodelled_ns"); got != 250 {
		t.Errorf("unmodelled_ns = %d, want 250", got)
	}
	byReason := laneLedgerObject(t, totals, "unmodelled_ns_by_reason")
	if got := laneLedgerNumber(t, byReason, "regs_unmeasured"); got != 250 {
		t.Errorf("regs_unmeasured ns = %d, want 250", got)
	}
	if got := laneLedgerNumber(t, totals, "time_weighted_idle_ns"); got != 1249 {
		t.Errorf("time_weighted_idle_ns = %d, want 833+416=1249", got)
	}
}

func TestBenchLaneObservedRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "missing profile refused", args: []string{"observed", "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps}, wantErr: "--profile is required"},
		{name: "missing ptxas log refused", args: []string{"observed", "--profile", benchLaneProfile, "--device-props", benchLaneProps}, wantErr: "--ptxas-log is required"},
		{name: "no hardware refused", args: []string{"observed", "--profile", benchLaneProfile, "--ptxas-log", benchLaneExcerpt}, wantErr: "--arch is missing"},
		{name: "empty profile refused", args: []string{"observed", "--profile", "testdata/bench_lane_ledger/observed_empty.json", "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps}, wantErr: "no launch rows"},
		{name: "zero top refused", args: []string{"observed", "--profile", benchLaneProfile, "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps, "--top", "0"}, wantErr: "--top must be positive"},
		{name: "sms disagreeing with props refused", args: []string{"observed", "--profile", benchLaneProfile, "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps, "--sms", "80"}, wantErr: "--sms disagrees"},
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

func TestBenchLaneObservedHumanTable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr, []string{
		"observed", "--profile", benchLaneProfile, "--ptxas-log", benchLaneExcerpt, "--device-props", benchLaneProps,
	})
	if code != 0 {
		t.Fatalf("exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak bench lane-ledger observed", "coverage:", "unmodelled ns by reason:", "time_weighted_idle_ns", "top modelled rows", "top unmodelled rows", "input conflicts", "caveats:"} {
		if !strings.Contains(out, want) {
			t.Errorf("observed table missing %q", want)
		}
	}
}

// ---- default mode with --device-props ----

// TestBenchLaneLedgerDevicePropsHandValues: 256 threads at 137 registers is 4608 registers per warp, 36864 per
// block, so one block fits per SM (65536 registers), register-limited. The capacity is 84 SMs times 1536.
func TestBenchLaneLedgerDevicePropsHandValues(t *testing.T) {
	doc := runLaneLedgerJSON(t, "--device-props", benchLaneProps, "--threads", "256", "--regs", "137", "--grid", "84")
	launch := laneLedgerObject(t, doc, "launch")
	if got := laneLedgerNumber(t, launch, "device_sms"); got != 84 {
		t.Errorf("device_sms = %d, want 84 from the file", got)
	}
	ledger := laneLedgerObject(t, doc, "ledger")
	if got := laneLedgerNumber(t, ledger, "blocks_per_sm"); got != 1 {
		t.Errorf("blocks_per_sm = %d, want 1", got)
	}
	if ledger["limiter"] != "register" {
		t.Errorf("limiter = %v, want register", ledger["limiter"])
	}
	if got := laneLedgerNumber(t, ledger, "capacity"); got != 129024 {
		t.Errorf("capacity = %d, want 129024", got)
	}
	check := laneLedgerObject(t, doc, "device_props")
	if check["verdict"] != "confirmed" {
		t.Errorf("device_props verdict = %v, want confirmed", check["verdict"])
	}
	caveats := doc["caveats"].([]any)
	for _, c := range caveats {
		if strings.Contains(c.(string), "the SM count is a caller input") {
			t.Error("caller-input SM caveat kept although the file supplied the SM count")
		}
	}
}

func TestBenchLaneLedgerDevicePropsRefusals(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{name: "sms disagreeing refused", args: []string{"--device-props", benchLaneProps, "--sms", "80", "--threads", "256", "--regs", "137", "--grid", "84"}, wantErr: "--sms disagrees"},
		{name: "arch disagreeing refused", args: []string{"--device-props", benchLaneProps, "--arch", "sm_89", "--threads", "256", "--regs", "137", "--grid", "84"}, wantErr: "--arch disagrees"},
		{name: "refuted refused", args: []string{"--device-props", "testdata/bench_lane_ledger/rtx5080_props_refuted.json", "--threads", "256", "--regs", "137", "--grid", "84"}, wantErr: "refutes"},
		{name: "missing file refused", args: []string{"--device-props", "testdata/bench_lane_ledger/none.json", "--threads", "256", "--regs", "137", "--grid", "84"}, wantErr: "--device-props"},
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
		})
	}
}

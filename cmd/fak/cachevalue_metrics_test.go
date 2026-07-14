package main

import (
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// sampleLine returns the first exposition sample line for a metric family + optional
// label substring, or "" if absent. It skips the # HELP/# TYPE header lines.
func sampleLine(t *testing.T, exposition, name, labelSubstr string) string {
	t.Helper()
	for _, ln := range strings.Split(exposition, "\n") {
		if strings.HasPrefix(ln, "#") || ln == "" {
			continue
		}
		if !strings.HasPrefix(ln, name) {
			continue
		}
		// guard against prefix collisions (fak_x vs fak_x_y): the char after the
		// name must be a space or '{'.
		rest := ln[len(name):]
		if rest == "" || (rest[0] != ' ' && rest[0] != '{') {
			continue
		}
		if labelSubstr == "" || strings.Contains(ln, labelSubstr) {
			return ln
		}
	}
	return ""
}

func fixedNow(t *testing.T) time.Time {
	t.Helper()
	n, err := time.Parse(time.RFC3339, "2026-07-04T12:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func ptr(f float64) *float64 { return &f }

// richReport is a fully-populated two-track report exercising every emitted family.
func richReport() cachevaluereport.TwoTrackReport {
	rep := cachevaluereport.TwoTrackReport{
		Schema:           cachevaluereport.Schema,
		GeneratedAt:      "2026-07-04T00:00:00Z",
		Verdict:          "MEASURED",
		LatestNetUSD:     530.6,
		CumulativeNetUSD: 5813.5,
		BrokeEven:        true,
		DollarBlindRows:  23,
	}
	rep.Track1.LatestReuseRatio = 0.6965
	rep.Track1.MultiTurnSessions = 6
	rep.Track1.TotalSessions = 11
	rep.FleetBenefit = cachevaluereport.FleetBenefitReport{
		ProviderPromptCacheTokenEq:  1.5e9,
		FakAuthoredTokenEq:          1.6e8,
		TotalSavedTokenEq:           1.66e9,
		ProviderAPICostAvoidedUSD:   7160.4,
		FakAPICostAvoidedUSD:        846.8,
		ObservedAPICostAvoidedUSD:   8007.2,
		ObservedActualSpendUSD:      2193.6,
		ObservedCounterfactualUSD:   10200.8,
		ObservedAPICostReductionPct: ptr(78.49),
		FakSharePct:                 ptr(9.97),
		ProviderUSDAvoidedPerDay:    810.5,
		FakUSDAvoidedPerDay:         95.8,
		USDAvoidedPerDay:            906.4,
		SpanDays:                    8.83,
		RateProvisional:             false,
		CompactionShedTokens:        170722283,
		ContextExtensionTokens:      170722283,
		UsageRows:                   2175,
		KernelDecisions:             420185,
	}
	return rep
}

// twoArmReport is a valid ablate report with a baseline + a faster arm.
func twoArmReport() ablate.Report {
	return ablate.Report{
		WorkloadHash: "9f1701415fb4a360",
		Baseline:     "all-off",
		Runs: []ablate.AblationRun{
			{ArmID: "all-off", WorkloadHash: "9f1701415fb4a360", WallSeconds: 0.07,
				Arm: metrics.Arm{MeanNs: 2904, P50Ns: 2586, P99Ns: 4252, VDSOHits: 0, EngineCalls: 12, InTokens: 687, OutTokens: 250}},
			{ArmID: "vdso", WorkloadHash: "9f1701415fb4a360", WallSeconds: 0.064,
				Arm: metrics.Arm{MeanNs: 2585, P50Ns: 2255, P99Ns: 3752, VDSOHits: 7, EngineCalls: 5, InTokens: 298, OutTokens: 119}},
		},
	}
}

// richCompaction is a compaction segmentation with one healthy shedding cell (headless
// 40-80, valid-denom fired rows → a shed percentile), one all-bail cell (interactive 0-20,
// no valid-denom rows → shed% must be ABSENT), and a quarantined phantom-100% row.
func richCompaction() gatewayusageledger.CompactionReport {
	return gatewayusageledger.CompactionReport{
		ExitRows:        204,
		QuarantinedRows: 3,
		Segments: []gatewayusageledger.CompactionSegment{
			{
				Budget: 48000, BudgetRegime: "interactive", Band: "0-20",
				Sessions: 120, FiredSessions: 0, Fires: 0, Bails: 44, BailRate: 1.0,
				ShedTokens: 0, ValidDenomRows: 0, DenomZeroRows: 3, ShedPctMedian: 0, ShedPctMean: 0,
				TopBailReason: "under_budget", TopBailShare: 1.0,
				BailReasons: map[string]uint64{"under_budget": 44},
			},
			{
				Budget: 96000, BudgetRegime: "headless", Band: "40-80",
				Sessions: 30, FiredSessions: 28, Fires: 900, Bails: 300, BailRate: 0.25,
				ShedTokens: 12_500_000, ValidDenomRows: 28, DenomZeroRows: 0,
				ShedPctMedian: 43.3, ShedPctMean: 41.7, TopBailReason: "under_budget", TopBailShare: 0.8,
				BailReasons: map[string]uint64{"under_budget": 240, "burst_unprofitable": 60},
			},
		},
	}
}

func TestRenderCachevalueExposition_Families(t *testing.T) {
	out := renderCachevalueExposition(richReport(), []ablate.Report{twoArmReport()}, richCompaction(), fixedNow(t))

	// presence + verdict + freshness
	if got := sampleLine(t, out, "fak_cachevalue_report_present", ""); got != "fak_cachevalue_report_present 1" {
		t.Errorf("report_present = %q", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_measured", ""); got != "fak_cachevalue_measured 1" {
		t.Errorf("measured = %q, want 1 for MEASURED verdict", got)
	}
	// GeneratedAt (2026-07-04T00:00:00Z) drives the timestamp, not `now` (12:00Z).
	genAt, _ := time.Parse(time.RFC3339, "2026-07-04T00:00:00Z")
	wantTS := strconv.FormatFloat(float64(genAt.Unix()), 'g', -1, 64)
	if got := sampleLine(t, out, "fak_cachevalue_generated_timestamp_seconds", ""); !strings.HasSuffix(got, " "+wantTS) {
		t.Errorf("generated_timestamp = %q, want suffix %q (from GeneratedAt, not now)", got, wantTS)
	}

	// Track 1 WITNESSED
	if got := sampleLine(t, out, "fak_cachevalue_latest_reuse_ratio", ""); !strings.Contains(got, "0.6965") {
		t.Errorf("latest_reuse_ratio = %q", got)
	}

	// owner splits must all three be present and disjoint
	for _, owner := range []string{"provider", "fak", "total"} {
		lbl := `owner="` + owner + `"`
		if got := sampleLine(t, out, "fak_cachevalue_saved_token_equiv", lbl); got == "" {
			t.Errorf("missing saved_token_equiv owner=%s", owner)
		}
		if got := sampleLine(t, out, "fak_cachevalue_api_cost_avoided_usd", lbl); got == "" {
			t.Errorf("missing api_cost_avoided_usd owner=%s", owner)
		}
		if got := sampleLine(t, out, "fak_cachevalue_usd_avoided_per_day", lbl); got == "" {
			t.Errorf("missing usd_avoided_per_day owner=%s", owner)
		}
	}

	// pointer fields present when set
	if got := sampleLine(t, out, "fak_cachevalue_fak_share_pct", ""); !strings.Contains(got, "9.97") {
		t.Errorf("fak_share_pct = %q", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_api_cost_reduction_pct", ""); !strings.Contains(got, "78.49") {
		t.Errorf("api_cost_reduction_pct = %q", got)
	}

	// ablation: presence, arm count, speedup ratio (baseline=1, faster arm>1)
	if got := sampleLine(t, out, "fak_ablation_report_present", ""); got != "fak_ablation_report_present 1" {
		t.Errorf("ablation_report_present = %q", got)
	}
	if got := sampleLine(t, out, "fak_ablation_arms", ""); got != "fak_ablation_arms 2" {
		t.Errorf("ablation_arms = %q, want 2", got)
	}
	if got := sampleLine(t, out, "fak_ablation_arm_speedup_ratio", `arm="all-off"`); !strings.HasSuffix(got, " 1") {
		t.Errorf("baseline speedup should be exactly 1: %q", got)
	}
	vdso := sampleLine(t, out, "fak_ablation_arm_speedup_ratio", `arm="vdso"`)
	if !strings.Contains(vdso, "1.12") {
		t.Errorf("vdso arm speedup = %q, want ~1.12 (faster than baseline)", vdso)
	}
	if got := sampleLine(t, out, "fak_ablation_baseline_info", `arm="all-off"`); got == "" {
		t.Error("baseline_info marker missing for baseline arm")
	}
	// workload label must be the short (<=12 char) hash
	if got := sampleLine(t, out, "fak_ablation_arm_mean_nanoseconds", `arm="vdso"`); !strings.Contains(got, `workload="9f1701415fb4"`) {
		t.Errorf("workload label not shortened: %q", got)
	}
}

func TestRenderCachevalueExposition_CompactionSegments(t *testing.T) {
	out := renderCachevalueExposition(richReport(), nil, richCompaction(), fixedNow(t))

	// corpus-level: present, exit rows, quarantined phantom-100% class all surfaced.
	if got := sampleLine(t, out, "fak_cachevalue_compaction_segments_present", ""); got != "fak_cachevalue_compaction_segments_present 1" {
		t.Errorf("segments_present = %q, want 1", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_exit_rows", ""); got != "fak_cachevalue_compaction_exit_rows 204" {
		t.Errorf("exit_rows = %q, want 204", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_quarantined_rows", ""); got != "fak_cachevalue_compaction_quarantined_rows 3" {
		t.Errorf("quarantined_rows = %q, want 3", got)
	}

	// the healthy headless 40-80 cell carries a shed percentile and its regime/budget/band labels.
	headless := sampleLine(t, out, "fak_cachevalue_compaction_shed_pct_median", `band="40-80"`)
	if !strings.Contains(headless, `regime="headless"`) || !strings.Contains(headless, `budget="96000"`) {
		t.Errorf("headless shed_pct_median missing regime/budget labels: %q", headless)
	}
	if !strings.Contains(headless, "43.3") {
		t.Errorf("headless shed_pct_median = %q, want ~43.3", headless)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_bail_rate", `regime="headless"`); !strings.HasSuffix(got, "0.25") {
		t.Errorf("headless bail_rate = %q, want 0.25", got)
	}

	// the all-bail interactive 0-20 cell must NOT emit a shed percentile (honest absence,
	// not a misleading 0), but its sessions/bails and quarantine count must be present.
	if got := sampleLine(t, out, "fak_cachevalue_compaction_shed_pct_median", `regime="interactive"`); got != "" {
		t.Errorf("interactive 0-20 (no valid-denom rows) must not emit shed_pct_median: %q", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_sessions", `regime="interactive"`); !strings.HasSuffix(got, "120") {
		t.Errorf("interactive sessions = %q, want 120", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_denom_zero_rows", `regime="interactive"`); !strings.HasSuffix(got, "3") {
		t.Errorf("interactive denom_zero_rows = %q, want 3", got)
	}

	// the bail-reason MIX is projected per (regime × band × reason): the headless cell's
	// burst_unprofitable slice (the tuning-sensitive one) must be its own series so an alert
	// can watch it independently of the correct-by-design under_budget mass.
	if got := sampleLine(t, out, "fak_cachevalue_compaction_bail_reason", `reason="burst_unprofitable"`); !strings.HasSuffix(got, "60") {
		t.Errorf("headless burst_unprofitable bail_reason = %q, want 60", got)
	}
	if got := sampleLine(t, out, "fak_cachevalue_compaction_bail_reason", `regime="headless",budget="96000",band="40-80",reason="under_budget"`); !strings.HasSuffix(got, "240") {
		t.Errorf("headless under_budget bail_reason = %q, want 240", got)
	}
	// the top reason's share rides its own gauge so a dashboard gates on "dominance fell"
	// without summing the per-reason series.
	if got := sampleLine(t, out, "fak_cachevalue_compaction_top_bail_share", `regime="headless"`); !strings.HasSuffix(got, "0.8") {
		t.Errorf("headless top_bail_share = %q, want 0.8", got)
	}
}

func TestRenderCachevalueExposition_HonestAbsence(t *testing.T) {
	// A nil pointer field is omitted (honest absence, not a zero sample).
	rep := richReport()
	rep.FleetBenefit.FakSharePct = nil
	rep.Verdict = "INSUFFICIENT"
	out := renderCachevalueExposition(rep, nil, gatewayusageledger.CompactionReport{}, fixedNow(t))

	if strings.Contains(out, "fak_cachevalue_fak_share_pct") {
		t.Error("fak_share_pct must be absent when the pointer is nil")
	}
	if got := sampleLine(t, out, "fak_cachevalue_measured", ""); got != "fak_cachevalue_measured 0" {
		t.Errorf("measured = %q, want 0 for INSUFFICIENT", got)
	}
	// no ablate reports => report_present 0, but the family is still declared
	if got := sampleLine(t, out, "fak_ablation_report_present", ""); got != "fak_ablation_report_present 0" {
		t.Errorf("ablation_report_present = %q, want 0", got)
	}
	if got := sampleLine(t, out, "fak_ablation_arms", ""); got != "fak_ablation_arms 0" {
		t.Errorf("ablation_arms = %q, want 0", got)
	}
	// report_present stays 1 even with the INSUFFICIENT verdict (dead exporter != real zero)
	if got := sampleLine(t, out, "fak_cachevalue_report_present", ""); got != "fak_cachevalue_report_present 1" {
		t.Errorf("report_present = %q, want 1", got)
	}
}

func TestRenderCachevalueExposition_NonFiniteSkipped(t *testing.T) {
	rep := richReport()
	rep.CumulativeNetUSD = math.Inf(1)
	rep.LatestNetUSD = math.NaN()
	out := renderCachevalueExposition(rep, nil, gatewayusageledger.CompactionReport{}, fixedNow(t))
	if sampleLine(t, out, "fak_cachevalue_cumulative_net_usd", "") != "" {
		t.Error("an +Inf value must be skipped, not emitted")
	}
	if sampleLine(t, out, "fak_cachevalue_latest_net_usd", "") != "" {
		t.Error("a NaN value must be skipped, not emitted")
	}
}

var sampleRE = regexp.MustCompile(`^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (.+)$`)

// TestExpositionWellFormed asserts every sample line parses and every sampled family
// carries a preceding # TYPE declaration (the contract Prometheus enforces on scrape).
func TestExpositionWellFormed(t *testing.T) {
	out := renderCachevalueExposition(richReport(), []ablate.Report{twoArmReport()}, richCompaction(), fixedNow(t))
	declared := map[string]bool{}
	for _, ln := range strings.Split(out, "\n") {
		if strings.HasPrefix(ln, "# TYPE ") {
			parts := strings.Fields(ln)
			declared[parts[2]] = true
			continue
		}
		if strings.HasPrefix(ln, "#") || ln == "" {
			continue
		}
		m := sampleRE.FindStringSubmatch(ln)
		if m == nil {
			t.Errorf("malformed sample line: %q", ln)
			continue
		}
		if !declared[m[1]] {
			t.Errorf("sample for undeclared family (no preceding # TYPE): %q", m[1])
		}
	}
}

func TestWriteFileAtomicProm(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache-value.prom")
	const content = "# TYPE x gauge\nx 1\n"
	if err := writeFileAtomicProm(path, content); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != content {
		t.Errorf("read back %q, want %q", got, content)
	}
	// the temp sibling must not linger
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Errorf(".tmp file lingered: %v", err)
	}
}

func TestCachevalueMetricsMux_ServesExposition(t *testing.T) {
	// Point at a temp dir with one valid ablate report and no ledgers: the /metrics
	// handler must still serve a 200 with a legal, present-but-mostly-zero exposition.
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "one.json"),
		`{"workload_hash":"abc","baseline_arm":"x","runs":[{"arm_id":"x","workload_hash":"abc","arm":{"mean_ns":10}}]}`)
	src := cachevalueMetricsSources{
		ledger:        filepath.Join(dir, "missing-cache-value.jsonl"),
		savingsLedger: filepath.Join(dir, "missing-savings.jsonl"),
		usageLedger:   filepath.Join(dir, "missing-usage.jsonl"),
		ablationDir:   dir,
		stderr:        io.Discard,
	}
	srv := httptest.NewServer(cachevalueMetricsMux(src))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/plain") {
		t.Errorf("content-type = %q, want text/plain exposition", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	out := string(body)
	if !strings.Contains(out, "fak_cachevalue_report_present 1") {
		t.Errorf("exposition missing report_present marker:\n%s", out)
	}
	if !strings.Contains(out, "fak_ablation_report_present 1") {
		t.Error("exposition missing ablation_report_present (the temp report should fold)")
	}
}

func TestAblationPathListFlag(t *testing.T) {
	var a ablationPathList
	if err := a.Set("one.json"); err != nil {
		t.Fatal(err)
	}
	if err := a.Set(""); err != nil { // empty is a no-op, not an error
		t.Fatal(err)
	}
	if err := a.Set("two.json"); err != nil {
		t.Fatal(err)
	}
	if got := a.String(); got != "one.json,two.json" {
		t.Errorf("String() = %q", got)
	}
}

func TestLoadAblationReports_SkipsInvalid(t *testing.T) {
	dir := t.TempDir()
	// a valid report, an empty-arms report (invalid), and unparseable JSON
	mustWrite(t, filepath.Join(dir, "a-valid.json"), `{"workload_hash":"abc","baseline_arm":"x","runs":[{"arm_id":"x","workload_hash":"abc","arm":{"mean_ns":10}}]}`)
	mustWrite(t, filepath.Join(dir, "b-empty.json"), `{"workload_hash":"abc","runs":[]}`)
	mustWrite(t, filepath.Join(dir, "c-garbage.json"), `{not json`)

	src := cachevalueMetricsSources{ablationDir: dir, stderr: io.Discard}
	got := src.loadAblationReports()
	if len(got) != 1 {
		t.Fatalf("loaded %d reports, want 1 (the other two are invalid/garbage)", len(got))
	}
	if got[0].WorkloadHash != "abc" {
		t.Errorf("wrong report loaded: %+v", got[0])
	}
}

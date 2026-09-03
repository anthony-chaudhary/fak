package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// runCachevalueMetrics handles `fak cachevalue metrics` — the Prometheus/Grafana
// exposition of the cache-value roll-up. It is the SAME two-track fold the Slack card
// (`fak cachevalue feed`) posts, re-projected into the Prometheus text-exposition
// format under the `fak_cachevalue_*` namespace, plus the offline feature-ablation
// arms (`fak_ablation_*`) read from the durable ablate report JSONs. It exists so the
// question the roll-up answers — "is fak's cache work paying off, and which feature
// moved the needle?" — becomes a live Grafana surface instead of a one-shot card.
//
//	fak cachevalue metrics                                   # render the exposition to stdout once
//	fak cachevalue metrics --serve --addr 127.0.0.1:9097     # serve /metrics; re-fold per scrape
//	fak cachevalue metrics --textfile cache-value.prom       # write a .prom for a textfile collector
//	fak cachevalue metrics --ablation-dir experiments/ablate # arms folded from these ablate reports
//	fak cachevalue metrics --ablation-report ablate.json     # add an explicit ablate report (repeatable)
//
// It never blends the tracks: Track-1 realized KV reuse (WITNESSED) and the Track-2
// OBSERVED/projected-$ P&L stay in separate metric families, mirroring the report's own
// honesty fence. Missing ledgers fold to an honest present-but-zero exposition (verdict
// gauge 0), never a hard error, so a dead exporter is distinguishable from a real zero.
//
//fak:ctxplan verb="cachevalue metrics" enters="nothing live — an offline fold over the durable cache-value/cache-savings/gateway-usage JSONL ledgers plus the ablate report JSONs on disk" pages="nothing into a model window — it renders a Prometheus text exposition to stdout, a .prom textfile, or a /metrics HTTP endpoint for Grafana to scrape" warms="nothing — it REPORTS whether the cache method pays off; it warms no prompt cache or KV itself"
func runCachevalueMetrics(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue metrics", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ ledger (.fak/nightrun/cache-savings.jsonl)")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger for cumulative fleet usage/session-extension counters (.fak/nightrun/gateway-usage.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	contextBudget := fs.Uint64("context-budget-tokens", 0, "optional session context budget denominator; normalizes witnessed shed tokens into window-equivalent extension")
	ablationDir := fs.String("ablation-dir", "experiments/ablate", "directory of ablate report JSONs to fold into the fak_ablation_* arms (empty to disable dir discovery)")
	var ablationReports ablationPathList
	fs.Var(&ablationReports, "ablation-report", "explicit ablate report JSON to fold (repeatable; adds to --ablation-dir discoveries)")
	serve := fs.Bool("serve", false, "serve the exposition on --addr /metrics, re-folding the ledgers on each scrape")
	addr := fs.String("addr", "127.0.0.1:9097", "with --serve, the host:port to bind /metrics on")
	textfile := fs.String("textfile", "", "write the exposition atomically to this .prom path (for a node_exporter textfile collector) instead of stdout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue metrics: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	src := cachevalueMetricsSources{
		ledger:        *ledger,
		savingsLedger: *savingsLedger,
		usageLedger:   *usageLedger,
		since:         *since,
		contextBudget: *contextBudget,
		ablationDir:   *ablationDir,
		ablationFiles: ablationReports,
		stderr:        stderr,
	}

	if *serve {
		return serveCachevalueMetrics(stdout, stderr, src, *addr)
	}

	exposition := src.render()
	if *textfile != "" {
		if err := writeFileAtomicProm(*textfile, exposition); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue metrics: write --textfile %s: %v\n", *textfile, err)
			return 1
		}
		fmt.Fprintf(stderr, "fak cachevalue metrics: wrote %d bytes to %s\n", len(exposition), *textfile)
		return 0
	}
	fmt.Fprint(stdout, exposition)
	return 0
}

// cachevalueMetricsSources is the resolved input set. Held as a value so both the
// one-shot path and the serve handler fold from the SAME recipe — the exposition a
// Grafana scrape reads is byte-for-byte the projection of the report the Slack card folds.
type cachevalueMetricsSources struct {
	ledger        string
	savingsLedger string
	usageLedger   string
	since         string
	contextBudget uint64
	ablationDir   string
	ablationFiles []string
	stderr        io.Writer
}

// render re-reads every ledger + ablate report from disk and returns the exposition
// text. Re-reading per call is what makes the --serve endpoint reflect ledger appends
// live; the fold itself is pure given the rows and the clock.
func (s cachevalueMetricsSources) render() string {
	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(s.ledger), s.since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(s.savingsLedger), s.since)
	usage := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(s.usageLedger), s.since)

	now := cachevalueReportNow()
	report := cachevaluereport.FoldTwoTrackWithUsage(track1, track2, usage, now, cachevaluereport.FleetBenefitOptions{
		ContextBudgetTokens: s.contextBudget,
	})
	report.Since = s.since

	// Fold the SAME usage rows into the compaction-by-regime segmentation so the
	// exposition carries the regime×band split, not just the blended fleet-total shed.
	// This is what keeps a deliberate 48k→96k budget switch from reading as a phantom
	// compaction regression on a Grafana panel (the failure mode the CLI table already
	// prevents; #887e8026f). Same fold `fak cachevalue compaction` renders.
	compaction := gatewayusageledger.FoldCompaction(usage, s.since)

	arms := s.loadAblationReports()
	return renderCachevalueExposition(report, arms, compaction, now)
}

// loadAblationReports gathers ablate report JSONs from --ablation-dir (globbed, sorted
// for determinism) plus any explicit --ablation-report files, and unmarshals each. A
// file that fails to read/parse/validate is skipped with a stderr note rather than
// aborting the exposition — an unreadable ablate snapshot must not blind the cache-value
// panels next to it.
func (s cachevalueMetricsSources) loadAblationReports() []ablate.Report {
	var paths []string
	if s.ablationDir != "" {
		globbed, _ := filepath.Glob(filepath.Join(s.ablationDir, "*.json"))
		sort.Strings(globbed)
		paths = append(paths, globbed...)
	}
	paths = append(paths, s.ablationFiles...)

	seen := map[string]bool{}
	var out []ablate.Report
	for _, p := range paths {
		if seen[p] {
			continue
		}
		seen[p] = true
		raw, err := os.ReadFile(p)
		if err != nil {
			s.note("read ablate report %s: %v", p, err)
			continue
		}
		var rep ablate.Report
		if err := json.Unmarshal(raw, &rep); err != nil {
			s.note("parse ablate report %s: %v", p, err)
			continue
		}
		if err := rep.Validate(); err != nil {
			s.note("skip invalid ablate report %s: %v", p, err)
			continue
		}
		out = append(out, rep)
	}
	return out
}

func (s cachevalueMetricsSources) note(format string, args ...any) {
	writeMetricsNote(s.stderr, "fak cachevalue metrics: ", format, args...)
}

// cachevalueMetricsMux builds the HTTP surface: /metrics re-folds the ledgers per scrape
// (so the endpoint reflects ledger appends live), and the root points a browser at it.
// Split from serveCachevalueMetrics so a test can exercise the handler without binding a
// port.
func cachevalueMetricsMux(src cachevalueMetricsSources) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		io.WriteString(w, src.render())
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, "fak cachevalue metrics — the cache-value & ablation roll-up exposition.\nScrape /metrics for the fak_cachevalue_* and fak_ablation_* families.\n")
	})
	return mux
}

// serveCachevalueMetrics binds an HTTP server exposing /metrics (re-folded per scrape)
// and a plain-text root pointing at it. It blocks until ListenAndServe returns.
func serveCachevalueMetrics(stdout, stderr io.Writer, src cachevalueMetricsSources, addr string) int {
	fmt.Fprintf(stdout, "fak cachevalue metrics: serving /metrics on http://%s/metrics\n", addr)
	srv := &http.Server{Addr: addr, Handler: cachevalueMetricsMux(src)}
	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintf(stderr, "fak cachevalue metrics: serve: %v\n", err)
		return 1
	}
	return 0
}

// ablationPathList is the repeatable --ablation-report flag.
type ablationPathList []string

func (a *ablationPathList) String() string { return strings.Join(*a, ",") }
func (a *ablationPathList) Set(v string) error {
	if v == "" {
		return nil
	}
	*a = append(*a, v)
	return nil
}

// writeFileAtomicProm writes content to path via a temp file + rename so a scraping
// textfile collector never reads a half-written exposition.
func writeFileAtomicProm(path, content string) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// ---- exposition renderer -------------------------------------------------------

// renderCachevalueExposition projects the two-track report + ablation arms into the
// Prometheus text-exposition format. It is PURE (report + arms + clock in, text out) so
// the golden test can pin it. Every family carries a HELP line naming its provenance
// (WITNESSED vs OBSERVED/projected) so the fence survives into Grafana.
func renderCachevalueExposition(rep cachevaluereport.TwoTrackReport, arms []ablate.Report, compaction gatewayusageledger.CompactionReport, now time.Time) string {
	w := newPromWriter()

	// --- meta: presence + freshness + verdict (so "No data" != "present but zero") ---
	w.gauge("fak_cachevalue_report_present", "1 when the cache-value roll-up exposition rendered (distinguishes a dead exporter from a real zero).", 1)
	w.gauge("fak_cachevalue_measured", "1 when the two-track fold reached the MEASURED verdict, 0 when INSUFFICIENT.", bool01(rep.Verdict == "MEASURED"))
	genTS := float64(now.Unix())
	if t, err := time.Parse(time.RFC3339, rep.GeneratedAt); err == nil {
		genTS = float64(t.Unix())
	}
	w.gauge("fak_cachevalue_generated_timestamp_seconds", "Unix time the exposition was folded (GeneratedAt of the report).", genTS)

	// --- Track 1: WITNESSED realized KV-prefix reuse ---
	t1 := rep.Track1
	w.gauge("fak_cachevalue_latest_reuse_ratio", "WITNESSED: most recent weekly bucket's realized KV-prefix reuse ratio (Track 1, not a vs-naive multiple).", t1.LatestReuseRatio)
	w.gauge("fak_cachevalue_multi_turn_sessions", "WITNESSED: multi-turn sessions folded into Track 1 (the only sessions that can realize reuse).", float64(t1.MultiTurnSessions))
	w.gauge("fak_cachevalue_total_sessions", "WITNESSED: total sessions folded into Track 1.", float64(t1.TotalSessions))

	// --- Track 2 / NET: OBSERVED-$ P&L (a cost projection, never a fak-witnessed claim) ---
	w.gauge("fak_cachevalue_latest_net_usd", "OBSERVED/projected $: net of the most recent period (rebate + shed - write premium - spend).", rep.LatestNetUSD)
	w.gauge("fak_cachevalue_cumulative_net_usd", "OBSERVED/projected $: running net total through the latest period (the P&L headline).", rep.CumulativeNetUSD)
	w.gauge("fak_cachevalue_broke_even", "1 when the cumulative net $ has crossed zero (broken even), else 0.", bool01(rep.BrokeEven))
	w.gauge("fak_cachevalue_dollar_blind_rows", "Rows carrying saved token-equiv but no trusted price (excluded from the $ columns).", float64(rep.DollarBlindRows))

	// --- fleet benefit: saved token-equiv + avoided $ split by owner (never summed across provenance) ---
	fb := rep.FleetBenefit
	const savedHelp = "Saved token-equivalent, split by owner. provider=OBSERVED prompt-cache; fak=WITNESSED kernel-authored; total=their sum."
	w.gauge("fak_cachevalue_saved_token_equiv", savedHelp, fb.ProviderPromptCacheTokenEq, "owner", "provider")
	w.gauge("fak_cachevalue_saved_token_equiv", savedHelp, fb.FakAuthoredTokenEq, "owner", "fak")
	w.gauge("fak_cachevalue_saved_token_equiv", savedHelp, fb.TotalSavedTokenEq, "owner", "total")

	const avoidedHelp = "OBSERVED/projected $ of API cost avoided, split by owner. provider=read rebate net of write premium; fak=compaction saving (WITNESSED shed, $ projected); total=blended sum."
	w.gauge("fak_cachevalue_api_cost_avoided_usd", avoidedHelp, fb.ProviderAPICostAvoidedUSD, "owner", "provider")
	w.gauge("fak_cachevalue_api_cost_avoided_usd", avoidedHelp, fb.FakAPICostAvoidedUSD, "owner", "fak")
	w.gauge("fak_cachevalue_api_cost_avoided_usd", avoidedHelp, fb.ObservedAPICostAvoidedUSD, "owner", "total")

	w.gauge("fak_cachevalue_observed_spend_usd", "OBSERVED $: actual API spend over the folded rows.", fb.ObservedActualSpendUSD)
	w.gauge("fak_cachevalue_counterfactual_usd", "OBSERVED/projected $: uncached/uncompacted counterfactual spend.", fb.ObservedCounterfactualUSD)
	if fb.ObservedAPICostReductionPct != nil {
		w.gauge("fak_cachevalue_api_cost_reduction_pct", "OBSERVED/projected: percent API-cost reduction vs the counterfactual.", *fb.ObservedAPICostReductionPct)
	}
	if fb.FakSharePct != nil {
		w.gauge("fak_cachevalue_fak_share_pct", "Percent of avoided $ attributable to fak-authored cache work (the rest is provider prompt-cache).", *fb.FakSharePct)
	}

	// --- run-rate (long-horizon lens) ---
	const rateHelp = "OBSERVED/projected $ avoided per day, split by owner, over the savings-row span (PROVISIONAL under a thin window — see fak_cachevalue_rate_provisional)."
	w.gauge("fak_cachevalue_usd_avoided_per_day", rateHelp, fb.ProviderUSDAvoidedPerDay, "owner", "provider")
	w.gauge("fak_cachevalue_usd_avoided_per_day", rateHelp, fb.FakUSDAvoidedPerDay, "owner", "fak")
	w.gauge("fak_cachevalue_usd_avoided_per_day", rateHelp, fb.USDAvoidedPerDay, "owner", "total")
	w.gauge("fak_cachevalue_span_days", "Span in days of the savings rows the run-rate is computed over.", fb.SpanDays)
	w.gauge("fak_cachevalue_rate_provisional", "1 when the run-rate span is under the honest-window floor (extrapolation not yet settled).", bool01(fb.RateProvisional))

	// --- session extension (WITNESSED compaction-shed only) ---
	w.gauge("fak_cachevalue_compaction_shed_tokens", "WITNESSED: context tokens shed by fak compaction (the session-extension source).", float64(fb.CompactionShedTokens))
	w.gauge("fak_cachevalue_context_extension_tokens", "WITNESSED: context tokens the session was extended by (compaction-shed only; provider reads never enlarge the window).", float64(fb.ContextExtensionTokens))

	// --- operational usage (WITNESSED, back-complete since the usage-row writer shipped) ---
	w.gauge("fak_cachevalue_usage_rows", "WITNESSED: gateway/guard usage rows folded into the fleet aggregate.", float64(fb.UsageRows))
	w.gauge("fak_cachevalue_kernel_decisions", "WITNESSED: cumulative kernel admission decisions over the usage rows.", float64(fb.KernelDecisions))

	renderCompactionSegments(w, compaction)
	renderAblationArms(w, arms)
	return w.String()
}

// renderCompactionSegments projects the compaction-by-(budget-regime × length-band)
// segmentation into the fak_cachevalue_compaction_* families, one sample per segment
// labelled (regime, budget, band). It exists so a Grafana panel can hold the interactive
// 48k and headless 96k regimes as SEPARATE series: the two budgets fire against
// structurally different resident shapes, so a blended fleet-total shed reads a deliberate
// budget switch as a compaction regression (see gatewayusageledger.FoldCompaction). These
// are WITNESSED gateway-usage counters, never a live provider claim.
//
// The shed-percent gauge is emitted ONLY for a segment with valid-denominator fired rows —
// a segment where every row bailed or was quarantined has no honest shed fraction, so it is
// absent (not a misleading 0), exactly like the table's dash. The quarantined-row count is
// surfaced corpus-level so the phantom-100% class (shed>0 but cached+input==0) stays visible
// rather than silently inflating a percentile.
func renderCompactionSegments(w *promWriter, rep gatewayusageledger.CompactionReport) {
	present := 0.0
	if rep.ExitRows > 0 {
		present = 1
	}
	w.gauge("fak_cachevalue_compaction_segments_present", "1 when the compaction segmentation folded at least one exit row (distinguishes a dead fold from a real zero).", present)
	w.gauge("fak_cachevalue_compaction_exit_rows", "WITNESSED: gateway-usage exit rows folded into the compaction segmentation.", float64(rep.ExitRows))
	w.gauge("fak_cachevalue_compaction_quarantined_rows", "WITNESSED: fired rows with shed>0 but cached+input==0 (phantom-100%% class), counted but excluded from every shed%% percentile.", float64(rep.QuarantinedRows))

	for _, s := range rep.Segments {
		lbl := []string{"regime", s.BudgetRegime, "budget", strconv.Itoa(s.Budget), "band", s.Band}
		w.gauge("fak_cachevalue_compaction_sessions", "WITNESSED: exit rows in this (regime × band) cell.", float64(s.Sessions), lbl...)
		w.gauge("fak_cachevalue_compaction_fired_sessions", "WITNESSED: rows in this cell that fired compaction at least once.", float64(s.FiredSessions), lbl...)
		w.gauge("fak_cachevalue_compaction_fires", "WITNESSED: compaction fires summed over this cell.", float64(s.Fires), lbl...)
		w.gauge("fak_cachevalue_compaction_bails", "WITNESSED: compaction bails summed over this cell.", float64(s.Bails), lbl...)
		w.gauge("fak_cachevalue_compaction_bail_rate", "WITNESSED: bails / (fires + bails) for this cell, NON-CANDIDATES INCLUDED; 0 when neither fired nor bailed. Near 1.0 by construction on mixed traffic — alert on candidate_bail_rate instead.", s.BailRate, lbl...)
		w.gauge("fak_cachevalue_compaction_non_candidate_bails", "WITNESSED: bails in this cell that were never compaction CANDIDATES (too_few_msgs, non_json, no_messages_key, decode_failed) — held out of candidate_bail_rate.", float64(s.NonCandidateBails), lbl...)
		w.gauge("fak_cachevalue_compaction_candidate_bail_rate", "WITNESSED: candidate bails / (fires + candidate bails) — declines over ELIGIBLE attempts, the alertable compaction-health rate; 0 when nothing was eligible.", s.CandidateBailRate, lbl...)
		w.gauge("fak_cachevalue_compaction_shed_tokens_by_segment", "WITNESSED: context tokens shed by compaction in this cell.", float64(s.ShedTokens), lbl...)
		w.gauge("fak_cachevalue_compaction_valid_denom_rows", "WITNESSED: fired rows with a usable shed denominator (cached+input>0) — the rows the shed%% folds over.", float64(s.ValidDenomRows), lbl...)
		w.gauge("fak_cachevalue_compaction_denom_zero_rows", "WITNESSED: fired rows quarantined from the shed%% (shed>0 but cached+input==0).", float64(s.DenomZeroRows), lbl...)
		// Honest absence: only a segment with valid-denominator fired rows has a real
		// shed fraction. A zero here would read as "0%% shed" rather than "no data".
		if s.ValidDenomRows > 0 {
			w.gauge("fak_cachevalue_compaction_shed_pct_median", "WITNESSED: median per-session shed fraction (%%) over this cell's valid-denominator fired rows.", s.ShedPctMedian, lbl...)
		}
		// Bail-reason mix: one sample per (regime × band × reason) so a panel/alert reads
		// the WHY the single bail_rate gauge cannot carry — an under_budget-dominated slice
		// is a band correctly declining under its budget, while a burst_unprofitable share
		// creeping up is a tuning call worth watching. Reasons sorted for a deterministic
		// exposition. The top reason's SHARE is emitted alongside so a dashboard can gate on
		// "the dominant reason no longer dominates" without summing the per-reason series.
		// The FULL mix is emitted, non-candidates included, so decode_failed stays assertable
		// on its own even though candidate_bail_rate holds it out.
		reasons := make([]string, 0, len(s.BailReasons))
		for r := range s.BailReasons {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			w.gauge("fak_cachevalue_compaction_bail_reason", "WITNESSED: compaction bails in this cell attributed to one CompactReason (under_budget is correct-by-design; burst_unprofitable is the tuning-sensitive slice, moving with --compact-history-budget/--assume-session-turns; too_few_msgs/non_json/no_messages_key/decode_failed are NOT CANDIDATES at all — no setting makes a 2-message request compactible — and are excluded from candidate_bail_rate).", float64(s.BailReasons[r]), "regime", s.BudgetRegime, "budget", strconv.Itoa(s.Budget), "band", s.Band, "reason", r)
		}
		if s.TopBailReason != "" {
			topLbl := []string{"regime", s.BudgetRegime, "budget", strconv.Itoa(s.Budget), "band", s.Band, "reason", s.TopBailReason}
			w.gauge("fak_cachevalue_compaction_top_bail_share", "WITNESSED: the top bail reason's fraction of this cell's classified CANDIDATE bails (1.0 = one dominant reason; a falling share means a second reason is eating attempts). Non-candidate reasons cannot win this label.", s.TopBailShare, topLbl...)
		}
	}
}

// renderAblationArms projects the offline feature-ablation reports into the
// fak_ablation_* families. Each arm is labelled by (arm, workload) so arms from
// different frozen traces never collide, and the baseline arm carries a speedup ratio
// of 1 with a marker gauge. Ablation is offline ($0 deterministic replay), so these are
// WITNESSED replay counters, never a live provider claim.
func renderAblationArms(w *promWriter, arms []ablate.Report) {
	present := 0.0
	total := 0
	if len(arms) > 0 {
		present = 1
	}
	w.gauge("fak_ablation_report_present", "1 when at least one ablate report was folded into the fak_ablation_* arms.", present)

	for _, rep := range arms {
		total += len(rep.Runs)
		workload := shortHash(rep.WorkloadHash)
		var baseMean float64
		if b := rep.ArmByID(rep.Baseline); b != nil {
			baseMean = float64(b.Arm.MeanNs)
		}
		for i := range rep.Runs {
			run := rep.Runs[i]
			lbl := []string{"arm", run.ArmID, "workload", workload}
			w.gauge("fak_ablation_arm_mean_nanoseconds", "WITNESSED replay: per-arm mean kernel decide latency (ns).", float64(run.Arm.MeanNs), lbl...)
			w.gauge("fak_ablation_arm_p50_nanoseconds", "WITNESSED replay: per-arm p50 kernel decide latency (ns).", float64(run.Arm.P50Ns), lbl...)
			w.gauge("fak_ablation_arm_p99_nanoseconds", "WITNESSED replay: per-arm p99 kernel decide latency (ns).", float64(run.Arm.P99Ns), lbl...)
			w.gauge("fak_ablation_arm_vdso_hits", "WITNESSED replay: per-arm vDSO fast-path hits.", float64(run.Arm.VDSOHits), lbl...)
			w.gauge("fak_ablation_arm_engine_calls", "WITNESSED replay: per-arm engine (slow-path) calls.", float64(run.Arm.EngineCalls), lbl...)
			w.gauge("fak_ablation_arm_input_tokens", "WITNESSED replay: per-arm input tokens over the frozen trace.", float64(run.Arm.InTokens), lbl...)
			w.gauge("fak_ablation_arm_output_tokens", "WITNESSED replay: per-arm output tokens over the frozen trace.", float64(run.Arm.OutTokens), lbl...)
			w.gauge("fak_ablation_arm_wall_seconds", "WITNESSED replay: per-arm wall-clock of the replay.", run.WallSeconds, lbl...)
			if baseMean > 0 && run.Arm.MeanNs > 0 {
				w.gauge("fak_ablation_arm_speedup_ratio", "WITNESSED replay: baseline mean / this arm's mean (>1 is faster than baseline).", baseMean/float64(run.Arm.MeanNs), lbl...)
			}
			if run.ArmID == rep.Baseline {
				w.gauge("fak_ablation_baseline_info", "1 marks the baseline arm each speedup ratio is measured against.", 1, lbl...)
			}
		}
	}
	w.gauge("fak_ablation_arms", "Total ablation arms folded across every report.", float64(total))
}

func bool01(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// ---- minimal Prometheus text-exposition writer ---------------------------------

// promWriter emits well-formed Prometheus text exposition: one HELP+TYPE per family
// (guarded so repeated samples of a family declare it once), then labelled samples.
type promWriter struct {
	b        strings.Builder
	declared map[string]bool
}

func newPromWriter() *promWriter { return &promWriter{declared: map[string]bool{}} }

func (w *promWriter) String() string { return w.b.String() }

// gauge emits one gauge sample. labelPairs are key,value,key,value...; a non-finite
// value is skipped (honest absence) rather than emitting NaN into a panel.
func (w *promWriter) gauge(name, help string, val float64, labelPairs ...string) {
	if math.IsNaN(val) || math.IsInf(val, 0) {
		return
	}
	if !w.declared[name] {
		w.declared[name] = true
		fmt.Fprintf(&w.b, "# HELP %s %s\n", name, escapeHelp(help))
		fmt.Fprintf(&w.b, "# TYPE %s gauge\n", name)
	}
	w.b.WriteString(name)
	w.writeLabels(labelPairs)
	w.b.WriteByte(' ')
	if val == math.Trunc(val) && math.Abs(val) < 1e15 {
		w.b.WriteString(strconv.FormatInt(int64(val), 10))
	} else {
		w.b.WriteString(strconv.FormatFloat(val, 'g', -1, 64))
	}
	w.b.WriteByte('\n')
}

func (w *promWriter) writeLabels(pairs []string) {
	if len(pairs) < 2 {
		return
	}
	w.b.WriteByte('{')
	first := true
	for i := 0; i+1 < len(pairs); i += 2 {
		if !first {
			w.b.WriteByte(',')
		}
		first = false
		w.b.WriteString(pairs[i])
		w.b.WriteString(`="`)
		w.b.WriteString(escapeLabelValue(pairs[i+1]))
		w.b.WriteByte('"')
	}
	w.b.WriteByte('}')
}

func escapeLabelValue(s string) string {
	if !strings.ContainsAny(s, "\\\"\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s)
}

func escapeHelp(s string) string {
	if !strings.ContainsAny(s, "\\\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, "\n", `\n`).Replace(s)
}

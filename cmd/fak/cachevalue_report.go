package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// cachevalueReportNow is the report's clock seam. It stamps GeneratedAt AND drives
// the #3394 freshness/drain-lag row (now − last-savings-row), so the report is no
// longer clock-independent: a recompute test must pin this to fold against the same
// instant the CLI used. Follows the dispatchProgressNow/releaseStatusNow convention.
var cachevalueReportNow = func() time.Time { return time.Now().UTC() }

// runCachevalueReport handles `fak cachevalue report` — the #1304 two-track P&L.
// It folds BOTH durable ledgers side by side, never blended:
//
//	Track 1 (WITNESSED kernel)  — docs/nightrun/cache-value.jsonl   (realized KV reuse)
//	Track 2 (OBSERVED $)        — .fak/nightrun/cache-savings.jsonl (provider rebate +
//	                              compaction token-shed − write premium − API spend)
//
// and prints both tracks plus a single NET line per period with the running total
// crossing break-even shown explicitly. --since floors both ledgers to rows on or
// after the date; --json emits the cachevaluereport.TwoTrackReport for downstream
// posting. A missing Track-2 ledger folds to the honest "rung B not appending yet"
// report rather than failing — Track 2's live append is epic #1301 rung B (#1303).
func runCachevalueReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ ledger (.fak/nightrun/cache-savings.jsonl)")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger for cumulative fleet usage/session-extension counters (.fak/nightrun/gateway-usage.jsonl)")
	recallInjectionLedger := fs.String("recall-injection-ledger", cachevaluereport.DefaultRecallInjectionLedger, "numbers-only recall injection debit ledger (.fak/recall-injections.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	contextBudget := fs.Uint64("context-budget-tokens", 0, "optional session context budget denominator; normalizes witnessed shed tokens into window-equivalent extension")
	asJSON := fs.Bool("json", false, "emit the two-track report as JSON instead of the table")
	markdown := fs.Bool("markdown", false, "emit the two-track report as markdown (mermaid xychart trends + sparklines + a provenance-labelled KPI table) instead of the terminal table")
	devSessions := fs.Bool("dev-sessions", false, "fold real, un-proxied Claude Code session transcripts into a separate Track-3 dev-session lens (may overlap the fleet aggregate; never summed with it); opt-in: it discovers/reads local transcript files, a real-FS side effect the plain ledger fold does not have")
	devSessionDays := fs.Float64("dev-session-days", 7, "with --dev-sessions, only include transcripts modified within N days")
	devSessionMax := fs.Int("dev-session-max", 40, "with --dev-sessions, maximum recent transcripts to analyze")
	devSessionAllNamespaces := fs.Bool("dev-sessions-all-namespaces", false, "with --dev-sessions, include transcripts from every project namespace instead of just this workspace")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue report: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)
	usage := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(*usageLedger), *since)

	now := cachevalueReportNow()
	report := cachevaluereport.FoldTwoTrackWithUsage(track1, track2, usage, now, cachevaluereport.FleetBenefitOptions{
		ContextBudgetTokens: *contextBudget,
	})
	report.Since = *since
	recallDebit, err := cachevaluereport.ReadRecallInjectionDebit(*recallInjectionLedger)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachevalue report: recall injection ledger: %v\n", err)
		return 1
	}
	report.RecallInjectionDebit = recallDebit
	if *devSessions {
		devRep := foldDevSessionBenefit(*devSessionDays, *devSessionMax, *devSessionAllNamespaces, now)
		report.DevSessionBenefit = &devRep
	}

	if *asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue report: marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	if *markdown {
		fmt.Fprint(stdout, cachevaluereport.RenderTwoTrackMarkdown(report))
		return 0
	}
	fmt.Fprint(stdout, cachevaluereport.RenderTwoTrack(report))
	return 0
}

// filterTrack1Since drops WITNESSED rows dated before `since` (empty since = keep
// all). The date compare is lexical on the YYYY-MM-DD string, which orders
// chronologically.
func filterTrack1Since(rows []cachevalueledger.Row, since string) []cachevalueledger.Row {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}

// filterTrack2Since drops OBSERVED-$ rows dated before `since` (empty since = keep all).
func filterTrack2Since(rows []cachevaluereport.SavingsRow, since string) []cachevaluereport.SavingsRow {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}

// filterGatewayUsageSince drops durable gateway-usage rows before since. Rows can be
// generated by old builds with only UnixMillis or by current builds with GeneratedAt; accept
// either timestamp and fall open for rows without a date so a partially-populated ledger does
// not disappear from an all-time report.
func filterGatewayUsageSince(rows []gatewayusageledger.Row, since string) []gatewayusageledger.Row {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		d := gatewayUsageRowDate(r)
		if d == "" || d >= since {
			out = append(out, r)
		}
	}
	return out
}

// foldDevSessionBenefit discovers + analyzes recent real Claude Code session transcripts
// (scoped to this workspace's namespace unless allNamespaces is set) and folds them into the
// Track-3 dev-session lens. A discovery error folds to the honest empty report rather than
// failing the whole `cachevalue report` command — this lens is additive, never load-bearing.
func foldDevSessionBenefit(sinceDays float64, max int, allNamespaces bool, now time.Time) cachevaluereport.DevSessionBenefitReport {
	nsPrefix := ""
	if !allNamespaces {
		if cwd, err := os.Getwd(); err == nil {
			nsPrefix = sessionaudit.ProjectNamespace(cwd)
		}
	}
	var since *float64
	if sinceDays >= 0 {
		v := sinceDays
		since = &v
	}
	recs, err := sessionaudit.Discover(sessionaudit.DiscoverOptions{SinceDays: since, NamespacePrefix: nsPrefix})
	if err != nil {
		return cachevaluereport.DevSessionBenefitReport{Finding: fmt.Sprintf("dev-session discovery error: %v", err)}
	}
	if max > 0 && len(recs) > max {
		recs = recs[:max]
	}
	sessions := make([]sessionaudit.Session, 0, len(recs))
	for _, rec := range recs {
		if rec.Kind == "subagent" {
			continue
		}
		sessions = append(sessions, sessionaudit.Analyze(rec.Path))
	}
	return cachevaluereport.FoldDevSessionBenefit(sessions, now)
}

func gatewayUsageRowDate(row gatewayusageledger.Row) string {
	if row.GeneratedAt != "" {
		if t, err := time.Parse(time.RFC3339, row.GeneratedAt); err == nil {
			return t.UTC().Format("2006-01-02")
		}
	}
	if row.UnixMillis > 0 {
		return time.UnixMilli(row.UnixMillis).UTC().Format("2006-01-02")
	}
	return ""
}

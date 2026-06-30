package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

const cacheFrontierReviewSchema = "fak-cache-frontier-review/1"

type cacheFrontierReviewRow struct {
	Schema           string              `json:"schema"`
	Date             string              `json:"date"`
	SourceMarkdown   string              `json:"source_markdown,omitempty"`
	EvidenceCommands []string            `json:"evidence_commands"`
	Track1           cacheFrontierTrack1 `json:"track1_witnessed_kernel"`
	Track2           cacheFrontierTrack2 `json:"track2_observed_usd"`
	DemoWalkthrough  string              `json:"demo_walkthrough"`
	Gaps             []string            `json:"gaps"`
	NextActions      []string            `json:"next_actions"`
	Verdict          string              `json:"verdict"`
}

type cacheFrontierTrack1 struct {
	Verdict                string         `json:"verdict"`
	Thin                   bool           `json:"thin"`
	TotalSessions          int            `json:"total_sessions"`
	MultiTurnSessions      int            `json:"multi_turn_sessions"`
	SingleTurnSessions     int            `json:"single_turn_sessions"`
	TotalTurns             uint64         `json:"total_turns"`
	MultiTurnTurns         uint64         `json:"multi_turn_turns"`
	PromptTokens           uint64         `json:"prompt_tokens"`
	ReusedTokens           uint64         `json:"reused_tokens"`
	GatePromptTokens       uint64         `json:"gate_prompt_tokens"`
	GateReusedTokens       uint64         `json:"gate_reused_tokens"`
	RealizedReuseRatio     float64        `json:"realized_reuse_ratio"`
	SessionTypeMix         map[string]int `json:"session_type_mix"`
	PublishableValueFamily string         `json:"publishable_value_family"`
}

type cacheFrontierTrack2 struct {
	Present bool   `json:"present"`
	Ledger  string `json:"ledger"`
	Reason  string `json:"reason,omitempty"`
}

func runCachevalueReview(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue review", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ ledger")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	date := fs.String("date", time.Now().UTC().Format("2006-01-02"), "review date (YYYY-MM-DD)")
	sourceMarkdown := fs.String("source-markdown", "", "relative markdown review path, e.g. reviews/2026-06-29.md")
	multiAgentDogfood := fs.Bool("multi-agent-dogfood", false, "mark the recurring multi-agent geometry as witnessed for this review")
	o1QuerySession := fs.Bool("o1-query-session", false, "mark an O(1) memory query over a real fak session as witnessed for this review")
	appendLedger := fs.String("append-ledger", "", "append the machine-readable review row to this JSONL ledger")
	markdownOut := fs.String("markdown-out", "", "write the generated human review markdown to this path")
	asJSON := fs.Bool("json", false, "emit one appendable JSONL row instead of markdown")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue review: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}
	if _, err := time.Parse("2006-01-02", *date); err != nil {
		fmt.Fprintf(stderr, "fak cachevalue review: --date must be YYYY-MM-DD: %v\n", err)
		return 2
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)
	report := cachevaluereport.FoldTwoTrack(track1, track2, time.Now().UTC())
	report.Since = *since

	row := buildCacheFrontierReview(report, *date, *since, *ledger, *savingsLedger, *sourceMarkdown, *multiAgentDogfood, *o1QuerySession)
	if err := rejectSameCacheFrontierOutput(*appendLedger, *markdownOut); err != nil {
		fmt.Fprintf(stderr, "fak cachevalue review: %v\n", err)
		return 2
	}
	if *appendLedger != "" {
		if err := appendCacheFrontierReviewLedger(*appendLedger, row); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue review: append --append-ledger: %v\n", err)
			return 1
		}
	}
	if *markdownOut != "" {
		if err := writeCacheFrontierReviewMarkdown(*markdownOut, row); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue review: write --markdown-out: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		b, err := json.Marshal(row)
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue review: marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprint(stdout, renderCacheFrontierReview(row))
	return 0
}

func buildCacheFrontierReview(report cachevaluereport.TwoTrackReport, date, since, track1Ledger, track2Ledger, sourceMarkdown string, multiAgentDogfood, o1QuerySession bool) cacheFrontierReviewRow {
	t1 := summarizeCacheFrontierTrack1(report.Track1)
	t2Present := len(report.Track2) > 0
	t2 := cacheFrontierTrack2{Present: t2Present, Ledger: track2Ledger}
	if !t2Present {
		t2.Reason = "ledger absent or empty; provider-dollar track has no OBSERVED rows yet"
	}

	gaps := cacheFrontierGaps(t1, t2Present, multiAgentDogfood, o1QuerySession)
	row := cacheFrontierReviewRow{
		Schema:         cacheFrontierReviewSchema,
		Date:           date,
		SourceMarkdown: sourceMarkdown,
		EvidenceCommands: []string{
			"go run ./cmd/fak nightrun score --json",
			cacheValueReportCommand(since),
		},
		Track1:          t1,
		Track2:          t2,
		DemoWalkthrough: "../run-the-demos.md#cache-frontier-walkthrough",
		Gaps:            gaps,
		NextActions:     cacheFrontierNextActions(gaps),
		Verdict:         cacheFrontierVerdict(gaps),
	}
	_ = track1Ledger // kept in the signature to make the two-ledger provenance explicit.
	return row
}

func summarizeCacheFrontierTrack1(report cachevaluereport.Report) cacheFrontierTrack1 {
	out := cacheFrontierTrack1{
		Verdict:                report.Verdict,
		TotalSessions:          report.TotalSessions,
		MultiTurnSessions:      report.MultiTurnSessions,
		SingleTurnSessions:     report.TotalSessions - report.MultiTurnSessions,
		RealizedReuseRatio:     report.LatestReuseRatio,
		SessionTypeMix:         map[string]int{},
		PublishableValueFamily: report.PublishableValueFamily,
	}
	for _, b := range report.Buckets {
		out.TotalTurns += b.Turns
		out.MultiTurnTurns += b.MultiTurnTurns
		out.PromptTokens += b.PromptTokens
		out.ReusedTokens += b.ReusedTokens
		out.GatePromptTokens += b.GatePromptTokens
		out.GateReusedTokens += b.GateReusedTokens
		if b.Thin {
			out.Thin = true
		}
		for k, v := range b.BySessionType {
			out.SessionTypeMix[k] += v
		}
	}
	return out
}

func cacheValueReportCommand(since string) string {
	if since == "" {
		return "go run ./cmd/fak cachevalue report --json"
	}
	return "go run ./cmd/fak cachevalue report --since " + since + " --json"
}

func cacheFrontierGaps(t1 cacheFrontierTrack1, track2Present, multiAgentDogfood, o1QuerySession bool) []string {
	var gaps []string
	if t1.Thin || t1.MultiTurnSessions == 0 {
		gaps = append(gaps, "thin_track1_corpus")
	}
	if t1.SessionTypeMix["guard"] == 0 && t1.SessionTypeMix["serve"] == 0 {
		gaps = append(gaps, "no_guard_serve_dogfood_rows")
	}
	if !track2Present {
		gaps = append(gaps, "no_track2_provider_dollar_rows")
	}
	if !multiAgentDogfood {
		gaps = append(gaps, "multi_agent_geometry_not_recurring_dogfood")
	}
	if !o1QuerySession {
		gaps = append(gaps, "o1_query_not_yet_real_session_workflow")
	}
	return gaps
}

func cacheFrontierNextActions(gaps []string) []string {
	actionByGap := map[string]string{
		"thin_track1_corpus":                         "accumulate more multi-turn cache-value rows",
		"no_guard_serve_dogfood_rows":                "capture multi-turn guard/serve cache-value rows from the dev loop",
		"no_track2_provider_dollar_rows":             "wire OBSERVED provider-dollar append",
		"multi_agent_geometry_not_recurring_dogfood": "record recurring multi-agent dogfood geometry",
		"o1_query_not_yet_real_session_workflow":     "query a real fak session through the chosen O(1) memory path",
	}
	out := make([]string, 0, len(gaps))
	for _, gap := range gaps {
		if action := actionByGap[gap]; action != "" {
			out = append(out, action)
		}
	}
	return out
}

func cacheFrontierVerdict(gaps []string) string {
	if len(gaps) == 0 {
		return "PRODUCT_DOGFOOD_READY"
	}
	return "VISIBLE_BUT_NOT_YET_PRODUCT_DOGFOOD"
}

func renderCacheFrontierReview(row cacheFrontierReviewRow) string {
	var b strings.Builder
	fmt.Fprintf(&b, "cache-frontier review - %s\n", row.Date)
	fmt.Fprintf(&b, "verdict: %s\n\n", row.Verdict)
	fmt.Fprintf(&b, "1. What did we use ourselves this week?\n")
	fmt.Fprintf(&b, "   Track 1: %s, reuse %.3f over %d multi-turn session(s), thin=%t\n",
		row.Track1.Verdict, row.Track1.RealizedReuseRatio, row.Track1.MultiTurnSessions, row.Track1.Thin)
	fmt.Fprintf(&b, "   session types: %s\n", renderSessionTypeMix(row.Track1.SessionTypeMix))
	if row.Track2.Present {
		fmt.Fprintf(&b, "   Track 2: OBSERVED-$ rows present in %s\n", row.Track2.Ledger)
	} else {
		fmt.Fprintf(&b, "   Track 2: missing (%s)\n", row.Track2.Reason)
	}
	fmt.Fprintf(&b, "\n2. What can a new person demo this week?\n")
	fmt.Fprintf(&b, "   %s\n", row.DemoWalkthrough)
	fmt.Fprintf(&b, "\n3. What is the next missing witness or product surface?\n")
	for i, gap := range row.Gaps {
		action := ""
		if i < len(row.NextActions) {
			action = " - " + row.NextActions[i]
		}
		fmt.Fprintf(&b, "   - %s%s\n", gap, action)
	}
	return b.String()
}

func rejectSameCacheFrontierOutput(appendLedger, markdownOut string) error {
	if appendLedger == "" || markdownOut == "" {
		return nil
	}
	left, err := filepath.Abs(appendLedger)
	if err != nil {
		return fmt.Errorf("resolve --append-ledger: %w", err)
	}
	right, err := filepath.Abs(markdownOut)
	if err != nil {
		return fmt.Errorf("resolve --markdown-out: %w", err)
	}
	if filepath.Clean(left) == filepath.Clean(right) {
		return fmt.Errorf("--append-ledger and --markdown-out must be different paths")
	}
	return nil
}

func appendCacheFrontierReviewLedger(path string, row cacheFrontierReviewRow) error {
	if err := ensureCacheFrontierParent(path); err != nil {
		return err
	}
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func writeCacheFrontierReviewMarkdown(path string, row cacheFrontierReviewRow) error {
	if err := ensureCacheFrontierParent(path); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(renderCacheFrontierReviewMarkdown(row)), 0o644)
}

func ensureCacheFrontierParent(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0o755)
}

func renderCacheFrontierReviewMarkdown(row cacheFrontierReviewRow) string {
	var b strings.Builder
	fmt.Fprintf(&b, "---\n")
	fmt.Fprintf(&b, "title: \"Cache frontier weekly review - %s\"\n", row.Date)
	fmt.Fprintf(&b, "description: \"Generated cache-frontier review from cache-value ledgers.\"\n")
	fmt.Fprintf(&b, "---\n\n")
	fmt.Fprintf(&b, "# Cache frontier weekly review - %s\n\n", row.Date)
	fmt.Fprintf(&b, "Generated by `fak cachevalue review` from the current cache-value ledgers. ")
	fmt.Fprintf(&b, "The matching machine row uses schema `%s`", row.Schema)
	if row.SourceMarkdown != "" {
		fmt.Fprintf(&b, " and records `source_markdown=%s`", row.SourceMarkdown)
	}
	fmt.Fprintf(&b, ".\n\n")
	fmt.Fprintf(&b, "## Evidence commands\n\n")
	fmt.Fprintf(&b, "```bash\n")
	for _, cmd := range row.EvidenceCommands {
		fmt.Fprintf(&b, "%s\n", cmd)
	}
	fmt.Fprintf(&b, "```\n\n")
	fmt.Fprintf(&b, "## 1. What did we use ourselves this week?\n\n")
	fmt.Fprintf(&b, "Track 1 has verdict `%s` with realized reuse ratio `%.6f`. The corpus is marked `thin=%t`.\n\n",
		row.Track1.Verdict, row.Track1.RealizedReuseRatio, row.Track1.Thin)
	fmt.Fprintf(&b, "| Field | Value |\n")
	fmt.Fprintf(&b, "|---|---:|\n")
	fmt.Fprintf(&b, "| Total cache-value sessions | %d |\n", row.Track1.TotalSessions)
	fmt.Fprintf(&b, "| Multi-turn sessions | %d |\n", row.Track1.MultiTurnSessions)
	fmt.Fprintf(&b, "| Single-turn sessions | %d |\n", row.Track1.SingleTurnSessions)
	fmt.Fprintf(&b, "| Total turns | %d |\n", row.Track1.TotalTurns)
	fmt.Fprintf(&b, "| Multi-turn turns | %d |\n", row.Track1.MultiTurnTurns)
	fmt.Fprintf(&b, "| Prompt tokens | %d |\n", row.Track1.PromptTokens)
	fmt.Fprintf(&b, "| Reused tokens | %d |\n", row.Track1.ReusedTokens)
	fmt.Fprintf(&b, "| Gate prompt tokens | %d |\n", row.Track1.GatePromptTokens)
	fmt.Fprintf(&b, "| Gate reused tokens | %d |\n", row.Track1.GateReusedTokens)
	fmt.Fprintf(&b, "| Session type mix | `%s` |\n\n", renderSessionTypeMix(row.Track1.SessionTypeMix))
	if row.Track2.Present {
		fmt.Fprintf(&b, "Track 2 OBSERVED-dollar rows are present in `%s`.\n\n", row.Track2.Ledger)
	} else {
		fmt.Fprintf(&b, "Track 2 is missing: %s.\n\n", row.Track2.Reason)
	}
	fmt.Fprintf(&b, "## 2. What can a new person demo this week?\n\n")
	fmt.Fprintf(&b, "The cache-frontier walkthrough is `%s`.\n\n", row.DemoWalkthrough)
	fmt.Fprintf(&b, "## 3. What is the next missing witness or product surface?\n\n")
	if len(row.Gaps) == 0 {
		fmt.Fprintf(&b, "No open gaps were found by the review fold.\n\n")
	} else {
		fmt.Fprintf(&b, "| Gap | Next action |\n")
		fmt.Fprintf(&b, "|---|---|\n")
		for i, gap := range row.Gaps {
			action := ""
			if i < len(row.NextActions) {
				action = row.NextActions[i]
			}
			fmt.Fprintf(&b, "| `%s` | %s |\n", gap, action)
		}
		fmt.Fprintf(&b, "\n")
	}
	fmt.Fprintf(&b, "## Verdict\n\n")
	fmt.Fprintf(&b, "`%s`\n", row.Verdict)
	return b.String()
}

func renderSessionTypeMix(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, m[k]))
	}
	return strings.Join(parts, ", ")
}

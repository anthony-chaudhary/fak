package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/sessionmine"
	"github.com/anthony-chaudhary/fak/internal/worktype"
	"io"
	"os"
	"sort"
	"strings"
)

type worktypeSpendClass struct {
	SessionID   string   `json:"session_id,omitempty"`
	TraceID     string   `json:"trace_id,omitempty"`
	PatternID   string   `json:"pattern_id"`
	Subpatterns []string `json:"subpatterns,omitempty"`
}

func runWorktypeSpend(out, errout io.Writer, args []string) int {
	fs := flag.NewFlagSet("worktype attribution", flag.ContinueOnError)
	fs.SetOutput(errout)
	usage := fs.String("usage-ledger", "", "gateway usage ledger")
	classes := fs.String("classifications", "", "fak-worktype/1 NDJSON")
	outcomes := fs.String("outcomes", "", "fak-session-outcomes/1 NDJSON")
	js := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	if *usage == "" {
		fmt.Fprintln(errout, "worktype attribution: --usage-ledger is required")
		return 2
	}
	rows, e := joinWorktypeSpend(*usage, *classes, *outcomes)
	if e != nil {
		fmt.Fprintln(errout, e)
		return 1
	}
	r := worktype.FoldSpend(rows, worktype.SeedPatternIDs())
	if *js {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
	} else {
		fmt.Fprint(out, formatWorktypeSpend(r))
	}
	return 0
}
func joinWorktypeSpend(usage, classes, outcomes string) ([]worktype.SpendRow, error) {
	cm := map[string]worktypeSpendClass{}
	om := map[string]sessionmine.SessionOutcome{}
	if e := scanSpendJSONL(classes, func(b []byte) error {
		var x worktypeSpendClass
		if e := json.Unmarshal(b, &x); e != nil {
			return e
		}
		id := x.TraceID
		if id == "" {
			id = x.SessionID
		}
		cm[id] = x
		return nil
	}); e != nil {
		return nil, e
	}
	if e := scanSpendJSONL(outcomes, func(b []byte) error {
		var x sessionmine.SessionOutcome
		if e := json.Unmarshal(b, &x); e != nil {
			return e
		}
		om[x.SessionID] = x
		return nil
	}); e != nil {
		return nil, e
	}
	latest := map[string]gatewayusageledger.Row{}
	for _, x := range gatewayusageledger.ReadLedgerFile(usage) {
		if x.SessionID != "" && latest[x.SessionID].UnixMillis <= x.UnixMillis {
			latest[x.SessionID] = x
		}
	}
	ids := make([]string, 0, len(latest))
	for id := range latest {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	rows := make([]worktype.SpendRow, 0, len(ids))
	for _, id := range ids {
		u := latest[id].Counters
		c := cm[id]
		o := om[id]
		rows = append(rows, worktype.SpendRow{SessionID: id, TraceID: id, PatternID: c.PatternID, Subpatterns: c.Subpatterns, Tokens: u.InputTokens + u.OutputTokens + u.CachedPromptTokens + u.CacheCreationTokens, Cost: float64(u.InputTokens) + float64(u.OutputTokens) + float64(u.CachedPromptTokens)*cacheprice.ReadMultiplier + float64(u.CacheCreationTokens)*cacheprice.Write5mMultiplier, Outcome: normalizeSpendOutcome(o.Outcome), OutcomeProof: o.Provenance})
	}
	return rows, nil
}
func scanSpendJSONL(path string, fn func([]byte) error) error {
	if path == "" {
		return nil
	}
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if strings.TrimSpace(s.Text()) != "" {
			if e := fn(s.Bytes()); e != nil {
				return e
			}
		}
	}
	return s.Err()
}
func normalizeSpendOutcome(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "shipped_commit", "accepted_witness":
		return "accepted_witness"
	case "no_change":
		return "no_change"
	case "failed", "cancelled", "lost", "reaped":
		return "failed"
	default:
		return "unknown"
	}
}
func formatWorktypeSpend(r worktype.SpendReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "worktype attribution — covered %d/%d sessions, %d/%d tokens (no extrapolation)\n", r.Coverage.ClassifiedRows, r.Coverage.RowCount, r.Coverage.CoveredTokens, r.Coverage.TotalTokens)
	if len(r.Groups) > 0 {
		g := r.Groups[0]
		fmt.Fprintf(&b, "highest spend: %s cost=%.2f %s accepted-witness=%.1f%%\n", g.PatternID, g.Cost, r.CostUnit, g.AcceptedRate*100)
	}
	return b.String()
}

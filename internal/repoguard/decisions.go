package repoguard

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The decision journal is the guard's durable value witness. Every live refusal
// the hook makes is otherwise ephemeral — the deny JSON is consumed by the
// harness and the stderr line scrolls away — so a session can be saved from a
// destructive out-of-tree write and leave no trace anyone can count later. The
// journal fixes that: one append-only JSONL row per enforced (or advisory)
// finding, so the value the guard delivered over a long session is a fact on
// disk, not a claim.
//
// Fail-OPEN is the invariant everywhere here: a journal that cannot be written
// must never change the guard's decision or wedge the tool call. The hook has
// already decided by the time we record; recording is best-effort.

// DecisionRecordSchema versions the row shape so a reader can evolve.
const DecisionRecordSchema = "repoguard.decision/v1"

// DefaultDecisionJournalRel is the workspace-relative path the hook appends to
// when no explicit override is set.
const DefaultDecisionJournalRel = ".fak/repoguard/decisions.jsonl"

// DecisionRecord is one durable row: a single finding the guard acted on, with
// enough context to count value (reason), attribute it (session, tool), and show
// what was stopped (target/resolved/why). Ts is an injected RFC3339 timestamp so
// the recorder stays deterministic and testable — the caller owns the clock.
type DecisionRecord struct {
	Schema   string `json:"schema"`
	Ts       string `json:"ts,omitempty"`
	Session  string `json:"session,omitempty"`
	Tool     string `json:"tool"`
	Decision string `json:"decision"` // "deny" | "advisory" | "record"
	Mode     string `json:"mode"`     // "enforce" | "warn"
	Reason   string `json:"reason"`
	Op       string `json:"op,omitempty"`
	Target   string `json:"target,omitempty"`
	Resolved string `json:"resolved,omitempty"`
	Why      string `json:"why,omitempty"`
}

// DecisionsFromViolations projects the guard's findings into durable rows. One
// row per violation so a multi-target command is fully accounted. severityOf maps
// each violation's reason to the severity the hook ACTUALLY applied (including any
// per-reason override), so the row's decision label — "record" | "advisory" |
// "deny" — reflects the real decision, not a recomputed default. A SeverityOff
// finding carries no row (the caller drops it before recording).
func DecisionsFromViolations(violations []Violation, tool, session, mode, ts string, severityOf func(reason string) Severity) []DecisionRecord {
	if len(violations) == 0 {
		return nil
	}
	rows := make([]DecisionRecord, 0, len(violations))
	for _, v := range violations {
		label := severityOf(v.Reason).DecisionLabel()
		if label == "" {
			continue // SeverityOff — nothing to record
		}
		rows = append(rows, DecisionRecord{
			Schema:   DecisionRecordSchema,
			Ts:       ts,
			Session:  session,
			Tool:     tool,
			Decision: label,
			Mode:     mode,
			Reason:   v.Reason,
			Op:       v.Op,
			Target:   v.Target,
			Resolved: v.Resolved,
			Why:      v.Why,
		})
	}
	return rows
}

// AppendDecisions appends rows to the JSONL journal at path, creating the parent
// directory if needed. Returns an error for the caller to swallow — never for the
// caller to act on. A zero-row slice is a no-op (no empty file is created).
func AppendDecisions(path string, rows []DecisionRecord) error {
	if len(rows) == 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // keep `->` in reason text readable, matching the deny JSON
	for i := range rows {
		if err := enc.Encode(rows[i]); err != nil {
			return err
		}
	}
	return w.Flush()
}

// DecisionJournalPath resolves the journal path: an explicit override
// (FAK_REPO_GUARD_DECISIONS) wins, else DefaultDecisionJournalRel under the
// workspace root.
func DecisionJournalPath(workspaceRoot string) string {
	if p := strings.TrimSpace(os.Getenv("FAK_REPO_GUARD_DECISIONS")); p != "" {
		return p
	}
	return filepath.Join(workspaceRoot, filepath.FromSlash(DefaultDecisionJournalRel))
}

// DecisionSummary is the accumulated-value view read back from the journal.
type DecisionSummary struct {
	Total      int              `json:"total"`
	Denies     int              `json:"denies"`
	Advisories int              `json:"advisories"`
	Recorded   int              `json:"recorded"`
	ByReason   map[string]int   `json:"by_reason"`
	FirstTs    string           `json:"first_ts,omitempty"`
	LastTs     string           `json:"last_ts,omitempty"`
	Recent     []DecisionRecord `json:"recent,omitempty"`
}

// SummarizeDecisions folds a journal reader into counts by reason plus the most
// recent rows (up to recentN). A malformed line is skipped, not fatal — a partial
// journal must still yield a usable value view.
func SummarizeDecisions(r io.Reader, recentN int) (DecisionSummary, error) {
	sum := DecisionSummary{ByReason: map[string]int{}}
	var all []DecisionRecord
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var rec DecisionRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue // tolerate a torn/foreign line
		}
		sum.Total++
		switch rec.Decision {
		case "advisory":
			sum.Advisories++
		case "record":
			sum.Recorded++
		default:
			sum.Denies++
		}
		if rec.Reason != "" {
			sum.ByReason[rec.Reason]++
		}
		if rec.Ts != "" {
			if sum.FirstTs == "" || rec.Ts < sum.FirstTs {
				sum.FirstTs = rec.Ts
			}
			if rec.Ts > sum.LastTs {
				sum.LastTs = rec.Ts
			}
		}
		all = append(all, rec)
	}
	if err := sc.Err(); err != nil {
		return sum, err
	}
	if recentN > 0 && len(all) > 0 {
		start := len(all) - recentN
		if start < 0 {
			start = 0
		}
		sum.Recent = append(sum.Recent, all[start:]...)
	}
	return sum, nil
}

// SummarizeDecisionsFile opens path and summarizes it. A missing journal is a
// valid empty summary, not an error — a session that never tripped the guard
// legitimately has zero rows.
func SummarizeDecisionsFile(path string, recentN int) (DecisionSummary, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return DecisionSummary{ByReason: map[string]int{}}, nil
		}
		return DecisionSummary{ByReason: map[string]int{}}, err
	}
	defer f.Close()
	return SummarizeDecisions(f, recentN)
}

// RenderSummary formats the accumulated-value view for a human reader.
func RenderSummary(sum DecisionSummary) string {
	var b strings.Builder
	if sum.Total == 0 {
		return "repo_guard: no recorded decisions yet (the guard has denied nothing this journal covers)."
	}
	fmt.Fprintf(&b, "repo_guard value: %d finding(s) recorded — %d denied, %d advisory, %d silent-record",
		sum.Total, sum.Denies, sum.Advisories, sum.Recorded)
	if sum.FirstTs != "" {
		fmt.Fprintf(&b, " (%s .. %s)", sum.FirstTs, sum.LastTs)
	}
	b.WriteString("\n")
	reasons := make([]string, 0, len(sum.ByReason))
	for k := range sum.ByReason {
		reasons = append(reasons, k)
	}
	sort.Slice(reasons, func(i, j int) bool {
		if sum.ByReason[reasons[i]] != sum.ByReason[reasons[j]] {
			return sum.ByReason[reasons[i]] > sum.ByReason[reasons[j]]
		}
		return reasons[i] < reasons[j]
	})
	for _, r := range reasons {
		fmt.Fprintf(&b, "  %-24s %d\n", r, sum.ByReason[r])
	}
	for _, rec := range sum.Recent {
		target := rec.Resolved
		if target == "" {
			target = rec.Target
		}
		fmt.Fprintf(&b, "  - [%s] %s %s: %s\n", rec.Decision, rec.Tool, rec.Reason, target)
	}
	return strings.TrimRight(b.String(), "\n")
}

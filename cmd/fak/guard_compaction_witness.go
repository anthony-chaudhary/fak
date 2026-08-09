package main

// guard_compaction_witness.go — the durable per-session compaction-health witness (#3152).
//
// THE GAP THIS CLOSES. Compaction health (fired / bailed / anchor-starved / shed / budget)
// lived only on the live gateway's counters (internal/gateway/metrics.go, folded by
// AdjudicationSummary) and was surfaced only through `/debug/vars` and the guard exit banner.
// Both die with the process. So once a guarded session ended there was NO checkable answer to
// "did compaction fire for THAT session?" — an auditor reaching for a number found only the
// fleet-cumulative usage ledgers, and reading a fleet `cache_read` next to an absent per-session
// shed is exactly the #3095 anti-pattern (never ratio shed against cache_read).
//
// WHY THE NUMBERS ARE HONESTLY PER-SESSION. `fak guard` constructs its OWN gateway.Server per
// launch (guard.go, gateway.New) and tears it down in finishGuardChildAndReport, so that
// server's counters are cumulative over exactly one guarded session. The counters were never
// wrong — they were merely UNDURABLE. This file changes nothing about how they are measured; it
// pins them, keyed by session id, to an append-only JSONL that outlives the process. That is
// the whole delta, and it is why the row and the banner's compaction block always agree.
//
// SCOPE. This is the post-hoc WITNESS OF RECORD, deliberately distinct from its two siblings:
// #3099 is the LIVE in-session verdict, #3095 is the honest shed ACCOUNTING. A row here answers
// only "what did compaction do for this session", and answers it without a live gateway.
//
// The ledger is the gitignored runtime path every other guard producer writes
// (.fak/nightrun/..., beside harness-resources.jsonl), and every entry point is BEST-EFFORT by
// contract, exactly like guardToolprocSummaryLine and guardNegframeSummary: an unwritable
// ledger, an unreadable one, or a malformed row all degrade to silence. An observability
// witness must never fail a session's exit.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	// guardCompactionWitnessSchema stamps every row so a later reader can tell which shape
	// wrote it. It moves only when a field changes meaning — appending a field does not.
	guardCompactionWitnessSchema = 1

	// guardCompactionWitnessLedgerRel is the gitignored runtime ledger, beside the
	// harness-resources ledger the sibling exit-summary producer appends to.
	guardCompactionWitnessLedgerRel = ".fak/nightrun/compaction-health.jsonl"

	// The two anchor modes --compact-anchor-head selects between. "head" is the DEFAULT-ON
	// #1407 fix (protect the stable system/tools head); "first-breakpoint" is the legacy
	// warm-only anchor a session gets with --compact-anchor-head=false. The row carries which
	// one ran because an anchor-starved count is uninterpretable without it: starvation under
	// "first-breakpoint" is the known #1407 trap, while starvation under "head" means the
	// traffic carries no stable head at all — operationally different findings.
	guardCompactionAnchorHead  = "head"
	guardCompactionAnchorFirst = "first-breakpoint"
)

// guardCompactionAnchorModeName maps the --compact-anchor-head flag onto the row's vocabulary.
func guardCompactionAnchorModeName(headAnchored bool) string {
	if headAnchored {
		return guardCompactionAnchorHead
	}
	return guardCompactionAnchorFirst
}

// guardCompactionAnchorMode records the anchor mode THIS launch runs under. cmdGuard sets it
// from the resolved --compact-anchor-head flag at boot; the exit funnel reads it when it stamps
// the row. A package var rather than a threaded parameter for the same reason
// guardAllowSessionScopeID is one: the exit funnel is reached through several call paths and
// none of them should grow a parameter for an observability field.
var guardCompactionAnchorMode = guardCompactionAnchorHead

// setGuardCompactionAnchorMode pins the anchor mode for this launch's witness row.
func setGuardCompactionAnchorMode(headAnchored bool) {
	guardCompactionAnchorMode = guardCompactionAnchorModeName(headAnchored)
}

// guardCompactionWitness is the durable per-session compaction-health row #3152 asks for:
// {session, anchor_mode, fired, bailed, anchor_starved, shed_tokens, budget, cache_read_at_fire}
// plus the two counts that keep the headline honest. Off separates "compaction never ran" from
// "compaction ran and shed nothing", and SolvencyForced is the subset of Fired the burst
// economics REFUSED and context solvency overrode — those must never be booked as cache wins,
// so a witness that folded them into Fired would overstate the win.
//
// CacheReadAtFire is the OBSERVED provider cache_read at this session's compaction fires — the
// warm witness a shed must be priced against. It is carried on the row precisely so an auditor
// never has to reach for a fleet-cumulative cache_read to interpret ShedTokens.
type guardCompactionWitness struct {
	Schema     int    `json:"schema"`
	RecordedAt string `json:"recorded_at"` // RFC3339 UTC — when the session ended
	Session    string `json:"session"`
	AnchorMode string `json:"anchor_mode"`

	Fired           uint64 `json:"fired"`
	Bailed          uint64 `json:"bailed"`
	Off             uint64 `json:"off"`
	AnchorStarved   uint64 `json:"anchor_starved"`
	SolvencyForced  uint64 `json:"solvency_forced"`
	ShedTokens      uint64 `json:"shed_tokens"`
	Budget          int    `json:"budget"`
	CacheReadAtFire uint64 `json:"cache_read_at_fire"`

	// BailReasons is the witnessed per-reason breakdown of the Bailed lump. Without it "N
	// bailed" is uninterpretable: an all-under_budget bail set is compaction correctly IDLE,
	// while a burst_unprofitable set is compaction refusing on economics.
	BailReasons map[string]uint64 `json:"bail_reasons,omitempty"`
}

// newGuardCompactionWitness folds one session's live summary into the durable row. Pure: no
// clock, no I/O — the caller supplies both, so the fold is unit-testable.
func newGuardCompactionWitness(session, anchorMode string, sum gateway.AdjudicationSummary, now time.Time) guardCompactionWitness {
	row := guardCompactionWitness{
		Schema:          guardCompactionWitnessSchema,
		RecordedAt:      now.UTC().Format(time.RFC3339),
		Session:         strings.TrimSpace(session),
		AnchorMode:      strings.TrimSpace(anchorMode),
		Fired:           sum.CompactionFired,
		Bailed:          sum.CompactionBailed,
		Off:             sum.CompactionOff,
		AnchorStarved:   sum.CompactionAnchorStarved,
		SolvencyForced:  sum.CompactionSolvencyForced,
		ShedTokens:      sum.CompactionShedTokens,
		Budget:          sum.CompactionBudget,
		CacheReadAtFire: sum.CompactionCacheReadTokens,
	}
	// Copy rather than alias: the summary's map belongs to the live gateway, and a durable row
	// must not share mutable state with the process it is meant to outlive.
	if len(sum.CompactionBailReasons) > 0 {
		row.BailReasons = make(map[string]uint64, len(sum.CompactionBailReasons))
		for k, v := range sum.CompactionBailReasons {
			row.BailReasons[k] = v
		}
	}
	return row
}

// appendGuardCompactionWitnessTo appends one row to the JSONL ledger at path, creating the
// parent directory if needed. Append-only by construction (O_APPEND): a later session adds a
// row, it never rewrites history — the same discipline as every other durable ledger here.
func appendGuardCompactionWitnessTo(path string, row guardCompactionWitness) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("compaction witness: empty ledger path")
	}
	line, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

// readGuardCompactionWitnesses reads every row back from the ledger at path, in append order.
// Blank lines are skipped and a malformed line is SKIPPED rather than fatal: this ledger is
// appended concurrently by every guarded session on the host, so a torn final line must never
// cost a reader the intact rows before it.
func readGuardCompactionWitnesses(path string) ([]guardCompactionWitness, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []guardCompactionWitness
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for sc.Scan() {
		b := sc.Bytes()
		if len(strings.TrimSpace(string(b))) == 0 {
			continue
		}
		var row guardCompactionWitness
		if json.Unmarshal(b, &row) != nil {
			continue
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return rows, err
	}
	return rows, nil
}

// latestGuardCompactionWitness returns the NEWEST row for session — the last one appended, since
// the ledger is append-only. This is the join the issue asks for: a session id in, that
// session's real compaction health out, with no live gateway process anywhere.
func latestGuardCompactionWitness(path, session string) (guardCompactionWitness, bool) {
	session = strings.TrimSpace(session)
	if session == "" {
		return guardCompactionWitness{}, false
	}
	rows, err := readGuardCompactionWitnesses(path)
	if err != nil {
		return guardCompactionWitness{}, false
	}
	for i := len(rows) - 1; i >= 0; i-- {
		if rows[i].Session == session {
			return rows[i], true
		}
	}
	return guardCompactionWitness{}, false
}

// formatGuardCompactionWitness renders the exit-summary block FROM the durable row — not from
// the live process counters. That is the point: the operator reads the same bytes a later audit
// will read, so the banner can never disagree with the witness of record.
func formatGuardCompactionWitness(row guardCompactionWitness, path string) string {
	var b strings.Builder
	b.WriteString(guardSection("compaction witness (this session)"))
	b.WriteString(guardRow("session", row.Session))
	b.WriteString(guardRow("anchor mode", row.AnchorMode))
	b.WriteString(guardRow("fired / bailed / off", fmt.Sprintf("%d / %d / %d", row.Fired, row.Bailed, row.Off)))
	b.WriteString(guardRow("shed", fmt.Sprintf("%d tok (budget %d)", row.ShedTokens, row.Budget)))
	// Price the shed against the cache_read OBSERVED AT THIS SESSION'S FIRES — never against a
	// fleet-cumulative cache_read, which is the #3095 anti-pattern this row exists to retire.
	b.WriteString(guardRow("cache_read at fire", fmt.Sprintf("%d tok", row.CacheReadAtFire)))
	if row.AnchorStarved > 0 {
		b.WriteString(guardRow("  ⚠ anchor-starved", fmt.Sprintf("x%d", row.AnchorStarved)))
	}
	if row.SolvencyForced > 0 {
		b.WriteString(guardRow("  solvency-forced", fmt.Sprintf("x%d of %d fired", row.SolvencyForced, row.Fired)))
	}
	b.WriteString(guardNote("durable per-session witness (#3152) — re-read it after this process is gone with `fak guard compaction-witness` or by joining on the session id in " + path))
	return b.String()
}

// recordGuardCompactionWitness is the exit-funnel entry point: pin THIS session's compaction
// health durably, then render the exit-summary block from the row that landed. It returns "" —
// staying silent, never erroring — when there is no session id, no live summary, or the ledger
// could not be written or read back. Rendering only from a row that survived the round trip is
// deliberate: the banner then doubles as proof the witness is actually durable, instead of
// printing numbers that may never have reached disk.
func recordGuardCompactionWitness(path, session string, sum gateway.AdjudicationSummary, now time.Time) string {
	if strings.TrimSpace(session) == "" {
		return ""
	}
	if err := appendGuardCompactionWitnessTo(path, newGuardCompactionWitness(session, guardCompactionAnchorMode, sum, now)); err != nil {
		return ""
	}
	row, ok := latestGuardCompactionWitness(path, session)
	if !ok {
		return ""
	}
	return formatGuardCompactionWitness(row, path)
}

// guardCompactionWitnessLedger resolves the workspace ledger path this launch writes.
func guardCompactionWitnessLedger() string {
	return nightrunLedgerPath(guardCompactionWitnessLedgerRel)
}

package cachevaluereport

import (
	"fmt"
	"sort"
	"time"
)

// CacheDidntHelpSchema tags the "cache didn't help" cohort report (#3652), distinct from
// the audit/score schemas so a consumer reads it apart.
const CacheDidntHelpSchema = "fak-cache-didnt-help-cohort/1"

// CacheDidntHelpEntry is one ACTIVE-but-unhelped session in the cohort: a session where
// managed cache was ACTIVE (it did cache work — a provider prompt-cache read/creation, or a
// compaction fire) yet the session's summed net-$ still came out <= 0. NetUSD is the
// inspectable figure the worklist routes to review; ColdWrite marks the "the write premium
// outweighed the read rebate" case audit.go already counts per-turn.
type CacheDidntHelpEntry struct {
	Date            string  `json:"date"`
	SessionType     string  `json:"session_type,omitempty"`
	Context         string  `json:"context,omitempty"`
	Provider        string  `json:"provider,omitempty"`
	Rows            int     `json:"rows"`
	NetUSD          float64 `json:"net_usd"`
	SavedTokenEquiv float64 `json:"saved_token_equiv"`
	ColdWrite       bool    `json:"cold_write,omitempty"`
	Reason          string  `json:"reason"`
}

// CacheDidntHelpCohort is the ranked worklist of ACTIVE-but-unhelped sessions (#3652). It
// turns the per-turn cold-write net-negative signal audit.go already counts (ColdWriteRows)
// into an actionable per-SESSION worklist: managed cache ran, the net-$ still landed <= 0,
// so route the worst offenders to review (the honest "sometimes caching costs more" cases).
// Unhelped is sorted worst-first (most-negative net-$). It is a REPORT, not a gate — a
// non-empty cohort is a worklist, never a CI failure.
type CacheDidntHelpCohort struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"`

	// ActiveSessions is the denominator: sessions where managed cache was ACTIVE and at
	// least one row carried a dollar figure (a fully dollar-blind session has no net-$ to
	// judge and is excluded from both the count and the cohort).
	ActiveSessions int                   `json:"active_sessions"`
	Unhelped       []CacheDidntHelpEntry `json:"unhelped,omitempty"`

	Verdict    string `json:"verdict"` // CLEAN | COHORT | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action,omitempty"`
}

// sessionKey identifies one session across its per-mechanism rows. A session's savings are
// split into separate rows (provider_prompt_cache vs compaction) that share these labels;
// the cohort sums net-$ across them so a session is judged whole, not per-mechanism.
type sessionKey struct {
	date        string
	sessionType string
	context     string
}

// rowManagedCacheActive reports whether managed cache did work on this row: a provider
// prompt-cache read/creation happened, or a compaction fire did. A row with none of these is
// a cold/inactive session with no cache behavior to hold responsible for a bad net-$.
func rowManagedCacheActive(r SavingsRow) bool {
	return r.CacheReadTokens > 0 || r.CacheCreationTokens > 0 || r.CompactionFired > 0
}

// cacheDidntHelpAgg accumulates one session's rows before it is judged.
type cacheDidntHelpAgg struct {
	provider        string
	rows            int
	netUSD          float64
	savedTokenEquiv float64
	active          bool
	priced          bool // at least one non-dollar-blind row → the session has a net-$ to judge
	coldWrite       bool
}

// FoldCacheDidntHelp folds Track-2 savings rows into the "cache didn't help" cohort (#3652).
// It is PURE and deterministic — bucketing comes from each row's own (date, session_type,
// context); `now` only stamps GeneratedAt. Rows with an unparseable date are skipped
// (mirroring FoldAudit). A session enters the cohort iff managed cache was ACTIVE on it, it
// carried at least one dollar-priced row, and its summed net-$ is <= 0. The cohort is sorted
// most-negative-first so the worst offenders route to review first.
func FoldCacheDidntHelp(rows []SavingsRow, now time.Time) CacheDidntHelpCohort {
	byKey := map[sessionKey]*cacheDidntHelpAgg{}
	var order []sessionKey

	for _, r := range rows {
		normalizeSavingsDimensions(&r)
		if _, err := time.Parse("2006-01-02", r.Date); err != nil {
			continue
		}
		key := sessionKey{date: r.Date, sessionType: r.SessionType, context: r.Context}
		a := byKey[key]
		if a == nil {
			a = &cacheDidntHelpAgg{provider: r.Provider}
			byKey[key] = a
			order = append(order, key)
		}
		a.rows++
		a.netUSD += r.NetUSD
		a.savedTokenEquiv += r.SavedTokenEquiv
		if rowManagedCacheActive(r) {
			a.active = true
		}
		if !isDollarBlindRow(r) {
			a.priced = true
		}
		// The per-turn cold-write signal: a provider row whose write premium outweighs its
		// read rebate (the write is not yet repaid by reuse).
		if r.Mechanism == "provider_prompt_cache" && r.RebateUSD < r.WritePremiumUSD {
			a.coldWrite = true
		}
		if a.provider == "" {
			a.provider = r.Provider
		}
	}

	rep := CacheDidntHelpCohort{
		Schema:      CacheDidntHelpSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Verdict:     "INSUFFICIENT",
	}
	for _, key := range order {
		a := byKey[key]
		if !a.active || !a.priced {
			continue // not an active, dollar-judged managed-cache session
		}
		rep.ActiveSessions++
		if a.netUSD > 0 {
			continue // cache helped (positive net-$)
		}
		rep.Unhelped = append(rep.Unhelped, CacheDidntHelpEntry{
			Date:            key.date,
			SessionType:     key.sessionType,
			Context:         key.context,
			Provider:        a.provider,
			Rows:            a.rows,
			NetUSD:          a.netUSD,
			SavedTokenEquiv: a.savedTokenEquiv,
			ColdWrite:       a.coldWrite,
			Reason:          cacheDidntHelpReason(a),
		})
	}

	// Worst-first: most-negative net-$, then oldest date, then context for determinism.
	sort.SliceStable(rep.Unhelped, func(i, j int) bool {
		a, b := rep.Unhelped[i], rep.Unhelped[j]
		if a.NetUSD != b.NetUSD {
			return a.NetUSD < b.NetUSD
		}
		if a.Date != b.Date {
			return a.Date < b.Date
		}
		return a.Context < b.Context
	})

	switch {
	case rep.ActiveSessions == 0:
		rep.Finding = "INSUFFICIENT — no active, dollar-priced managed-cache session to judge"
	case len(rep.Unhelped) > 0:
		rep.Verdict = "COHORT"
		rep.Finding = fmt.Sprintf("COHORT — %d of %d active managed-cache session(s) netted <= $0 (cache did not pay off)",
			len(rep.Unhelped), rep.ActiveSessions)
		rep.NextAction = "route the worst-first cohort to review: confirm each was genuinely worth caching (long-lived reuse) or accept it as an honest cold/short session where caching costs more than it saves"
	default:
		rep.Verdict = "CLEAN"
		rep.Finding = fmt.Sprintf("CLEAN — all %d active managed-cache session(s) netted > $0", rep.ActiveSessions)
	}
	return rep
}

// cacheDidntHelpReason explains why a session landed in the cohort, naming the cold-write
// case explicitly so the reader knows whether the write premium or the residual spend drove
// the non-positive net.
func cacheDidntHelpReason(a *cacheDidntHelpAgg) string {
	if a.coldWrite {
		return fmt.Sprintf("net $%.4f over %d row(s); cold-write (provider write premium outweighed the read rebate — reuse has not repaid the write)", a.netUSD, a.rows)
	}
	return fmt.Sprintf("net $%.4f over %d row(s); active cache yielded no positive net saving this session", a.netUSD, a.rows)
}

// ScoreCacheDidntHelpFile reads the Track-2 savings ledger at path and folds the cohort. It
// is the file-reading convenience over FoldCacheDidntHelp; a missing/unreadable file yields
// an empty INSUFFICIENT report, never an error, mirroring ReadSavingsLedgerFile.
func ScoreCacheDidntHelpFile(path string, now time.Time) CacheDidntHelpCohort {
	return FoldCacheDidntHelp(ReadSavingsLedgerFile(path), now)
}

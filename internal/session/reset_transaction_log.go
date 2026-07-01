package session

import (
	"strconv"
	"sync"
)

// ResetTransactionLog is the append-only, replayable ledger of budget-triggered reset
// transactions — the runtime-continuity audit trail issue #1582 asks for, one level
// above the single ResetTransaction row a child State carries. A State only ever holds
// the transaction that minted IT (the latest reset in its own lineage); a long-lived
// goal that survives many hidden restarts needs the FULL chain, so a caller (a host's
// reset boundary, e.g. cmd/fak's resetServedSessionOnBudget) appends every transaction
// it produces here in occurrence order. The zero value is a usable empty log. Mirrors
// ctxplan.ObjectiveLog / ctxplan.PageFaultLog in shape (Entries/Latest/Replay/Summary),
// but — unlike those single-goroutine-owned siblings — this log is wired into a served
// gateway's reset hook where concurrent traces can reset at the same time, so it guards
// its own state with a mutex rather than leaving that to the caller.
type ResetTransactionLog struct {
	mu      sync.Mutex
	entries []ResetTransaction
}

// Append records tx as the next entry and returns it unchanged — the one call sites
// need to both persist the row and keep using the value they just built. A zero tx
// (IsZero()) is still recorded: the log is a raw event history, not a filtered one; a
// caller that wants to skip no-op resets should check IsZero() itself before calling.
func (l *ResetTransactionLog) Append(tx ResetTransaction) ResetTransaction {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries = append(l.entries, tx)
	return tx
}

// Entries returns a defensive copy of the logged transactions in occurrence order.
func (l *ResetTransactionLog) Entries() []ResetTransaction {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]ResetTransaction(nil), l.entries...)
}

// Latest returns the most recently appended transaction and whether the log has any
// entries at all. A caller chaining resets across an arbitrarily long-lived goal uses
// this to find "what did the last reset arm", without needing to track it separately.
func (l *ResetTransactionLog) Latest() (ResetTransaction, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.entries) == 0 {
		return ResetTransaction{}, false
	}
	return l.entries[len(l.entries)-1], true
}

// ResetTransactionReplayVerdict is the outcome of replaying one logged entry: whether
// the row is internally well-formed and, when it is not the first entry in the chain,
// whether its OldTrace connects to the prior entry's NewTrace. A ResetTransaction is
// deliberately payload-free (SeedDigest/OmittedSpans carry digests, never transcript
// text), so there is no original content left to re-hash and compare — replay here
// checks the SHAPE of the audit trail is coherent, the same class of non-forgeable
// evidence a git-diff witness gives a commit claim, not a re-derivation of dropped
// bytes the schema exists specifically to avoid persisting.
type ResetTransactionReplayVerdict struct {
	Index       int    `json:"index"`
	OldTrace    string `json:"old_trace,omitempty"`
	NewTrace    string `json:"new_trace,omitempty"`
	WellFormed  bool   `json:"well_formed"`
	ChainLinked bool   `json:"chain_linked"`
	Diverged    bool   `json:"diverged"`
	DivergeNote string `json:"diverge_note,omitempty"`
}

// Replay recomputes the well-formedness and chain-linkage of every logged entry and
// reports any DIVERGED entry, so a caller can tell "is this audit trail coherent" from
// the stored rows themselves instead of trusting that the log was assembled correctly.
// A row is well-formed when it carries the schema token and both trace ids; the chain
// link check only applies from the second entry onward and only when consecutive
// entries share the same lineage (entry N's OldTrace == entry N-1's NewTrace) — a log
// that interleaves independent lineages is expected to show ChainLinked=false for the
// first row of each new lineage, which is not itself a divergence.
func (l *ResetTransactionLog) Replay() (verdicts []ResetTransactionReplayVerdict, allMatch bool) {
	allMatch = true
	var prevNewTrace string
	for i, tx := range l.Entries() {
		v := ResetTransactionReplayVerdict{
			Index:      i,
			OldTrace:   tx.OldTrace,
			NewTrace:   tx.NewTrace,
			WellFormed: tx.Schema == ResetTransactionSchema && tx.OldTrace != "" && tx.NewTrace != "",
		}
		if i == 0 {
			v.ChainLinked = true
		} else {
			v.ChainLinked = tx.OldTrace == prevNewTrace
		}
		if !v.WellFormed {
			v.Diverged = true
			v.DivergeNote = "missing schema token or trace id"
		}
		if v.Diverged {
			allMatch = false
		}
		verdicts = append(verdicts, v)
		prevNewTrace = tx.NewTrace
	}
	return verdicts, allMatch
}

// ResetTransactionSummary folds the log into counts — the O(1) health signal a debug
// surface prints instead of walking every entry (mirrors ObjectiveSummary/PageFaultSummary).
type ResetTransactionSummary struct {
	Total          int `json:"total"`
	WithSeedDigest int `json:"with_seed_digest"`
	WithWarmPrefix int `json:"with_warm_prefix"`
	OmittedSpans   int `json:"omitted_spans"`
}

// Summary computes the aggregate counts over every logged entry.
func (l *ResetTransactionLog) Summary() ResetTransactionSummary {
	entries := l.Entries()
	var s ResetTransactionSummary
	s.Total = len(entries)
	for _, tx := range entries {
		if tx.SeedDigest != "" {
			s.WithSeedDigest++
		}
		if tx.WarmPrefixDigest != "" {
			s.WithWarmPrefix++
		}
		s.OmittedSpans += len(tx.OmittedSpans)
	}
	return s
}

// Explain renders the log as an operator-readable report, in the
// ObjectiveLog.Explain / PageFaultLog.Explain style: one line per transaction plus the
// count footer.
func (l *ResetTransactionLog) Explain() string {
	entries := l.Entries()
	var b []byte
	b = append(b, "session reset-transaction log: "+strconv.Itoa(len(entries))+" reset(s)\n"...)
	for i, tx := range entries {
		b = append(b, strconv.Itoa(i)+": "+tx.OldTrace+" -> "+tx.NewTrace+
			" seed="+shortHash(tx.SeedDigest)+" contributors="+strconv.Itoa(len(tx.Contributors))+
			" omitted="+strconv.Itoa(len(tx.OmittedSpans))+"\n"...)
	}
	return string(b)
}

func shortHash(digest string) string {
	if digest == "" {
		return "(none)"
	}
	if len(digest) <= 12 {
		return digest
	}
	return digest[:12]
}

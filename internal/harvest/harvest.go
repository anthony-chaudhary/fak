// Package harvest is the LabelRow harvester — the data-collection rung of the
// compiled defender-side loop.
//
// THE LOOP. The papers in this space describe an ATTACKER loop (generate attacks,
// measure success). The defender side they leave open is the symmetric loop that
// LEARNS from outcomes: red-team (internal/agentdojo) produces adaptive attacks ->
// the kernel's gates adjudicate them, producing typed VERDICTS -> harvest collects
// each (call, verdict, reason) as a frozen abi.LabelRow -> shipgate
// (internal/shipgate) keeps-or-reverts a candidate policy change only if a
// non-author witness confirms a strict gain (e.g. lower ASR) on that corpus. This
// package is the third arrow: it turns the kernel's own adjudication stream into
// the supervised corpus the syscall-tuned model (and the keep-or-revert gate) train
// against — the LabelRow shape is frozen in the ABI precisely so this corpus cannot
// drift across drivers.
//
// It is a pure Emitter (the abi.Emitter seam): attach it to a kernel and every
// adjudication becomes a labeled example. A non-Allow verdict (deny / quarantine /
// escalate) is a POSITIVE (a catch); an Allow is a NEGATIVE. A call that passed a
// cheap rung but failed an expensive one (the pre-flight ladder's EvRungLabel,
// carrying an explicit typed LabelRow) is the HARD NEGATIVE the ladder was built to
// mine. harvest does not register globally — it is opt-in, attached by the bench /
// the compiled loop, so it never taxes the normal dispatch path.
package harvest

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// defaultMaxCorpusRows bounds how many rows a Corpus retains by default. A
// Harvester attached to a live kernel folds a row per adjudication for as long as
// it stays registered; without a bound the rows slice grows without limit. The
// default holds the corpus to its most RECENT window. A batch caller that needs
// the FULL corpus for a digest or export (e.g. the red-team bench over a fixed
// battery) opts out with SetMaxRows(-1); a long-lived attach point leaves the
// default in place. Below the bound every accessor is byte-identical.
const defaultMaxCorpusRows = 1024

// Corpus is the accumulated labeled training data — a thread-safe append log of
// abi.LabelRow, the frozen supervised signal. Retention is bounded to a recent
// window by default (defaultMaxCorpusRows); see SetMaxRows to tune or disable it.
type Corpus struct {
	mu   sync.Mutex
	rows []abi.LabelRow
	// maxRows bounds the retained window: 0 = the default cap; <0 = unbounded; >0 =
	// a custom cap. See rowCap.
	maxRows int
}

// NewCorpus returns an empty thread-safe corpus ready to collect LabelRows.
func NewCorpus() *Corpus { return &Corpus{} }

// rowCap resolves the effective retained-row cap: (cap, true) when bounded,
// (_, false) when the caller disabled the bound (maxRows<0). Caller holds the lock.
func (c *Corpus) rowCap() (max int, bounded bool) {
	switch {
	case c.maxRows < 0:
		return 0, false
	case c.maxRows == 0:
		return defaultMaxCorpusRows, true
	default:
		return c.maxRows, true
	}
}

// SetMaxRows bounds the corpus to at most n retained rows, evicting the OLDEST
// beyond n as new rows arrive (a recent-window corpus). n<0 disables the bound
// (unbounded — every row retained), n==0 restores the default. A batch caller that
// needs the FULL corpus for a digest/export calls SetMaxRows(-1) before collecting;
// a long-lived attach point leaves the default so the collector cannot grow without
// limit. Re-applies the bound immediately, so lowering it trims existing rows.
// Concurrency-safe.
func (c *Corpus) SetMaxRows(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.maxRows = n
	if max, bounded := c.rowCap(); bounded && len(c.rows) > max {
		c.rows = c.rows[len(c.rows)-max:]
	}
}

func (c *Corpus) add(r abi.LabelRow) {
	c.mu.Lock()
	c.rows = append(c.rows, r)
	if max, bounded := c.rowCap(); bounded && len(c.rows) > max {
		// Drop the oldest rows beyond the window (recent-window retention).
		c.rows = c.rows[len(c.rows)-max:]
	}
	c.mu.Unlock()
}

// Rows returns a snapshot copy of the collected rows.
func (c *Corpus) Rows() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]abi.LabelRow(nil), c.rows...)
}

// Len is the number of collected rows.
func (c *Corpus) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.rows)
}

// filterLocked returns the rows matching pred, under the (already held) corpus
// lock — the shared iterate-and-collect body behind Positives / HardNegatives.
func (c *Corpus) filterLocked(pred func(abi.LabelRow) bool) []abi.LabelRow {
	var out []abi.LabelRow
	for _, r := range c.rows {
		if pred(r) {
			out = append(out, r)
		}
	}
	return out
}

// Positives returns the rows that are CATCHES (a non-Allow verdict) — the labeled
// examples of the gates firing.
func (c *Corpus) Positives() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.filterLocked(func(r abi.LabelRow) bool { return r.Verdict != abi.VerdictAllow })
}

// HardNegatives returns rows where a cheap rung PASSED but an expensive rung FAILED
// (RungFailed > RungPassed >= 0) — the pre-flight ladder's signal, the most
// informative training examples (a model that fails these is over-permissive).
func (c *Corpus) HardNegatives() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.filterLocked(func(r abi.LabelRow) bool { return r.RungPassed >= 0 && r.RungFailed > r.RungPassed })
}

// ByReason tallies catches per reason name (forensics / a class-balance view of the
// corpus).
func (c *Corpus) ByReason() map[string]int {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := map[string]int{}
	for _, r := range c.rows {
		if r.Verdict != abi.VerdictAllow {
			out[abi.ReasonName(r.Reason)]++
		}
	}
	return out
}

// Harvester is the Emitter that folds the kernel's adjudication stream into a
// Corpus. Attach with kernel registration: abi.RegisterEmitter(h).
type Harvester struct{ corpus *Corpus }

// New returns a Harvester that folds adjudication events into c; register it with abi.RegisterEmitter.
func New(c *Corpus) *Harvester { return &Harvester{corpus: c} }

// Emit collects a LabelRow from each label-bearing event. An explicit typed
// Event.Label (the pre-flight ladder's EvRungLabel) is taken verbatim — its rung
// structure is the richest signal. Otherwise a decision/deny event with a Verdict
// is DERIVED into a row, so a security catch becomes labeled data even when no
// explicit LabelRow rode the event.
func (h *Harvester) Emit(ev abi.Event) {
	if ev.Label != nil {
		h.corpus.add(*ev.Label)
		return
	}
	if ev.Verdict == nil {
		return
	}
	// The kernel pairs a deny's EvDecide with a dedicated EvDeny (and a Submit-path
	// require-witness intermediate with its resolved event); deriving a LabelRow from
	// BOTH would double the deny/require-witness rows in the training corpus
	// (Positives/ByReason then over-count each catch). Skip the redundant re-emit so a
	// decision contributes one row — the shared rule the decision-stream folders use.
	if abi.RedundantDecisionEvent(ev) {
		return
	}
	switch ev.Kind {
	case abi.EvDecide, abi.EvDeny, abi.EvResultDeny, abi.EvQuarantine:
		h.corpus.add(abi.LabelRow{
			CallHash:   callHash(ev.Call),
			RungPassed: -1, // unknown without an explicit ladder label
			RungFailed: -1,
			Verdict:    ev.Verdict.Kind,
			Reason:     ev.Verdict.Reason,
		})
	}
}

// callHash is a stable identity for a call (tool + args digest) so duplicate
// adjudications collapse to one training identity. Uses the args Ref digest when
// present, else a fnv of the tool + inline args.
func callHash(c *abi.ToolCall) string {
	if c == nil {
		return ""
	}
	if c.Args.Digest != "" {
		return c.Tool + "@" + c.Args.Digest
	}
	return c.Tool + "@" + itohex(fnv1a(append([]byte(c.Tool), c.Args.Inline...)))
}

func fnv1a(b []byte) uint64 {
	const off = 1469598103934665603
	const prime = 1099511628211
	h := uint64(off)
	for _, x := range b {
		h ^= uint64(x)
		h *= prime
	}
	return h
}

func itohex(n uint64) string {
	const hexd = "0123456789abcdef"
	var b [16]byte
	for i := 15; i >= 0; i-- {
		b[i] = hexd[n&0xf]
		n >>= 4
	}
	return string(b[:])
}

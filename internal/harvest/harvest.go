// Package harvest collects kernel adjudication events into labeled training corpora
// of frozen abi.LabelRow records for model training and policy evaluation.
package harvest

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// defaultMaxCorpusRows bounds retained corpus size to a recent sliding window.
const defaultMaxCorpusRows = 1024

// Corpus collects and retains labeled training rows with thread-safe sliding window bounds.
type Corpus struct {
	mu      sync.Mutex
	rows    []abi.LabelRow
	maxRows int
}

// NewCorpus initializes an empty corpus with default sliding window retention.
func NewCorpus() *Corpus { return &Corpus{} }

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

// SetMaxRows sets maximum row retention, trimming oldest rows immediately if exceeded.
// Passing n < 0 disables bounds, while n == 0 restores defaultMaxCorpusRows.
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
		c.rows = c.rows[len(c.rows)-max:]
	}
	c.mu.Unlock()
}

// Rows returns an independent snapshot slice of all currently retained rows.
func (c *Corpus) Rows() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]abi.LabelRow(nil), c.rows...)
}

// Len returns the total count of currently retained rows in the corpus.
func (c *Corpus) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.rows)
}

func (c *Corpus) filterLocked(pred func(abi.LabelRow) bool) []abi.LabelRow {
	var out []abi.LabelRow
	for _, r := range c.rows {
		if pred(r) {
			out = append(out, r)
		}
	}
	return out
}

// Positives returns all retained rows where verdict is not VerdictAllow.
func (c *Corpus) Positives() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.filterLocked(func(r abi.LabelRow) bool { return r.Verdict != abi.VerdictAllow })
}

// HardNegatives returns ladder rows where an earlier rung passed but a later rung failed.
func (c *Corpus) HardNegatives() []abi.LabelRow {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.filterLocked(func(r abi.LabelRow) bool { return r.RungPassed >= 0 && r.RungFailed > r.RungPassed })
}

// ByReason aggregates positive non-allow catches grouped by canonical reason name.
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

// Harvester implements abi.Emitter to fold kernel adjudication events into a Corpus.
type Harvester struct{ corpus *Corpus }

// New constructs a Harvester attached to the target corpus.
func New(c *Corpus) *Harvester { return &Harvester{corpus: c} }

// Emit ingests label-bearing and actionable adjudication events into the corpus.
func (h *Harvester) Emit(ev abi.Event) {
	if ev.Label != nil {
		h.corpus.add(*ev.Label)
		return
	}
	if ev.Verdict == nil || abi.RedundantDecisionEvent(ev) {
		return
	}
	switch ev.Kind {
	case abi.EvDecide, abi.EvDeny, abi.EvResultDeny, abi.EvQuarantine:
		h.corpus.add(abi.LabelRow{
			CallHash:   callHash(ev.Call),
			RungPassed: -1,
			RungFailed: -1,
			Verdict:    ev.Verdict.Kind,
			Reason:     ev.Verdict.Reason,
		})
	}
}

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

// Package preflight is the pre-flight rung ladder: cheapest-first
// well-formedness checks that catch a malformed/unsafe call BEFORE it fires, so a
// dead branch never spawns a process or burns a model turn. Each rung is run in
// order and escalation only happens on a pass (unit 49). A catch is recorded as a
// typed hard-negative (passed cheap rung k, failed expensive rung k+1) — the
// self-labeling signal that trains the syscall-tuned model (unit 50).
//
// Rungs in v0.1:
//
//	rung 0  static parse  — are the args even valid JSON? (unit 47)
//	rung 1  schema check  — required fields present + types match a JSON Schema
//	                        (unit 48)
//
// It registers as a low-rank Adjudicator (runs before the authoritative monitor);
// the kernel's fold takes the most-restrictive verdict, so a rung Deny wins over
// a later Allow. A well-formed call returns Defer (the rung has nothing to prove).
package preflight

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// FieldType is a minimal JSON Schema scalar type.
type FieldType string

const (
	TypeString FieldType = "string"
	TypeNumber FieldType = "number"
	TypeBool   FieldType = "boolean"
	TypeObject FieldType = "object"
	TypeArray  FieldType = "array"
	TypeAny    FieldType = ""
)

// Schema is a tiny JSON-Schema subset: required keys + their expected types.
type Schema struct {
	Required map[string]FieldType
}

// DefaultMaxNegatives bounds the hard-negative ledger so a long-lived ladder
// driven by sustained malformed/adversarial traffic — precisely the workload
// preflight exists to catch — cannot grow negatives without bound. Every
// caughtAt appended a JSONL row with no removal path, so on the registered
// rank-10 Default ladder (it serves every tool call) this grew for the life of
// the process. It matches the repo's other process-lifetime ledgers
// (ctxmmu.DefaultMaxHeld, normgate.DefaultMaxHeld, ifc.DefaultLedgerLimit,
// ratelimit.defaultMaxKeys = 8192). When the cap is reached the OLDEST negatives
// are dropped first; the dropped rows are pure observability/training samples
// (the unit-50 harvest), so eviction shortens the harvest, never a verdict.
const DefaultMaxNegatives = 8192

// Ladder is the rung ladder. Construct with New; Default is registered.
type Ladder struct {
	mu        sync.RWMutex
	schemas   map[string]Schema
	total     int64
	caught    int64
	negatives [][]byte // labeled hard-negative JSONL rows (unit 50), FIFO-bounded by maxNeg
	negHead   int      // consumed-prefix index into negatives (compacted in place)
	maxNeg    int      // cap on resident negatives; 0 in the zero value, set by constructors
	evicted   int64    // negatives dropped by the maxNeg bound (observability)
}

// New builds a ladder with the standard hard-negative-ledger bound
// (DefaultMaxNegatives).
func New() *Ladder { return NewWithLimit(DefaultMaxNegatives) }

// NewWithLimit builds a ladder whose hard-negative ledger holds at most maxNeg
// resident rows (oldest dropped first). A non-positive maxNeg falls back to
// DefaultMaxNegatives. This is the seam the leak-regression test uses to exercise
// eviction with a small bound.
func NewWithLimit(maxNeg int) *Ladder {
	if maxNeg < 1 {
		maxNeg = DefaultMaxNegatives
	}
	return &Ladder{schemas: map[string]Schema{}, maxNeg: maxNeg}
}

// SetSchema installs a schema for a tool (rung-1 input).
func (l *Ladder) SetSchema(tool string, s Schema) {
	l.mu.Lock()
	l.schemas[tool] = s
	l.mu.Unlock()
}

// Schemas returns a deep copy of the installed per-tool schemas. The static tool
// linter (internal/toollint) reads these to check what contract the kernel actually
// ENFORCES at rung-1 — versus the contract a tool merely advertises to the model —
// and to catch a Required field whose declared type falls outside the supported
// subset (typeOK would silently treat it as TypeAny and never validate it). The
// returned maps are copies; mutating them does not affect the ladder.
func (l *Ladder) Schemas() map[string]Schema {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make(map[string]Schema, len(l.schemas))
	for tool, s := range l.schemas {
		req := make(map[string]FieldType, len(s.Required))
		for k, v := range s.Required {
			req[k] = v
		}
		out[tool] = Schema{Required: req}
	}
	return out
}

func (l *Ladder) Caps() []abi.Capability { return nil }

// Adjudicate runs the rungs cheapest-first.
func (l *Ladder) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	l.mu.Lock()
	l.total++
	l.mu.Unlock()

	args := refBytes(ctx, c.Args)

	// rung 0: static parse. Empty args is well-formed (no body). Non-empty must
	// parse as a JSON object.
	if len(args) > 0 {
		var probe map[string]any
		if err := json.Unmarshal(args, &probe); err != nil {
			return l.caughtAt(c, -1, 0, abi.ReasonMalformed)
		}
	}

	// rung 1: schema validation (only if a schema is known; else escalate-by-defer).
	l.mu.RLock()
	s, known := l.schemas[c.Tool]
	l.mu.RUnlock()
	if known && len(s.Required) > 0 {
		var m map[string]any
		_ = json.Unmarshal(args, &m)
		if m == nil {
			m = map[string]any{}
		}
		for k, ty := range s.Required {
			v, present := m[k]
			if !present {
				return l.caughtAt(c, 0, 1, abi.ReasonMalformed)
			}
			if !typeOK(v, ty) {
				return l.caughtAt(c, 0, 1, abi.ReasonMalformed)
			}
		}
	}

	// passed every rung: the ladder has nothing to prove, defer to the monitor.
	return abi.Verdict{Kind: abi.VerdictDefer, By: "preflight"}
}

func (l *Ladder) caughtAt(c *abi.ToolCall, passed, failed int, r abi.ReasonCode) abi.Verdict {
	row := abi.LabelRow{
		CallHash:   callHash(c),
		RungPassed: passed,
		RungFailed: failed,
		Verdict:    abi.VerdictDeny,
		Reason:     r,
	}
	b, _ := json.Marshal(struct {
		CallHash   string `json:"call_hash"`
		RungPassed int    `json:"rung_passed"`
		RungFailed int    `json:"rung_failed"`
		Verdict    string `json:"verdict"`
		Reason     string `json:"reason"`
	}{row.CallHash, row.RungPassed, row.RungFailed, "deny", abi.ReasonName(r)})
	l.mu.Lock()
	l.caught++
	l.negatives = append(l.negatives, b)
	l.evictExcessLocked()
	l.mu.Unlock()
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: r, By: "preflight"}
}

// evictExcessLocked drops the oldest negatives (FIFO) until the resident count
// (len(negatives) - negHead) is within maxNeg, niling each dropped slot so the
// row's bytes are released for GC immediately. The consumed prefix is compacted
// in place once it reaches half the slice, so the backing array stays ≈2·maxNeg
// and never leaks. The caller holds l.mu. Dropped rows are observability/training
// samples (unit 50), so dropping the oldest never affects a verdict. caught/total
// are lifetime counters (the catch-rate numerators) and are intentionally NOT
// bounded — only the resident ledger is.
func (l *Ladder) evictExcessLocked() {
	for len(l.negatives)-l.negHead > l.maxNeg {
		l.negatives[l.negHead] = nil // release the dropped row's bytes (GC)
		l.negHead++
		l.evicted++
	}
	if l.negHead > 0 && l.negHead*2 >= len(l.negatives) {
		n := copy(l.negatives, l.negatives[l.negHead:])
		for i := n; i < len(l.negatives); i++ {
			l.negatives[i] = nil // clear the vacated tail so no moved row is retained twice
		}
		l.negatives = l.negatives[:n]
		l.negHead = 0
	}
}

// CatchRate returns caught/total (unit 51).
func (l *Ladder) CatchRate() (caught, total int64, rate float64) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.total > 0 {
		rate = float64(l.caught) / float64(l.total)
	}
	return l.caught, l.total, rate
}

// Negatives returns the labeled hard-negative JSONL rows still resident in the
// bounded ledger (the freshest ≤ maxNeg, in FIFO order), as a defensive copy so a
// caller can never mutate the live ledger. Semantics are unchanged from the
// pre-bound version except that rows evicted by the maxNeg cap are no longer
// present (use Evicted for the lifetime drop count).
func (l *Ladder) Negatives() [][]byte {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return append([][]byte(nil), l.negatives[l.negHead:]...)
}

// NegativesLen reports the current number of resident hard-negative rows in the
// bounded ledger (≤ maxNeg). Evicted reports how many were dropped by the bound
// over the ladder's lifetime. Together they make the leak fix observable:
// NegativesLen plateaus at the cap while Evicted climbs, instead of the resident
// ledger growing without bound.
func (l *Ladder) NegativesLen() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.negatives) - l.negHead
}

// Evicted reports the lifetime count of negatives dropped by the maxNeg bound.
func (l *Ladder) Evicted() int64 {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.evicted
}

func typeOK(v any, ty FieldType) bool {
	switch ty {
	case TypeAny:
		return true
	case TypeString:
		_, ok := v.(string)
		return ok
	case TypeNumber:
		_, ok := v.(float64)
		return ok
	case TypeBool:
		_, ok := v.(bool)
		return ok
	case TypeObject:
		_, ok := v.(map[string]any)
		return ok
	case TypeArray:
		_, ok := v.([]any)
		return ok
	}
	return true
}

func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

func callHash(c *abi.ToolCall) string {
	return c.Tool + ":" + c.Args.Digest
}

// Default is the registered ladder.
var Default = New()

func init() {
	abi.RegisterAdjudicator(10, Default) // before the rank-100 monitor
	abi.RegisterCapability("preflight.v1")
}

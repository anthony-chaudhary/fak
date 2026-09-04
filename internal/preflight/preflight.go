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

	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// Invariant: FieldType must identify a valid JSON primitive or composite type contract.
// FieldType defines the canonical scalar or composite JSON-Schema type identifier
// recognized by the preflight rung-ladder validation engine.
type FieldType string

// Invariant: Type string constants map directly to expected JSON scalar and container wire types.
const (
	// TypeString identifies JSON string scalar payload values.
	TypeString FieldType = "string"
	// TypeNumber identifies JSON numeric (float64 or integer) scalar payload values.
	TypeNumber FieldType = "number"
	// TypeBool identifies JSON boolean scalar payload values.
	TypeBool FieldType = "boolean"
	// TypeObject identifies JSON key-value map or object container values.
	TypeObject FieldType = "object"
	// TypeArray identifies JSON sequence or array container values.
	TypeArray FieldType = "array"
	// TypeAny indicates an unconstrained payload value where any valid JSON type is permitted.
	TypeAny FieldType = ""
)

// Invariant: Required argument specifications map field names to their expected validation types.
// Precondition: Keys in Required must correspond to valid parameter names expected in the payload.
// Schema specifies a lightweight JSON Schema contract defining required argument
// keys along with their expected field types for rung-1 validation.
type Schema struct {
	Required map[string]FieldType
}

// Invariant: DefaultMaxNegatives provides a strict upper bound preventing unbounded ledger memory growth.
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

// Invariant: Thread-safe state synchronization across concurrent calls is guarded by the internal mutex.
// Invariant: Resident negative entries count never exceeds maxNeg after eviction cycles complete.
// Ladder implements the multi-rung pre-flight validation pipeline, evaluating tool
// invocations cheapest-first to intercept invalid arguments before model turns fire.
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

// Postcondition: Returns an initialized Ladder bounded by DefaultMaxNegatives with empty schema registry.
// New constructs a preflight rung Ladder configured with the default maximum resident
// hard-negative capacity (DefaultMaxNegatives) to prevent unbounded memory growth.
func New() *Ladder { return NewWithLimit(DefaultMaxNegatives) }

// Invariant: Resident capacity limit maxNeg must be strictly positive; non-positive parameters fall back to DefaultMaxNegatives.
// Contract: The allocated ladder instance must initialize an empty schemas map and retain its configured capacity bound.
// Postcondition: Returns a fully initialized Ladder with valid capacity bounds and allocated schema mappings.
// NewWithLimit constructs a preflight Ladder enforcing an explicit resident capacity
// bound on the hard-negative ledger, falling back to DefaultMaxNegatives if non-positive.
// This is the seam the leak-regression test uses to exercise eviction with a small bound.
func NewWithLimit(maxNeg int) *Ladder {
	if maxNeg < 1 {
		maxNeg = DefaultMaxNegatives
	}
	return &Ladder{schemas: map[string]Schema{}, maxNeg: maxNeg}
}

// Contract: Updating tool schemas requires acquiring the ladder write lock to prevent concurrent adjudication race conditions.
// Expectation: Registered schemas must only declare supported FieldType definitions to ensure deterministic rung-1 validation.
// Precondition: Tool name must identify the target tool call to match during rung-1 validation.
// Postcondition: Registers the schema contract enforced on subsequent tool invocations matching the name.
// SetSchema registers the expected argument validation schema for a specific named tool,
// establishing the rung-1 schema validation contract for subsequent calls.
func (l *Ladder) SetSchema(tool string, s Schema) {
	l.mu.Lock()
	l.schemas[tool] = s
	l.mu.Unlock()
}

// Contract: Schema retrieval must return deep copies of registered schemas to guarantee caller isolation from internal state.
// Invariant: Internal schema registrations remain completely immutable against caller modification of returned structures.
// Postcondition: Returns a defensive deep copy of all registered tool schema specifications.
// Schemas returns a deep defensive copy of all currently installed tool schemas,
// ensuring caller inspection or mutation cannot alter the internal adjudication state.
// The static tool linter (internal/toollint) reads these to check what contract the kernel
// actually ENFORCES at rung-1 — versus the contract a tool merely advertises to the model —
// and to catch a Required field whose declared type falls outside the supported subset (typeOK
// would silently treat it as TypeAny and never validate it). The returned maps are copies;
// mutating them does not affect the ladder.
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

// Contract: Preflight adjudication acts as a structural filter without requiring extra capabilities.
// Postcondition: Always returns nil to indicate that no supplemental capability tokens are advertised.
// Caps returns the slice of capabilities advertised by this adjudicator; the preflight
// ladder operates as a pure structural filter and advertises no extra capabilities.
func (l *Ladder) Caps() []abi.Capability { return nil }

// Contract: Rung execution order must remain strictly monotonic from cheapest to most expensive: rung 0 (syntax) before rung 1 (schema).
// Invariant: Short-circuiting on any failed rung guarantees that a malformed payload never escalates to later rungs or burns compute.
// Fail-closed guard: Unparseable non-empty payload arguments must trigger an immediate VerdictDeny with ReasonMalformed.
// Precondition: Invocations must provide a non-nil ToolCall reference; empty arguments are treated as valid nil-body payloads.
// Postcondition: Returns VerdictDefer when all validation rungs pass, or VerdictDeny with ReasonMalformed on validation failure.
// Adjudicate evaluates tool invocation arguments through ascending rungs (syntax parse
// followed by schema conformance) to intercept malformed calls before monitor execution.
func (l *Ladder) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	l.mu.Lock()
	l.total++
	l.mu.Unlock()

	args := refutil.Bytes(ctx, c.Args)

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

// Invariant: Resident ledger count (len(negatives) - negHead) must never exceed maxNeg after eviction completes.
// Contract: Evicted negative entries must be nil-cleared immediately so garbage collection reclaims dropped row memory.
// Assumption: Callers of evictExcessLocked must hold the ladder write lock (l.mu) throughout eviction and slice compaction.
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

// Invariant: Catch rate ratio is normalized to [0.0, 1.0] when total is positive, and 0.0 otherwise.
// Postcondition: Returns lifetime caught count, total count, and the computed catch rate floating point ratio.
// CatchRate calculates and returns the lifetime telemetry metrics: total calls evaluated,
// malformed calls caught by rungs, and the computed catch-rate ratio (unit 51).
func (l *Ladder) CatchRate() (caught, total int64, rate float64) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.total > 0 {
		rate = float64(l.caught) / float64(l.total)
	}
	return l.caught, l.total, rate
}

// Contract: Returns a defensive slice copy preserving FIFO insertion order of hard-negative JSONL records.
// Invariant: Resident count of returned rows never exceeds the configured maxNeg capacity limit.
// Postcondition: Returns a copy of active resident hard-negative rows without exposing internal slices.
// Negatives retrieves a defensive copy of resident labeled hard-negative JSONL rows,
// preserving FIFO insertion order up to the configured resident ledger capacity limit.
func (l *Ladder) Negatives() [][]byte {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return append([][]byte(nil), l.negatives[l.negHead:]...)
}

// Invariant: Resident negative count is strictly non-negative and never exceeds maxNeg capacity.
// Postcondition: Returns the instantaneous count of resident hard-negative rows held in memory.
// NegativesLen returns the count of resident hard-negative rows currently held in
// the bounded memory buffer (always bounded by maxNeg). Evicted reports how many
// were dropped by the bound over the ladder's lifetime.
func (l *Ladder) NegativesLen() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.negatives) - l.negHead
}

// Invariant: Cumulative eviction count is monotonically non-decreasing over the lifetime of the ladder.
// Postcondition: Returns total count of hard-negative rows discarded to satisfy the resident ledger cap.
// Evicted reports the cumulative lifetime count of hard-negative sample rows dropped
// when ledger occupancy exceeded the resident maximum capacity threshold.
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

func callHash(c *abi.ToolCall) string {
	return c.Tool + ":" + c.Args.Digest
}

// Invariant: Default instance is registered as a rank-10 adjudicator before kernel monitor evaluation.
// Default provides the globally registered preflight Ladder instance wired into the
// kernel adjudication sequence at priority rank 10.
var Default = New()

func init() {
	abi.RegisterAdjudicator(10, Default) // before the rank-100 monitor
	abi.RegisterCapability("preflight.v1")
}

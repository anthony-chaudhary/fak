// Package grammar is the tool-invocation grammar rung: the well-formedness axis
// the trust/effect rungs can't see. It catches the "--name is a flag, not
// positional" class of failure — the model guessed a tool's ARGUMENT SHAPE wrong
// — BEFORE the call spawns, and where the fix is mechanical it auto-repairs the
// call in-syscall (a TRANSFORM) instead of burning a model turn (unit 54).
//
// Three behaviours:
//   - well-formed call            -> Defer (nothing to prove)
//   - malformed but repairable    -> Transform (positional args zipped into the
//     grammar's named params, arity-matched)
//   - malformed & unrepairable    -> Deny(MISROUTE)  (model-fixable disposition)
//   - no grammar for the tool      -> Defer (FAIL-OPEN, unit 55: never over-refuse)
//
// Grammars are content-addressed and deduped (unit 57): a grammar learned once is
// shared fleet-wide as a single entry. A grammar can be derived from an MCP tool's
// JSON Schema (unit 53), the build-first free source.
package grammar

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Param is one named parameter in a tool's grammar.
type Param struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Required bool   `json:"required"`
}

// Grammar is a tool's argument shape: an ordered list of named params. v0.1
// models the common case (all named), the positional-misuse repair, and the
// argument-NAME-alias repair (the model used a synonym key — e.g. "from" for the
// required "from_currency" — the single most common shape error a competent model
// makes on a strict schema). Aliases maps a synonym key -> the canonical param it
// should be renamed to; a call that satisfies a required param only via an alias
// is repaired in-syscall (a rename TRANSFORM) instead of bouncing back as an error
// the model must spend a turn to fix.
type Grammar struct {
	Params  []Param           `json:"params"`
	Aliases map[string]string `json:"aliases,omitempty"` // synonym key -> canonical param
}

func (g Grammar) digest() string {
	b, _ := json.Marshal(g)
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])[:24]
}

// Rung is the grammar adjudicator. Grammars dedupe by content hash.
type Rung struct {
	mu       sync.RWMutex
	byTool   map[string]string  // tool -> grammar digest
	byDigest map[string]Grammar // digest -> canonical grammar (deduped)
	repairs  int64
	denies   int64
}

func New() *Rung {
	return &Rung{byTool: map[string]string{}, byDigest: map[string]Grammar{}}
}

// Add installs a grammar for a tool, deduping identical grammars to one entry.
func (r *Rung) Add(tool string, g Grammar) {
	d := g.digest()
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.byDigest[d]; !ok {
		r.byDigest[d] = g
	}
	r.byTool[tool] = d
}

// Has reports whether a grammar (alias/repair contract) is registered for a tool.
// The static tool linter reads this to know a tool's args are enforced in-kernel by
// a grammar even when no pre-flight schema is installed (the convert_currency case).
func (r *Rung) Has(tool string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.byTool[tool]
	return ok
}

// UniqueGrammarCount is the number of distinct (deduped) grammars (unit 57).
func (r *Rung) UniqueGrammarCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.byDigest)
}

// LoadFromJSONSchema derives a grammar from an MCP-style JSON Schema (unit 53):
// {"properties":{"name":{"type":"string"}},"required":["name"]}.
func (r *Rung) LoadFromJSONSchema(tool string, schema []byte) error {
	var doc struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &doc); err != nil {
		return err
	}
	reqd := map[string]bool{}
	for _, k := range doc.Required {
		reqd[k] = true
	}
	g := Grammar{}
	names := make([]string, 0, len(doc.Properties))
	for n := range doc.Properties {
		names = append(names, n)
	}
	sort.Strings(names) // deterministic param order
	for _, n := range names {
		g.Params = append(g.Params, Param{Name: n, Type: doc.Properties[n].Type, Required: reqd[n]})
	}
	r.Add(tool, g)
	return nil
}

func (r *Rung) Caps() []abi.Capability { return nil }

// Adjudicate enforces the grammar.
func (r *Rung) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	r.mu.RLock()
	d, ok := r.byTool[c.Tool]
	g := r.byDigest[d]
	r.mu.RUnlock()
	if !ok {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"} // fail-open
	}

	args := refBytes(ctx, c.Args)
	var m map[string]any
	if len(args) > 0 {
		_ = json.Unmarshal(args, &m)
	}
	if m == nil {
		m = map[string]any{}
	}

	// Already well-formed? Every required named param present => defer.
	if countMissing(m, g.Params) == 0 {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}
	}

	// Argument-NAME-alias repair: the model used a synonym key for a required
	// param (e.g. "from" for "from_currency"). If renaming the known aliases makes
	// every required param present, repair the call in-syscall (a rename TRANSFORM)
	// — no model turn burned. We never trust the alias blindly: the rename only
	// fires when it actually closes the well-formedness gap.
	if len(g.Aliases) > 0 {
		if repaired, renamed := applyAliases(m, g.Aliases); renamed && countMissing(repaired, g.Params) == 0 {
			if ref, ok := putJSON(ctx, repaired); ok {
				r.mu.Lock()
				r.repairs++
				r.mu.Unlock()
				return abi.Verdict{Kind: abi.VerdictTransform, By: "grammar",
					Reason: abi.ReasonMisroute, Payload: abi.TransformPayload{NewArgs: ref}}
			}
		}
	}

	// Malformed: did the model pass POSITIONAL args (the --name-as-positional bug)?
	if pos, ok := m["_positional"].([]any); ok {
		if len(pos) == len(g.Params) {
			// arity matches => mechanical repair: zip positional -> named.
			repaired := map[string]any{}
			for i, p := range g.Params {
				repaired[p.Name] = pos[i]
			}
			if ref, ok := putJSON(ctx, repaired); ok {
				r.mu.Lock()
				r.repairs++
				r.mu.Unlock()
				return abi.Verdict{Kind: abi.VerdictTransform, By: "grammar",
					Reason: abi.ReasonMisroute, Payload: abi.TransformPayload{NewArgs: ref}}
			}
		}
	}

	// Unrepairable well-formedness error: refuse with a MODEL-FIXABLE disposition.
	r.mu.Lock()
	r.denies++
	r.mu.Unlock()
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMisroute, By: "grammar"}
}

// countMissing counts required params absent from the arg map (a present
// "named:X" or "X" key both satisfy param X).
func countMissing(m map[string]any, params []Param) int {
	missing := 0
	for _, p := range params {
		if !p.Required {
			continue
		}
		if _, ok := m["named:"+p.Name]; ok {
			continue
		}
		if _, ok := m[p.Name]; ok {
			continue
		}
		missing++
	}
	return missing
}

// applyAliases returns a copy of m with every known synonym key renamed to its
// canonical param (only when the canonical is not already present), plus whether
// any rename happened. The original keys that are not aliases are preserved.
func applyAliases(m map[string]any, aliases map[string]string) (map[string]any, bool) {
	out := make(map[string]any, len(m))
	renamed := false
	for k, v := range m {
		if canon, ok := aliases[k]; ok {
			if _, present := m[canon]; !present {
				out[canon] = v
				renamed = true
				continue
			}
		}
		out[k] = v
	}
	return out, renamed
}

// Stats reports repairs + denies (forensics / KPI).
func (r *Rung) Stats() (repairs, denies int64) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.repairs, r.denies
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

func putJSON(ctx context.Context, m map[string]any) (abi.Ref, bool) {
	b, err := json.Marshal(m)
	if err != nil {
		return abi.Ref{}, false
	}
	res := abi.ActiveResolver()
	if res == nil {
		return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}, true
	}
	ref, err := res.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, false
	}
	return ref, true
}

// Default is the registered grammar rung.
var Default = New()

func init() {
	abi.RegisterAdjudicator(5, Default) // the very first rung (cheapest)
	abi.RegisterCapability("grammar.v1")
}

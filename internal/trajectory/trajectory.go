package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// Turn is one analysis-shaped record of an agent action: the trace it belongs to,
// its order within the trace, the query that drove it, the tool and the kernel's
// verdict over it, the result taint, the digest identities, and the per-turn cost.
// QueryEmbedding is an optional deterministic simhash vector over Query — present
// only when the Recorder was built EmbedQueries(true) — so a downstream similarity
// query needs no second pass. The JSON tags are the stable export schema; new
// fields are additive (omitempty) so an older reader keeps parsing.
type Turn struct {
	TraceID        string            `json:"trace_id"`
	Seq            int               `json:"seq"`             // order within the trace, 1-based
	TSUnixNano     int64             `json:"ts_unix_nano"`    // wall-clock anchor (from Fields["ts_unix_nano"], else 0)
	Query          string            `json:"query,omitempty"` // the human-meaningful query/prompt that drove the turn
	Tool           string            `json:"tool,omitempty"`
	Verdict        string            `json:"verdict,omitempty"` // ALLOW | DENY | QUARANTINE | WITNESS | ...
	Reason         string            `json:"reason,omitempty"`
	Taint          string            `json:"taint,omitempty"`           // result provenance taint
	Materialized   string            `json:"materialized,omitempty"`    // HIT | FAULT | ... (Fields["materialized"])
	ArgsDigest     string            `json:"args_digest,omitempty"`     // content identity of the call args
	ResultDigest   string            `json:"result_digest,omitempty"`   // content identity of the result payload
	TokenEstimate  int               `json:"token_estimate,omitempty"`  // turn cost in tokens (Fields["tokens"])
	Bytes          int64             `json:"bytes,omitempty"`           // turn cost in bytes (Fields["bytes"])
	CacheHit       bool              `json:"cache_hit,omitempty"`       // served from the KV/view cache
	QueryEmbedding []float32         `json:"query_embedding,omitempty"` // deterministic simhash vector over Query
	Labels         map[string]string `json:"labels,omitempty"`          // OPEN: producer-stamped Meta carried through
}

// Recorder folds the kernel's adjudication stream into per-trace Turn rows. It is an
// abi.Emitter: register it once (Enable) and every audit-relevant lifecycle event
// becomes an analysis row. It never blocks the kernel and never panics — the fan-out
// contract is fire-and-forget. Safe for concurrent Emit; queries (Trace/Traces/
// Export) take a snapshot.
type Recorder struct {
	mu     sync.Mutex
	byID   map[string][]Turn // trace id -> ordered turns
	order  []string          // trace ids in first-seen order (stable export)
	embed  bool              // stamp QueryEmbedding from simhash
	clock  func() int64      // injectable wall clock (UnixNano); nil => Fields-only
	maxLen int               // cap on retained turns per trace (0 = unbounded)
}

// New returns an empty Recorder. By default it does NOT stamp query embeddings (a
// large corpus stays lean) and reads timestamps only from Fields. Configure with the
// option setters before registering.
func New() *Recorder {
	return &Recorder{byID: map[string][]Turn{}}
}

// EmbedQueries turns on per-turn simhash embedding of Query. A gardening skill that
// will run similarity queries wants this on; an audit-only consumer leaves it off to
// keep rows small. Returns the recorder for chaining.
func (r *Recorder) EmbedQueries(on bool) *Recorder {
	r.mu.Lock()
	r.embed = on
	r.mu.Unlock()
	return r
}

// MaxPerTrace caps the retained turns per trace (oldest dropped). 0 = unbounded.
// Bounds memory for a long-lived process recording many traces.
func (r *Recorder) MaxPerTrace(n int) *Recorder {
	r.mu.Lock()
	r.maxLen = n
	r.mu.Unlock()
	return r
}

// Emit implements abi.Emitter. It records every decision-bearing lifecycle event
// (DECIDE/DENY/QUARANTINE/VDSO_HIT) as a Turn appended to its trace. Operational-only
// kinds (Submit/Dispatch/Complete without a verdict) are skipped — Complete is folded
// in when it carries cost Fields, otherwise it is operational noise.
func (r *Recorder) Emit(ev abi.Event) {
	t, ok := turnFromEvent(ev)
	if !ok {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.embed && t.Query != "" {
		t.QueryEmbedding = simhash.Embed(t.Query)
	}
	if _, seen := r.byID[t.TraceID]; !seen {
		r.order = append(r.order, t.TraceID)
	}
	turns := append(r.byID[t.TraceID], t)
	if r.maxLen > 0 && len(turns) > r.maxLen {
		turns = turns[len(turns)-r.maxLen:]
	}
	// Renumber Seq so it stays 1-based and contiguous after any cap drop.
	for i := range turns {
		turns[i].Seq = i + 1
	}
	r.byID[t.TraceID] = turns
}

// Trace returns a snapshot of one trace's turns in order (nil if unknown).
func (r *Recorder) Trace(id string) []Turn {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Turn(nil), r.byID[id]...)
}

// Traces returns all trace ids in first-seen order.
func (r *Recorder) Traces() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.order...)
}

// Turns returns every recorded turn across all traces, trace-ordered then turn-
// ordered — the flat corpus a similarity index or scorer consumes.
func (r *Recorder) Turns() []Turn {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Turn, 0)
	for _, id := range r.order {
		out = append(out, r.byID[id]...)
	}
	return out
}

// Len is the total number of recorded turns.
func (r *Recorder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, ts := range r.byID {
		n += len(ts)
	}
	return n
}

// ExportTo writes every recorded turn as JSONL (one Turn per line), trace-ordered.
// This is the stable on-disk corpus a `fak traj` verb or an external trajectory
// analyzer reads. Returns the number of rows written.
func (r *Recorder) ExportTo(w io.Writer) (int, error) {
	turns := r.Turns()
	n := 0
	for _, t := range turns {
		b, err := json.Marshal(t)
		if err != nil {
			return n, err
		}
		if _, err := w.Write(append(b, '\n')); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// ImportFrom reads a JSONL corpus written by ExportTo back into a Recorder, grouping
// by TraceID in encounter order. It is the read side a stateless `fak traj` verb uses
// to load a corpus file produced by an earlier run. Malformed lines are skipped (a
// torn final line never aborts a load); returns the number of turns ingested.
func ImportFrom(rd io.Reader) (*Recorder, int, error) {
	r := New()
	dec := json.NewDecoder(rd)
	n := 0
	for {
		var t Turn
		if err := dec.Decode(&t); err != nil {
			if err == io.EOF {
				break
			}
			// Skip a malformed record but keep going to the next newline-delimited one.
			// json.Decoder advances past the bad token; if it can't, stop.
			if dec.More() {
				continue
			}
			break
		}
		if t.TraceID == "" {
			continue
		}
		if _, seen := r.byID[t.TraceID]; !seen {
			r.order = append(r.order, t.TraceID)
		}
		r.byID[t.TraceID] = append(r.byID[t.TraceID], t)
		n++
	}
	return r, n, nil
}

// Index builds a simhash.Index over the recorded turns' queries, keyed by
// "trace:seq", so a caller can ask "what past turns look like this query?". Turns
// with an empty query are skipped (nothing to embed). A turn that already carries a
// QueryEmbedding reuses it; otherwise it is embedded on the fly.
func (r *Recorder) Index() *simhash.Index {
	turns := r.Turns()
	ix := &simhash.Index{}
	for _, t := range turns {
		if t.Query == "" {
			continue
		}
		vec := t.QueryEmbedding
		if len(vec) == 0 {
			vec = simhash.Embed(t.Query)
		}
		ix.Add(turnKey(t), vec, t.Query)
	}
	return ix
}

// turnKey is the stable id of a turn within a corpus index.
func turnKey(t Turn) string { return t.TraceID + ":" + itoa(t.Seq) }

// turnFromEvent projects a lifecycle Event into an analysis Turn, returning false for
// a kind that carries no analysis signal. It reads the human query and per-turn cost
// from the OPEN Event.Fields / ToolCall.Meta channels the producer stamps, defaulting
// cleanly when absent — so the same recorder works whether or not the producer
// enriches the stream.
func turnFromEvent(ev abi.Event) (Turn, bool) {
	switch ev.Kind {
	case abi.EvDecide, abi.EvDeny, abi.EvQuarantine, abi.EvVDSOHit:
		// analysis-bearing kinds proceed
	default:
		return Turn{}, false
	}

	t := Turn{Labels: map[string]string{}}
	if c := ev.Call; c != nil {
		t.TraceID = c.TraceID
		t.Tool = c.Tool
		t.ArgsDigest = refDigest(c.Args)
		// Producer-stamped query/labels ride the call's OPEN Meta.
		if c.Meta != nil {
			if q, ok := c.Meta["query"]; ok {
				t.Query = q
			}
			for k, v := range c.Meta {
				if k == "query" {
					continue
				}
				t.Labels[k] = v
			}
		}
	}
	// Default the verdict label from the event kind so a kind without a Verdict
	// object (a VDSO hit) is still legible; a real Verdict overrides it.
	switch ev.Kind {
	case abi.EvDeny:
		t.Verdict = "DENY"
	case abi.EvQuarantine:
		t.Verdict = "QUARANTINE"
	case abi.EvVDSOHit:
		t.Verdict = "VDSO_HIT"
	}
	if v := ev.Verdict; v != nil {
		t.Verdict = verdictName(v.Kind)
		t.Reason = abi.ReasonName(v.Reason)
	}
	if res := ev.Result; res != nil {
		t.ResultDigest = refDigest(res.Payload)
		t.Taint = taintName(res.Payload.Taint)
	}
	if ev.Kind == abi.EvVDSOHit {
		t.CacheHit = true
		t.Materialized = "HIT"
	}
	// OPEN telemetry: query text, cost, timestamp, materialization, cache-hit can all
	// ride Event.Fields (the producer's enrichment channel), overriding/augmenting Meta.
	applyFields(&t, ev.Fields)
	if len(t.Labels) == 0 {
		t.Labels = nil
	}
	// A turn with no trace id has no analysis home; drop it.
	if t.TraceID == "" {
		return Turn{}, false
	}
	return t, true
}

// applyFields folds the OPEN Event.Fields telemetry into the Turn. Each key is
// optional and typed defensively (a wrong-typed value is ignored, never panics).
func applyFields(t *Turn, f map[string]any) {
	if f == nil {
		return
	}
	if q, ok := f["query"].(string); ok && q != "" {
		t.Query = q
	}
	if ts, ok := asInt64(f["ts_unix_nano"]); ok {
		t.TSUnixNano = ts
	}
	if n, ok := asInt(f["tokens"]); ok {
		t.TokenEstimate = n
	}
	if b, ok := asInt64(f["bytes"]); ok {
		t.Bytes = b
	}
	if m, ok := f["materialized"].(string); ok && m != "" {
		t.Materialized = m
	}
	if hit, ok := f["cache_hit"].(bool); ok {
		t.CacheHit = hit
	}
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	}
	return 0, false
}

func asInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int64:
		return n, true
	case int:
		return int64(n), true
	case float64:
		return int64(n), true
	}
	return 0, false
}

func refDigest(r abi.Ref) string {
	if r.Digest != "" {
		return r.Digest
	}
	if r.Kind == abi.RefInline && len(r.Inline) > 0 {
		sum := sha256.Sum256(r.Inline)
		return "sha256:" + hex.EncodeToString(sum[:])
	}
	return ""
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return "K" + itoa(int(k))
}

func taintName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return ""
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// ---------------------------------------------------------------------------
// Process-global registered instance — opt-in via Enable (mirrors journal).
// ---------------------------------------------------------------------------

var (
	activeMu sync.Mutex
	active   *Recorder
)

// Default returns the registered process-global Recorder, or nil if none was enabled.
// A front door (the gateway / fak guard) reads this to export the trajectory corpus.
func Default() *Recorder {
	activeMu.Lock()
	defer activeMu.Unlock()
	return active
}

// Enable turns trajectory recording ON: it builds a Recorder (EmbedQueries follows
// the argument), registers it as ONE emitter against the frozen ABI, and returns it.
// It is IDEMPOTENT — a second Enable returns the existing recorder without double-
// registering (the ABI has no unregister), so the first enablement wins.
func Enable(embedQueries bool) *Recorder {
	activeMu.Lock()
	defer activeMu.Unlock()
	if active != nil {
		return active
	}
	r := New().EmbedQueries(embedQueries)
	active = r
	abi.RegisterEmitter(r)
	return r
}

// sortedTraceIDs is a small helper for deterministic iteration when callers want
// traces in lexical (not encounter) order — used by report renderers/tests.
func sortedTraceIDs(r *Recorder) []string {
	ids := r.Traces()
	sort.Strings(ids)
	return ids
}

var _ = sortedTraceIDs // exported-equivalent helper retained for report use

// init wires the opt-in front door, mirroring the journal: trajectory recording is
// OFF unless FAK_TRAJECTORY is set to a truthy value, so a benchmark or a unit test
// never pays to record. FAK_TRAJECTORY_EMBED additionally turns on per-turn simhash
// embedding (a gardening deployment wants vectors; an audit-only one does not). The
// blank import of this package in internal/registrations is what makes this init
// fire before the kernel boots — the same enablement seam as FAK_AUDIT_JOURNAL.
func init() {
	if !truthy(os.Getenv("FAK_TRAJECTORY")) {
		return
	}
	Enable(truthy(os.Getenv("FAK_TRAJECTORY_EMBED")))
}

// truthy reads an env toggle: "1", "true", "yes", "on" (case-insensitive) enable;
// everything else (including unset) is off.
func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

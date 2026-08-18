package wirescreen

import (
	"bytes"
	"context"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/strictjson"
)

// jsonwalk.go is the JSON-leaf-aware sibling of the flat pre-send redaction path
// (redactor.go). The flat path treats a tool result / message body as one opaque
// byte string and runs the Redactor over all of it — which means it also scans the
// JSON scaffolding (keys, braces, the base64 of an inline image, an identifier like
// tool_use_id) that is never prose a human wrote. On a structured body that is the
// wrong grain: a false-positive on an id is churn, and a secret split across two
// JSON string leaves (or nested inside a stringified-JSON payload) is invisible to a
// flat regex that sees the escaped bytes.
//
// RedactJSONLeaves walks the decoded JSON, renders ONLY the string leaves worth
// screening (skipping structural/identifier keys), runs the Redactor over that prose
// with a leaf-offset table, then writes each redaction back into the exact leaf it
// came from and re-marshals — so the output is still valid JSON that round-trips.
// It reaches one level further than a flat scan: a string leaf whose content is
// itself JSON (the stringified-arguments / stringified-payload shape) is decoded,
// its leaves screened, and re-serialized on a finalizer.
//
// Contract parity with Apply (redactor.go): this is the SAME witnessed-lossy-proposer
// discipline — strictly one-sided (a leaf's secret span becomes "[REDACTED:<kind>]",
// nothing is injected or reordered), and reversible when the caller pins the original
// body in CAS exactly as Apply does before calling. RedactJSONLeaves is the pure
// transform (no CAS coupling, so it is directly testable); the reversible outbound
// entry that pins the original first is the #555-gated wiring follow-on, matching the
// fence redactor.go's Apply already shipped under. The flat path and its CAS contract
// are unchanged; this is an additive, opt-in grain for structured bodies.

// maxStringifiedDepth bounds how many levels of stringified-JSON the walker will
// decode-and-reach-into before treating a payload as an ordinary string leaf. It
// caps the work a hostile deeply-nested "{\"a\":\"{\\\"b\\\":...}\"}" body can cost
// while still covering the common one-hop stringified-arguments shape.
const maxStringifiedDepth = 6

// structuralKeys are object keys whose values are control/identifier fields, never
// prose worth screening: the message/block discriminators, the tool-call ids and
// name, model/stop metadata, and the image `source` subtree (base64 image data is
// phash's domain, not text redaction). Skipping them avoids both false-positive
// churn on an id and mangling a value another rung round-trips on.
var structuralKeys = map[string]bool{
	"type":          true,
	"role":          true,
	"id":            true,
	"tool_use_id":   true,
	"tool_call_id":  true,
	"name":          true,
	"model":         true,
	"stop_reason":   true,
	"stop_sequence": true,
	"finish_reason": true,
	"index":         true,
	"cache_control": true,
	"source":        true,
}

// JSONHit is one redaction the leaf-aware pass applied, addressed by an RFC-6901 JSON
// Pointer to the leaf (not a byte offset — the flat body's offsets are meaningless
// once the tree is re-marshalled) plus the Span kind that matched.
type JSONHit struct {
	Pointer string `json:"pointer"`
	Kind    string `json:"kind"`
}

// JSONRedaction is the result of RedactJSONLeaves: the re-marshalled body with each
// matched leaf span replaced by a placeholder, and the pointer-addressed audit of
// what was redacted. Redacted equals the input body verbatim when ok is false.
type JSONRedaction struct {
	Redacted []byte    `json:"-"`
	Hits     []JSONHit `json:"hits,omitempty"`
}

// RedactJSONLeaves runs r over the string leaves of a JSON body and returns the
// re-marshalled body with each proposed span redacted in place. It returns ok=false
// (and Redacted == body) when r is nil, the body is empty, the body is not a JSON
// object/array/string, or nothing was redacted — every case in which the caller
// should fall back to the flat path unchanged.
func RedactJSONLeaves(ctx context.Context, r Redactor, body []byte, tool string) (JSONRedaction, bool) {
	if r == nil || len(body) == 0 {
		return JSONRedaction{Redacted: body}, false
	}
	// Only a JSON object/array/string carries screenable string leaves; a bare
	// number/bool/null (or non-JSON) has none, so defer to the flat path.
	trimmed := bytes.TrimLeft(body, " \t\r\n")
	if len(trimmed) == 0 || (trimmed[0] != '{' && trimmed[0] != '[' && trimmed[0] != '"') {
		return JSONRedaction{Redacted: body}, false
	}
	root, err := strictjson.NumberValue(body)
	if err != nil {
		return JSONRedaction{Redacted: body}, false
	}
	w := &jsonWalker{}
	w.walk(root, "", "", func(nv any) { root = nv })
	if len(w.leaves) == 0 {
		return JSONRedaction{Redacted: body}, false
	}
	spans := coalesce(r.Propose(ctx, []byte(w.prose.String()), tool), w.prose.Len())
	if len(spans) == 0 {
		return JSONRedaction{Redacted: body}, false
	}
	hits := w.applySpans(spans)
	if len(hits) == 0 {
		return JSONRedaction{Redacted: body}, false
	}
	// Re-serialize stringified-JSON payloads innermost-first: a finalizer registered
	// on descent (outer before inner) must run inner before outer so the outer
	// container sees the already-re-marshalled inner string.
	for i := len(w.finalizers) - 1; i >= 0; i-- {
		w.finalizers[i]()
	}
	out, err := marshalJSONValue(root)
	if err != nil {
		return JSONRedaction{Redacted: body}, false
	}
	return JSONRedaction{Redacted: out, Hits: hits}, true
}

// jsonLeaf is one rendered string leaf: where its value sits in the prose buffer, the
// value itself, and the closure that writes a redacted replacement back into the tree.
type jsonLeaf struct {
	pointer string
	start   int // offset of val within the prose buffer
	val     string
	write   func(string)
}

// jsonWalker accumulates the rendered prose, the leaf-offset table, and the
// stringified-JSON re-marshal finalizers over one walk.
type jsonWalker struct {
	prose      strings.Builder
	leaves     []*jsonLeaf
	finalizers []func()
	depth      int // current stringified-JSON nesting (not object nesting)
}

// walk descends v, appending screenable string leaves to the prose/offset table.
// setSelf writes a replacement value back into v's slot in the parent container.
func (w *jsonWalker) walk(v any, pointer, keyLabel string, setSelf func(any)) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys) // stable prose order regardless of map iteration
		for _, k := range keys {
			if structuralKeys[k] {
				continue // never render or descend a structural/identifier value
			}
			key := k
			w.walk(t[key], pointer+"/"+escapePointer(key), key, func(nv any) { t[key] = nv })
		}
	case []any:
		for i := range t {
			idx := i
			w.walk(t[idx], pointer+"/"+strconv.Itoa(idx), strconv.Itoa(idx), func(nv any) { t[idx] = nv })
		}
	case string:
		w.walkString(t, pointer, keyLabel, setSelf)
	default:
		// json.Number, bool, nil: no screenable text; leave the slot untouched.
	}
}

// walkString handles a string value: either reach into a stringified-JSON payload
// (decode, screen its leaves, re-marshal on a finalizer) or render it as a plain leaf.
func (w *jsonWalker) walkString(s, pointer, keyLabel string, setSelf func(any)) {
	if w.depth < maxStringifiedDepth {
		if inner, ok := decodeStringifiedJSON(s); ok {
			holder := inner
			w.depth++
			w.walk(holder, pointer, keyLabel, func(nv any) { holder = nv })
			w.depth--
			w.finalizers = append(w.finalizers, func() {
				if b, err := marshalJSONValue(holder); err == nil {
					setSelf(string(b))
				}
			})
			return
		}
	}
	// Plain string leaf: render "<key>: <value>" (or the bare value at the root) and
	// record its offset so a prose span maps back to exactly this leaf.
	if w.prose.Len() > 0 {
		w.prose.WriteString("\n\n")
	}
	if keyLabel != "" {
		w.prose.WriteString(keyLabel)
		w.prose.WriteString(": ")
	}
	start := w.prose.Len()
	w.prose.WriteString(s)
	w.leaves = append(w.leaves, &jsonLeaf{pointer: pointer, start: start, val: s, write: func(redacted string) { setSelf(redacted) }})
}

// applySpans maps coalesced prose spans back onto the leaves they overlap and rewrites
// each affected leaf, replacing the intersected sub-range with "[REDACTED:<kind>]".
// A span that straddles two leaves redacts the covered part of each. Returns the
// pointer-addressed hits in prose order.
func (w *jsonWalker) applySpans(spans []Span) []JSONHit {
	type piece struct {
		start, end int
		kind       string
	}
	perLeaf := make(map[int][]piece)
	for _, sp := range spans {
		for li, lf := range w.leaves {
			ls, le := lf.start, lf.start+len(lf.val)
			if sp.End <= ls || sp.Start >= le {
				continue // no overlap with this leaf
			}
			cs, ce := sp.Start, sp.End
			if cs < ls {
				cs = ls
			}
			if ce > le {
				ce = le
			}
			perLeaf[li] = append(perLeaf[li], piece{cs - lf.start, ce - lf.start, sp.Kind})
		}
	}
	var hits []JSONHit
	for li, lf := range w.leaves {
		ps := perLeaf[li]
		if len(ps) == 0 {
			continue
		}
		sort.Slice(ps, func(i, j int) bool { return ps[i].start < ps[j].start })
		var b strings.Builder
		prev := 0
		for _, p := range ps {
			if p.start < prev {
				continue // defensive: coalesce already made spans disjoint
			}
			b.WriteString(lf.val[prev:p.start])
			b.WriteString("[REDACTED:")
			b.WriteString(p.kind)
			b.WriteByte(']')
			prev = p.end
			atomic.AddInt64(&redactions, 1)
			hits = append(hits, JSONHit{Pointer: lf.pointer, Kind: p.kind})
		}
		b.WriteString(lf.val[prev:])
		lf.write(b.String())
	}
	return hits
}

// decodeStringifiedJSON decodes s when its content is itself a JSON object/array (the
// stringified-arguments / stringified-payload shape). A plain string returns ok=false.
func decodeStringifiedJSON(s string) (any, bool) {
	t := strings.TrimLeft(s, " \t\r\n")
	if t == "" || (t[0] != '{' && t[0] != '[') {
		return nil, false
	}
	v, err := strictjson.NumberValue([]byte(s))
	if err != nil {
		return nil, false
	}
	return v, true
}

// decodeJSONValue decodes one JSON value with numbers preserved (json.Number) so
// re-marshalling does not reformat integers, and rejects trailing garbage so only a
// marshalJSONValue re-serializes a walked tree without HTML escaping (so '<', '>', '&'
// in prose survive byte-exact) and without the trailing newline json.Encoder appends.
func marshalJSONValue(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// escapePointer encodes one JSON Pointer reference token per RFC 6901 ('~' -> '~0',
// '/' -> '~1') so a key containing those characters yields an unambiguous pointer.
func escapePointer(k string) string {
	k = strings.ReplaceAll(k, "~", "~0")
	k = strings.ReplaceAll(k, "/", "~1")
	return k
}

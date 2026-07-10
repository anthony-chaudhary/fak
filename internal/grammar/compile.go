// compile.go is the tokenizer-aware schema->mask compiler: the minimal native
// spine of the structured-generation backlog (#2596), the exact follow-on
// CLAIMS.md names after #929 shipped the model-side sink. It lowers fak tool
// schemas — one or two tools, tool_choice=required, required scalar fields —
// through the canonical oneOf tool-call JSON-Schema shape into a byte-FSM-backed
// per-step token mask for the in-kernel engine's GenerateConstrained seam.
//
// Layering (the same dependency inversion as model.GuidedByteMask, seen from the
// other side of the seam): internal/model must not import internal/tokenizer or
// internal/grammar, and this compiler must not drag in model weights. So Compile
// emits a *CallMask whose MaskLogits(history, logits) satisfies model.LogitMask
// STRUCTURALLY — no internal/model import — and the id->bytes decode is injected
// as the small TokenDecoder interface that tokenizer.(*Tokenizer) already
// satisfies. The envelope skeleton region ({"name":<enum>,"arguments":) is
// delegated verbatim to the tested guideddecode byte FSM; this file adds only
// the ARGUMENT region that guideddecode's package doc defers to a later slice.
//
// Honest scope — supported v0 surface (everything else returns an error wrapping
// ErrUnsupportedSchema that NAMES the feature, never a silently weaker mask):
//   - argument objects with ONLY required properties (additionalProperties
//     false or absent), emitted in sorted property order — the same
//     deterministic order LoadFromJSONSchema uses;
//   - scalar property types string, boolean, integer, number;
//   - string enum/const with printable-ASCII values needing no JSON escaping;
//   - generated free strings are printable ASCII without escapes;
//   - numbers are canonical JSON without exponents.
//
// Non-goals (research note "Minimal native spine"): XGrammar-2 performance
// parity or full JSON Schema coverage. Admission recomputes the byte FSM per
// candidate token — fine off the hot path, unapologetically not a compiled
// token-trie cache.
package grammar

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guideddecode"
)

// ErrUnsupportedSchema is wrapped by every compile error that rejects a JSON
// Schema feature outside the v0 surface. The error text names the tool, the
// property, and the feature — an unsupported schema is refused loudly, never
// lowered to a mask that silently under-constrains.
var ErrUnsupportedSchema = errors.New("unsupported JSON Schema feature")

// ToolSpec is one callable tool as the compiler receives it: the tool name and
// its MCP-style argument JSON Schema (the same document LoadFromJSONSchema
// accepts, restricted to the v0 surface above).
type ToolSpec struct {
	Name   string
	Schema []byte
}

// TokenDecoder maps a token id to the exact bytes it decodes to, or nil for an
// id that is not a decodable token. tokenizer.(*Tokenizer) satisfies it; tests
// use a trivial byte-identity decoder. This is the "tokenizer interface" seam:
// the compiler never imports internal/tokenizer.
type TokenDecoder interface {
	TokenBytes(id int) []byte
}

// CompileOptions carries the per-vocabulary knobs that are not part of the
// schema. EOS is the tokenizer's end-of-sequence id: once the envelope is
// complete the mask admits ONLY EOS, which is what lets GenerateConstrained
// terminate instead of free-running past the closing brace. EOS < 0 declares
// no EOS contract: a complete envelope leaves the logits unconstrained
// (mirroring guideddecode's nil = unconstrained region).
type CompileOptions struct {
	EOS int
}

// fieldKind is the closed set of v0 value grammars.
type fieldKind int

const (
	fieldString     fieldKind = iota // free printable-ASCII string, no escapes
	fieldStringEnum                  // one of a closed set of quoted literals
	fieldBool                        // true | false
	fieldInt                         // -? (0 | [1-9][0-9]*)
	fieldNumber                      // integer with optional .[0-9]+ fraction
)

// compiledField is one required argument property in canonical (sorted) order.
type compiledField struct {
	name       string
	kind       fieldKind
	enum       []string // fieldStringEnum: sorted, deduped variants
	quotedEnum [][]byte // enum variants with their JSON quotes, prefix-free
}

// segment is one step of a tool's argument-region grammar: either a fixed
// literal (lit non-nil) or a scalar value region (field non-nil). Literals are
// merged maximally at build time, so a value segment is ALWAYS followed by a
// literal segment (","<key>": or the closing "}}") — the walker relies on it.
type segment struct {
	lit   []byte
	field *compiledField
}

// compiledTool is one tool's full envelope grammar: the guideddecode skeleton
// {"name":"<name>","arguments": followed by the argument-region segments.
type compiledTool struct {
	name     string
	skeleton []byte
	fields   []compiledField
	segs     []segment
}

// CallMask is the compiled trie/FSM-backed token mask over one or two tool
// envelopes. It implements model.LogitMask structurally: MaskLogits(history,
// logits) decodes the generated-token history into the emitted byte prefix,
// asks the byte FSM which bytes may come next, and sets every token whose
// decoded bytes would leave the envelope path to -inf. Like every LogitMask it
// is dormant unless the decode loop runs with FAK_NATIVE_GUIDED_DECODE=1 —
// flag-off decodes stay bit-exact with the mask compiled in.
type CallMask struct {
	tools     []compiledTool
	names     guideddecode.ToolSchema
	dec       TokenDecoder
	eos       int
	canonical []byte
	digest    string
}

// Compile lowers specs into a *CallMask over dec's vocabulary. Order of specs
// does not matter: tools are sorted by name and identical duplicates are
// deduped (the same content-addressed normalization discipline as Rung.Add);
// two specs sharing a name with DIFFERENT schemas are an error.
func Compile(specs []ToolSpec, dec TokenDecoder, opts CompileOptions) (*CallMask, error) {
	if dec == nil {
		return nil, errors.New("grammar: Compile needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	tools, err := normalizeTools(specs)
	if err != nil {
		return nil, err
	}
	canon := canonicalOneOf(tools)
	names := make([]string, len(tools))
	for i := range tools {
		names[i] = tools[i].name
	}
	sum := sha256.Sum256(canon)
	return &CallMask{
		tools:     tools,
		names:     guideddecode.ToolSchema{Names: names},
		dec:       dec,
		eos:       opts.EOS,
		canonical: canon,
		digest:    hex.EncodeToString(sum[:])[:16],
	}, nil
}

// CanonicalSchema returns the canonical oneOf tool-call JSON Schema the specs
// normalized to: {"oneOf":[{"type":"object","properties":{"name":{"const":..},
// "arguments":<schema>},...}]}. Byte-deterministic (sorted tools, sorted
// properties), so equal tool sets always produce equal bytes.
func (m *CallMask) CanonicalSchema() []byte {
	return append([]byte(nil), m.canonical...)
}

// Digest is the content address of the canonical schema (16 hex chars): equal
// tool sets share a digest, which is what lets a fleet dedup compiled masks the
// way the Rung dedups grammars.
func (m *CallMask) Digest() string { return m.digest }

// MaskLogits implements the model.LogitMask contract: it sets to -inf every
// token whose decoded bytes cannot legally extend the emitted envelope prefix
// (history = the generated ids this turn, NOT the prompt). Once the envelope is
// complete, only EOS survives (or, with no EOS declared, the logits are left
// unconstrained). A dead-end prefix — unreachable under masked decoding —
// declines to mask at all, mirroring model.GuidedByteMask.
func (m *CallMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.dec == nil {
		return
	}
	var prefix []byte
	for _, id := range history {
		prefix = append(prefix, m.dec.TokenBytes(id)...)
	}
	allowed, complete := m.allowedNext(prefix)
	neg := float32(math.Inf(-1))
	if complete {
		if m.eos >= 0 && m.eos < len(logits) {
			for i := range logits {
				if i != m.eos {
					logits[i] = neg
				}
			}
		}
		return
	}
	if len(allowed) == 0 {
		return // off-path prefix: decline to mask (mirrors GuidedByteMask)
	}
	for id := range logits {
		if id == m.eos {
			logits[id] = neg // EOS is admissible ONLY at a complete envelope
			continue
		}
		if !m.admits(prefix, allowed, m.dec.TokenBytes(id)) {
			logits[id] = neg
		}
	}
}

// admits replays a candidate token's decoded bytes one at a time from the
// current prefix, re-deriving the byte admission set at every step. A token is
// admitted iff every byte stays on an envelope path; bytes that continue past a
// COMPLETE envelope are admitted only when no EOS contract was declared.
func (m *CallMask) admits(prefix []byte, first map[byte]bool, tb []byte) bool {
	if len(tb) == 0 {
		return false // undecodable / zero-byte ids never advance the envelope
	}
	buf := append([]byte(nil), prefix...)
	allowed, complete := first, false
	for _, b := range tb {
		if complete {
			return m.eos < 0
		}
		if !allowed[b] {
			return false
		}
		buf = append(buf, b)
		allowed, complete = m.allowedNext(buf)
	}
	return true
}

// allowedNext is the whole-envelope byte FSM: which bytes may extend prefix,
// and whether prefix already IS a complete envelope. The skeleton + name-enum
// region is guideddecode's tested FSM verbatim; a nil return from it means some
// tool's skeleton is fully consumed, and exactly one tool can own the prefix
// from there (names contain no quote, so one skeleton is never a prefix of
// another). Dead ends return an empty set, never nil.
func (m *CallMask) allowedNext(prefix []byte) (map[byte]bool, bool) {
	if a := guideddecode.AllowedNextBytes(prefix, m.names); a != nil {
		return a, false
	}
	for i := range m.tools {
		t := &m.tools[i]
		if bytes.HasPrefix(prefix, t.skeleton) {
			return t.argsAllowed(prefix[len(t.skeleton):])
		}
	}
	return map[byte]bool{}, false
}

// argsAllowed walks rest (the bytes after this tool's skeleton) through the
// argument-region segments. Returns the admission set for the next byte, or
// (nil/empty, true) when rest is exactly a complete envelope.
func (t *compiledTool) argsAllowed(rest []byte) (map[byte]bool, bool) {
	pos := 0
	for si, seg := range t.segs {
		if seg.field == nil {
			n := len(seg.lit)
			if rem := len(rest) - pos; rem < n {
				if !bytes.Equal(rest[pos:], seg.lit[:rem]) {
					return map[byte]bool{}, false
				}
				return map[byte]bool{seg.lit[rem]: true}, false
			}
			if !bytes.Equal(rest[pos:pos+n], seg.lit) {
				return map[byte]bool{}, false
			}
			pos += n
			continue
		}
		v := scanValueField(*seg.field, rest[pos:])
		switch v.state {
		case vDead:
			return map[byte]bool{}, false
		case vPartial:
			// The value consumed rest to its end. If the bytes so far ALSO form
			// a complete value (a number can end without a delimiter), the
			// following literal's first byte is admissible too — the segment
			// after a value is always a literal by construction.
			out := v.allowed
			if v.canEnd {
				out[t.segs[si+1].lit[0]] = true
			}
			return out, false
		default: // vComplete
			pos += v.n
		}
	}
	if pos == len(rest) {
		return map[byte]bool{}, true // complete envelope
	}
	return map[byte]bool{}, false // trailing bytes beyond a complete envelope
}

// --- scalar value scanners -------------------------------------------------

const (
	vDead     = iota // rest diverged from this value's grammar
	vPartial         // rest ends inside the value; allowed/canEnd describe next
	vComplete        // a full value occupies rest[:n]
)

// valueScan is one scalar scanner verdict over the remaining bytes.
type valueScan struct {
	state   int
	n       int           // vComplete: bytes consumed
	allowed map[byte]bool // vPartial: bytes that may extend the value
	canEnd  bool          // vPartial: bytes so far also form a complete value
}

func scanValueField(f compiledField, rest []byte) valueScan {
	switch f.kind {
	case fieldString:
		return scanFreeString(rest)
	case fieldStringEnum:
		return scanLiterals(rest, f.quotedEnum)
	case fieldBool:
		return scanLiterals(rest, boolLits)
	case fieldInt:
		return scanNumber(rest, true)
	default:
		return scanNumber(rest, false)
	}
}

// stringContentOK is the free-string generation alphabet: printable ASCII
// minus the two bytes that would need escaping. Escapes and non-ASCII are
// outside v0 (labeled in the package doc), so the mask never admits them —
// every maskable string is valid JSON with no escape handling.
func stringContentOK(b byte) bool {
	return b >= 0x20 && b <= 0x7e && b != '"' && b != '\\'
}

func stringContentSet() map[byte]bool {
	a := make(map[byte]bool, 0x7f-0x20)
	for b := byte(0x20); b <= 0x7e; b++ {
		if b != '"' && b != '\\' {
			a[b] = true
		}
	}
	return a
}

func scanFreeString(rest []byte) valueScan {
	if len(rest) == 0 {
		return valueScan{state: vPartial, allowed: map[byte]bool{'"': true}}
	}
	if rest[0] != '"' {
		return valueScan{state: vDead}
	}
	for i := 1; i < len(rest); i++ {
		switch {
		case rest[i] == '"':
			return valueScan{state: vComplete, n: i + 1}
		case !stringContentOK(rest[i]):
			return valueScan{state: vDead}
		}
	}
	a := stringContentSet()
	a['"'] = true // the string may close at any point (empty content is legal)
	return valueScan{state: vPartial, allowed: a}
}

var boolLits = [][]byte{[]byte("true"), []byte("false")}

// scanLiterals matches rest against a prefix-free candidate set (quoted enum
// variants disambiguate at the closing quote; true/false share no prefix), so
// at most one candidate can complete and completion is never ambiguous.
func scanLiterals(rest []byte, cands [][]byte) valueScan {
	allowed := map[byte]bool{}
	for _, c := range cands {
		if len(rest) >= len(c) {
			if bytes.Equal(rest[:len(c)], c) {
				return valueScan{state: vComplete, n: len(c)}
			}
			continue
		}
		if bytes.Equal(rest, c[:len(rest)]) {
			allowed[c[len(rest)]] = true
		}
	}
	if len(allowed) == 0 {
		return valueScan{state: vDead}
	}
	return valueScan{state: vPartial, allowed: allowed}
}

// scanNumber walks canonical JSON number syntax: -? (0 | [1-9][0-9]*) with an
// optional .[0-9]+ fraction when integerOnly is false. No exponents, no leading
// zeros — the mask only ever generates numbers encoding/json will accept. A
// number has no closing delimiter of its own, so completion is decided by the
// FOLLOWING segment byte (canEnd) or by rest continuing past the number.
func scanNumber(rest []byte, integerOnly bool) valueScan {
	const (
		start      = iota // nothing consumed yet
		afterMinus        // '-' consumed
		intZero           // integer part is exactly "0"
		intDigits         // [1-9][0-9]*
		afterDot          // '.' consumed, no fraction digit yet
		fracDigits        // at least one fraction digit
	)
	s := start
	for i, b := range rest {
		digit := b >= '0' && b <= '9'
		switch s {
		case start, afterMinus:
			switch {
			case b == '-' && s == start:
				s = afterMinus
			case b == '0':
				s = intZero
			case digit:
				s = intDigits
			default:
				return valueScan{state: vDead}
			}
		case intZero:
			if !integerOnly && b == '.' {
				s = afterDot
				continue
			}
			return valueScan{state: vComplete, n: i}
		case intDigits:
			switch {
			case digit:
			case !integerOnly && b == '.':
				s = afterDot
			default:
				return valueScan{state: vComplete, n: i}
			}
		case afterDot:
			if !digit {
				return valueScan{state: vDead}
			}
			s = fracDigits
		case fracDigits:
			if !digit {
				return valueScan{state: vComplete, n: i}
			}
		}
	}
	allowed := map[byte]bool{}
	digitsInto := func() {
		for b := byte('0'); b <= '9'; b++ {
			allowed[b] = true
		}
	}
	canEnd := false
	switch s {
	case start:
		allowed['-'] = true
		digitsInto()
	case afterMinus, afterDot:
		digitsInto()
	case intZero:
		if !integerOnly {
			allowed['.'] = true
		}
		canEnd = true
	case intDigits:
		digitsInto()
		if !integerOnly {
			allowed['.'] = true
		}
		canEnd = true
	case fracDigits:
		digitsInto()
		canEnd = true
	}
	return valueScan{state: vPartial, allowed: allowed, canEnd: canEnd}
}

// --- normalization ----------------------------------------------------------

// envPre/envSuf mirror guideddecode's envelope skeleton byte-for-byte; the
// composition test pins the agreement so a drift in either side fails loudly.
const (
	envPre = `{"name":"`
	envSuf = `","arguments":`
)

func normalizeTools(specs []ToolSpec) ([]compiledTool, error) {
	if len(specs) == 0 {
		return nil, errors.New("grammar: Compile needs at least one tool spec")
	}
	seen := map[string]string{} // name -> canonical arg-schema bytes
	var tools []compiledTool
	for _, s := range specs {
		if s.Name == "" {
			return nil, errors.New("grammar: tool spec with empty name")
		}
		if strings.ContainsAny(s.Name, "\"\\") || !asciiPrintable(s.Name) {
			return nil, unsupported(s.Name, "", "tool name needs JSON string escaping")
		}
		fields, err := normalizeFields(s.Name, s.Schema)
		if err != nil {
			return nil, err
		}
		var buf bytes.Buffer
		writeArgsSchema(&buf, fields)
		if prev, dup := seen[s.Name]; dup {
			if prev == buf.String() {
				continue // identical duplicate: dedup, same discipline as Rung.Add
			}
			return nil, fmt.Errorf("grammar: tool %q declared twice with different schemas", s.Name)
		}
		seen[s.Name] = buf.String()
		tools = append(tools, compiledTool{name: s.Name, fields: fields})
	}
	sort.Slice(tools, func(i, j int) bool { return tools[i].name < tools[j].name })
	for i := range tools {
		tools[i].skeleton = []byte(envPre + tools[i].name + envSuf)
		tools[i].segs = buildSegments(tools[i].fields)
	}
	return tools, nil
}

// buildSegments lowers a tool's fields into the argument-region segment list,
// merging literals maximally so every value segment has a literal successor.
func buildSegments(fields []compiledField) []segment {
	var segs []segment
	lit := []byte{'{'}
	for i := range fields {
		if i > 0 {
			lit = append(lit, ',')
		}
		lit = append(lit, '"')
		lit = append(lit, fields[i].name...)
		lit = append(lit, '"', ':')
		segs = append(segs, segment{lit: lit})
		segs = append(segs, segment{field: &fields[i]})
		lit = nil
	}
	lit = append(lit, '}', '}') // close the arguments object and the envelope
	return append(segs, segment{lit: lit})
}

func normalizeFields(tool string, schema []byte) ([]compiledField, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(schema, &top); err != nil {
		return nil, fmt.Errorf("grammar: tool %q: parse schema: %w", tool, err)
	}
	for k := range top {
		switch k {
		case "type", "properties", "required", "additionalProperties", "title", "description", "$schema":
		default:
			return nil, unsupported(tool, "", "top-level keyword %q", k)
		}
	}
	if raw, ok := top["type"]; ok {
		var ty string
		if err := json.Unmarshal(raw, &ty); err != nil || ty != "object" {
			return nil, unsupported(tool, "", "top-level type %s (arguments must be an object)", raw)
		}
	}
	if raw, ok := top["additionalProperties"]; ok {
		var ap bool
		if err := json.Unmarshal(raw, &ap); err != nil || ap {
			return nil, unsupported(tool, "", "additionalProperties other than false")
		}
	}
	props := map[string]map[string]json.RawMessage{}
	if raw, ok := top["properties"]; ok {
		if err := json.Unmarshal(raw, &props); err != nil {
			return nil, fmt.Errorf("grammar: tool %q: parse properties: %w", tool, err)
		}
	}
	var required []string
	if raw, ok := top["required"]; ok {
		if err := json.Unmarshal(raw, &required); err != nil {
			return nil, fmt.Errorf("grammar: tool %q: parse required: %w", tool, err)
		}
	}
	reqd := map[string]bool{}
	for _, r := range required {
		if _, ok := props[r]; !ok {
			return nil, fmt.Errorf("grammar: tool %q: required property %q not declared in properties", tool, r)
		}
		reqd[r] = true
	}
	names := make([]string, 0, len(props))
	for n := range props {
		names = append(names, n)
	}
	sort.Strings(names) // canonical field order, matching LoadFromJSONSchema
	fields := make([]compiledField, 0, len(names))
	for _, n := range names {
		if !reqd[n] {
			return nil, unsupported(tool, n, "optional property (v0 emits required-only argument objects)")
		}
		if strings.ContainsAny(n, "\"\\") || !asciiPrintable(n) {
			return nil, unsupported(tool, n, "property name needs JSON string escaping")
		}
		f, err := normalizeField(tool, n, props[n])
		if err != nil {
			return nil, err
		}
		fields = append(fields, f)
	}
	return fields, nil
}

func normalizeField(tool, name string, prop map[string]json.RawMessage) (compiledField, error) {
	for k := range prop {
		switch k {
		case "type", "enum", "const", "title", "description":
		default:
			return compiledField{}, unsupported(tool, name, "keyword %q", k)
		}
	}
	var ty string
	if raw, ok := prop["type"]; ok {
		if err := json.Unmarshal(raw, &ty); err != nil {
			return compiledField{}, unsupported(tool, name, "non-string type %s", raw)
		}
	}
	_, hasEnum := prop["enum"]
	_, hasConst := prop["const"]
	if (hasEnum || hasConst) && ty != "string" {
		return compiledField{}, unsupported(tool, name, "enum/const on type %q (v0 supports string enums only)", ty)
	}
	if hasEnum && hasConst {
		return compiledField{}, unsupported(tool, name, "both enum and const")
	}
	switch ty {
	case "string":
		if !hasEnum && !hasConst {
			return compiledField{name: name, kind: fieldString}, nil
		}
		var variants []string
		if hasConst {
			var v string
			if err := json.Unmarshal(prop["const"], &v); err != nil {
				return compiledField{}, unsupported(tool, name, "non-string const %s", prop["const"])
			}
			variants = []string{v}
		} else {
			if err := json.Unmarshal(prop["enum"], &variants); err != nil {
				return compiledField{}, unsupported(tool, name, "non-string enum %s", prop["enum"])
			}
		}
		if len(variants) == 0 {
			return compiledField{}, unsupported(tool, name, "empty enum")
		}
		sort.Strings(variants)
		variants = dedupSorted(variants)
		f := compiledField{name: name, kind: fieldStringEnum, enum: variants}
		for _, v := range variants {
			if strings.ContainsAny(v, "\"\\") || !asciiPrintable(v) {
				return compiledField{}, unsupported(tool, name, "enum value %q needs JSON string escaping", v)
			}
			f.quotedEnum = append(f.quotedEnum, []byte(`"`+v+`"`))
		}
		return f, nil
	case "boolean":
		return compiledField{name: name, kind: fieldBool}, nil
	case "integer":
		return compiledField{name: name, kind: fieldInt}, nil
	case "number":
		return compiledField{name: name, kind: fieldNumber}, nil
	case "":
		return compiledField{}, unsupported(tool, name, "property without a type")
	default:
		return compiledField{}, unsupported(tool, name, "type %q", ty)
	}
}

func asciiPrintable(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < 0x20 || s[i] > 0x7e {
			return false
		}
	}
	return true
}

func dedupSorted(s []string) []string {
	out := s[:0]
	for i, v := range s {
		if i == 0 || v != s[i-1] {
			out = append(out, v)
		}
	}
	return out
}

func unsupported(tool, prop, format string, a ...any) error {
	loc := fmt.Sprintf("grammar: tool %q", tool)
	if prop != "" {
		loc += fmt.Sprintf(" property %q", prop)
	}
	return fmt.Errorf("%s: %s: %w", loc, fmt.Sprintf(format, a...), ErrUnsupportedSchema)
}

// --- canonical schema -------------------------------------------------------

// canonicalOneOf renders the normalized tool set as the canonical oneOf
// tool-call JSON Schema, built with an ordered writer (not json.Marshal of a
// map) so the bytes are deterministic and digest-stable.
func canonicalOneOf(tools []compiledTool) []byte {
	var b bytes.Buffer
	b.WriteString(`{"oneOf":[`)
	for i := range tools {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"type":"object","properties":{"name":{"const":`)
		writeJSONString(&b, tools[i].name)
		b.WriteString(`},"arguments":`)
		writeArgsSchema(&b, tools[i].fields)
		b.WriteString(`},"required":["name","arguments"],"additionalProperties":false}`)
	}
	b.WriteString(`]}`)
	return b.Bytes()
}

func writeArgsSchema(b *bytes.Buffer, fields []compiledField) {
	b.WriteString(`{"type":"object","properties":{`)
	for i, f := range fields {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(b, f.name)
		b.WriteByte(':')
		switch f.kind {
		case fieldString:
			b.WriteString(`{"type":"string"}`)
		case fieldStringEnum:
			b.WriteString(`{"type":"string","enum":[`)
			for j, v := range f.enum {
				if j > 0 {
					b.WriteByte(',')
				}
				writeJSONString(b, v)
			}
			b.WriteString(`]}`)
		case fieldBool:
			b.WriteString(`{"type":"boolean"}`)
		case fieldInt:
			b.WriteString(`{"type":"integer"}`)
		default:
			b.WriteString(`{"type":"number"}`)
		}
	}
	b.WriteString(`},"required":[`)
	for i, f := range fields {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(b, f.name)
	}
	b.WriteString(`],"additionalProperties":false}`)
}

func writeJSONString(b *bytes.Buffer, s string) {
	enc, _ := json.Marshal(s) // a string never fails to marshal
	b.Write(enc)
}

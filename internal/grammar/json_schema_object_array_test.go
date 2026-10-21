package grammar

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// objectArrayVocab extends the cursor vocabulary with pieces that cross item,
// array and member boundaries inside one token.
func objectArrayVocab(seed int64) jsonSchemaTokenDecoder {
	rng := rand.New(rand.NewSource(seed))
	dec := cursorTestVocab(rng)
	for _, s := range []string{
		`[{"`, `},{"`, `}]`, `}]}`, `"}]`, `1}]`, `],"`, `[]`, `[`, `]}`, `},`, `{"a":1},{"a":`, `[true,`,
		`"},{"`, `]}]`, `[{}`, `{}]`, `},{}`, `[1,`, `1]}`, `]},{"`, `true]`, `"]}`, `[]},{`, `]}]}`,
	} {
		dec = append(dec, []byte(s))
	}
	const alphabet = `{}[]":,.-0123456789abtrufelsxyz`
	for len(dec) < 1300 {
		b := make([]byte, 1+rng.Intn(6))
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		dec = append(dec, b)
	}
	return dec
}

// enumerateObjectArrayDocs independently lists every canonical document of a
// finite-domain schema. It reports false once more than limit documents exist.
func enumerateObjectArrayDocs(t *testing.T, s map[string]any, limit int) ([]string, bool) {
	t.Helper()
	ty, _ := s["type"].(string)
	switch ty {
	case "object":
		props, _ := s["properties"].(map[string]any)
		req := map[string]bool{}
		if r, ok := s["required"].([]any); ok {
			for _, x := range r {
				req[x.(string)] = true
			}
		}
		names := make([]string, 0, len(props))
		for n := range props {
			names = append(names, n)
		}
		sort.Strings(names)
		combos := [][]string{{}}
		for _, n := range names {
			vals, ok := enumerateObjectArrayDocs(t, props[n].(map[string]any), limit)
			if !ok {
				return nil, false
			}
			var next [][]string
			for _, c := range combos {
				if !req[n] {
					next = append(next, c)
				}
				for _, v := range vals {
					next = append(next, append(append([]string(nil), c...), `"`+n+`":`+v))
				}
			}
			if len(next) > limit {
				return nil, false
			}
			combos = next
		}
		out := make([]string, 0, len(combos))
		for _, c := range combos {
			out = append(out, "{"+strings.Join(c, ",")+"}")
		}
		return out, true
	case "boolean":
		return []string{"false", "true"}, true
	case "string", "integer":
		var out []string
		add := func(v any) {
			if str, ok := v.(string); ok {
				out = append(out, `"`+str+`"`)
			} else {
				out = append(out, strconv.FormatInt(int64(v.(float64)), 10))
			}
		}
		if c, ok := s["const"]; ok {
			add(c)
		}
		if e, ok := s["enum"].([]any); ok {
			for _, v := range e {
				add(v)
			}
		}
		if out == nil {
			t.Fatalf("enumerate: %s %v has no finite domain", ty, s)
		}
		sort.Strings(out)
		return dedupSorted(out), true
	case "array":
		items, ok := enumerateObjectArrayDocs(t, s["items"].(map[string]any), limit)
		if !ok {
			return nil, false
		}
		minItems, maxItems := 0, maxResponseArrayItems
		if v, ok := s["minItems"].(float64); ok {
			minItems = int(v)
		}
		if v, ok := s["maxItems"].(float64); ok {
			maxItems = int(v)
		}
		var out []string
		overflow := false
		var rec func(cur []string)
		rec = func(cur []string) {
			if overflow {
				return
			}
			if len(cur) >= minItems {
				out = append(out, "["+strings.Join(cur, ",")+"]")
			}
			if len(out) > limit {
				overflow = true
				return
			}
			if len(cur) == maxItems {
				return
			}
			for _, it := range items {
				rec(append(append([]string(nil), cur...), it))
			}
		}
		rec(nil)
		return out, !overflow
	}
	t.Fatalf("enumerate: unsupported schema %v", s)
	return nil, false
}

// checkExactLanguage asserts that after every prefix of every document the mask
// admits exactly the tokens that keep some document reachable, and admits EOS
// exactly at complete documents.
func checkExactLanguage(t *testing.T, schema string, mask *JSONSchemaMask, dec jsonSchemaTokenDecoder, docs []string) {
	t.Helper()
	lang := map[string]bool{}
	prefixSet := map[string]bool{}
	for _, d := range docs {
		if lang[d] || !json.Valid([]byte(d)) {
			t.Fatalf("%s: enumerator produced duplicate or invalid %q", schema, d)
		}
		lang[d] = true
		for i := 0; i <= len(d); i++ {
			prefixSet[d[:i]] = true
		}
	}
	prefixes := make([]string, 0, len(prefixSet))
	for p := range prefixSet {
		prefixes = append(prefixes, p)
	}
	sort.Strings(prefixes)
	bad := 0
	logits := make([]float32, len(dec))
	for _, p := range prefixes {
		history := make([]int, len(p))
		for i := range p {
			history[i] = cursorSingleByteBase + int(p[i])
		}
		clear(logits)
		mask.MaskLogits(history, logits)
		for id := range logits {
			admitted := !math.IsInf(float64(logits[id]), -1)
			want := lang[p]
			if id != 0 {
				want = len(dec[id]) > 0 && prefixSet[p+string(dec[id])]
			}
			if admitted != want {
				t.Errorf("%s: after %q token %q admitted=%v, want %v", schema, p, dec[id], admitted, want)
				if bad++; bad > 10 {
					t.FailNow()
				}
			}
		}
	}
}

var objectArrayFiniteSchemas = []string{
	`{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string","enum":["a","ab"]},"qty":{"type":"integer","enum":[1,10]}},"required":["name","qty"],"additionalProperties":false},"maxItems":2}},"required":["items"],"additionalProperties":false}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{}},"minItems":1,"maxItems":3},"z":{"type":"boolean"}},"required":["r"]}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{"p":{"type":"object","properties":{"x":{"type":"integer","enum":[0,5,50]}},"required":["x"]},"t":{"type":"boolean"}},"required":["p","t"]},"minItems":1,"maxItems":2}},"required":["r"]}`,
	`{"type":"object","properties":{"m":{"type":"object","properties":{"rs":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string","enum":["}",",","]"]}},"required":["k"]},"maxItems":2}},"required":["rs"]},"n":{"type":"integer","enum":[1,12]}}}`,
	`{"type":"object","properties":{"a":{"type":"array","items":{"type":"object","properties":{"v":{"type":"integer","enum":[1,10,100]}},"required":["v"]},"maxItems":2},"b":{"type":"array","items":{"type":"integer","enum":[2,20]},"maxItems":2}},"required":["b"]}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{"q":{"type":"boolean"}},"required":["q"]},"minItems":2,"maxItems":2}},"required":["r"]}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{"q":{"type":"boolean"}},"required":["q"]},"maxItems":0}},"required":["r"]}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string","enum":["a","ab"]},"l":{"type":"array","items":{"type":"integer","enum":[1,10]},"maxItems":2},"z":{"type":"string","const":"}"}},"required":["k","l","z"]},"maxItems":2}},"required":["r"]}`,
	`{"type":"object","properties":{"r":{"type":"array","items":{"type":"object","properties":{"o":{"type":"object","properties":{"l":{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":2}},"required":["l"]}},"required":["o"]},"minItems":1,"maxItems":2},"s":{"type":"integer","enum":[3,30]}},"required":["r","s"]}`,
}

// fak-test:runtime fast est=1500ms lane=default
func TestJSONSchemaObjectArrayFiniteLanguageOracle(t *testing.T) {
	dec := objectArrayVocab(41)
	for _, schema := range objectArrayFiniteSchemas {
		var s map[string]any
		if err := json.Unmarshal([]byte(schema), &s); err != nil {
			t.Fatal(err)
		}
		docs, ok := enumerateObjectArrayDocs(t, s, 5000)
		if !ok {
			t.Fatalf("%s: language too large for the oracle", schema)
		}
		mask, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: 0})
		if err != nil {
			t.Fatalf("%s: %v", schema, err)
		}
		checkExactLanguage(t, schema, mask, dec, docs)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestJSONSchemaObjectArrayItemDelimiters(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(51)))
	mask := compileArraySchema(t, dec, `{"type":"array","items":{"type":"object","properties":{"k":{"type":"string","enum":["a"]},"n":{"type":"integer"},"t":{"type":"array","items":{"type":"boolean"},"maxItems":1}},"required":["k","n","t"]},"minItems":1,"maxItems":2}`, true)
	for _, tc := range []struct {
		prefix string
		want   string
		eos    bool
	}{
		{`{"v":`, `[`, false},
		{`{"v":[`, `{`, false},
		{`{"v":[{`, `"`, false},
		{`{"v":[{"k":"a","n":1`, `,0123456789`, false},
		{`{"v":[{"k":"a","n":1,"t":[`, `]ft`, false},
		{`{"v":[{"k":"a","n":1,"t":[true`, `]`, false},
		{`{"v":[{"k":"a","n":1,"t":[true]`, `}`, false},
		{`{"v":[{"k":"a","n":1,"t":[]}`, `,]`, false},
		{`{"v":[{"k":"a","n":1,"t":[]},`, `{`, false},
		{`{"v":[{"k":"a","n":1,"t":[]},{"k":"a","n":0,"t":[]}`, `]`, false},
		{`{"v":[{"k":"a","n":1,"t":[]}]`, `}`, false},
		{`{"v":[{"k":"a","n":1,"t":[]}]}`, ``, true},
	} {
		got, eos := arrayNextBytes(t, mask, dec, tc.prefix)
		if got != tc.want || eos != tc.eos {
			t.Errorf("after %s: bytes %q eos=%v, want %q eos=%v", tc.prefix, got, eos, tc.want, tc.eos)
		}
	}
	for _, dead := range []string{
		`{"v":[]`,
		`{"v":[{"n":`,
		`{"v":[{"k":"a"}`,
		`{"v":[{"k":"a","n":1,"t":[true,`,
		`{"v":[{"k":"a","n":1,"t":[]},{"k":"a","n":1,"t":[]},`,
		`{"v":[{"k":"a","n":1,"t":[]}}`,
	} {
		if got, eos := arrayNextBytes(t, mask, dec, dead); got != "" || eos {
			t.Errorf("dead prefix %s admitted %q eos=%v", dead, got, eos)
		}
	}
}

var (
	objectArrayIntRe    = regexp.MustCompile(`^-?(0|[1-9][0-9]*)$`)
	objectArrayNumberRe = regexp.MustCompile(`^-?(0|[1-9][0-9]*)(\.[0-9]+)?$`)
)

// canonicalObjectArrayValue renders v in the canonical form the schema
// demands, reporting false when v violates the schema.
func canonicalObjectArrayValue(s map[string]any, v any) (string, bool) {
	ty, _ := s["type"].(string)
	switch ty {
	case "object":
		obj, ok := v.(map[string]any)
		if !ok {
			return "", false
		}
		props, _ := s["properties"].(map[string]any)
		if r, ok := s["required"].([]any); ok {
			for _, x := range r {
				if _, ok := obj[x.(string)]; !ok {
					return "", false
				}
			}
		}
		keys := make([]string, 0, len(obj))
		for k := range obj {
			if _, ok := props[k]; !ok {
				return "", false
			}
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			lit, ok := canonicalObjectArrayValue(props[k].(map[string]any), obj[k])
			if !ok {
				return "", false
			}
			parts = append(parts, `"`+k+`":`+lit)
		}
		return "{" + strings.Join(parts, ",") + "}", true
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return "", false
		}
		if n, ok := s["minItems"].(float64); ok && len(arr) < int(n) {
			return "", false
		}
		if n, ok := s["maxItems"].(float64); ok && len(arr) > int(n) {
			return "", false
		}
		parts := make([]string, 0, len(arr))
		for _, it := range arr {
			lit, ok := canonicalObjectArrayValue(s["items"].(map[string]any), it)
			if !ok {
				return "", false
			}
			parts = append(parts, lit)
		}
		return "[" + strings.Join(parts, ",") + "]", true
	case "string":
		str, ok := v.(string)
		if !ok {
			return "", false
		}
		for i := 0; i < len(str); i++ {
			if !stringContentOK(str[i]) {
				return "", false
			}
		}
		if e, ok := s["enum"].([]any); ok {
			found := false
			for _, x := range e {
				found = found || x.(string) == str
			}
			if !found {
				return "", false
			}
		}
		return `"` + str + `"`, true
	case "boolean":
		b, ok := v.(bool)
		return strconv.FormatBool(b), ok
	case "integer", "number":
		n, ok := v.(json.Number)
		if !ok {
			return "", false
		}
		re := objectArrayNumberRe
		if ty == "integer" {
			re = objectArrayIntRe
		}
		if !re.MatchString(string(n)) {
			return "", false
		}
		if e, ok := s["enum"].([]any); ok {
			found := false
			for _, x := range e {
				found = found || strconv.FormatInt(int64(x.(float64)), 10) == string(n)
			}
			if !found {
				return "", false
			}
		}
		return string(n), true
	}
	return "", false
}

func validObjectArrayDocument(s map[string]any, doc []byte) bool {
	d := json.NewDecoder(bytes.NewReader(doc))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return false
	}
	if _, err := d.Token(); err == nil {
		return false
	}
	canon, ok := canonicalObjectArrayValue(s, v)
	return ok && canon == string(doc)
}

var objectArrayOpenSchemas = []string{
	`{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"qty":{"type":"integer"}},"required":["name","qty"],"additionalProperties":false}}},"required":["items"],"additionalProperties":false}`,
	`{"type":"object","properties":{"rows":{"type":"array","items":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]},"minItems":2,"maxItems":4},"z":{"type":"string"}},"required":["rows"]}`,
	`{"type":"object","properties":{"o":{"type":"object","properties":{"l":{"type":"array","items":{"type":"object","properties":{"a":{"type":"integer","enum":[1,10]},"b":{"type":"string","enum":["x","xy"]}},"required":["a","b"]},"minItems":1}},"required":["l"]},"p":{"type":"array","items":{"type":"integer"},"maxItems":2}},"required":["o"]}`,
	`{"type":"object","properties":{"e":{"type":"array","items":{"type":"object"}},"n":{"type":"integer"}},"required":["n"]}`,
	`{"type":"object","properties":{"rows":{"type":"array","items":{"type":"object","properties":{"n":{"type":"number"},"tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3},"w":{"type":"array","items":{"type":"integer"}}},"required":["n","tags","w"]},"maxItems":2}},"required":["rows"]}`,
}

// fak-test:runtime fast est=800ms lane=default
func TestJSONSchemaObjectArrayStateSpaceOracle(t *testing.T) {
	dec := jsonSchemaTokenDecoder{nil}
	for b := 0; b < 256; b++ {
		dec = append(dec, []byte{byte(b)})
	}
	for si, schema := range objectArrayOpenSchemas {
		var s map[string]any
		if err := json.Unmarshal([]byte(schema), &s); err != nil {
			t.Fatal(err)
		}
		mask, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: 0})
		if err != nil {
			t.Fatalf("schema %d: %v", si, err)
		}
		type state struct {
			cursors []jsonCursor
			path    []byte
		}
		type edge struct {
			from, to int
			b        byte
		}
		start := make([]jsonCursor, len(mask.programs))
		for i := range mask.programs {
			start[i] = mask.programs[i].start()
		}
		keyOf := func(c []jsonCursor) string { return (&jsonMaskState{cursors: c}).key() }
		complete := func(c []jsonCursor) bool {
			for i := range c {
				if mask.programs[i].complete(&c[i]) {
					return true
				}
			}
			return false
		}
		index := map[string]int{keyOf(start): 0}
		states := []state{{cursors: start}}
		var edges []edge
		for q := 0; q < len(states); q++ {
			if len(states) > 20000 {
				t.Fatalf("schema %d: cursor state space is not finite", si)
			}
			for b := 0; b < 256; b++ {
				next := append([]jsonCursor(nil), states[q].cursors...)
				alive := false
				for i := range next {
					if mask.programs[i].step(&next[i], byte(b)) {
						alive = true
					}
				}
				if !alive {
					continue
				}
				k := keyOf(next)
				id, ok := index[k]
				if !ok {
					id = len(states)
					index[k] = id
					states = append(states, state{cursors: next, path: append(append([]byte(nil), states[q].path...), byte(b))})
				}
				edges = append(edges, edge{from: q, to: id, b: byte(b)})
			}
		}
		reverse := make([][]edge, len(states))
		for _, e := range edges {
			reverse[e.to] = append(reverse[e.to], e)
		}
		dist := make([]int, len(states))
		via := make([]edge, len(states))
		var queue []int
		for i := range states {
			dist[i] = -1
			if complete(states[i].cursors) {
				dist[i] = 0
				queue = append(queue, i)
			}
		}
		for len(queue) > 0 {
			x := queue[0]
			queue = queue[1:]
			for _, e := range reverse[x] {
				if dist[e.from] < 0 {
					dist[e.from], via[e.from] = dist[x]+1, e
					queue = append(queue, e.from)
				}
			}
		}
		completion := func(i int) []byte {
			var out []byte
			for dist[i] > 0 {
				out = append(out, via[i].b)
				i = via[i].to
			}
			return out
		}
		bad := 0
		report := func(format string, a ...any) {
			if bad++; bad <= 8 {
				t.Errorf("schema %d: "+format, append([]any{si}, a...)...)
			}
		}
		for i := range states {
			if dist[i] < 0 {
				report("dead-end state reached by %q", states[i].path)
			}
			if complete(states[i].cursors) && !validObjectArrayDocument(s, states[i].path) {
				report("complete state is not a valid document: %q", states[i].path)
			}
		}
		for _, e := range edges {
			if dist[e.to] < 0 {
				continue
			}
			doc := append(append(append([]byte(nil), states[e.from].path...), e.b), completion(e.to)...)
			if !validObjectArrayDocument(s, doc) {
				report("admitted %q after %q completes to invalid %q", e.b, states[e.from].path, doc)
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestJSONSchemaObjectArrayLongArrayStaysFinite(t *testing.T) {
	dec := objectArrayVocab(43)
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"qty":{"type":"integer"},"tags":{"type":"array","items":{"type":"integer"}}},"required":["name","qty","tags"]}}},"required":["items"]}`), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	item := `{"name":"widget","qty":12,"tags":[1,23,4]}`
	prefix := `{"items":[` + item
	for i := 0; i < 10; i++ {
		prefix += `,` + item
	}
	visit := func(p string) {
		for i := len(`{"items":[`); i <= len(p); i++ {
			arrayNextBytes(t, mask, dec, p[:i])
		}
	}
	visit(prefix)
	before := len(mask.cache)
	visit(prefix + `,` + item)
	if len(mask.cache) != before {
		t.Fatalf("a later item added cache states: %d -> %d", before, len(mask.cache))
	}
	if got, eos := arrayNextBytes(t, mask, dec, prefix); got != `,]` || eos {
		t.Fatalf("after an item: bytes %q eos=%v, want only , and ]", got, eos)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestCompileJSONSchemaObjectArraySubset(t *testing.T) {
	dec := jsonSchemaTokenDecoder{[]byte(`{}`)}
	compile := func(schema string) error {
		_, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: -1})
		return err
	}
	wrap := func(items string) string {
		return `{"type":"object","properties":{"v":{"type":"array","items":` + items + `}},"required":["v"]}`
	}
	for _, items := range []string{
		`{"type":"object"}`,
		`{"type":"object","properties":{}}`,
		`{"type":"object","properties":{"a":{"type":"array","items":{"type":"string"},"maxItems":3}},"required":["a"]}`,
		`{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"array","items":{"type":"integer","enum":[1,2]}}},"required":["b"]}},"required":["a"],"additionalProperties":false}`,
	} {
		if err := compile(wrap(items)); err != nil {
			t.Errorf("items %s: %v", items, err)
		}
	}
	for _, schema := range []string{
		wrap(`{"type":"object","properties":{"a":{"type":"string"}}}`),
		wrap(`{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"integer"}}}},"required":["a"]}`),
		wrap(`{"type":"object","properties":{"a":{"type":"array","items":{"type":"object","properties":{}}}},"required":["a"]}`),
		wrap(`{"type":"object","properties":{"a":{"type":"array","items":{"type":"array","items":{"type":"string"}}}},"required":["a"]}`),
		wrap(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":true}`),
		wrap(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"minProperties":1}`),
		wrap(`{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"object","properties":{}}},"required":["b"]}},"required":["a"]}`),
		`{"type":"object","properties":{"m":{"type":"object","properties":{"n":{"type":"object","properties":{"v":{"type":"array","items":{"type":"object","properties":{}}}},"required":["v"]}},"required":["n"]}},"required":["m"]}`,
		wrap(`{"type":"array","items":{"type":"object","properties":{}}}`),
		wrap(`{"type":["object","null"],"properties":{}}`),
	} {
		if err := compile(schema); !errors.Is(err, ErrUnsupportedSchema) {
			t.Errorf("%s: err = %v, want ErrUnsupportedSchema", schema, err)
		}
	}
	props := make([]string, 0, 16)
	req := make([]string, 0, 16)
	for i := range 15 {
		props = append(props, fmt.Sprintf(`"p%02d":{"type":"integer"}`, i))
		req = append(req, fmt.Sprintf(`"p%02d"`, i))
	}
	item := func() string {
		return `{"type":"object","properties":{` + strings.Join(props, ",") + `},"required":[` + strings.Join(req, ",") + `]}`
	}
	if err := compile(wrap(item())); err != nil {
		t.Fatalf("16 properties counting item members: %v", err)
	}
	props = append(props, `"p15":{"type":"integer"}`)
	req = append(req, `"p15"`)
	if err := compile(wrap(item())); !errors.Is(err, ErrUnsupportedSchema) {
		t.Fatalf("17 properties counting item members: err = %v, want ErrUnsupportedSchema", err)
	}
	if _, err := Compile([]ToolSpec{{Name: "f", Schema: []byte(wrap(`{"type":"object","properties":{}}`))}}, dec, CompileOptions{EOS: -1}); !errors.Is(err, ErrUnsupportedSchema) {
		t.Fatalf("tool schema with an object array: err = %v, want ErrUnsupportedSchema", err)
	}
}

// fak-test:runtime fast est=1500ms lane=default
func TestJSONSchemaObjectArrayRandomOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(777))
	dec := objectArrayVocab(45)
	type stats struct {
		props, optional, maxDepth int
		refused                   bool
	}
	var gen func(depth int, inItem bool, st *stats) string
	gen = func(depth int, inItem bool, st *stats) string {
		st.maxDepth = max(st.maxDepth, depth)
		names := []string{"a", "b", "k", "", "a b", "}", ",", "]", "["}
		rng.Shuffle(len(names), func(i, j int) { names[i], names[j] = names[j], names[i] })
		var props, req []string
		for i := range rng.Intn(3) {
			st.props++
			var v string
			switch r := rng.Intn(8); {
			case r == 0 && depth < 4:
				v = gen(depth+1, inItem, st)
			case r == 1 && depth < 4:
				v = `{"type":"array","items":` + gen(depth+1, true, st) + `,"maxItems":` + strconv.Itoa(1+rng.Intn(2)) + `}`
				st.refused = st.refused || inItem
			case r == 2:
				v = `{"type":"array","items":{"type":"integer","enum":[2,20]},"minItems":` + strconv.Itoa(rng.Intn(2)) + `,"maxItems":2}`
			case r == 3:
				v = `{"type":"boolean"}`
			case r == 4:
				v = `{"type":"integer","enum":[1,10]}`
			case r == 5:
				v = `{"type":"string","enum":["x","xy"]}`
			default:
				v = `{"type":"string","const":"}"}`
			}
			props = append(props, fmt.Sprintf("%q:%s", names[i], v))
			if rng.Intn(3) != 0 {
				req = append(req, fmt.Sprintf("%q", names[i]))
			} else {
				st.optional++
				st.refused = st.refused || inItem
			}
		}
		return `{"type":"object","properties":{` + strings.Join(props, ",") + `},"required":[` + strings.Join(req, ",") + `]}`
	}
	checked := 0
	for range 400 {
		st := stats{}
		schema := gen(1, false, &st)
		mask, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: 0})
		if st.refused || st.props > maxJSONSchemaProperties || st.optional > maxJSONSchemaOptional || st.maxDepth > maxJSONSchemaDepth {
			if !errors.Is(err, ErrUnsupportedSchema) {
				t.Fatalf("%s (%+v): err = %v, want ErrUnsupportedSchema", schema, st, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s (%+v): %v", schema, st, err)
		}
		var s map[string]any
		_ = json.Unmarshal([]byte(schema), &s)
		docs, ok := enumerateObjectArrayDocs(t, s, 300)
		if !ok {
			continue
		}
		checkExactLanguage(t, schema, mask, dec, docs)
		checked++
	}
	if checked < 100 {
		t.Fatalf("only %d random schemas reached the language oracle", checked)
	}
}

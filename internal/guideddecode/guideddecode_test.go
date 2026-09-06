package guideddecode

import (
	"sort"
	"testing"
)

// preLit and sufLit mirror the package's fixed skeleton so the tests read as the
// spec does: full = preLit + name + sufLit.
const (
	preLit = `{"name":"`
	sufLit = `","arguments":`
)

// set is a tiny helper to build an expected allowed-set from a byte string.
func set(bs string) map[byte]bool {
	m := map[byte]bool{}
	for i := 0; i < len(bs); i++ {
		m[bs[i]] = true
	}
	return m
}

// eqSet compares two allowed-sets, distinguishing nil (UNCONSTRAINED) from an
// empty non-nil map (DEAD END) — the two carry different meaning in the return
// convention, so a test must never treat them as equal.
func eqSet(a, b map[byte]bool) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func showSet(m map[byte]bool) string {
	if m == nil {
		return "nil (UNCONSTRAINED)"
	}
	if len(m) == 0 {
		return "{} (DEAD END)"
	}
	bs := make([]byte, 0, len(m))
	for k := range m {
		bs = append(bs, k)
	}
	sort.Slice(bs, func(i, j int) bool { return bs[i] < bs[j] })
	return string(bs)
}

func TestAllowedNextBytes(t *testing.T) {
	// The primary schema used across most cases. "get" and "get_weather" overlap
	// so the shared-prefix branch is exercised; "list" is disjoint.
	schema := ToolSchema{Names: []string{"get", "get_weather", "list"}}

	cases := []struct {
		name   string
		schema ToolSchema
		prefix string
		want   map[byte]bool
	}{
		// --- empty prefix -----------------------------------------------------
		{"empty prefix admits open brace", schema, "", set("{")},

		// --- proper prefixes of PRE, consumed byte-by-byte --------------------
		{"PRE step 1", schema, `{`, set(`"`)},
		{"PRE step 2", schema, `{"`, set("n")},
		{"PRE step 3", schema, `{"n`, set("a")},
		{"PRE step 4", schema, `{"na`, set("m")},
		{"PRE step 5", schema, `{"nam`, set("e")},
		{"PRE step 6", schema, `{"name`, set(`"`)},
		{"PRE step 7", schema, `{"name"`, set(":")},
		{"PRE step 8", schema, `{"name":`, set(`"`)},

		// --- the enum branch: prefix == PRE -----------------------------------
		// First byte of every name; "get"/"get_weather" both start 'g', "list" 'l'.
		{"enum branch first bytes", schema, preLit, set("gl")},

		// --- inside a name, overlapping candidates ----------------------------
		{"name g -> e (only get*)", schema, preLit + "g", set("e")},
		{"name ge -> t", schema, preLit + "ge", set("t")},
		// After the shared "get": close it (") OR continue to get_weather (_).
		{"overlap get: close-or-continue", schema, preLit + "get", set(`"_`)},
		{"name get_ -> w", schema, preLit + "get_", set("w")},
		{"name li -> s", schema, preLit + "li", set("s")},
		{"full name list -> close quote only", schema, preLit + "list", set(`"`)},

		// --- full name + closing quote, then MID consumed byte-by-byte --------
		// After PRE+get+`"` the remaining fixed bytes are `,"arguments":`.
		{"after get\": MID ,", schema, preLit + "get" + `"`, set(",")},
		{"MID ,\" -> \"", schema, preLit + "get" + `",`, set(`"`)},
		{"MID ,\"a...", schema, preLit + "get" + `","`, set("a")},
		{"MID ...argument", schema, preLit + "get" + `","argument`, set("s")},
		{"MID ...arguments -> close quote", schema, preLit + "get" + `","arguments`, set(`"`)},
		{"MID ...arguments\" -> colon", schema, preLit + "get" + `","arguments"`, set(":")},

		// --- the UNCONSTRAINED (nil) transition -------------------------------
		{"full skeleton is unconstrained", schema, preLit + "get" + sufLit, nil},
		{"beyond skeleton stays unconstrained", schema, preLit + "get" + sufLit + `{"x":1}`, nil},

		// --- dead-end divergences (empty non-nil) -----------------------------
		{"diverge inside PRE", schema, `{"nXme`, map[byte]bool{}},
		{"diverge at enum (no such first byte)", schema, preLit + "z", map[byte]bool{}},
		{"diverge mid-name", schema, preLit + "gex", map[byte]bool{}},
		{"diverge in MID", schema, preLit + "get" + `"X`, map[byte]bool{}},

		// --- empty Names ------------------------------------------------------
		{"empty names still admits brace", ToolSchema{}, "", set("{")},
		{"empty names dead-end at enum", ToolSchema{}, preLit, map[byte]bool{}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := AllowedNextBytes([]byte(c.prefix), c.schema)
			if !eqSet(got, c.want) {
				t.Fatalf("AllowedNextBytes(%q)\n  got  %s\n  want %s",
					c.prefix, showSet(got), showSet(c.want))
			}
		})
	}
}

// TestSoundnessSpotCheck drives a full valid envelope prefix byte-by-byte and
// asserts the soundness contract directly: at every step until the skeleton is
// fully consumed, the ACTUAL next byte of the valid envelope is present in the
// allowed set (the mask never forbids a byte on a valid path). Once the fixed
// skeleton is consumed the result is nil (UNCONSTRAINED), and every subsequent
// byte of the arguments JSON is trivially admitted.
func TestSoundnessSpotCheck(t *testing.T) {
	schema := ToolSchema{Names: []string{"get_weather", "list"}}
	full := preLit + "get_weather" + sufLit + `{"city":"NYC"}}`
	skeletonLen := len(preLit) + len("get_weather") + len(sufLit)

	for i := 0; i < len(full); i++ {
		prefix := full[:i]
		next := full[i]
		got := AllowedNextBytes([]byte(prefix), schema)

		if i < skeletonLen {
			if got == nil {
				t.Fatalf("step %d prefix=%q: got nil (UNCONSTRAINED) before skeleton complete", i, prefix)
			}
			if len(got) == 0 {
				t.Fatalf("step %d prefix=%q: got DEAD END on a valid envelope path", i, prefix)
			}
			if !got[next] {
				t.Fatalf("step %d prefix=%q: valid next byte %q NOT in allowed set %s — soundness violated",
					i, prefix, string(next), showSet(got))
			}
		} else {
			if got != nil {
				t.Fatalf("step %d prefix=%q: expected nil (UNCONSTRAINED) past skeleton, got %s",
					i, prefix, showSet(got))
			}
		}
	}

	// The exact boundary: one byte short of the skeleton constrains to the final
	// ':' ; the complete skeleton is UNCONSTRAINED (nil).
	if got := AllowedNextBytes([]byte(full[:skeletonLen-1]), schema); !eqSet(got, set(":")) {
		t.Fatalf("one byte short of skeleton: want {':'}, got %s", showSet(got))
	}
	if got := AllowedNextBytes([]byte(full[:skeletonLen]), schema); got != nil {
		t.Fatalf("complete skeleton: want nil (UNCONSTRAINED), got %s", showSet(got))
	}
}

// TestDeadEndIsDistinctFromUnconstrained guards the return-convention invariant
// that a DEAD END is an empty NON-nil map, never nil, and never conflated with
// the UNCONSTRAINED nil.
func TestDeadEndIsDistinctFromUnconstrained(t *testing.T) {
	schema := ToolSchema{Names: []string{"get"}}
	dead := AllowedNextBytes([]byte(preLit+"zzz"), schema)
	if dead == nil {
		t.Fatal("dead end returned nil; must be an empty non-nil map")
	}
	if len(dead) != 0 {
		t.Fatalf("dead end should be empty, got %s", showSet(dead))
	}
	unc := AllowedNextBytes([]byte(preLit+"get"+sufLit), schema)
	if unc != nil {
		t.Fatalf("unconstrained should be nil, got %s", showSet(unc))
	}
}

// TestNamesWithSpecialBytesDoNotPanic checks the slice-1 out-of-scope note: a
// name containing a quote/backslash is handled literally (each byte is just the
// next byte of the name skeleton, never a JSON delimiter/escape) and never panics.
func TestNamesWithSpecialBytesDoNotPanic(t *testing.T) {
	schema := ToolSchema{Names: []string{`a"b`, `c\d`}}
	// Exercise several prefixes: the contract is first that none of them panic.
	for _, p := range []string{"", preLit, preLit + "a", preLit + `a"`, preLit + "c"} {
		_ = AllowedNextBytes([]byte(p), schema)
	}
	// Beyond "no panic": the special bytes must be admitted LITERALLY. At the name
	// start both names are enrolled by their first byte ('a' and 'c') — a regression
	// that mishandled special-byte names would drop one from the constrained set.
	if got := AllowedNextBytes([]byte(preLit), schema); !eqSet(got, set("ac")) {
		t.Fatalf("enum first bytes: got %s, want %s", showSet(got), showSet(set("ac")))
	}
	// A quote INSIDE a name is the literal next name byte, not a string terminator.
	if got := AllowedNextBytes([]byte(preLit+"a"), schema); !eqSet(got, set(`"`)) {
		t.Fatalf(`after "a" (name a"b): got %s, want %s`, showSet(got), showSet(set(`"`)))
	}
	// A backslash INSIDE a name is likewise a literal name byte, not an escape lead-in.
	if got := AllowedNextBytes([]byte(preLit+"c"), schema); !eqSet(got, set(`\`)) {
		t.Fatalf(`after "c" (name c\d): got %s, want %s`, showSet(got), showSet(set(`\`)))
	}
}

// TestAllowedNextBytes_WhitespaceTolerance witnesses issue #11719: optional JSON
// whitespace (spaces, tabs, newlines) within structural delimiters must be tolerated
// without dead-ending.
func TestAllowedNextBytes_WhitespaceTolerance(t *testing.T) {
	schema := ToolSchema{Names: []string{"get", "get_weather", "list"}}

	cases := []struct {
		name   string
		prefix string
		want   map[byte]bool
	}{
		{"space after colon before quote", `{"name": "`, set("gl")},
		{"space around colon", `{"name" : "`, set("gl")},
		{"space before brace", ` {"name":"`, set("gl")},
		{"space and newline after brace", "{\n  \"name\": \"", set("gl")},
		{"tab after colon before quote", "{\"name\":\t\"", set("gl")},
		{"space after colon only", `{"name": `, set(`"`)},
		{"space before colon only", `{"name" `, set(":")},
		{"space after brace only", `{ `, set(`"`)},
		{"space before brace only", ` `, set("{")},
		{"space after colon transitions to tool prefix", `{"name": "g`, set("e")},
		{"space after colon transitions to tool mid", `{"name": "ge`, set("t")},
		{"space after colon transitions to tool overlap", `{"name": "get`, set(`"_`)},
		{"suffix space after comma", `{"name":"get", "arguments":`, nil},
		{"suffix space around comma and colon", `{"name":"get" , "arguments" :`, nil},
		{"suffix space after colon", `{"name":"get","arguments": `, nil},
		{"suffix whitespace padded prefix", `{"name": "get", "arguments": `, nil},
		{"suffix space after tool closing quote", `{"name":"get" `, set(",")},
		{"suffix space after comma only", `{"name":"get", `, set(`"`)},
		{"suffix space after comma to arguments opening quote", `{"name": "get", "`, set("a")},
		{"suffix arguments prefix", `{"name":"get", "arg`, set("u")},
		{"suffix space before arguments colon", `{"name":"get", "arguments" `, set(":")},
		{"suffix tab around comma and colon", "{\"name\":\"get\"\t,\t\"arguments\"\t:", nil},
		{"crlf whitespace in delimiters", "{\r\n  \"name\": \"get\",\r\n  \"arguments\": ", nil},
		{"full envelope with whitespace and args", `{"name": "get", "arguments": {"city": "NYC"}}`, nil},
		{"list tool envelope with whitespace", `{"name": "list", "arguments": []}`, nil},
		{"get_weather tool envelope with whitespace", `{"name": "get_weather", "arguments": {"loc": "SF"}}`, nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := AllowedNextBytes([]byte(c.prefix), schema)
			if !eqSet(got, c.want) {
				t.Fatalf("AllowedNextBytes(%q)\n  got  %s\n  want %s",
					c.prefix, showSet(got), showSet(c.want))
			}
		})
	}
}

// TestWhitespacePaddedPrefixTransitions witnesses issue #11719 by walking through
// whitespace-padded envelope prefixes byte-by-byte, verifying that at every step
// the decoder transitions toward valid tool names and arguments payloads without
// dead-ending.
func TestWhitespacePaddedPrefixTransitions(t *testing.T) {
	schema := ToolSchema{Names: []string{"get", "get_weather", "list"}}

	// Sequence of prefixes simulating a decoder completing a tool call with
	// realistic whitespace emitted by frontier tokenizers:
	// {"name": "get", "arguments": {"query": "test"}}
	steps := []struct {
		prefix string
		want   map[byte]bool
	}{
		{`{"name": "`, set("gl")},
		{`{"name": "g`, set("e")},
		{`{"name": "ge`, set("t")},
		{`{"name": "get`, set(`"_`)},
		{`{"name": "get"`, set(",")},
		{`{"name": "get", `, set(`"`)},
		{`{"name": "get", "`, set("a")},
		{`{"name": "get", "a`, set("r")},
		{`{"name": "get", "ar`, set("g")},
		{`{"name": "get", "arg`, set("u")},
		{`{"name": "get", "argu`, set("m")},
		{`{"name": "get", "argum`, set("e")},
		{`{"name": "get", "argume`, set("n")},
		{`{"name": "get", "argumen`, set("t")},
		{`{"name": "get", "argument`, set("s")},
		{`{"name": "get", "arguments`, set(`"`)},
		{`{"name": "get", "arguments"`, set(":")},
		{`{"name": "get", "arguments":`, nil},
		{`{"name": "get", "arguments": `, nil},
		{`{"name": "get", "arguments": {`, nil},
		{`{"name": "get", "arguments": {"query": "test"}}`, nil},
	}

	for _, s := range steps {
		got := AllowedNextBytes([]byte(s.prefix), schema)
		if !eqSet(got, s.want) {
			t.Fatalf("prefix %q:\n  got  %s\n  want %s", s.prefix, showSet(got), showSet(s.want))
		}
		if got != nil && len(got) == 0 {
			t.Fatalf("prefix %q reached unexpected DEAD END", s.prefix)
		}
	}
}

// TestByteBitset exercises bitset creation, membership, removal, counting,
// and single-byte extraction across all 4 64-bit words (0-255).
func TestByteBitset(t *testing.T) {
	var bs ByteBitset
	if !bs.Empty() {
		t.Fatal("new bitset must be empty")
	}
	if bs.Count() != 0 {
		t.Fatalf("expected count 0, got %d", bs.Count())
	}
	if _, ok := bs.SingleByte(); ok {
		t.Fatal("empty bitset SingleByte must report ok=false")
	}

	testBytes := []byte{0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 254, 255}
	for _, b := range testBytes {
		if bs.Contains(b) {
			t.Fatalf("Contains(%d) reported true before Add", b)
		}
	}

	// Add testBytes one by one and test SingleByte & Count
	var singleBs ByteBitset
	singleBs.Add('x')
	if b, ok := singleBs.SingleByte(); !ok || b != 'x' {
		t.Fatalf("SingleByte on single element want ('x', true), got (%q, %v)", b, ok)
	}

	for i, b := range testBytes {
		bs.Add(b)
		if !bs.Contains(b) || !bs.Has(b) {
			t.Fatalf("Contains/Has(%d) reported false after Add", b)
		}
		if bs.Empty() {
			t.Fatal("bitset must not be empty after Add")
		}
		if bs.Count() != i+1 {
			t.Fatalf("count after %d adds: want %d, got %d", i+1, i+1, bs.Count())
		}
	}

	// Verify Bytes() matches testBytes in ascending order
	bytesList := bs.Bytes()
	if len(bytesList) != len(testBytes) {
		t.Fatalf("Bytes() len want %d, got %d", len(testBytes), len(bytesList))
	}
	for i, b := range bytesList {
		if b != testBytes[i] {
			t.Fatalf("Bytes()[%d] want %d, got %d", i, testBytes[i], b)
		}
	}

	// Test Set and Clear
	var bs2 ByteBitset
	bs2.Set('z')
	if !bs2.Has('z') {
		t.Fatal("Has('z') should be true after Set('z')")
	}
	bs2.Clear('z')
	if bs2.Has('z') || !bs2.Empty() {
		t.Fatal("bitset should be empty after Clear('z')")
	}

	// Test SingleByteBitset
	sbb := SingleByteBitset('q')
	if !sbb.Has('q') || sbb.Count() != 1 {
		t.Fatalf("SingleByteBitset('q') invalid: Has=%v, Count=%d", sbb.Has('q'), sbb.Count())
	}
	if b, ok := sbb.SingleByte(); !ok || b != 'q' {
		t.Fatalf("SingleByte on SingleByteBitset want ('q', true), got (%q, %v)", b, ok)
	}

	if _, ok := bs.SingleByte(); ok {
		t.Fatal("multi-element bitset SingleByte must report ok=false")
	}

	// Verify ToMap produces an identical map
	m := bs.ToMap()
	if len(m) != len(testBytes) {
		t.Fatalf("ToMap len want %d, got %d", len(testBytes), len(m))
	}
	for _, b := range testBytes {
		if !m[b] {
			t.Fatalf("ToMap missing byte %d", b)
		}
	}

	// Remove bytes and check
	for i, b := range testBytes {
		bs.Remove(b)
		if bs.Contains(b) {
			t.Fatalf("Contains(%d) reported true after Remove", b)
		}
		remaining := len(testBytes) - (i + 1)
		if bs.Count() != remaining {
			t.Fatalf("count after %d removes: want %d, got %d", i+1, remaining, bs.Count())
		}
	}
	if !bs.Empty() {
		t.Fatal("bitset should be empty after removing all elements")
	}
}

// TestAllowedNextByteBitset_Equivalence verifies that AllowedNextByteBitset and
// AllowedNextBytes produce equivalent decisions across all test prefixes.
func TestAllowedNextByteBitset_Equivalence(t *testing.T) {
	schema := ToolSchema{Names: []string{"get", "get_weather", "list"}}
	prefixes := []string{
		"",
		"{",
		`{"`,
		preLit,
		preLit + "g",
		preLit + "get",
		preLit + "get_",
		preLit + "get_weather",
		preLit + "get_weather" + `"`,
		preLit + "get_weather" + sufLit,
		preLit + "get_weather" + sufLit + `{"city":"NYC"}}`,
		preLit + "unknown",
		`{"wrong_key`,
		` {"name" : "get" , "arguments" : `,
		`{"name": "`,
		`{"name": "get", "arguments": `,
	}

	for _, p := range prefixes {
		prefixBytes := []byte(p)
		legacy := AllowedNextBytes(prefixBytes, schema)
		bitset, unconstrained := AllowedNextByteBitset(prefixBytes, schema)

		if unconstrained != (legacy == nil) {
			t.Fatalf("prefix %q unconstrained mismatch: bitset unconstrained=%v, legacy nil=%v",
				p, unconstrained, legacy == nil)
		}
		if unconstrained {
			if !bitset.Empty() {
				t.Fatalf("prefix %q: unconstrained bitset must be empty, got count=%d", p, bitset.Count())
			}
		} else {
			gotMap := bitset.ToMap()
			if !eqSet(gotMap, legacy) {
				t.Fatalf("prefix %q map mismatch:\n  bitset map: %s\n  legacy map: %s",
					p, showSet(gotMap), showSet(legacy))
			}
		}
	}
}

// TestAllowedNextBitset_ZeroAllocs witnesses issue #11858: AllowedNextBitset
// executes with 0 heap allocations across the entire tool envelope skeleton walk.
func TestAllowedNextBitset_ZeroAllocs(t *testing.T) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files"},
	}
	envelope := []byte(preLit + "get_weather" + sufLit + `{"city":"San Francisco"}}`)

	allocs := testing.AllocsPerRun(100, func() {
		for j := 0; j <= len(envelope); j++ {
			sinkBitset, sinkUnc = AllowedNextByteBitset(envelope[:j], schema)
		}
	})

	if allocs != 0 {
		t.Fatalf("AllowedNextBitset incurred %v allocs/run during envelope walk; want 0", allocs)
	}
}

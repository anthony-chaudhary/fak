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
		{"space after colon only", `{"name": `, set(`"`)},
		{"space before colon only", `{"name" `, set(":")},
		{"space after brace only", `{ `, set(`"`)},
		{"space before brace only", ` `, set("{")},
		{"suffix space after comma", `{"name":"get", "arguments":`, nil},
		{"suffix space around comma and colon", `{"name":"get" , "arguments" :`, nil},
		{"suffix space after colon", `{"name":"get","arguments": `, nil},
		{"suffix whitespace padded prefix", `{"name": "get", "arguments": `, nil},
		{"suffix space after tool closing quote", `{"name":"get" `, set(",")},
		{"suffix space after comma only", `{"name":"get", `, set(`"`)},
		{"suffix arguments prefix", `{"name":"get", "arg`, set("u")},
		{"suffix space before arguments colon", `{"name":"get", "arguments" `, set(":")},
		{"full envelope with whitespace and args", `{"name": "get", "arguments": {"city": "NYC"}}`, nil},
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

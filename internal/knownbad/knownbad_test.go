package knownbad

import (
	"strings"
	"testing"
)

// Signature must be stable across glob order and redundant "/**" suffixes (two
// agents hitting the same shared cause produce the same id) and must diverge on a
// different reason class, a disjoint tree, or a different failure hash.
func TestSignatureStability(t *testing.T) {
	base := Signature("build", []string{"internal/foo/**", "internal/bar/**"}, "")

	// Same trees, reversed order -> same signature (sort makes it order-free).
	if got := Signature("build", []string{"internal/bar/**", "internal/foo/**"}, ""); got != base {
		t.Errorf("signature is order-sensitive: %s != %s", got, base)
	}
	// Redundant "/**" vs bare dir normalize to the same prefix -> same signature.
	if got := Signature("build", []string{"internal/foo", "internal/bar/**/*"}, ""); got != base {
		t.Errorf("signature is not normalization-stable: %s != %s", got, base)
	}
	// Duplicate glob collapses -> same signature.
	if got := Signature("build", []string{"internal/foo/**", "internal/foo/**", "internal/bar/**"}, ""); got != base {
		t.Errorf("signature is not dedup-stable: %s != %s", got, base)
	}
	// A "sha256:"+hex shape like guardrsi.
	if !strings.HasPrefix(base, "sha256:") || len(base) != len("sha256:")+64 {
		t.Errorf("signature shape = %q, want sha256:<64 hex>", base)
	}

	// Distinct cause -> distinct signature on each dimension.
	for name, sig := range map[string]string{
		"reason":  Signature("test", []string{"internal/foo/**", "internal/bar/**"}, ""),
		"tree":    Signature("build", []string{"internal/other/**"}, ""),
		"failure": Signature("build", []string{"internal/foo/**", "internal/bar/**"}, "sha256:deadbeef"),
	} {
		if sig == base {
			t.Errorf("%s dimension did not change the signature (collision with base)", name)
		}
	}
}

func TestNormalizeTree(t *testing.T) {
	cases := map[string]string{
		"internal/foo/**":     "internal/foo",
		"internal/foo/*":      "internal/foo",
		"internal/foo/":       "internal/foo",
		"internal/foo/bar.go": "internal/foo/bar.go",
		`internal\foo\bar.go`: "internal/foo/bar.go",
		"internal/foo/**/*":   "internal/foo",
		"./internal/foo/**":   "internal/foo",
		"":                    "",
		"**":                  "",
		"*":                   "",
		"/etc/passwd":         "",
		"../escape":           "",
		"..":                  "",
	}
	for in, want := range cases {
		if got := NormalizeTree(in); got != want {
			t.Errorf("NormalizeTree(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTreesIntersect(t *testing.T) {
	cases := []struct {
		name string
		a, b []string
		want bool
	}{
		{"dir contains file", []string{"internal/foo/**"}, []string{"internal/foo/bar.go"}, true},
		{"file under dir reversed", []string{"internal/foo/bar.go"}, []string{"internal/foo/**"}, true},
		{"equal prefixes", []string{"internal/foo"}, []string{"internal/foo/**"}, true},
		{"sibling dirs disjoint", []string{"internal/foo/**"}, []string{"internal/other/**"}, false},
		{"prefix-name not a path boundary", []string{"internal/foo/**"}, []string{"internal/foobar/x.go"}, false},
		{"one of many overlaps", []string{"internal/a/**", "internal/foo/**"}, []string{"internal/foo/bar.go"}, true},
		{"empty query never matches", []string{"internal/foo/**"}, []string{}, false},
		{"all-invalid globs drop out", []string{"internal/foo/**"}, []string{"../nope", ""}, false},
	}
	for _, c := range cases {
		if got := TreesIntersect(c.a, c.b); got != c.want {
			t.Errorf("%s: TreesIntersect(%v,%v)=%v, want %v", c.name, c.a, c.b, got, c.want)
		}
	}
}

func TestRecordLiveness(t *testing.T) {
	const now = int64(1_000_000)
	cases := []struct {
		name string
		rec  Record
		want bool
	}{
		{"open no ttl", Record{Status: "open"}, true},
		{"open case-insensitive", Record{Status: "OPEN"}, true},
		{"open unexpired", Record{Status: "open", DiscoveredAtUnix: now - 10, TTLSeconds: 100}, true},
		{"open exactly expired", Record{Status: "open", DiscoveredAtUnix: now - 100, TTLSeconds: 100}, false},
		{"open expired", Record{Status: "open", DiscoveredAtUnix: now - 200, TTLSeconds: 100}, false},
		{"closed ignored", Record{Status: "closed"}, false},
		{"empty status ignored", Record{Status: ""}, false},
	}
	for _, c := range cases {
		if got := c.rec.Live(now); got != c.want {
			t.Errorf("%s: Live(%d)=%v, want %v", c.name, now, got, c.want)
		}
	}
}

// Match is the end-to-end done-condition of the spine: record a tree, an
// intersecting query matches, a disjoint one does not, and an expired/closed row
// is skipped.
func TestMatch(t *testing.T) {
	const now = int64(1_000_000)
	recs := []Record{
		NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "", now-10, 0),
		NewRecord("test", []string{"internal/baz/**"}, "expired", "agent-2", "", now-500, 100), // expired
		func() Record {
			r := NewRecord("lint", []string{"internal/qux/**"}, "closed", "a3", "", now-1, 0)
			r.Status = "closed"
			return r
		}(),
	}

	// Intersecting, live -> exactly the foo record.
	got := Match(recs, Query{TreeGlobs: []string{"internal/foo/bar.go"}}, now)
	if len(got) != 1 || got[0].ReasonClass != "build" {
		t.Fatalf("intersecting match = %+v, want the one live build record", got)
	}
	if got[0].Schema != Schema || got[0].Signature == "" || got[0].Status != StatusOpen {
		t.Errorf("matched record not well-formed: %+v", got[0])
	}

	// Disjoint -> no match.
	if got := Match(recs, Query{TreeGlobs: []string{"internal/other/**"}}, now); len(got) != 0 {
		t.Errorf("disjoint query matched: %+v", got)
	}
	// A query intersecting the expired row must NOT match (liveness gate).
	if got := Match(recs, Query{TreeGlobs: []string{"internal/baz/x.go"}}, now); len(got) != 0 {
		t.Errorf("expired row matched: %+v", got)
	}
	// A query intersecting the closed row must NOT match.
	if got := Match(recs, Query{TreeGlobs: []string{"internal/qux/x.go"}}, now); len(got) != 0 {
		t.Errorf("closed row matched: %+v", got)
	}
}

// ParseLedger round-trips a written line and skips blank, malformed, and
// foreign-schema rows without erroring — the shared-ledger robustness property.
func TestParseLedgerRobust(t *testing.T) {
	rec := NewRecord("build", []string{"internal/foo/**"}, "n", "a1", "", 42, 0)
	line, err := MarshalLine(rec)
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	blob := "\n" + line + "\n" +
		"{not valid json\n" +
		`{"schema":"some.other.v1","signature":"x"}` + "\n" +
		"   \n"
	got := ParseLedger([]byte(blob))
	if len(got) != 1 {
		t.Fatalf("ParseLedger kept %d rows, want 1 (foreign/torn rows skipped): %+v", len(got), got)
	}
	if got[0].Signature != rec.Signature || len(got[0].TreeGlobs) != 1 || got[0].TreeGlobs[0] != "internal/foo" {
		t.Errorf("round-trip lost data: %+v", got[0])
	}
}

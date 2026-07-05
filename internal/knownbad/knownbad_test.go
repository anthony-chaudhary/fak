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

// LeaseID turns a signature into one safe ref segment ("knownbad-<hex>"), strips
// the "sha256:" scheme (a colon is ref-illegal), is stable for the same signature
// (two agents derive the same mutex), and returns "" for a signature with no
// ref-safe content.
func TestLeaseID(t *testing.T) {
	sig := Signature("build", []string{"internal/foo/**"}, "")
	id := LeaseID(sig)
	if !strings.HasPrefix(id, "knownbad-") {
		t.Fatalf("LeaseID(%q) = %q, want a knownbad- prefix", sig, id)
	}
	if strings.ContainsRune(id, ':') || strings.ContainsAny(id, "/ \t") {
		t.Errorf("LeaseID produced a non-ref-safe id: %q", id)
	}
	if len(id) != len("knownbad-")+64 {
		t.Errorf("LeaseID(%q) = %q, want knownbad- + 64 hex", sig, id)
	}
	// Stable: the same signature yields the same mutex id.
	if LeaseID(sig) != id {
		t.Errorf("LeaseID is not deterministic for %q", sig)
	}
	// The colon scheme is stripped, not carried through.
	if LeaseID("sha256:deadBEEF01") != "knownbad-deadBEEF01" {
		t.Errorf("LeaseID did not strip the sha256: scheme: %q", LeaseID("sha256:deadBEEF01"))
	}
	// No usable content -> "" (the shell refuses rather than lease a degenerate id).
	for _, empty := range []string{"", "   ", "sha256:", "::::", "/\\ "} {
		if got := LeaseID(empty); got != "" {
			t.Errorf("LeaseID(%q) = %q, want empty", empty, got)
		}
	}
}

// WithClaim stamps the fixer without mutating the signature/tree, Claimed reflects
// the stamp, and FindLatestLive returns the LATEST live row for a signature (so a
// superseding claim row wins over the original) while skipping other signatures and
// dead rows.
func TestClaimStampAndFindLatestLive(t *testing.T) {
	const now = int64(1_000_000)
	orig := NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "", now-10, 0)
	if orig.Claimed() {
		t.Fatalf("a fresh record must not be Claimed()")
	}
	claim := orig.WithClaim("fixer-7", now)
	if !claim.Claimed() || claim.ClaimedBy != "fixer-7" || claim.ClaimedAtUnix != now {
		t.Fatalf("WithClaim did not stamp the fixer: %+v", claim)
	}
	if claim.Signature != orig.Signature || len(claim.TreeGlobs) != len(orig.TreeGlobs) || claim.ReasonClass != orig.ReasonClass {
		t.Errorf("WithClaim mutated the failure it points at: %+v vs %+v", claim, orig)
	}
	if orig.Claimed() {
		t.Errorf("WithClaim mutated the receiver instead of returning a copy")
	}

	other := NewRecord("test", []string{"internal/bar/**"}, "", "a2", "", now-5, 0)
	dead := NewRecord("lint", []string{"internal/qux/**"}, "expired", "a3", "", now-500, 100) // TTL elapsed by `now`
	records := []Record{orig, other, claim, dead}

	got, ok := FindLatestLive(records, orig.Signature, now)
	if !ok || !got.Claimed() || got.ClaimedBy != "fixer-7" {
		t.Fatalf("FindLatestLive did not return the superseding claim row: %+v ok=%v", got, ok)
	}
	// An unknown signature is not found.
	if _, ok := FindLatestLive(records, "sha256:nope", now); ok {
		t.Errorf("FindLatestLive found a signature that is not in the ledger")
	}
	// A signature whose only rows are dead is not live/claimable.
	if _, ok := FindLatestLive([]Record{dead}, dead.Signature, now); ok {
		t.Errorf("FindLatestLive returned a signature with no live row")
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

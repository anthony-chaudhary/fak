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
		"internal/foo/**":              "internal/foo",
		"internal/foo/*":               "internal/foo",
		"internal/foo/":                "internal/foo",
		"internal/foo/bar.go":          "internal/foo/bar.go",
		`internal\foo\bar.go`:          "internal/foo/bar.go",
		"internal/foo/**/*":            "internal/foo",
		"./internal/foo/**":            "internal/foo",
		"":                             "",
		"**":                           "",
		"*":                            "",
		"/etc/passwd":                  "",
		"../escape":                    "",
		"..":                           "",
		"C:/work/fak/internal/gateway": "",
		`C:\work\fak\internal\gateway`: "",
		"C:":                           "",
		"C:/":                          "",
		`C:\`:                          "",
		"c:/work/fak/internal/gateway": "",
		`c:\work\fak\internal\gateway`: "",
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

// TestMatchSupersedeByResolve is the pure-fold witness of the W6 (#2718) retract
// invariant: a signature's state is its LATEST row. An open row FOLLOWED BY a
// resolved row for the SAME signature no longer matches (the resolve superseded the
// open row), EVEN THOUGH the earlier open row still sits on the ledger — otherwise a
// resolve could never clear the W4 hold. A disjoint signature that is still open must
// keep matching, so the supersede is scoped to the one signature, not the ledger.
func TestMatchSupersedeByResolve(t *testing.T) {
	const now = int64(1_000_000)
	open := NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "", now-10, 0)
	other := NewRecord("test", []string{"internal/bar/**"}, "bar broke", "agent-2", "", now-10, 0)
	// The ledger keeps the original open row AND appends a resolved row for the same
	// signature — the real append-to-supersede shape a resolve writes.
	recs := []Record{
		open,
		other,
		open.WithResolve("fixer", now, "tests"),
	}

	// foo was resolved: its LATEST row is not live, so it no longer matches even though
	// the earlier open foo row is still present.
	if got := Match(recs, Query{TreeGlobs: []string{"internal/foo/x.go"}}, now); len(got) != 0 {
		t.Fatalf("a resolved signature must stop matching, got %+v", got)
	}
	if _, live := FindLatestLive(recs, open.Signature, now); live {
		t.Errorf("FindLatestLive must report a resolved signature as not live")
	}
	// bar is untouched and still open: the supersede did not leak across signatures.
	if got := Match(recs, Query{TreeGlobs: []string{"internal/bar/y.go"}}, now); len(got) != 1 {
		t.Fatalf("a disjoint still-open signature must keep matching, got %+v", got)
	}
	if _, live := FindLatestLive(recs, other.Signature, now); !live {
		t.Errorf("the untouched open signature must still be live")
	}
}

func TestLiveRecords(t *testing.T) {
	const now = int64(1_000_000)
	openFoo := NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "", now-10, 0)
	openBar := NewRecord("test", []string{"internal/bar/**"}, "bar broke", "agent-2", "", now-10, 0)
	expired := NewRecord("lint", []string{"internal/baz/**"}, "aged out", "agent-3", "", now-500, 100)
	recs := []Record{
		openFoo,
		openBar,
		expired,
		// foo is superseded by a resolved row — its LATEST row is not live.
		openFoo.WithResolve("fixer", now, "tests"),
		// bar picks up a claim but stays open+live; the claimed fixer must survive.
		openBar.WithClaim("fixer-9", now-5),
	}

	live := LiveRecords(recs, now)
	if len(live) != 1 {
		t.Fatalf("LiveRecords should collapse to the single still-live signature, got %d: %+v", len(live), live)
	}
	got := live[0]
	if got.Signature != openBar.Signature {
		t.Fatalf("the surviving live signature should be bar (foo resolved, baz expired), got %s", got.Signature)
	}
	if got.ClaimedBy != "fixer-9" {
		t.Fatalf("LiveRecords must return the LATEST row per signature (the claimed one), got claimant %q", got.ClaimedBy)
	}
	// Ordering is first-seen: prove it by adding a second live signature discovered later.
	openQux := NewRecord("build", []string{"internal/qux/**"}, "qux broke", "agent-4", "", now-1, 0)
	live2 := LiveRecords(append(recs, openQux), now)
	if len(live2) != 2 || live2[0].Signature != openBar.Signature || live2[1].Signature != openQux.Signature {
		t.Fatalf("LiveRecords must preserve first-seen order, got %+v", live2)
	}
	if LiveRecords(nil, now) != nil {
		t.Fatalf("an empty ledger should fold to no live records")
	}
}

func TestWithResolveStampsTerminalWitnessRow(t *testing.T) {
	const now = int64(1_000_000)
	open := NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "sha256:bad", now-10, 0)
	claim := open.WithClaim("fixer-7", now-5)

	resolved := claim.WithResolve(" fixer-7 ", now, " tests ")
	if !resolved.Resolved() || resolved.Live(now) {
		t.Fatalf("resolved row terminal state = Resolved:%v Live:%v, want resolved and non-live", resolved.Resolved(), resolved.Live(now))
	}
	if resolved.Status != StatusResolved || resolved.ResolvedBy != "fixer-7" || resolved.ResolvedAtUnix != now || resolved.Witness != "tests" {
		t.Fatalf("WithResolve did not stamp the witness-backed release: %+v", resolved)
	}
	if resolved.Signature != open.Signature || resolved.ReasonClass != open.ReasonClass || resolved.FailureHash != open.FailureHash {
		t.Errorf("WithResolve mutated the failure identity: %+v vs %+v", resolved, open)
	}
	if len(resolved.TreeGlobs) != len(open.TreeGlobs) || resolved.TreeGlobs[0] != open.TreeGlobs[0] {
		t.Errorf("WithResolve mutated the release tree: %+v vs %+v", resolved.TreeGlobs, open.TreeGlobs)
	}
	if resolved.ClaimedBy != "fixer-7" || resolved.ClaimedAtUnix != now-5 {
		t.Errorf("WithResolve must preserve fixer claim bookkeeping for operator read-back: %+v", resolved)
	}
	if open.Resolved() || !open.Live(now) {
		t.Errorf("WithResolve mutated the original open row: %+v", open)
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

// WithDerivedFrom stamps the attempt-genealogy edge (#4100): a non-empty parent
// sets DerivedFrom (trimmed) and lights Derived(); it survives a JSONL round-trip;
// and — the load-bearing backward-compat property — an empty parent leaves the row
// byte-identical to a pre-#4100 row (omitempty drops the key entirely).
func TestWithDerivedFromGenealogy(t *testing.T) {
	root := NewRecord("build", []string{"internal/foo/**"}, "root attempt", "agent-1", "", 100, 0)

	// A root row is flat: no edge, and the JSON carries no derived_from key.
	if root.Derived() {
		t.Errorf("a root row reports Derived()=true with DerivedFrom=%q", root.DerivedFrom)
	}
	rootLine, err := MarshalLine(root)
	if err != nil {
		t.Fatalf("MarshalLine(root): %v", err)
	}
	if strings.Contains(rootLine, "derived_from") {
		t.Errorf("empty DerivedFrom must be omitted (not byte-identical to a pre-#4100 row): %s", rootLine)
	}

	// A child derived from the root's signature carries the edge, trimmed.
	child := NewRecord("build", []string{"internal/foo/bar.go"}, "mutated variant", "agent-2", "", 200, 0).
		WithDerivedFrom("  " + root.Signature + "  ")
	if !child.Derived() || child.DerivedFrom != root.Signature {
		t.Fatalf("WithDerivedFrom did not stamp+trim the parent: Derived=%v DerivedFrom=%q want %q",
			child.Derived(), child.DerivedFrom, root.Signature)
	}

	// The edge survives a ledger round-trip.
	childLine, err := MarshalLine(child)
	if err != nil {
		t.Fatalf("MarshalLine(child): %v", err)
	}
	got := ParseLedger([]byte(childLine))
	if len(got) != 1 || got[0].DerivedFrom != root.Signature {
		t.Fatalf("round-trip lost the genealogy edge: %+v", got)
	}

	// A whitespace-only parent collapses to empty (flat row), same as a root.
	if blank := root.WithDerivedFrom("   "); blank.Derived() || blank.DerivedFrom != "" {
		t.Errorf("whitespace-only parent must collapse to empty, got %q", blank.DerivedFrom)
	}

	// WithDerivedFrom is a birth stamp: it must not disturb the signature/tree/reason
	// or the lifecycle bookkeeping the supersede fold reads.
	if child.Signature != NewRecord("build", []string{"internal/foo/bar.go"}, "mutated variant", "agent-2", "", 200, 0).Signature {
		t.Errorf("WithDerivedFrom must not change the signature")
	}
}

// TestCompact folds a mixed ledger (superseded rows, an expired signature, a
// resolved and a revoked terminal) and asserts the GC keeps every live signature's
// LATEST row, drops expired + superseded rows, bounds the resolved/revoked tail, and
// is a no-op on a second pass (#3471).
func TestCompact(t *testing.T) {
	const now = int64(1_000_000)
	const ttl = int64(100)
	mk := func(reason, tree string, at int64) Record {
		return NewRecord(reason, []string{tree}, "", "agent", "", at, ttl)
	}

	live := mk("build", "internal/live/**", now)            // open, unexpired
	liveClaim := live.WithClaim("fixer", now+1)             // supersedes live's open row (still open -> live)
	expired := mk("test", "internal/expired/**", now-200)   // open but TTL lapsed: now-200+100 < now
	resolvedOpen := mk("lint", "internal/resolved/**", now) // superseded by the resolve below
	resolved := resolvedOpen.WithResolve("fixer", now+2, "tests")
	revokedOpen := mk("vet", "internal/revoked/**", now) // superseded by the revoke below
	revoked := revokedOpen.WithRevoke("op", now+3, "was flaky, not shared")

	records := []Record{live, liveClaim, expired, resolvedOpen, resolved, revokedOpen, revoked}

	// keepTerminal < 0 keeps all terminal history: live + resolved + revoked survive,
	// expired + the 3 superseded rows drop.
	kept, stats := Compact(records, now, -1)
	if stats.InputRows != 7 || stats.KeptRows != 3 || stats.Signatures != 4 {
		t.Fatalf("keep-all stats = %+v, want Input 7 Kept 3 Signatures 4", stats)
	}
	if stats.LiveKept != 1 || stats.TerminalKept != 2 || stats.SupersededDropped != 3 ||
		stats.ExpiredDropped != 1 || stats.TerminalDropped != 0 {
		t.Fatalf("keep-all stats breakdown = %+v", stats)
	}
	if len(kept) != 3 {
		t.Fatalf("keep-all kept %d rows, want 3", len(kept))
	}
	// The kept live row must be the LATEST (claimed) row, not the original open row.
	if kept[0].ClaimedBy != "fixer" {
		t.Errorf("compact kept a superseded live row, not the latest: %+v", kept[0])
	}
	// The dropped expired signature must be gone entirely.
	for _, r := range kept {
		if strings.Contains(strings.Join(r.TreeGlobs, ","), "internal/expired") {
			t.Errorf("expired signature survived compaction: %+v", r)
		}
	}
	// Balance invariant.
	if stats.InputRows-stats.KeptRows != stats.SupersededDropped+stats.ExpiredDropped+stats.TerminalDropped {
		t.Errorf("stats do not balance: %+v", stats)
	}

	// A second compaction of an already-compact ledger is a no-op (stable rewrite).
	kept2, stats2 := Compact(kept, now, -1)
	if stats2.KeptRows != stats2.InputRows || len(kept2) != len(kept) {
		t.Errorf("re-compact was not a no-op: %+v", stats2)
	}

	// keepTerminal = 1 keeps only the MOST-recently-retracted terminal (the revoke at
	// now+3 beats the resolve at now+2), dropping the older resolved signature.
	kept1, stats1 := Compact(records, now, 1)
	if stats1.KeptRows != 2 || stats1.TerminalKept != 1 || stats1.TerminalDropped != 1 {
		t.Fatalf("keep-1 stats = %+v, want Kept 2 TerminalKept 1 TerminalDropped 1", stats1)
	}
	var sawRevoke, sawResolve bool
	for _, r := range kept1 {
		if r.Revoked() {
			sawRevoke = true
		}
		if r.Resolved() {
			sawResolve = true
		}
	}
	if !sawRevoke || sawResolve {
		t.Errorf("keep-1 kept the wrong terminal tail (sawRevoke=%v sawResolve=%v)", sawRevoke, sawResolve)
	}

	// keepTerminal = 0 is live-only: just the one live signature survives.
	keptLiveOnly, stats0 := Compact(records, now, 0)
	if stats0.KeptRows != 1 || len(keptLiveOnly) != 1 || !keptLiveOnly[0].Live(now) {
		t.Fatalf("keep-0 = %d rows / stats %+v, want 1 live row", len(keptLiveOnly), stats0)
	}

	// An empty ledger compacts to nothing without panicking.
	if out, s := Compact(nil, now, -1); len(out) != 0 || s.InputRows != 0 {
		t.Errorf("empty compact = %d rows / %+v", len(out), s)
	}
}

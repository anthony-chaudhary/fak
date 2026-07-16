package toolproc

// repeatclass_test.go — the FAILURE-CLASS proof for #4764's repeat classifier
// (the DoD "fixtures" bullet and the issue Witness). Each test pins one class of
// the safety contract the classifier must never get wrong:
//
//   - IMMUTABLE reuse: equivalent skill-read spellings fold to one identity and
//     every repeat after the first is avoidable (KEYED).
//   - INVALIDATION after mutation: a re-read whose content digest changed forms a
//     NEW identity, so the stale entry is never served — proven against the
//     digest-blind control where the same stream over-counts the saving.
//   - NO reuse for writes / unknowns: a push and an unrecognized command are
//     NEVER served locally (zero avoidable), fail-closed.
//   - BOUNDED freshness for status polls: only repeats inside the freshness window
//     of the last real fetch coalesce — a poll past the window is a fresh fetch,
//     so the saving is bounded, not blanket.
//   - SECRET redaction: no retained analytics field ever carries a credential.
//
// All of it is the pure decision spine: same records + same config ⇒ identical
// report (asserted below).

import (
	"reflect"
	"strings"
	"testing"
)

// findGroup returns the single group whose Identity matches, failing if absent.
func findGroup(t *testing.T, rep RepeatReport, identity string) RepeatGroup {
	t.Helper()
	for _, g := range rep.Groups {
		if g.Identity == identity {
			return g
		}
	}
	t.Fatalf("identity %q not in report (%d groups)", identity, len(rep.Groups))
	return RepeatGroup{}
}

// TestImmutableReadFoldsSpellingsAndServesEveryRepeat proves the 640+204 skill-read
// audit line: three equivalent spellings of one immutable file (cat, Get-Content
// -Raw, Get-Content -LiteralPath, mixed path separators) fold to ONE identity, and
// every repeat after the first is an avoidable, KEYED-reusable spawn.
func TestImmutableReadFoldsSpellingsAndServesEveryRepeat(t *testing.T) {
	const digest = "sha256:abc123"
	recs := []CallRecord{
		{Tool: "shell_command", Raw: `cat C:/x/super-loop/SKILL.md`, AtMS: 0, OutputBytes: 1000, Digest: digest},
		{Tool: "shell_command", Raw: `Get-Content -Raw C:\x\super-loop\SKILL.md`, AtMS: 10, OutputBytes: 1000, Digest: digest},
		{Tool: "shell_command", Raw: `Get-Content -LiteralPath C:/x/super-loop/SKILL.md -Raw`, AtMS: 20, OutputBytes: 1000, Digest: digest},
		{Tool: "shell_command", Raw: `cat C:/x/super-loop/SKILL.md`, AtMS: 30, OutputBytes: 1000, Digest: digest},
	}
	rep := ClassifyRepeats(recs, RepeatConfig{})

	if len(rep.Groups) != 1 {
		t.Fatalf("equivalent spellings must fold to one group, got %d: %+v", len(rep.Groups), rep.Groups)
	}
	g := rep.Groups[0]
	wantID := "read:C:/x/super-loop/SKILL.md@" + digest
	if g.Identity != wantID {
		t.Errorf("identity: want %q, got %q", wantID, g.Identity)
	}
	if g.Class != ClassImmutableRead || g.Reuse != ReuseKeyed {
		t.Errorf("class/reuse: want IMMUTABLE_READ/KEYED, got %s/%s", g.Class, g.Reuse)
	}
	if g.Count != 4 {
		t.Errorf("count: want 4, got %d", g.Count)
	}
	// obs4 exactly re-spells obs1; obs2/obs3 are new equivalent spellings.
	if g.ExactDups != 1 || g.NearDups != 2 {
		t.Errorf("dups: want exact=1 near=2, got exact=%d near=%d", g.ExactDups, g.NearDups)
	}
	// Immutable: every repeat after the first serves from the digest cache.
	if g.AvoidableSpawns != 3 || g.AvoidableInputBytes != 3000 {
		t.Errorf("avoidable: want 3 spawns / 3000 bytes, got %d / %d", g.AvoidableSpawns, g.AvoidableInputBytes)
	}
}

// TestMutationInvalidatesImmutableReuse proves invalidation-after-mutation at the
// key layer: two reads of one path at digest d1, then the file mutates (digest d2)
// and is read twice more. The digest fold splits this into two identities, so the
// FIRST read of the mutated file is a real fetch (not avoided) — only 2 repeats are
// avoidable, not 3. The digest-blind control over the identical stream over-counts
// to 3, which is exactly the unsafe blanket-cache behaviour the fold prevents.
func TestMutationInvalidatesImmutableReuse(t *testing.T) {
	const path = `C:/x/config.json`
	keyed := []CallRecord{
		{Tool: "shell_command", Raw: "cat " + path, AtMS: 0, OutputBytes: 100, Digest: "d1"},
		{Tool: "shell_command", Raw: "cat " + path, AtMS: 10, OutputBytes: 100, Digest: "d1"},
		{Tool: "shell_command", Raw: "cat " + path, AtMS: 20, OutputBytes: 100, Digest: "d2"}, // mutated → fresh fetch
		{Tool: "shell_command", Raw: "cat " + path, AtMS: 30, OutputBytes: 100, Digest: "d2"},
	}
	rep := ClassifyRepeats(keyed, RepeatConfig{})
	if len(rep.Groups) != 2 {
		t.Fatalf("a digest change must open a new identity, want 2 groups, got %d: %+v", len(rep.Groups), rep.Groups)
	}
	g1 := findGroup(t, rep, "read:"+path+"@d1")
	g2 := findGroup(t, rep, "read:"+path+"@d2")
	if g1.AvoidableSpawns != 1 || g2.AvoidableSpawns != 1 {
		t.Errorf("each digest cohort avoids exactly its own repeat: got d1=%d d2=%d", g1.AvoidableSpawns, g2.AvoidableSpawns)
	}
	if rep.Totals.AvoidableSpawns != 2 {
		t.Errorf("mutation forces a fresh fetch: total avoidable want 2, got %d", rep.Totals.AvoidableSpawns)
	}

	// Control: the SAME stream with no observed digest folds path-only and would
	// serve the post-mutation read from the stale entry — 3 avoidable. This is the
	// unsafe behaviour the digest fold exists to prevent; asserting the delta makes
	// the invalidation the point of the test, not an accident of the fixture.
	blind := make([]CallRecord, len(keyed))
	copy(blind, keyed)
	for i := range blind {
		blind[i].Digest = ""
	}
	blindRep := ClassifyRepeats(blind, RepeatConfig{})
	if len(blindRep.Groups) != 1 || blindRep.Totals.AvoidableSpawns != 3 {
		t.Fatalf("digest-blind control: want 1 group / 3 avoidable, got %d groups / %d avoidable",
			len(blindRep.Groups), blindRep.Totals.AvoidableSpawns)
	}
	if blindRep.Totals.AvoidableSpawns <= rep.Totals.AvoidableSpawns {
		t.Errorf("digest fold must REDUCE avoidable across a mutation (blind=%d keyed=%d)",
			blindRep.Totals.AvoidableSpawns, rep.Totals.AvoidableSpawns)
	}
}

// TestWritesAndUnknownsAreNeverReused proves the fail-closed floor: a repeated push
// (idempotent write) and a repeated unrecognized command are classified NEVER-reuse
// with zero avoidable spawns — a write's effect is not a value to serve, and an
// unmatched command is fail-closed.
func TestWritesAndUnknownsAreNeverReused(t *testing.T) {
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 0, OutputBytes: 40},
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 1000, OutputBytes: 40},
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 2000, OutputBytes: 40},
		{Tool: "shell_command", Raw: "frobnicate --wibble 3", AtMS: 100, OutputBytes: 9},
		{Tool: "shell_command", Raw: "frobnicate --wibble 3", AtMS: 1100, OutputBytes: 9},
	}
	rep := ClassifyRepeats(recs, RepeatConfig{})

	push := findGroup(t, rep, "write:git push origin main")
	if push.Class != ClassIdempotentWrite || push.Reuse != ReuseNever {
		t.Errorf("push: want IDEMPOTENT_WRITE/NEVER, got %s/%s", push.Class, push.Reuse)
	}
	if push.AvoidableSpawns != 0 || push.AvoidableInputBytes != 0 {
		t.Errorf("push must never be reused, got %d spawns / %d bytes", push.AvoidableSpawns, push.AvoidableInputBytes)
	}

	unk := findGroup(t, rep, "unknown:shell_command|frobnicate --wibble 3")
	if unk.Class != ClassUnknown || unk.Reuse != ReuseNever {
		t.Errorf("unknown: want UNKNOWN/NEVER, got %s/%s", unk.Class, unk.Reuse)
	}
	if unk.AvoidableSpawns != 0 {
		t.Errorf("unknown must be fail-closed, got %d avoidable", unk.AvoidableSpawns)
	}
	if rep.Totals.AvoidableSpawns != 0 {
		t.Errorf("no write/unknown repeat is ever avoidable, total want 0, got %d", rep.Totals.AvoidableSpawns)
	}
}

// TestStatusPollIsFreshnessBoundedNotBlanket proves the mutable-status contract:
// a tight `git status` poll storm is promoted to POLLING_LOOP with FRESHNESS_BOUNDED
// reuse, flag order folds to one identity, and — critically — only the polls INSIDE
// the freshness window of the last real fetch coalesce. A poll past the window is a
// fresh fetch, so the saving is bounded (4 of 6), never the blanket 5-of-6 a stable
// cache of mutable state would unsafely claim.
func TestStatusPollIsFreshnessBoundedNotBlanket(t *testing.T) {
	// Freshness window is the 2s default. Spacings: 0,1s,2s inside the first
	// window; a 3s jump opens a fresh fetch at 5s; then 1s,1s inside the second.
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 0, OutputBytes: 200},
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 1000, OutputBytes: 200},
		{Tool: "shell_command", Raw: "git status --branch --short", AtMS: 2000, OutputBytes: 200}, // reordered flags fold
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 5000, OutputBytes: 200}, // past window → fresh fetch
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 6000, OutputBytes: 200},
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 7000, OutputBytes: 200},
	}
	rep := ClassifyRepeats(recs, RepeatConfig{})
	if len(rep.Groups) != 1 {
		t.Fatalf("flag order must fold to one identity, got %d groups: %+v", len(rep.Groups), rep.Groups)
	}
	g := findGroup(t, rep, "query:git status --branch --short")
	if g.Class != ClassPollStorm || g.Reuse != ReuseFreshnessBounded {
		t.Errorf("class/reuse: want POLLING_LOOP/FRESHNESS_BOUNDED, got %s/%s", g.Class, g.Reuse)
	}
	if g.FreshMS != DefaultFreshnessWindowMS {
		t.Errorf("freshness window: want %d, got %d", DefaultFreshnessWindowMS, g.FreshMS)
	}
	if g.Count != 6 {
		t.Errorf("count: want 6, got %d", g.Count)
	}
	// Two real fetches (t=0, t=5000); the other four polls coalesce.
	if g.AvoidableSpawns != 4 || g.AvoidableInputBytes != 800 {
		t.Errorf("bounded saving: want 4 spawns / 800 bytes, got %d / %d", g.AvoidableSpawns, g.AvoidableInputBytes)
	}
	if g.AvoidableSpawns >= g.Count-1 {
		t.Errorf("freshness must be BOUNDED, not blanket: avoidable %d must be < count-1 %d", g.AvoidableSpawns, g.Count-1)
	}
}

// TestInfrequentMutableQueryStaysMutableNotPollingLoop proves the POLLING_LOOP
// promotion is gated on frequency+regularity: a mutable query repeated only twice,
// far apart, stays MUTABLE_QUERY (still freshness-bounded, but not flagged as a poll
// storm) and — spaced well past the window — yields no avoidable saving.
func TestInfrequentMutableQueryStaysMutableNotPollingLoop(t *testing.T) {
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git diff", AtMS: 0, OutputBytes: 500},
		{Tool: "shell_command", Raw: "git diff", AtMS: 600_000, OutputBytes: 500},
	}
	rep := ClassifyRepeats(recs, RepeatConfig{})
	g := findGroup(t, rep, "query:git diff")
	if g.Class != ClassMutableQuery {
		t.Errorf("class: want MUTABLE_QUERY (too few/too sparse for a poll loop), got %s", g.Class)
	}
	if g.Reuse != ReuseFreshnessBounded {
		t.Errorf("reuse: want FRESHNESS_BOUNDED, got %s", g.Reuse)
	}
	if g.AvoidableSpawns != 0 {
		t.Errorf("a repeat 10min apart is past every freshness window: want 0 avoidable, got %d", g.AvoidableSpawns)
	}
}

// TestSecretsNeverReachAnalytics proves the redaction boundary: a command carrying a
// flag-borne token, a KEY=VALUE credential, and a self-identifying key is normalized
// with every secret replaced by the placeholder — no retained field (Canonical or
// Identity, directly and through the full report) ever carries the raw credential.
func TestSecretsNeverReachAnalytics(t *testing.T) {
	const ghToken = "ghp_0123456789abcdefghijABCD"
	const pwVal = "SuperSecret123"
	const skKey = "sk-0123456789abcdefghijKL"
	raw := "deploy --token " + ghToken + " PASSWORD=" + pwVal + " " + skKey

	nc := Normalize(CallRecord{Tool: "shell_command", Raw: raw}, RepeatConfig{})
	for _, field := range []string{nc.Canonical, nc.Identity} {
		for _, secret := range []string{ghToken, pwVal, skKey} {
			if strings.Contains(field, secret) {
				t.Errorf("secret %q leaked into retained field %q", secret, field)
			}
		}
		if !strings.Contains(field, redactedPlaceholder) {
			t.Errorf("field %q should carry the redaction placeholder", field)
		}
	}

	// The same guarantee must hold through the full classifier, not just Normalize.
	rep := ClassifyRepeats([]CallRecord{{Tool: "shell_command", Raw: raw, OutputBytes: 10}}, RepeatConfig{})
	for _, g := range rep.Groups {
		for _, secret := range []string{ghToken, pwVal, skKey} {
			if strings.Contains(g.Canonical, secret) || strings.Contains(g.Identity, secret) {
				t.Fatalf("secret %q leaked into report group %+v", secret, g)
			}
		}
	}
}

// TestClassifyRepeatsIsDeterministic pins the pure-fold contract the doc block
// claims: same records + same config ⇒ byte-identical report.
func TestClassifyRepeatsIsDeterministic(t *testing.T) {
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: 100, Digest: "d"},
		{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 10, OutputBytes: 100, Digest: "d"},
		{Tool: "shell_command", Raw: "git status", AtMS: 20, OutputBytes: 50},
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 30, OutputBytes: 40},
	}
	a := ClassifyRepeats(recs, RepeatConfig{})
	b := ClassifyRepeats(recs, RepeatConfig{})
	if !reflect.DeepEqual(a, b) {
		t.Errorf("classifier is not deterministic:\n a=%+v\n b=%+v", a, b)
	}
	if a.Schema != RepeatReportSchema {
		t.Errorf("schema: want %q, got %q", RepeatReportSchema, a.Schema)
	}
}

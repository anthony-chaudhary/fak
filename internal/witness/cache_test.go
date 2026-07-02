package witness

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// cacheFakeGit is a canned-evidence Runner for the verdict-cache tests: a rev store
// (rev -> sha) for the key derivation, canned rung outputs, and a per-verb call
// counter so a test asserts a HIT by the rung NOT running — the deterministic form
// of the #2152 "returns from cache" witness (a wall-clock assertion would flake).
type cacheFakeGit struct {
	commonDir string
	revs      map[string]string // rev spec -> resolved sha ("HEAD", "abc^{commit}", ...)
	showOut   string            // `show --name-only` output (the notests/symptom rung)
	grepOut   string            // `log --grep` output
	calls     map[string]int    // verb -> count
}

func (f *cacheFakeGit) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	f.calls[args[0]]++
	switch args[0] {
	case "rev-parse":
		if args[1] == "--git-common-dir" {
			return f.commonDir + "\n", 0, nil
		}
		var out []string
		for _, a := range args[1:] {
			if strings.HasPrefix(a, "--") {
				continue
			}
			sha, ok := f.revs[a]
			if !ok {
				return "", 1, nil
			}
			out = append(out, sha)
		}
		return strings.Join(out, "\n") + "\n", 0, nil
	case "merge-base":
		return "", 0, nil // arg IS an ancestor of HEAD
	case "show":
		return f.showOut, 0, nil
	case "log":
		return f.grepOut, 0, nil
	}
	return "", 0, nil
}

func newCacheFakeGit(t *testing.T) *cacheFakeGit {
	t.Helper()
	return &cacheFakeGit{
		commonDir: t.TempDir(),
		revs:      map[string]string{"HEAD": "headsha1", "abc^{commit}": "abcsha1"},
		calls:     map[string]int{},
	}
}

func TestResolveRefCacheKeyPinsResolvedRefs(t *testing.T) {
	f := newCacheFakeGit(t)
	f.revs["HEAD^{commit}"] = "headsha1"

	key1, ok := ResolveRefCacheKey(context.Background(), f.run, "", "dos_commit_audit", []string{"HEAD^{commit}"}, "diff-witnessed")
	if !ok {
		t.Fatalf("first key did not resolve")
	}
	key2, ok := ResolveRefCacheKey(context.Background(), f.run, "", "dos_commit_audit", []string{"HEAD^{commit}"}, "diff-witnessed")
	if !ok {
		t.Fatalf("second key did not resolve")
	}
	if key1 != key2 {
		t.Fatalf("unchanged ref produced different keys:\n%s\n%s", key1, key2)
	}
	if strings.Contains(key1, "HEAD") {
		t.Fatalf("cache key must pin resolved SHAs, not symbolic refs: %s", key1)
	}

	f.revs["HEAD^{commit}"] = "headsha2"
	key3, ok := ResolveRefCacheKey(context.Background(), f.run, "", "dos_commit_audit", []string{"HEAD^{commit}"}, "diff-witnessed")
	if !ok {
		t.Fatalf("moved ref key did not resolve")
	}
	if key3 == key1 {
		t.Fatalf("moved ref reused stale key %s", key3)
	}
}

func TestVerdictCacheRoundTripsDeterministically(t *testing.T) {
	c := NewVerdictCache(t.TempDir())
	key := WitnessCacheKey("dos_commit_audit", []string{"abcsha"}, "diff-witnessed")
	entry := CachedVerdict{
		Key:     key,
		Verb:    "dos_commit_audit",
		Subject: "abcsha",
		Verdict: "OK",
	}
	if !c.Put(entry) {
		t.Fatalf("put verdict")
	}
	got, hit := c.Get(key)
	if !hit {
		t.Fatalf("cache miss after put")
	}
	if got.Verb != entry.Verb || got.Subject != entry.Subject || got.Verdict != entry.Verdict {
		t.Fatalf("cached verdict did not round-trip: %+v", got)
	}

	first, err := os.ReadFile(c.path(key))
	if err != nil {
		t.Fatalf("read cache entry: %v", err)
	}
	if bytes.Contains(first, []byte("created_unix")) {
		t.Fatalf("cache entry contains nondeterministic timestamp: %s", first)
	}
	if !c.Put(entry) {
		t.Fatalf("second put verdict")
	}
	second, err := os.ReadFile(c.path(key))
	if err != nil {
		t.Fatalf("read second cache entry: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("same verdict wrote nondeterministic bytes:\nfirst:  %s\nsecond: %s", first, second)
	}
}

// TestCacheHitSkipsRung is the #2152 acceptance witness in deterministic form: a
// second resolve of the same claim at the same ref returns the IDENTICAL verdict
// WITHOUT re-running the evidence rung — one `show` for two resolves; only the O(1)
// rev-parse key derivation repeats.
func TestCacheHitSkipsRung(t *testing.T) {
	f := newCacheFakeGit(t)
	f.showOut = "internal/gateway/gateway.go\n" // the commit touched no gating test
	r := NewWithRunner(f.run, "")

	first := r.Resolve(context.Background(), nil, "notests:abc")
	if first != abi.WitnessConfirmed {
		t.Fatalf("first resolve = %v, want Confirmed", first)
	}
	if f.calls["show"] != 1 {
		t.Fatalf("first resolve ran show %d times, want 1", f.calls["show"])
	}

	second := r.Resolve(context.Background(), nil, "notests:abc")
	if second != first {
		t.Fatalf("cached verdict %v != original %v", second, first)
	}
	if f.calls["show"] != 1 {
		t.Fatalf("second resolve re-ran the rung (show=%d), want the cache hit to skip it", f.calls["show"])
	}
}

// TestCacheSharedAcrossResolvers: the cache is FLEET-SHARED by placement — a second
// Resolver (a peer agent session on the same clone) hits the entry the first wrote.
func TestCacheSharedAcrossResolvers(t *testing.T) {
	f := newCacheFakeGit(t)
	f.showOut = "cmd/fak/main.go\n"
	if got := NewWithRunner(f.run, "").Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessConfirmed {
		t.Fatalf("seed resolve = %v", got)
	}
	shows := f.calls["show"]
	if got := NewWithRunner(f.run, "").Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessConfirmed {
		t.Fatalf("peer resolve = %v", got)
	}
	if f.calls["show"] != shows {
		t.Fatalf("peer resolver re-ran the rung, want a shared-cache hit")
	}
}

// TestCacheMovedRefMisses: the key pins resolved SHAs, so a moved ref (same claim
// string, new underlying commit) MISSES and recomputes — content-addressed
// invalidation, no bookkeeping.
func TestCacheMovedRefMisses(t *testing.T) {
	f := newCacheFakeGit(t)
	r := NewWithRunner(f.run, "")

	if got := r.Resolve(context.Background(), nil, "ancestor:abc"); got != abi.WitnessConfirmed {
		t.Fatalf("first resolve = %v", got)
	}
	if f.calls["merge-base"] != 1 {
		t.Fatalf("merge-base = %d, want 1", f.calls["merge-base"])
	}
	// A cache hit at the same HEAD: no second merge-base.
	r.Resolve(context.Background(), nil, "ancestor:abc")
	if f.calls["merge-base"] != 1 {
		t.Fatalf("same-HEAD re-verify re-ran merge-base (%d), want cache hit", f.calls["merge-base"])
	}
	// HEAD moves (a new commit landed): the same claim must recompute.
	f.revs["HEAD"] = "headsha2"
	r.Resolve(context.Background(), nil, "ancestor:abc")
	if f.calls["merge-base"] != 2 {
		t.Fatalf("moved-HEAD re-verify hit the stale entry (merge-base=%d), want recompute", f.calls["merge-base"])
	}
}

func TestCacheNonexistentCommonDirDisablesCache(t *testing.T) {
	f := newCacheFakeGit(t)
	f.commonDir = filepath.Join(t.TempDir(), "missing-common-dir")
	f.showOut = "internal/gateway/gateway.go\n"
	r := NewWithRunner(f.run, "")

	if got := r.Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessConfirmed {
		t.Fatalf("first resolve = %v, want Confirmed", got)
	}
	if got := r.Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessConfirmed {
		t.Fatalf("second resolve = %v, want Confirmed", got)
	}
	if f.calls["show"] != 2 {
		t.Fatalf("nonexistent common dir should disable cache, show=%d want 2", f.calls["show"])
	}
}

// TestCacheAbstainNotCached: an abstain is uncertainty, not evidence — it is never
// memoized, so the rung re-runs once the environment recovers.
func TestCacheAbstainNotCached(t *testing.T) {
	f := newCacheFakeGit(t)
	failingShow := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if args[0] == "show" {
			f.calls["show"]++
			return "", 128, nil // a transient git failure -> the rung abstains
		}
		return f.run(ctx, dir, args...)
	}
	r := NewWithRunner(failingShow, "")
	if got := r.Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessAbstain {
		t.Fatalf("failing rung = %v, want Abstain", got)
	}
	if got := r.Resolve(context.Background(), nil, "notests:abc"); got != abi.WitnessAbstain {
		t.Fatalf("second failing resolve = %v, want Abstain", got)
	}
	if f.calls["show"] != 2 {
		t.Fatalf("abstain was cached (show=%d), want the rung re-run every time", f.calls["show"])
	}
}

// TestCacheDisabledByEnv: FAK_WITNESS_CACHE=off is the escape hatch — every resolve
// re-runs the rung and nothing is written.
func TestCacheDisabledByEnv(t *testing.T) {
	t.Setenv(CacheFlagEnv, "off")
	f := newCacheFakeGit(t)
	f.showOut = "internal/gateway/gateway.go\n"
	r := NewWithRunner(f.run, "")
	r.Resolve(context.Background(), nil, "notests:abc")
	r.Resolve(context.Background(), nil, "notests:abc")
	if f.calls["show"] != 2 {
		t.Fatalf("disabled cache still memoized (show=%d), want 2 rung runs", f.calls["show"])
	}
	entries, _ := os.ReadDir(f.commonDir)
	for _, e := range entries {
		if e.Name() == "fak" {
			t.Fatalf("disabled cache still wrote entries under %s", f.commonDir)
		}
	}
}

// TestCacheMutableClaimsNeverCached: a working-tree claim (path:/clean:/committed:)
// has no content address — it must re-resolve every call and derive no key at all.
func TestCacheMutableClaimsNeverCached(t *testing.T) {
	f := newCacheFakeGit(t)
	r := NewWithRunner(f.run, "")
	r.Resolve(context.Background(), nil, "clean:docs/")
	r.Resolve(context.Background(), nil, "clean:docs/")
	if f.calls["status"] != 0 && f.calls["rev-parse"] != 0 {
		t.Fatalf("mutable claim derived a cache key (rev-parse=%d), want none", f.calls["rev-parse"])
	}
	if f.calls["status"] != 2 {
		t.Fatalf("clean rung ran %d times, want 2 (never cached)", f.calls["status"])
	}
}

// TestCacheSymptomModeSplitsKey: the symptom verdict depends on which rung was armed
// (structural vs red-then-green execution), so the mode is part of the key — arming
// FAK_WITNESS_SYMPTOM must not hit a struct-mode entry.
func TestCacheSymptomModeSplitsKey(t *testing.T) {
	f := newCacheFakeGit(t)
	f.showOut = "internal/gateway/gateway.go\n" // no _test.go touched -> structurally REFUTED
	r := NewWithRunner(f.run, "")
	if got := r.Resolve(context.Background(), nil, "symptom:abc"); got != abi.WitnessRefuted {
		t.Fatalf("struct-mode resolve = %v, want Refuted", got)
	}
	shows := f.calls["show"]
	t.Setenv(SymptomFlagEnv, "1")
	if got := r.Resolve(context.Background(), nil, "symptom:abc"); got != abi.WitnessRefuted {
		t.Fatalf("exec-mode resolve = %v, want Refuted (still no test touched)", got)
	}
	if f.calls["show"] <= shows {
		t.Fatalf("exec-mode resolve hit the struct-mode entry, want a distinct key and a re-run")
	}
}

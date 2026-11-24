package workerworktree

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processstart"
)

func useDisambiguationScratchBase(t *testing.T) string {
	t.Helper()
	base := t.TempDir()
	old := disambiguationScratchBase
	disambiguationScratchBase = func() string { return base }
	t.Cleanup(func() {
		waitDisambiguationScratchReapIdle(t)
		disambiguationScratchBase = old
	})
	return base
}

func waitDisambiguationScratchReapIdle(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for disambiguationScratchReap.Load() {
		if time.Now().After(deadline) {
			t.Fatal("background scratch reap did not finish")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func disambiguationScratchEntries(t *testing.T, base string) []string {
	t.Helper()
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), disambiguationScratchPrefix) {
			names = append(names, entry.Name())
		}
	}
	return names
}

func stubDisambiguationCorpus(t *testing.T, analyzer func(ctx context.Context, root, generated string) ([]byte, error)) {
	t.Helper()
	oldList, oldRead, oldAnalyzer := listDisambiguationTree, readDisambiguationObject, runAnalyzer
	t.Cleanup(func() {
		listDisambiguationTree, readDisambiguationObject, runAnalyzer = oldList, oldRead, oldAnalyzer
	})
	listDisambiguationTree = func(context.Context, string, string) ([]byte, error) {
		return disambiguationTreeListing(
			disambiguationTreeEntry{Mode: "100644", ObjectID: "script", Size: 1, Path: disambiguationAnalyzerContractPath},
			disambiguationTreeEntry{Mode: "100644", ObjectID: "doc", Size: 1, Path: "docs/a.md"},
		), nil
	}
	readDisambiguationObject = func(context.Context, string, string) ([]byte, error) { return []byte("x"), nil }
	runAnalyzer = analyzer
}

func TestComputeDisambiguationWitnessLeavesNoScratchTree(t *testing.T) {
	for _, tc := range []struct {
		name    string
		payload []byte
		err     error
	}{
		{name: "analyzer-ok", payload: []byte(`{"ok":true,"corpus":{}}`)},
		{name: "analyzer-error", err: errors.New("analyzer exploded")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := useDisambiguationScratchBase(t)
			self := currentDisambiguationScratchOwner()
			var sawRoot string
			stubDisambiguationCorpus(t, func(_ context.Context, root, _ string) ([]byte, error) {
				sawRoot = root
				if _, err := os.Stat(filepath.Join(root, "docs", "a.md")); err != nil {
					t.Errorf("corpus file not materialized under %s: %v", root, err)
				}
				return tc.payload, tc.err
			})

			w, _ := computeDisambiguationWitness(context.Background(), "/repo", "tree-id", DisambiguationWitness{}, func(string) {})

			scratch := filepath.Base(filepath.Dir(sawRoot))
			if filepath.Dir(filepath.Dir(sawRoot)) != base {
				t.Fatalf("scratch root %s not under the injected base %s", sawRoot, base)
			}
			owner, tagged := parseDisambiguationScratchOwner(scratch)
			if !tagged || owner != self {
				t.Fatalf("scratch %q owner = %+v tagged=%v, want %+v", scratch, owner, tagged, self)
			}
			if left := disambiguationScratchEntries(t, base); len(left) != 0 {
				t.Fatalf("scratch trees left behind: %v", left)
			}
			if w.ScratchCleanup != "" {
				t.Fatalf("ScratchCleanup = %q, want empty", w.ScratchCleanup)
			}
			waitDisambiguationScratchReapIdle(t)
			disambiguationScratchMu.Lock()
			held := len(disambiguationScratchHeld)
			disambiguationScratchMu.Unlock()
			if held != 0 {
				t.Fatalf("held scratch set not released: %d entries", held)
			}
		})
	}
}

func TestComputeDisambiguationWitnessScratchCleanupIsNotCached(t *testing.T) {
	path := filepath.Join(t.TempDir(), "entry.json")
	w := DisambiguationWitness{Fresh: true, ScratchCleanup: "scratch tree still present"}
	if err := writeCachedDisambiguation(path, "key", w); err != nil {
		t.Fatal(err)
	}
	cached, _, ok := readCachedDisambiguation(path, "key")
	if !ok || cached.ScratchCleanup != "" || !cached.Fresh {
		t.Fatalf("cached = %+v ok=%v, want Fresh and no ScratchCleanup", cached, ok)
	}
}

func TestReapDisambiguationScratchReclaimsOnlyProvablyOrphanedTrees(t *testing.T) {
	base := useDisambiguationScratchBase(t)
	self := currentDisambiguationScratchOwner()
	if self.Start == 0 {
		t.Skip("process-start identity unavailable on this platform")
	}
	parent := disambiguationScratchOwner{PID: os.Getppid()}
	if started, ok := processstart.Start(parent.PID); ok && !started.IsZero() {
		parent.Start = started.UnixNano()
	}
	reusedPID := disambiguationScratchOwner{PID: self.PID, Start: self.Start + int64(time.Hour)}
	clockStep := disambiguationScratchOwner{PID: self.PID, Start: self.Start + int64(time.Second)}

	mk := func(name string) string {
		path := filepath.Join(base, name)
		if err := os.MkdirAll(filepath.Join(path, "tree", "docs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "tree", "docs", "a.md"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	tagged := func(o disambiguationScratchOwner, suffix string) string {
		return strings.TrimSuffix(o.pattern(), "*") + suffix
	}
	reused := mk(tagged(reusedPID, "1"))
	leakedSelf := mk(tagged(self, "2"))
	steppedSelf := mk(tagged(clockStep, "5"))
	heldSelf := mk(tagged(self, "3"))
	liveParent := mk(tagged(parent, "4"))
	legacyOld := mk(disambiguationScratchPrefix + "111")
	legacyFresh := mk(disambiguationScratchPrefix + "222")
	unrelated := mk("fak-concept-fresh-333")
	old := time.Now().Add(-2 * disambiguationLegacyScratchAge)
	if err := os.Chtimes(legacyOld, old, old); err != nil {
		t.Fatal(err)
	}

	disambiguationScratchMu.Lock()
	disambiguationScratchHeld[heldSelf] = true
	disambiguationScratchMu.Unlock()
	t.Cleanup(func() {
		disambiguationScratchMu.Lock()
		delete(disambiguationScratchHeld, heldSelf)
		disambiguationScratchMu.Unlock()
	})

	waitDisambiguationScratchReapIdle(t)
	reapDisambiguationScratch(base, self, time.Now())

	for path, wantGone := range map[string]bool{
		reused: true, leakedSelf: true, legacyOld: true,
		heldSelf: false, liveParent: false, steppedSelf: false, legacyFresh: false, unrelated: false,
	} {
		_, err := os.Stat(path)
		if gone := errors.Is(err, os.ErrNotExist); gone != wantGone {
			t.Errorf("%s: gone=%v, want %v (stat err %v)", filepath.Base(path), gone, wantGone, err)
		}
	}
}

func TestParseDisambiguationScratchOwnerRejectsUntaggedNames(t *testing.T) {
	for _, name := range []string{
		disambiguationScratchPrefix + "123456",
		disambiguationScratchPrefix + "x-1-2",
		disambiguationScratchPrefix + "0-1-2",
		"other-12-34-56",
	} {
		if owner, ok := parseDisambiguationScratchOwner(name); ok {
			t.Errorf("%q parsed as tagged owner %+v", name, owner)
		}
	}
	want := disambiguationScratchOwner{PID: 42, Start: 1700000000123456789}
	got, ok := parseDisambiguationScratchOwner(strings.TrimSuffix(want.pattern(), "*") + "987")
	if !ok || got != want {
		t.Fatalf("round trip = %+v ok=%v, want %+v", got, ok, want)
	}
}

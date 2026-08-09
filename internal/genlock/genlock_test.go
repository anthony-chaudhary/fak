package genlock_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/genlock"
)

// artifact is the repo-relative path used throughout; a nested one so the
// MkdirAll path is exercised too.
const artifact = "docs/marketing/updates.json"

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func read(t *testing.T, root, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// stamp captures the two things a "no write at all" claim has to survive.
type stamp struct {
	mtime time.Time
	body  string
}

func snap(t *testing.T, root, rel string) stamp {
	t.Helper()
	fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return stamp{mtime: fi.ModTime(), body: read(t, root, rel)}
}

func openLock(t *testing.T, root string) *genlock.Lock {
	t.Helper()
	l, err := genlock.Open(root, "marketing-aeo")
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// TestUnchangedInputWritesNothingAtAll is the ticket's central claim, and the reason
// the whole package exists: a skip is not a write of identical bytes, it is the absence
// of a write.
//
// Three separate witnesses, because each one alone can be satisfied by a cheat. The
// content check alone passes if the file was rewritten with the same bytes. The mtime
// check alone passes if the render was skipped but the lock still churned. The
// render-not-called check alone passes if the render ran and its output was discarded.
// On a generator that stamps a clock the difference is not academic — a rewrite would
// have produced DIFFERENT bytes.
func TestUnchangedInputWritesNothingAtAll(t *testing.T) {
	root := t.TempDir()
	input := []byte("ships: a,b,c")

	l := openLock(t, root)
	renders := 0
	out, err := l.Sync(artifact, input, func() ([]byte, error) {
		renders++
		return []byte("built at 10:00:00\n"), nil
	})
	if err != nil || out != genlock.Wrote {
		t.Fatalf("first Sync = %v, %v; want Wrote with no error (nothing on disk yet)", out, err)
	}
	if err := l.Save(); err != nil {
		t.Fatal(err)
	}

	before := snap(t, root, artifact)
	lockBefore := snap(t, root, l.Rel())
	// A filesystem whose mtime resolution is coarse would let a rewrite hide inside the
	// same tick; sleeping past it makes the mtime witness real rather than lucky.
	time.Sleep(20 * time.Millisecond)

	// Second run, same input, reloaded from the committed lock exactly as a fresh
	// process in another checkout would.
	l2 := openLock(t, root)
	out, err = l2.Sync(artifact, input, func() ([]byte, error) {
		renders++
		return []byte("built at 10:00:01\n"), nil // a clock-stamped generator: different bytes
	})
	if err != nil {
		t.Fatal(err)
	}
	if out != genlock.Skipped {
		t.Fatalf("second Sync over unchanged input = %v, want Skipped", out)
	}
	if renders != 1 {
		t.Errorf("render ran %d times, want 1 — the skip must not even build the bytes it "+
			"is not going to write; for a subprocess generator that build IS the cost", renders)
	}
	if err := l2.Save(); err != nil {
		t.Fatal(err)
	}

	after := snap(t, root, artifact)
	if after.body != before.body {
		t.Errorf("artifact content changed on a skip:\n before %q\n after  %q", before.body, after.body)
	}
	if !after.mtime.Equal(before.mtime) {
		t.Errorf("artifact mtime moved on a skip (%v -> %v). Identical bytes rewritten is still "+
			"a write: it re-dirties the stat cache for every peer sharing the checkout, which is "+
			"exactly the churn this lock exists to remove.", before.mtime, after.mtime)
	}
	lockAfter := snap(t, root, l.Rel())
	if !lockAfter.mtime.Equal(lockBefore.mtime) || lockAfter.body != lockBefore.body {
		t.Errorf("the lock file itself was rewritten on a no-op run — that moves the churn "+
			"rather than removing it (mtime %v -> %v)", lockBefore.mtime, lockAfter.mtime)
	}
}

func TestSyncRebuildsWhenSomethingActuallyChanged(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mutate  func(t *testing.T, root string, l *genlock.Lock)
		input   []byte
		wantOut genlock.Outcome
	}{
		{
			name:    "input changed",
			mutate:  func(*testing.T, string, *genlock.Lock) {},
			input:   []byte("ships: a,b,c,d"),
			wantOut: genlock.Wrote,
		},
		{
			name: "artifact deleted",
			mutate: func(t *testing.T, root string, _ *genlock.Lock) {
				if err := os.Remove(filepath.Join(root, filepath.FromSlash(artifact))); err != nil {
					t.Fatal(err)
				}
			},
			input:   []byte("ships: a,b,c"),
			wantOut: genlock.Wrote,
		},
		{
			name: "artifact hand-edited under the lock",
			mutate: func(t *testing.T, root string, _ *genlock.Lock) {
				write(t, root, artifact, "someone edited this by hand\n")
			},
			input:   []byte("ships: a,b,c"),
			wantOut: genlock.Wrote,
		},
		{
			name: "lock entry dropped (the documented force-rebuild)",
			mutate: func(_ *testing.T, _ string, l *genlock.Lock) {
				delete(l.Input, artifact)
			},
			input:   []byte("ships: a,b,c"),
			wantOut: genlock.Wrote,
		},
		{
			name:    "nothing changed",
			mutate:  func(*testing.T, string, *genlock.Lock) {},
			input:   []byte("ships: a,b,c"),
			wantOut: genlock.Skipped,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			l := openLock(t, root)
			if _, err := l.Sync(artifact, []byte("ships: a,b,c"), func() ([]byte, error) {
				return []byte("v1\n"), nil
			}); err != nil {
				t.Fatal(err)
			}
			if err := l.Save(); err != nil {
				t.Fatal(err)
			}

			l2 := openLock(t, root)
			tc.mutate(t, root, l2)
			got, err := l2.Sync(artifact, tc.input, func() ([]byte, error) { return []byte("v2\n"), nil })
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.wantOut {
				t.Fatalf("Sync = %v, want %v", got, tc.wantOut)
			}
			want := "v1\n"
			if tc.wantOut == genlock.Wrote {
				want = "v2\n"
			}
			if body := read(t, root, artifact); body != want {
				t.Errorf("artifact = %q, want %q", body, want)
			}
		})
	}
}

// TestSumIsCheckoutPortable pins the CRLF fold. Without it a committed lock would
// disagree with itself across the Windows and Linux sessions that share this repo, and
// every checkout would rebuild for a reason that has nothing to do with the source.
func TestSumIsCheckoutPortable(t *testing.T) {
	if lf, crlf := genlock.Sum([]byte("a\nb\n")), genlock.Sum([]byte("a\r\nb\r\n")); lf != crlf {
		t.Errorf("Sum disagrees across line endings (%s vs %s); a committed lock would be "+
			"unusable on a checkout with the other convention", lf[:12], crlf[:12])
	}
	// Binary is hashed verbatim: folding CR LF inside it would corrupt the identity.
	bin, other := []byte{0x00, 0x0d, 0x0a, 0x01}, []byte{0x00, 0x0a, 0x01}
	if genlock.Sum(bin) == genlock.Sum(other) {
		t.Error("Sum folded CRLF inside binary bytes; that changes the artifact's identity")
	}
}

// TestCanonicalIsNotConcatenation guards the classic length-prefix bug: two different
// multi-part inputs that concatenate to the same string must not hash the same, or a
// caller could move a byte across the boundary and the lock would call it unchanged.
func TestCanonicalIsNotConcatenation(t *testing.T) {
	a := genlock.Canonical([]byte("ab"), []byte("c"))
	b := genlock.Canonical([]byte("a"), []byte("bc"))
	if genlock.Sum(a) == genlock.Sum(b) {
		t.Error(`Canonical("ab","c") and Canonical("a","bc") hash alike — the fragment boundary is forgeable`)
	}
	if genlock.Sum(a) != genlock.Sum(genlock.Canonical([]byte("ab"), []byte("c"))) {
		t.Error("Canonical is not stable across calls")
	}
}

// TestStaleAnswersForAnyCheckout is the gate half. It never renders, so it is the
// question a CI job or a peer's fresh clone can afford to ask — and it can only be
// asked at all because the lock is committed.
func TestStaleAnswersForAnyCheckout(t *testing.T) {
	root := t.TempDir()
	inputs := map[string][]byte{
		artifact:              []byte("ships: a,b,c"),
		"llms-updates.txt":    []byte("ships: a,b,c"),
		"llms-terms.txt":      []byte("terms roster v1"),
		"docs/never-built.md": []byte("nothing built this"),
	}
	l := openLock(t, root)
	for rel, in := range inputs {
		if rel == "docs/never-built.md" {
			continue
		}
		in := in
		if _, err := l.Sync(rel, in, func() ([]byte, error) { return append([]byte("body of "), in...), nil }); err != nil {
			t.Fatal(err)
		}
	}
	if err := l.Save(); err != nil {
		t.Fatal(err)
	}

	// A fresh process — the peer who just cloned — reads only committed facts.
	fresh := openLock(t, root)
	if got := fresh.Stale(inputs); len(got) != 1 || got[0] != "docs/never-built.md" {
		t.Fatalf("Stale = %v, want only the artifact nothing ever built", got)
	}

	// Now the tree moves under the artifacts, which is the question the gate exists for.
	inputs["llms-terms.txt"] = []byte("terms roster v2")
	if got := fresh.Stale(inputs); len(got) != 2 || got[0] != "docs/never-built.md" || got[1] != "llms-terms.txt" {
		t.Fatalf("Stale after an input change = %v, want the never-built one plus llms-terms.txt", got)
	}
}

func TestPruneForgetsDeletedArtifacts(t *testing.T) {
	root := t.TempDir()
	l := openLock(t, root)
	for _, rel := range []string{artifact, "llms-updates.txt"} {
		if _, err := l.Sync(rel, []byte("in"), func() ([]byte, error) { return []byte("out\n"), nil }); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Remove(filepath.Join(root, "llms-updates.txt")); err != nil {
		t.Fatal(err)
	}
	if got := l.Prune(); len(got) != 1 || got[0] != "llms-updates.txt" {
		t.Fatalf("Prune = %v, want [llms-updates.txt]", got)
	}
	if _, ok := l.Input["llms-updates.txt"]; ok {
		t.Error("Prune left the input hash of a deleted artifact behind")
	}
	if _, ok := l.Output["llms-updates.txt"]; ok {
		t.Error("Prune left the output hash of a deleted artifact behind")
	}
	if _, ok := l.Input[artifact]; !ok {
		t.Error("Prune dropped an artifact that still exists")
	}
}

// TestCorruptLockRebuildsRatherThanTrusting: the failure mode of guessing "stale" is a
// slow run; the failure mode of guessing "current" is shipping an artifact that does not
// match its source. Both directions are chosen deliberately, so both are pinned.
func TestCorruptLockRebuildsRatherThanTrusting(t *testing.T) {
	root := t.TempDir()
	write(t, root, genlock.PathFor("marketing-aeo"), "{ this is not json")
	write(t, root, artifact, "stale body\n")

	l := openLock(t, root)
	got, err := l.Sync(artifact, []byte("in"), func() ([]byte, error) { return []byte("fresh body\n"), nil })
	if err != nil {
		t.Fatal(err)
	}
	if got != genlock.Wrote {
		t.Fatalf("Sync over an unreadable lock = %v, want Wrote — an unparseable lock must mean "+
			"'rebuild everything', never 'everything is current'", got)
	}
	if body := read(t, root, artifact); body != "fresh body\n" {
		t.Errorf("artifact = %q, want the rebuilt body", body)
	}
}

func TestSyncSurfacesRenderFailureWithoutTouchingTheArtifact(t *testing.T) {
	root := t.TempDir()
	write(t, root, artifact, "previous good body\n")
	before := snap(t, root, artifact)
	time.Sleep(20 * time.Millisecond)

	l := openLock(t, root)
	boom := errors.New("generator exited 1")
	if _, err := l.Sync(artifact, []byte("in"), func() ([]byte, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Fatalf("Sync error = %v, want the render's own error", err)
	}
	if after := snap(t, root, artifact); after.body != before.body || !after.mtime.Equal(before.mtime) {
		t.Error("a failed render clobbered the artifact that was already there")
	}
	if _, ok := l.Input[artifact]; ok {
		t.Error("a failed render recorded an input hash; the next run would skip a build that never happened")
	}
}

func TestOpenRejectsAToolNameThatIsNotOneSegment(t *testing.T) {
	for _, bad := range []string{"", "a/b", `a\b`, ".", ".."} {
		if _, err := genlock.Open(t.TempDir(), bad); err == nil {
			t.Errorf("Open(%q) succeeded; a separator in the tool name silently relocates the "+
				"lock, which is a lane decision, not a naming one", bad)
		}
	}
}

func TestPathForIsUnderTheToolsDir(t *testing.T) {
	if got, want := genlock.PathFor("marketing-aeo"), "tools/genlock/marketing-aeo.lock.json"; got != want {
		t.Errorf("PathFor = %q, want %q", got, want)
	}
	if genlock.Dir != "tools/genlock" {
		t.Errorf("Dir = %q; see TestLockPathTakesANamedLaneNotTheGlobalCatchAll before moving it", genlock.Dir)
	}
}

package corelocks

// Tests for the positionally-resolved lease journal (#5933).
//
// ⛔ Every test here is a KNOWN POSITIVE with a control beside it, and the
// pairing is the point rather than test hygiene. The defect under repair is a
// probe whose BLIND answer and whose CLEAN answer are the same bytes, so a test
// asserting "authoritative" proves nothing about it: the old behaviour was
// "authoritative" everywhere, including from a shadow root. Only "this input
// SHOULD be refused, and it is" separates the two.
//
// ⭐ The fixtures are CONSTRUCTED rather than measured against this repository.
// The shadow roots that exist today are untracked peer state of unknown
// provenance and somebody may legitimately remove them, at which point a test
// pinned to them goes red over a tree that got BETTER. The live tree is used
// exactly twice below, and only as a source of known positives — never as the
// assertion.

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// stateFixture builds a repository-shaped tree in a temp dir: a toplevel plus
// whichever extra state roots the case asks for.
//
// It is a REAL directory tree, not a synthetic index, because the walk under
// test is a filesystem walk — the whole finding is that no git enumeration can
// see these paths, so a fixture made of strings would not exercise the code path
// that matters.
func stateFixture(t *testing.T, top string, at ...string) {
	t.Helper()
	for _, rel := range at {
		dir := filepath.Join(top, filepath.FromSlash(rel), StateDir)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
		// project.json is what a real state root carries, and its `root` field
		// is the machine-readable witness that a shadow records the wrong root.
		body := `{"root":"` + filepath.ToSlash(filepath.Dir(dir)) + `","schema":1}` + "\n"
		if err := os.WriteFile(filepath.Join(dir, "project.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestCheckRootRefusesEveryNonAuthoritativePosition is the table the ticket's
// first acceptance item states: the same repository, one row per place a caller
// can stand, and what a census claim is worth from there.
//
// The wantAuth column IS the regression. Before this file existed the answer was
// effectively `true` for every row — the census was read wherever the caller
// happened to be — so every `false` here is a position that used to return a
// confident wrong answer.
func TestCheckRootRefusesEveryNonAuthoritativePosition(t *testing.T) {
	for _, tc := range []struct {
		name string
		// at are repo-relative dirs given a StateDir child; "." is the
		// toplevel's own, i.e. the correct one.
		at []string
		// startAt is the repo-relative dir the caller stands in; "" is the
		// toplevel itself.
		startAt string
		// outsideStart runs the check from a directory not in the repo at all,
		// with its own root above it.
		outsideStart bool
		// noGitTop passes an empty toplevel: the caller could not resolve one.
		noGitTop   bool
		wantAuth   bool
		wantShadow bool
		// wantCause are substrings the refusal must contain. A refusal nobody
		// can act on is the reason this column exists.
		wantCause []string
	}{
		{
			// The control. Everything else is a departure from it; if this row
			// ever fails, the check refuses correct work.
			name:     "at the git toplevel, with only the toplevel's own root",
			at:       []string{"."},
			wantAuth: true,
		},
		{
			// ⭐ THE known positive: the live `docs/.dos` shape. The caller is
			// INSIDE the repository, which is exactly the condition the
			// "run it from the repo root" mitigation says is safe.
			name:       "a shadow root one level down shadows the real one",
			at:         []string{".", "docs"},
			startAt:    "docs",
			wantAuth:   false,
			wantShadow: true,
			wantCause:  []string{"SHADOW", "docs", "#5933", "laneadmit"},
		},
		{
			// The blast radius is not the shadow directory itself: everything
			// BENEATH it resolves there too, which is how a worker writing an
			// artifact three directories deep gets the rubber stamp.
			name:       "a directory BENEATH a shadow root resolves to the shadow",
			at:         []string{".", "tools"},
			startAt:    "tools/concept_disambiguation_scorecard.data/inner",
			wantAuth:   false,
			wantShadow: true,
			wantCause:  []string{"SHADOW", "tools"},
		},
		{
			// The NEAREST root wins, not the outermost — which is what makes a
			// shadow able to shadow anything at all.
			name:       "the NEAREST root wins when roots nest",
			at:         []string{".", "a", "a/b"},
			startAt:    "a/b/c",
			wantAuth:   false,
			wantShadow: true,
			wantCause:  []string{"SHADOW", "a/b"},
		},
		{
			// A shadow root ELSEWHERE must not make the toplevel unsafe. This
			// row is why the refusal is positional: if it failed, the check
			// would deny every session in the fleet the moment any agent ran a
			// tool from a subdirectory.
			name:     "a shadow root elsewhere does not spoil the toplevel",
			at:       []string{".", "tools"},
			wantAuth: true,
		},
		{
			// An ordinary subdirectory with no root of its own walks up to the
			// real one and is authoritative. The check must not punish ordinary
			// work.
			name:     "an ordinary subdirectory walks up to the real root",
			at:       []string{"."},
			startAt:  "internal/corelocks",
			wantAuth: true,
		},
		{
			// ⭐ The verdict most easily conflated with a clean tree: no root at
			// all. "Every lane is free" and "there is no journal" are opposite
			// claims that used to print the same bytes.
			name:      "no root anywhere is a refusal, not a clean census",
			at:        nil,
			wantAuth:  false,
			wantCause: []string{"no .dos directory", "found NOTHING"},
		},
		{
			name:         "a root OUTSIDE the repository is refused and is not a shadow",
			at:           []string{"."},
			outsideStart: true,
			wantAuth:     false,
			wantCause:    []string{"OUTSIDE the repository", "DIFFERENT workspace"},
		},
		{
			name:      "no git toplevel is its own refusal, not an accidental match",
			at:        []string{"."},
			noGitTop:  true,
			wantAuth:  false,
			wantCause: []string{"no git toplevel", "unknown workspace"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			top := t.TempDir()
			stateFixture(t, top, tc.at...)

			start := filepath.Join(top, filepath.FromSlash(tc.startAt))
			// ⛔ The fixture is the whole world: the walk stops at the
			// directory the case built, because a temp directory does NOT
			// control its own ancestors. On the host this was measured on, the
			// OS temp directory itself carries a `.dos` — a rogue journal
			// somebody left by running a lease verb from /tmp, i.e. the
			// finding again — and without the ceiling the no-root row below
			// silently turns into the outside-the-repo row and stops testing
			// the verdict it names.
			ceiling := top
			if tc.outsideStart {
				other := t.TempDir()
				stateFixture(t, other, ".")
				start, ceiling = other, other
			}
			if err := os.MkdirAll(start, 0o755); err != nil {
				t.Fatal(err)
			}
			gitTop := top
			if tc.noGitTop {
				gitTop = ""
			}

			v := checkRootWithin(start, gitTop, ceiling)
			if v.Authoritative != tc.wantAuth {
				t.Fatalf("Authoritative=%v, want %v (resolved %q, gitTop %q, cause %q)",
					v.Authoritative, tc.wantAuth, v.Resolved, v.GitTop, v.Cause)
			}
			if v.Shadowed != tc.wantShadow {
				t.Errorf("Shadowed=%v, want %v — the remedy differs between a shadow root somebody "+
					"owns and a caller standing outside the tree", v.Shadowed, tc.wantShadow)
			}
			// The invariant that keeps a refusal printable: Cause is set if and
			// only if the verdict is a refusal.
			if v.Authoritative && v.Cause != "" {
				t.Errorf("an authoritative verdict carries a cause: %q", v.Cause)
			}
			if !v.Authoritative && v.Cause == "" {
				t.Fatal("a refusal with no cause — the caller can print nothing, which is how a " +
					"blind probe comes to look like a clean one")
			}
			for _, want := range tc.wantCause {
				if !strings.Contains(v.Cause, want) {
					t.Errorf("cause does not mention %q:\n%s", want, v.Cause)
				}
			}
		})
	}
}

// TestTheFourRefusalsPrintDifferentBytes is the ticket's headline acceptance
// item stated where it can fail: an empty lease set and a wrong root must not
// print the same bytes. A single shared refusal message would satisfy every
// assertion in the table above and still leave the reader unable to tell which
// of four unrelated problems they have — and one of the four (no root) is the
// one that most resembles a healthy quiet repository.
func TestTheFourRefusalsPrintDifferentBytes(t *testing.T) {
	shadowTop := t.TempDir()
	stateFixture(t, shadowTop, ".", "sub")

	bare := t.TempDir()

	outTop := t.TempDir()
	stateFixture(t, outTop, ".")
	elsewhere := t.TempDir()
	stateFixture(t, elsewhere, ".")

	// Each verdict is bounded by its own fixture (see the table test above): the
	// no-root case is only a no-root case if the walk cannot escape into an
	// ambient `.dos` somewhere above the temp directory.
	cases := map[string]RootVerdict{
		"shadow-root":     checkRootWithin(filepath.Join(shadowTop, "sub"), shadowTop, shadowTop),
		"no-root":         checkRootWithin(bare, bare, bare),
		"outside-repo":    checkRootWithin(elsewhere, outTop, elsewhere),
		"no-git-toplevel": checkRootWithin(shadowTop, "", shadowTop),
	}
	seenCause := map[string]string{}
	seenLine := map[string]string{}
	for name, v := range cases {
		if v.Authoritative {
			t.Fatalf("%s: expected a refusal, got an authoritative verdict", name)
		}
		if prev, dup := seenCause[v.Cause]; dup {
			t.Errorf("%s and %s print identical causes; they are different problems with "+
				"different remedies:\n%s", name, prev, v.Cause)
		}
		seenCause[v.Cause] = name

		c, err := NewCensus(v, 0)
		if err == nil {
			t.Fatalf("%s: NewCensus admitted a non-authoritative root", name)
		}
		if !errors.Is(err, ErrBlindCensus) {
			t.Errorf("%s: refusal does not wrap ErrBlindCensus: %v", name, err)
		}
		if c.Counted {
			t.Errorf("%s: a refused census reports Counted=true", name)
		}
		if c.Held != -1 {
			t.Errorf("%s: a refused census reports Held=%d; it must not be 0, which is the very "+
				"number a caller would act on", name, c.Held)
		}
		line := c.Line()
		if prev, dup := seenLine[line]; dup {
			t.Errorf("%s and %s render identical lines:\n%s", name, prev, line)
		}
		seenLine[line] = name
		// The whole ticket in one assertion: the refusal names the journal it
		// actually read (or says plainly that it read none).
		if v.Resolved != "" && !strings.Contains(line, v.Resolved) {
			t.Errorf("%s: the rendered refusal does not name the resolved root %q:\n%s",
				name, v.Resolved, line)
		}
		if v.Resolved == "" && !strings.Contains(line, "NOTHING was read") {
			t.Errorf("%s: a no-root refusal must say nothing was read:\n%s", name, line)
		}
	}

	// And the two SUCCESSES must differ from each other and from every refusal:
	// a proven zero is a real answer and must not read like a blind one.
	provenZero, err := NewCensus(CheckRoot(shadowTop, shadowTop), 0)
	if err != nil {
		t.Fatalf("the toplevel is not its own root: %v", err)
	}
	provenMany, err := NewCensus(CheckRoot(shadowTop, shadowTop), 22)
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range []string{provenZero.Line(), provenMany.Line()} {
		if prev, dup := seenLine[line]; dup {
			t.Errorf("a PROVEN census renders identically to the %s refusal:\n%s", prev, line)
		}
		seenLine[line] = "proven"
		if !strings.Contains(line, shadowTop) {
			t.Errorf("a proven census does not name its journal root:\n%s", line)
		}
	}
	if provenZero.Line() == provenMany.Line() {
		t.Error("a proven zero and a proven 22 render identically")
	}
}

// TestUnderDirDoesNotAcceptASiblingPrefix pins the containment predicate. A
// strings.HasPrefix spelling passes every other test in this file and calls
// /work/repo-backup a directory inside /work/repo — which would report a wholly
// separate checkout's root as a shadow of this one, in a refusal that reads
// authoritative.
func TestUnderDirDoesNotAcceptASiblingPrefix(t *testing.T) {
	sep := string(filepath.Separator)
	base := filepath.Join(sep, "work", "repo")
	for _, tc := range []struct {
		path string
		want bool
	}{
		{filepath.Join(base, "internal"), true},
		{filepath.Join(base, "a", "b"), true},
		{base, false},                                      // not a STRICT descendant
		{filepath.Join(sep, "work", "repo-backup"), false}, // the prefix trap
		{filepath.Join(sep, "work"), false},                // the parent
		{filepath.Join(sep, "elsewhere", "repo"), false},   // unrelated
	} {
		if got := underDir(tc.path, base); got != tc.want {
			t.Errorf("underDir(%q, %q) = %v, want %v", tc.path, base, got, tc.want)
		}
	}
}

// TestShadowRootsCensus covers the enumeration half. The negative controls are
// the interesting rows: the toplevel's own root must NEVER be reported (it is
// the correct one, and a census naming it sends someone to delete the real
// journal), and nothing under .git may be.
func TestShadowRootsCensus(t *testing.T) {
	for _, tc := range []struct {
		name string
		at   []string
		// mkdirs are extra directories created with no state child.
		mkdirs []string
		want   []string
	}{
		{
			name: "the toplevel's own root is not a shadow",
			at:   []string{"."},
			want: nil,
		},
		{
			// The live shape on this checkout, measured 2026-08-08.
			name: "every nested root is reported, repo-relative",
			at:   []string{".", "docs", "tools", "tools/concept_disambiguation_scorecard.data"},
			want: []string{"docs/.dos", "tools/.dos", "tools/concept_disambiguation_scorecard.data/.dos"},
		},
		{
			name: "a nested root is found with no toplevel root present",
			at:   []string{"sub"},
			want: []string{"sub/.dos"},
		},
		{
			name:   "an administrative .git directory is skipped",
			at:     []string{"."},
			mkdirs: []string{".git/worktrees/x/.dos"},
			want:   nil,
		},
		{
			name: "nesting below a shadow root is not enumerated twice",
			at:   []string{".", "a"},
			// A state dir inside a state dir is not a second project root.
			mkdirs: []string{"a/.dos/inner/.dos"},
			want:   []string{"a/.dos"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			top := t.TempDir()
			stateFixture(t, top, tc.at...)
			for _, d := range tc.mkdirs {
				if err := os.MkdirAll(filepath.Join(top, filepath.FromSlash(d)), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			got, err := ShadowRoots(top)
			if err != nil {
				t.Fatalf("ShadowRoots: %v", err)
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Errorf("ShadowRoots = %v, want %v", got, tc.want)
			}
		})
	}
}

// leaseFixture writes n lease records into the journal of the state root at
// <top>/<rel>. It is the fixture the fail-closed test needs: a canonical root
// that genuinely HOLDS leases and a shadow root that genuinely holds none, so
// the shadow's honest answer is 0 and the only thing separating the two is the
// root check.
func leaseFixture(t *testing.T, top, rel string, n int) {
	t.Helper()
	dir := filepath.Join(top, filepath.FromSlash(rel), StateDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, `{"lane":"lane-%d","holder":"worker-%d"}`+"\n", i, i)
	}
	if err := os.WriteFile(filepath.Join(dir, "lease-journal.jsonl"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

// countLeaseJournal is the stand-in for the real counter: it reads ONLY the root
// it is handed, which is the discipline ReadCensus enforces on every counter.
func countLeaseJournal(root string) (int, error) {
	f, err := os.Open(filepath.Join(root, StateDir, "lease-journal.jsonl"))
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil // an absent journal honestly holds zero leases
		}
		return 0, err
	}
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) != "" {
			n++
		}
	}
	return n, sc.Err()
}

// TestReadCensusFailsClosedFromAShadowRoot is the ticket's fail-closed
// acceptance item, stated exactly as it asks: ONE fixture, TWO cwds.
//
// ⭐ The fixture is built so the shadow's answer is HONESTLY zero — its journal
// really is empty — which is what makes the two indistinguishable by count. The
// canonical root holds 22, the number the upstream measurement was taken at.
func TestReadCensusFailsClosedFromAShadowRoot(t *testing.T) {
	const held = 22

	top := t.TempDir()
	stateFixture(t, top, ".", "docs")
	leaseFixture(t, top, ".", held)
	leaseFixture(t, top, "docs", 0)
	shadowStart := filepath.Join(top, "docs")

	// ⛔ The control that makes the rest of this test mean something: a naive
	// counter run from the shadow position returns a clean, confident 0. This is
	// the bug, reproduced. Everything below is about not shipping that 0.
	shadowRoot, ok := ResolveRoot(shadowStart)
	if !ok {
		t.Fatal("the fixture's shadow root did not resolve")
	}
	if blind, err := countLeaseJournal(shadowRoot); err != nil || blind != 0 {
		t.Fatalf("the shadow journal should honestly count 0 (got %d, %v) — without that, this "+
			"test is not exercising the confusion it exists to refuse", blind, err)
	}

	// cwd 1: the canonical root. The POSITIVE CONTROL — 22 held must report as
	// 22, from the canonical root, so a future 0 here reads as a regression
	// rather than as quiet.
	good, err := ReadCensus(top, top, countLeaseJournal)
	if err != nil {
		t.Fatalf("the canonical root was refused: %v", err)
	}
	if !good.Counted || good.Held != held {
		t.Fatalf("canonical census = %d (counted=%v), want %d", good.Held, good.Counted, held)
	}
	if !sameDir(good.Root, top) {
		t.Errorf("canonical census reports root %q, want %q", good.Root, top)
	}
	if !strings.Contains(good.Line(), strconv.Itoa(held)) || !strings.Contains(good.Line(), good.Root) {
		t.Errorf("the census line must carry BOTH the count and the root:\n%s", good.Line())
	}

	// cwd 2: the shadow. It must NOT return a bare 0.
	bad, err := ReadCensus(shadowStart, top, countLeaseJournal)
	if err == nil {
		t.Fatalf("the shadow root was admitted and reported %d lease(s) — this is the failure "+
			"the ticket exists for", bad.Held)
	}
	if !errors.Is(err, ErrBlindCensus) {
		t.Errorf("shadow refusal does not wrap ErrBlindCensus: %v", err)
	}
	if bad.Counted {
		t.Error("the shadow census claims to have counted")
	}
	if bad.Held == 0 {
		t.Error("the shadow census reports Held=0 — a caller that ignored the error would admit " +
			"every lane on the strength of it")
	}
	if !bad.Shadowed {
		t.Error("the shadow census is not classified as shadowed, so the remedy is unclear")
	}
	// And the two outputs must be distinguishable, which is the whole point.
	if bad.Line() == good.Line() {
		t.Fatal("the shadow read and the canonical read print identical lines")
	}
	if !strings.Contains(bad.Line(), bad.Root) {
		t.Errorf("the shadow refusal does not name the journal it read:\n%s", bad.Line())
	}
	if !strings.Contains(bad.Line(), top) {
		t.Errorf("the shadow refusal does not name the canonical root the reader should have "+
			"used:\n%s", bad.Line())
	}
}

// TestReadCensusNeverCallsTheCounterFromAWrongRoot pins the STRUCTURAL half of
// fail-closed. Refusing after counting would still leave the number in memory
// for a caller to log; refusing BEFORE counting means a blind number is never
// produced at all, so no amount of caller sloppiness can surface one.
func TestReadCensusNeverCallsTheCounterFromAWrongRoot(t *testing.T) {
	top := t.TempDir()
	stateFixture(t, top, ".", "docs")

	calls := 0
	counter := func(string) (int, error) { calls++; return 0, nil }

	if _, err := ReadCensus(filepath.Join(top, "docs"), top, counter); err == nil {
		t.Fatal("the shadow root was admitted")
	}
	if calls != 0 {
		t.Fatalf("the counter ran %d time(s) from a non-authoritative root — a blind count was "+
			"produced and only politely discarded", calls)
	}
	if _, err := ReadCensus(top, top, counter); err != nil {
		t.Fatalf("the canonical root was refused: %v", err)
	}
	if calls != 1 {
		t.Fatalf("the counter ran %d time(s) from the canonical root, want 1", calls)
	}
}

// TestAnUnreadableJournalIsUnknownNotZero pins the other substitution that turns
// this file back into the bug: a counter that errors must not degrade to 0.
func TestAnUnreadableJournalIsUnknownNotZero(t *testing.T) {
	top := t.TempDir()
	stateFixture(t, top, ".")
	c, err := ReadCensus(top, top, func(string) (int, error) { return 0, errors.New("journal is corrupt") })
	if err == nil {
		t.Fatal("an unreadable journal was reported as a successful census")
	}
	if !errors.Is(err, ErrCensusUnreadable) {
		t.Errorf("refusal does not wrap ErrCensusUnreadable: %v", err)
	}
	if errors.Is(err, ErrBlindCensus) {
		t.Error("an unreadable canonical journal is classified as a wrong-root read; the two have " +
			"different remedies and must not share a sentinel")
	}
	if c.Counted || c.Held == 0 {
		t.Errorf("an unreadable journal reported Held=%d counted=%v — unknown must not become zero",
			c.Held, c.Counted)
	}
	if !strings.Contains(c.Line(), c.Root) {
		t.Errorf("the unreadable-journal refusal does not name the root:\n%s", c.Line())
	}
	// A nil counter is the same class of unknown, never a zero.
	if _, err := ReadCensus(top, top, nil); !errors.Is(err, ErrCensusUnreadable) {
		t.Errorf("a nil counter did not refuse as unreadable: %v", err)
	}
}

// --- the git-blindness proof ------------------------------------------------

// gitInTemp runs git in dir with a hermetic config: the developer's global
// excludesfile must not decide the outcome of a test about ignore patterns.
func gitInTemp(t *testing.T, dir string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL="+filepath.Join(dir, "no-such-gitconfig"),
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.invalid",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.invalid",
	)
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

// TestGitEnumerationIsBlindToAShadowRootTheWalkFinds is the ticket's second
// acceptance item, and it is the reason ShadowRoots is a filesystem walk.
//
// ⛔ THE CLAIM: with an UNANCHORED ignore pattern, every git enumeration returns
// ZERO for a nested state root — `ls-files` (tracked), `ls-files --others
// --exclude-standard` (untracked-and-not-ignored), and `status --porcelain` all
// see nothing — while the filesystem walk finds it. A discovery tool built on a
// git subprocess would therefore report a confidently empty answer, which is the
// same failure mode one level up.
//
// ⭐ And the converse, in the same fixture: ANCHORING the pattern to `/.dos/`
// makes the nested root visible to `git status`. That is the mechanical proof
// behind this repository's .gitignore change, not an assertion about it.
func TestGitEnumerationIsBlindToAShadowRootTheWalkFinds(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH; the blindness claim is only checkable with a real git")
	}
	top := t.TempDir()
	if out, err := gitInTemp(t, top, "init", "-q"); err != nil {
		t.Skipf("git init failed in the sandbox (%v): %s", err, out)
	}

	// An ordinary tracked file, so the enumerations below are demonstrably
	// working rather than trivially empty.
	if err := os.WriteFile(filepath.Join(top, "README.md"), []byte("# fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeIgnore := func(body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(top, ".gitignore"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// The unanchored pattern, exactly as it read in this repository before
	// #5933: it matches at EVERY depth.
	writeIgnore(StateDir + "/\n**/" + StateDir + "/\n")
	stateFixture(t, top, ".", "docs")
	if out, err := gitInTemp(t, top, "add", "-A"); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}

	// The walk finds the nested root.
	shadows, err := ShadowRoots(top)
	if err != nil {
		t.Fatalf("ShadowRoots: %v", err)
	}
	want := "docs/" + StateDir
	if len(shadows) != 1 || shadows[0] != want {
		t.Fatalf("ShadowRoots = %v, want [%s]", shadows, want)
	}

	// The control: git enumeration is not broken — it sees the ordinary file.
	if out, err := gitInTemp(t, top, "ls-files", "--", "README.md"); err != nil || out == "" {
		t.Fatalf("git ls-files cannot see a tracked file (%v): %q — the blindness assertions "+
			"below would then be vacuous", err, out)
	}

	// ⛔ And returns ZERO for the shadow root, three different ways.
	for _, probe := range [][]string{
		{"ls-files", "--", "docs/" + StateDir},
		{"ls-files", "--others", "--exclude-standard", "--", "docs/" + StateDir},
		{"status", "--porcelain", "--untracked-files=all", "--", "docs/" + StateDir},
	} {
		out, err := gitInTemp(t, top, probe...)
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(probe, " "), err, out)
		}
		if out != "" {
			t.Errorf("git %s unexpectedly SAW the shadow root: %q", strings.Join(probe, " "), out)
		}
	}

	// ⭐ The converse: anchor the pattern and git can see it.
	writeIgnore("/" + StateDir + "/\n")
	out, err := gitInTemp(t, top, "status", "--porcelain", "--untracked-files=all", "--", "docs/"+StateDir)
	if err != nil {
		t.Fatalf("git status: %v: %s", err, out)
	}
	if out == "" {
		t.Error("with an ANCHORED /.dos/ pattern git status still cannot see the nested root — " +
			"the anchoring remedy would then be worthless")
	}
	// ...and the toplevel's own root stays ignored, which is why anchoring is
	// safe: only the shadows become visible.
	out, err = gitInTemp(t, top, "status", "--porcelain", "--untracked-files=all", "--", StateDir)
	if err != nil {
		t.Fatalf("git status: %v: %s", err, out)
	}
	if out != "" {
		t.Errorf("anchoring exposed the toplevel's own state root, which is regenerated runtime "+
			"state and must stay ignored: %q", out)
	}
}

// --- the live-tree probes ---------------------------------------------------

// repoTop walks up from this package to the module root.
func repoTop(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skipf("no go.mod above %s; the live-tree probes are only checkable in-repo", dir)
		}
		dir = parent
	}
}

// TestLiveTreeInvariantAndShadowCensus is the one test that touches this
// repository, deliberately split into a claim that CANNOT rot and a report that
// cannot fail.
//
// ⭐ THE CLAIM: resolving from the repository root yields the repository root.
// That is the property every lease reader depends on and the one a shadow root
// can never break, so it is safe to assert on a shared tree ~20 sessions write.
//
// ⛔ WHAT THIS DELIBERATELY DOES NOT ASSERT: that the shadow census is empty.
// Three shadow roots existed on 2026-08-08 and a fourth can appear at any
// moment, because creating one requires nothing more than running a kernel verb
// from a subdirectory. Asserting zero would redden the suite for the whole fleet
// over a condition no single worker can clear — the directories are peers'
// untracked state and removing them belongs to whoever owns them.
//
// ⭐ Instead, each live shadow root is used as a KNOWN POSITIVE: standing in it
// must be REFUSED, with a cause naming it. That assertion gets stronger as
// shadow roots accumulate and stays green when they are cleaned up.
func TestLiveTreeInvariantAndShadowCensus(t *testing.T) {
	top := repoTop(t)
	if _, err := os.Stat(filepath.Join(top, StateDir)); errors.Is(err, os.ErrNotExist) {
		t.Skipf("no runtime %s root in this clean checkout; missing-root refusal is covered by the constructed fixtures", StateDir)
	}

	if v := CheckRoot(top, top); !v.Authoritative {
		t.Fatalf("the repository root is not its own state root: %s\n"+
			"Every lease census in this repository is read from here, so this is the invariant the "+
			"fix rests on. If %s/ is genuinely absent, the census is BLIND rather than clean, and "+
			"that is the finding.", v.Cause, StateDir)
	}

	shadows, err := ShadowRoots(top)
	if err != nil {
		// A peer's scratch directory can vanish mid-walk on this tree; that is
		// not this test's finding.
		t.Skipf("ShadowRoots over the live tree: %v", err)
	}
	if len(shadows) == 0 {
		t.Logf("no shadow %s roots in the tree — #5933's three have been resolved", StateDir)
		return
	}
	t.Logf("%d shadow %s root(s) live in this repository (#5933); each is used below as a known "+
		"positive:\n  %s", len(shadows), StateDir, strings.Join(shadows, "\n  "))

	for _, rel := range shadows {
		dir := filepath.Dir(filepath.Join(top, filepath.FromSlash(rel)))
		v := CheckRoot(dir, top)
		if v.Authoritative {
			t.Errorf("%s: standing in a shadow root was called authoritative — this is the exact "+
				"position from which a lease census reports every lane free while leases are held", rel)
			continue
		}
		if !v.Shadowed {
			t.Errorf("%s: refused, but not classified as a shadow root (cause: %s)", rel, v.Cause)
		}
		if !strings.Contains(v.Cause, filepath.ToSlash(filepath.Dir(rel))) {
			t.Errorf("%s: the cause does not name the offending directory, so a reader cannot find "+
				"it: %s", rel, v.Cause)
		}
	}
}

// TestStateIgnorePatternIsAnchored pins the ticket's ignore-anchoring item on
// the live .gitignore. An unanchored pattern makes every nested state root
// invisible to `git status`, which is the condition that let three of them
// accumulate here unnoticed.
func TestStateIgnorePatternIsAnchored(t *testing.T) {
	top := repoTop(t)
	body, err := os.ReadFile(filepath.Join(top, ".gitignore"))
	if err != nil {
		t.Skipf("no .gitignore at %s (%v)", top, err)
	}
	if bad := UnanchoredStateIgnores(body, StateDir); len(bad) > 0 {
		t.Errorf(".gitignore hides %s at EVERY depth via %v — anchor it to %q so a nested state "+
			"root is visible to `git status` (the toplevel's own stays ignored). An unanchored "+
			"pattern makes every git enumeration return 0 for a shadow root (#5933).",
			StateDir, bad, "/"+StateDir+"/")
	}
	// The control: the toplevel's own root must still be ignored, or this
	// repository's runtime state floods every peer's `git status`.
	if !strings.Contains(string(body), "/"+StateDir+"/") {
		t.Errorf(".gitignore no longer ignores the toplevel %s/ at all", StateDir)
	}
}

// TestUnanchoredStateIgnoresClassifies is the unit table behind the live pin.
func TestUnanchoredStateIgnoresClassifies(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want []string
	}{
		{"anchored is clean", "/.dos/\n", nil},
		{"bare directory matches at every depth", ".dos/\n", []string{".dos/"}},
		{"bare name with no slash matches at every depth", ".dos\n", []string{".dos"}},
		{"globstar prefix matches at every depth", "**/.dos/\n", []string{"**/.dos/"}},
		{"both live spellings are reported", ".dos/\n**/.dos/\n", []string{".dos/", "**/.dos/"}},
		{"comments and blanks are ignored", "# .dos/\n\n/.dos/\n", nil},
		{"a negation is still a depth-blind rule", "!.dos/\n", []string{"!.dos/"}},
		{"an unrelated pattern is not claimed", "dos.runs/\n.dosier/\n", nil},
		{"a deeper anchored path is not claimed", "tools/.dos/\n", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := UnanchoredStateIgnores([]byte(tc.body), StateDir)
			if strings.Join(got, "|") != strings.Join(tc.want, "|") {
				t.Errorf("UnanchoredStateIgnores = %v, want %v", got, tc.want)
			}
		})
	}
}

// --- the caller audit -------------------------------------------------------

// dosExecSite matches a direct shell-out to the trust kernel: `exec.Command`/
// `exec.CommandContext` whose PROGRAM argument is the kernel binary, either as
// the literal "dos" or as a resolved *dos* path variable.
//
// ⛔ THE TRAILING DELIMITER IS LOAD-BEARING, and it replaces a `\b` that could
// never fire. `\b` asserts a word/non-word boundary; the literal alternative
// ends in a QUOTE and every real call site continues with `,` or `)`. Both are
// non-word characters, so there is no boundary between them and `..."dos"\b`
// matches NOTHING — the scanner silently degraded to `dosBin`-only, and this
// whole audit passed green over 11 literal `"dos"` call sites, one of which
// (cmd/fak/loop_drive.go's loop-gate witness) genuinely relied on the positional
// walk. A vacuous scanner is worse than no scanner: it is an acceptance item
// that reports itself satisfied. TestDosExecSiteMatchesRealCallSites below is
// the known-positive table that keeps it from going quiet again.
var dosExecSite = regexp.MustCompile(`exec\.Command(?:Context)?\((?:[A-Za-z_][A-Za-z0-9_.]*,\s*)?(?:"dos"|dosBin)\s*[,)]`)

// TestDosExecSiteMatchesRealCallSites is the control for the scanner that the
// caller audit is built on. Without it, TestNoCallerReliesOnThePositionalWalk
// can only ever prove "the pattern found nothing", which is the same sentence
// whether the tree is clean or the pattern is broken — the identical-bytes
// failure this whole file exists to refuse, one layer up in the harness.
//
// ⭐ The positive rows are spellings COPIED FROM THIS TREE, so a future
// refactor of the call sites reds this table rather than quietly emptying the
// audit.
func TestDosExecSiteMatchesRealCallSites(t *testing.T) {
	for _, tc := range []struct {
		name string
		line string
		want bool
	}{
		// ⭐ The row that was broken: the exact line at cmd/fak/loop_drive.go:918.
		{"ctx + literal, variadic argv", `	cmd := exec.CommandContext(ctx, "dos", req.Argv()...)`, true},
		{"ctx + literal, spread args", `	cmd := exec.CommandContext(ctx, "dos", args...)`, true},
		{"ctx + literal, fixed args", `	cmd := exec.CommandContext(ctx, "dos", "lease-lane", "live")`, true},
		{"no ctx + literal", `	cmd := exec.Command("dos", "commit-audit", "--json", ref)`, true},
		{"resolved path variable", `	cmd := exec.CommandContext(ctx, dosBin, args...)`, true},
		{"literal with no further args", `	cmd := exec.Command("dos")`, true},

		// Negative controls: a scanner that matched these would flood the audit
		// with rows nobody can act on, and the audit would be abandoned.
		{"a different program", `	cmd := exec.Command("git", "status")`, false},
		{"a program merely prefixed dos", `	cmd := exec.Command("dostoevsky", "read")`, false},
		{"not an exec at all", `	run("dos", "arbitrate")`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := dosExecSite.MatchString(tc.line); got != tc.want {
				t.Errorf("dosExecSite.MatchString(%q) = %v, want %v\n"+
					"The caller audit is only as good as this pattern: a pattern that matches no "+
					"real call site turns TestNoCallerReliesOnThePositionalWalk into a test that "+
					"passes by finding nothing (#5933).", tc.line, got, tc.want)
			}
		})
	}
}

// enclosingFuncPins reports whether the function containing line i pins the
// workspace anywhere in its body — `cmd.Dir = <root>` or an explicit
// `--workspace` in the argv it builds.
//
// ⛔ IT IS FUNCTION-SCOPED, and the previous FIXED 8-LINE WINDOW is why. Two
// ways that window lied, in opposite directions:
//
//   - FALSE POSITIVE. The pin is one line, but the COMMENT explaining it need
//     not be. cmd/fak/loop_drive.go's fix carries an eight-line rationale, which
//     pushed `cmd.Dir = repoRoot()` to the ninth line and out of the window — so
//     the scanner reported correctly-pinned code as an offender. A checker that
//     punishes documenting the fix gets the documentation deleted, or gets
//     itself deleted.
//   - FALSE NEGATIVE. An argv assembled ABOVE the call site (`args := []string{
//     ..., "--workspace", root}`, then `exec.CommandContext(ctx, "dos",
//     args...)`) is pinned, but the pin is behind the window, not ahead of it.
//
// The enclosing function is the honest unit: that is the scope in which the
// command is built, configured and run, so a pin anywhere in it governs this
// call. The walk back stops at a `func ` at column 0 and the walk forward at a
// `}` at column 0 — gofmt guarantees both for top-level declarations, and this
// repository is gofmt-clean.
func enclosingFuncPins(lines []string, i int) bool {
	isPin := func(s string) bool {
		return strings.Contains(s, ".Dir =") || strings.Contains(s, "--workspace")
	}
	start := 0
	for j := i; j >= 0; j-- {
		if strings.HasPrefix(lines[j], "func ") || strings.HasPrefix(lines[j], "func(") {
			start = j
			break
		}
	}
	end := len(lines) - 1
	for j := i; j < len(lines); j++ {
		if lines[j] == "}" || lines[j] == "}\r" {
			end = j
			break
		}
	}
	for j := start; j <= end; j++ {
		if isPin(lines[j]) {
			return true
		}
	}
	return false
}

// TestEnclosingFuncPinsSeesBothSidesOfTheCall is the control for the window
// widening. Each row is a shape the old fixed-8-line scan got WRONG, so a
// regression to any bounded window reds here rather than in a confusing sweep
// over the live tree.
func TestEnclosingFuncPinsSeesBothSidesOfTheCall(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  string
		want bool
	}{
		{
			// ⭐ The false positive that motivated this helper: a long comment
			// between the call and its pin.
			name: "a pin behind a long rationale comment still counts",
			src: "func run() {\n\tcmd := exec.Command(\"dos\")\n" +
				"\t// 1\n\t// 2\n\t// 3\n\t// 4\n\t// 5\n\t// 6\n\t// 7\n\t// 8\n" +
				"\tcmd.Dir = repoRoot()\n\treturn\n}\n",
			want: true,
		},
		{
			// ⭐ The false negative: the workspace flag is in an argv built
			// ABOVE the call, which no forward-only window can see.
			name: "a --workspace assembled above the call counts",
			src: "func run() {\n\targs := []string{\"lease-lane\", \"--workspace\", root}\n" +
				"\tcmd := exec.CommandContext(ctx, \"dos\", args...)\n\treturn\n}\n",
			want: true,
		},
		{
			name: "an immediate pin still counts",
			src:  "func run() {\n\tcmd := exec.Command(\"dos\")\n\tcmd.Dir = root\n}\n",
			want: true,
		},
		{
			// ⛔ The negative control that keeps the widening honest: a pin in a
			// DIFFERENT function must not launder this one.
			name: "a pin in the next function does not count",
			src: "func run() {\n\tcmd := exec.Command(\"dos\")\n\treturn\n}\n" +
				"\nfunc other() {\n\tcmd.Dir = root\n}\n",
			want: false,
		},
		{
			name: "no pin anywhere is unpinned",
			src:  "func run() {\n\tcmd := exec.Command(\"dos\")\n\treturn\n}\n",
			want: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lines := strings.Split(tc.src, "\n")
			at := -1
			for i, l := range lines {
				if dosExecSite.MatchString(l) {
					at = i
					break
				}
			}
			if at < 0 {
				t.Fatalf("the fixture has no dos exec site; the scanner pattern and this "+
					"fixture have drifted apart:\n%s", tc.src)
			}
			if got := enclosingFuncPins(lines, at); got != tc.want {
				t.Errorf("enclosingFuncPins = %v, want %v for:\n%s", got, tc.want, tc.src)
			}
		})
	}
}

// TestNoCallerReliesOnThePositionalWalk is the ticket's third acceptance item.
//
// ⛔ THE RULE: a caller that shells out to the trust kernel must pin the
// workspace — either `cmd.Dir = <root>` or an explicit `--workspace <root>` in
// the argv. A caller that pins neither inherits the process's cwd, and the
// kernel then resolves its journal by the upward positional walk. That is not a
// hypothetical: two sites in this tree did exactly that before #5933
// (cmd/fak/loop_drive.go's loop-gate witness and cmd/fak/guard_plan_oracles.go's
// `dos arbitrate` oracle — the fleet REFEREE), and an empty census admits every
// lane.
//
// The scan is deliberately shallow (a bounded window after the call site) so its
// failure message is always actionable: it names the file and line, and the fix
// is one line.
func TestNoCallerReliesOnThePositionalWalk(t *testing.T) {
	top := repoTop(t)
	var offenders []string
	for _, sub := range []string{"cmd", "internal"} {
		root := filepath.Join(top, sub)
		err := filepath.WalkDir(root, func(p string, d os.DirEntry, werr error) error {
			if werr != nil {
				return nil // a peer's directory vanished mid-walk; not this test's finding
			}
			name := d.Name()
			if d.IsDir() {
				if p != root && (strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "testdata") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				return nil
			}
			body, err := os.ReadFile(p)
			if err != nil {
				return nil
			}
			lines := strings.Split(string(body), "\n")
			for i, line := range lines {
				if !dosExecSite.MatchString(line) {
					continue
				}
				if strings.Contains(line, "--workspace") {
					continue
				}
				if !enclosingFuncPins(lines, i) {
					rel, _ := filepath.Rel(top, p)
					offenders = append(offenders, fmt.Sprintf("%s:%d: %s",
						filepath.ToSlash(rel), i+1, strings.TrimSpace(line)))
				}
			}
			return nil
		})
		if err != nil {
			t.Skipf("walking %s: %v", root, err)
		}
	}
	if len(offenders) > 0 {
		t.Errorf("these kernel shell-outs pin neither cmd.Dir nor --workspace, so the journal they "+
			"read is decided by whatever cwd the process happens to have — and an empty lease "+
			"census ADMITS every lane (#5933):\n  %s\nFix: set `cmd.Dir = repoRoot()` (or pass "+
			"`--workspace <root>`) on the line after the exec.Command call.",
			strings.Join(offenders, "\n  "))
	}
}

// TestDestructiveReaderAuditIsLive keeps the ticket's last acceptance item —
// "every reader that acts destructively on an empty census is audited and
// listed" — from rotting into prose. Each row must name a file that exists and a
// symbol that is still in it, so a rename reds the audit instead of silently
// leaving a claim about a function nobody has.
func TestDestructiveReaderAuditIsLive(t *testing.T) {
	top := repoTop(t)
	rows := DestructiveReaders()
	if len(rows) == 0 {
		t.Fatal("the destructive-reader audit is empty")
	}
	seen := map[string]bool{}
	for _, r := range rows {
		key := r.Path + "::" + r.Symbol
		if seen[key] {
			t.Errorf("duplicate audit row for %s", key)
		}
		seen[key] = true
		if r.Note == "" {
			t.Errorf("%s: an audit row with no argument is not an audit", key)
		}
		switch r.Disposition {
		case DispositionFailsClosed, DispositionFailOpenIsCorrect, DispositionFailsOpen:
		default:
			t.Errorf("%s: unknown disposition %q", key, r.Disposition)
		}
		body, err := os.ReadFile(filepath.Join(top, filepath.FromSlash(r.Path)))
		if err != nil {
			t.Errorf("%s: audited file is unreadable: %v", r.Path, err)
			continue
		}
		if !strings.Contains(string(body), r.Symbol) {
			t.Errorf("%s: the audit claims %q but that symbol is not in the file — the row is "+
				"stale and its safety argument is unproven", r.Path, r.Symbol)
		}
	}
	// A fail-open row is the hazard, so it must carry its mitigation in prose;
	// an unexplained fail-open row is exactly the silence this ticket is about.
	for _, r := range rows {
		if r.Disposition == DispositionFailsOpen && !strings.Contains(r.Note, "mitigation") {
			t.Errorf("%s: a fails-open reader with no stated mitigation", r.Path)
		}
	}
}

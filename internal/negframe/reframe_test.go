package negframe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReframeMechanical checks that each unambiguous negative idiom is flipped to the positive
// inverse its lexicon rule declares, while surrounding prose is left byte-for-byte intact.
func TestReframeMechanical(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"forget-to", "Do not forget to stamp the commit.", "remember to stamp the commit."},
		{"dont-forget", "don't forget to run the test.", "remember to run the test."},
		{"hesitate", "Do not hesitate to ask.", "feel free to ask."},
		{"no-need", "No need to rebuild here.", "you can skip rebuild here."},
		{"hedge", "The result is not unclear.", "The result is clear."},
		{"mid-sentence", "First land it, and do not forget to push after.", "First land it, and remember to push after."},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Reframe(c.in); got != c.want {
				t.Fatalf("Reframe(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestReframeLeavesJudgement pins that a load-bearing judgement-tier negative — a bare prohibition
// with no confident mechanical inverse — is emitted byte-identical and only counted, never guessed.
func TestReframeLeavesJudgement(t *testing.T) {
	in := "Do not re-propose a refused call unchanged; never merge without review."
	r := ReframePass(in)
	if r.Text != in {
		t.Fatalf("judgement prose was rewritten: %q -> %q", in, r.Text)
	}
	if r.Applied != 0 {
		t.Fatalf("Applied = %d, want 0 (no mechanical idiom present)", r.Applied)
	}
	if r.ResidualNegatives == 0 {
		t.Fatalf("ResidualNegatives = 0, want the judgement negatives counted")
	}
}

// TestReframeTokenSupersetFailSafe is the load-bearing safety fixture: a mechanical rewrite that
// would fold an ALL-CAPS contract token into a gerund (dropping the standalone token) is REFUSED,
// the original span is emitted verbatim, and the refusal is counted as a VerbatimFallback.
func TestReframeTokenSupersetFailSafe(t *testing.T) {
	in := "Make sure that you do not DEPLOY."
	r := ReframePass(in)
	if r.Text != in {
		t.Fatalf("token-dropping rewrite was applied: %q -> %q (must stay verbatim)", in, r.Text)
	}
	if r.Applied != 0 || r.VerbatimFallback != 1 {
		t.Fatalf("Applied=%d VerbatimFallback=%d, want 0 and 1", r.Applied, r.VerbatimFallback)
	}
}

// TestReframeKeepsReasonToken checks the common runtime shape: a mechanical idiom sitting next to a
// structured reason token is flipped while the token (OFF_TRUNK) survives byte-for-byte.
func TestReframeKeepsReasonToken(t *testing.T) {
	in := "do not forget to clear the OFF_TRUNK blocker before you retry."
	got := Reframe(in)
	want := "remember to clear the OFF_TRUNK blocker before you retry."
	if got != want {
		t.Fatalf("Reframe = %q, want %q", got, want)
	}
}

// TestReframeIdempotent pins the fixed-point property the emit-time pass depends on: a positive
// rewrite carries no mechanical idiom, so a second pass is a no-op.
func TestReframeIdempotent(t *testing.T) {
	inputs := []string{
		"Do not forget to stamp the commit.",
		"No need to rebuild; do not hesitate to ask. The result is not unclear.",
		"Make sure that you do not DEPLOY.",
		"Do not re-propose a refused call unchanged.",
		SampleRuntimeSteer,
	}
	for _, in := range inputs {
		once := Reframe(in)
		twice := Reframe(once)
		if once != twice {
			t.Fatalf("not idempotent:\n once = %q\n twice = %q", once, twice)
		}
	}
}

// TestReframePreservesFencesAndBlanks checks that fenced code and blank lines are copied verbatim
// (a "don't" inside a code sample is not steer prose) and the newline structure is preserved.
func TestReframePreservesFencesAndBlanks(t *testing.T) {
	in := "Do not forget to build.\n\n```\ndo not forget to X\n```\ndone."
	want := "remember to build.\n\n```\ndo not forget to X\n```\ndone."
	if got := Reframe(in); got != want {
		t.Fatalf("Reframe = %q, want %q", got, want)
	}
}

// TestReframeEmptyAndPlain covers the trivial inputs: empty text and text with no negatives are
// returned unchanged with zero counts.
func TestReframeEmptyAndPlain(t *testing.T) {
	for _, in := range []string{"", "Land it and move on.", "Commit what compiles."} {
		r := ReframePass(in)
		if r.Text != in || r.Applied != 0 || r.VerbatimFallback != 0 {
			t.Fatalf("ReframePass(%q) = %+v, want unchanged/zero", in, r)
		}
	}
}

// SampleRuntimeSteer is a representative runtime-assembled steer string (idempotency input).
const SampleRuntimeSteer = "fak posture: keep going while checkable work remains. " +
	"Do not forget to land durable state before a context event. " +
	"Do not re-propose a refused call unchanged; the OFF_TRUNK blocker must clear first."

func TestReframePolarityPreservation(t *testing.T) {
	tests := []struct {
		name   string
		before string
		after  string
		want   bool
	}{
		{"never softened", "Never deploy `OFF_TRUNK`.", "Deploy `OFF_TRUNK`.", false},
		{"must-not softened", "You must not force-push `MAIN`.", "Force-push `MAIN`.", false},
		{"do-not preserved", "Do not bypass POLICY_BLOCK.", "Never bypass POLICY_BLOCK.", true},
		{"no-token softened", "No `OFF_TRUNK` deployment.", "Allow `OFF_TRUNK` deployment.", false},
		{"obligation is not prohibition", "Do not forget to preserve `OFF_TRUNK`.", "Remember to preserve `OFF_TRUNK`.", true},
		{"unbound token", "Never guess. Report `OFF_TRUNK` separately.", "Avoid guessing. Report `OFF_TRUNK` separately.", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := polarityPreserved(tc.before, tc.after); got != tc.want {
				t.Fatalf("polarityPreserved(%q, %q) = %v, want %v", tc.before, tc.after, got, tc.want)
			}
		})
	}
}

func TestReframeLineRefusesPolarityFlip(t *testing.T) {
	before := "Never deploy `OFF_TRUNK`."
	if got := Reframe(before); got != before {
		t.Fatalf("Reframe softened load-bearing prohibition: %q", got)
	}
	candidate := "Deploy `OFF_TRUNK`."
	if !tokenSuperset(mustKeepSet(before), mustKeepSet(candidate)) {
		t.Fatal("fixture must pass the older token-superset gate")
	}
	if polarityPreserved(before, candidate) {
		t.Fatal("polarity gate admitted a permissive rewrite")
	}
}

func TestReframeCorpusIdempotentFailSafe(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	targets := ResolveTargets(root)
	if len(targets) == 0 {
		t.Skip("reframe corpus absent in partial checkout")
	}
	files := 0
	fallbackLines := 0
	for _, target := range targets {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(target)))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			t.Errorf("read %s: %v", target, err)
			continue
		}
		files++
		input := string(data)
		once := Reframe(input)
		if twice := Reframe(once); twice != once {
			t.Errorf("reframe not idempotent: %s", target)
		}
		for lineNo, line := range strings.Split(input, "\n") {
			pass := ReframePass(line)
			if pass.VerbatimFallback == 0 {
				continue
			}
			fallbackLines++
			if pass.Text != line {
				t.Errorf("%s:%d fallback changed bytes:\n got %q\nwant %q", target, lineNo+1, pass.Text, line)
			}
			if !tokenSuperset(mustKeepSet(line), mustKeepSet(pass.Text)) {
				t.Errorf("%s:%d fallback dropped must-keep token", target, lineNo+1)
			}
			if !polarityPreserved(line, pass.Text) {
				t.Errorf("%s:%d fallback softened prohibition", target, lineNo+1)
			}
		}
	}
	if files == 0 {
		t.Skip("reframe corpus files absent in partial checkout")
	}
	t.Logf("checked %d corpus files (%d fallback lines)", files, fallbackLines)
}

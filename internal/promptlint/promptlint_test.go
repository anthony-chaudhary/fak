package promptlint

import (
	"reflect"
	"testing"
)

// A representative slice of a rendered worker prompt: it names real fak verbs, real
// refusal tokens (with underscores), the `(fak <lane>)` commit trailer, bare ALLCAPS
// prose, and dotted filenames — everything the extractors must tell apart.
const samplePrompt = "read first: run `gh issue view 12`, then orient with `AGENTS.md`.\n" +
	"Then run `fak memory-read` for the committed fleet memory.\n" +
	"how to work it:\n" +
	"- Take the lane lease: `dos arbitrate --workspace . --lane docs`.\n" +
	"- Commit with `fak commit -s -- <paths>`. NEVER `git add -A`.\n" +
	"git laws (enforced below the agent):\n" +
	"- Work on `main` ONLY (the OFF_TRUNK guard refuses a branch).\n" +
	"- Reference `#12` and end with a `(fak docs)` trailer.\n" +
	"- If a hook reports `PATHSPEC_RACE`, refresh and recommit by path.\n" +
	"- An upward import reds architest (`ARCH_LAYER_VIOLATION`)."

func knownFixture() Known {
	// Only the verbs/tokens a real registry would vouch for. `docs` (the trailer lane)
	// and `git` are deliberately absent — a correct extractor must not ask about them.
	return NewKnown(
		[]string{"memory-read", "commit", "issue", "index"},
		[]string{"OFF_TRUNK", "PATHSPEC_RACE", "ARCH_LAYER_VIOLATION", "CORE_SELF_MODIFY"},
	)
}

func TestLintCleanPrompt(t *testing.T) {
	if got := Lint(samplePrompt, knownFixture()); len(got) != 0 {
		t.Fatalf("clean prompt should have no findings, got %+v", got)
	}
	if !OK(samplePrompt, knownFixture()) {
		t.Fatal("OK should be true for a clean prompt")
	}
}

// The headline false-positive trap: the `(fak docs)` commit-stamp trailer names the
// owning LANE, not a verb. It must never be reported as STALE_VERB even though `docs` is
// not a fak verb.
func TestTrailerLaneNotFlaggedAsVerb(t *testing.T) {
	for _, m := range ExtractFakVerbs(samplePrompt) {
		if m.Token == "docs" {
			t.Fatalf("`(fak docs)` trailer lane was extracted as a verb: %+v", m)
		}
	}
	// And a bare `(fak cmd)` in isolation stays silent.
	if got := Lint("end it with a `(fak cmd)` trailer", knownFixture()); len(got) != 0 {
		t.Fatalf("`(fak cmd)` trailer should not lint, got %+v", got)
	}
}

func TestStaleVerbFlagged(t *testing.T) {
	// A renamed/typo'd verb the catalog no longer carries.
	p := "then run `fak memory-reed` for the fleet memory."
	got := Lint(p, knownFixture())
	if len(got) != 1 || got[0].Kind != StaleVerb || got[0].Token != "memory-reed" {
		t.Fatalf("want one STALE_VERB memory-reed, got %+v", got)
	}
}

func TestStaleRefusalTokenFlagged(t *testing.T) {
	// A token no fak registry declares (the PATHSPEC_RACE-style drift we want caught).
	p := "if a hook reports `SANDBOX_ESCAPE`, stop." // all-caps, but not a real token
	got := Lint(p, knownFixture())
	if len(got) != 1 || got[0].Kind != StaleRefusalToken {
		t.Fatalf("want one STALE_REFUSAL_TOKEN, got %+v", got)
	}
}

// Bare ALLCAPS prose words and dotted filenames are not reason tokens and must not be
// extracted (they have no underscore-joined segments).
func TestUpperSnakeExtractionExcludesNonTokens(t *testing.T) {
	toks := ExtractRefusalTokens("NEVER do this. Read AGENTS.md and CLAIMS.md. Only ALLOW or DENY.")
	if len(toks) != 0 {
		t.Fatalf("bare ALLCAPS / filenames must not be tokens, got %+v", toks)
	}
	real := ExtractRefusalTokens("the `OFF_TRUNK` and `NO_PATHS` guards")
	if len(real) != 2 {
		t.Fatalf("want OFF_TRUNK and NO_PATHS, got %+v", real)
	}
}

func TestNilSetSkipsDimension(t *testing.T) {
	// Only reasons loaded: a stale verb must be ignored, a stale token still caught.
	k := Known{Reasons: map[string]bool{"OFF_TRUNK": true}}
	p := "run `fak bogusverb`; if `WEIRD_TOKEN` fires, stop."
	got := Lint(p, k)
	if len(got) != 1 || got[0].Kind != StaleRefusalToken || got[0].Token != "WEIRD_TOKEN" {
		t.Fatalf("nil Verbs should skip verb dimension; want one token finding, got %+v", got)
	}
}

func TestNewKnownNormalizesCase(t *testing.T) {
	k := NewKnown([]string{"Memory-Read", " commit "}, []string{"off_trunk", ""})
	want := Known{
		Verbs:   map[string]bool{"memory-read": true, "commit": true},
		Reasons: map[string]bool{"OFF_TRUNK": true},
	}
	if !reflect.DeepEqual(k, want) {
		t.Fatalf("NewKnown normalization: got %+v want %+v", k, want)
	}
}

func TestFindingsAreSortedAndStable(t *testing.T) {
	p := "run `fak zeta` and `fak alpha`; tokens `ZZ_TOP` and `AA_BB` may fire."
	got := Lint(p, NewKnown([]string{"commit"}, []string{"OFF_TRUNK"}))
	// Assert only the stable contract (kind, then token, both ascending) — not the
	// context snippet, whose exact padding is an implementation detail.
	type kt struct {
		k Kind
		t string
	}
	var gotKT []kt
	for _, f := range got {
		gotKT = append(gotKT, kt{f.Kind, f.Token})
	}
	// Sort is by Kind string ascending, so "STALE_REFUSAL_TOKEN" precedes "STALE_VERB",
	// then by token within each kind.
	want := []kt{
		{StaleRefusalToken, "AA_BB"},
		{StaleRefusalToken, "ZZ_TOP"},
		{StaleVerb, "alpha"},
		{StaleVerb, "zeta"},
	}
	if !reflect.DeepEqual(gotKT, want) {
		t.Fatalf("sorted findings mismatch:\n got %+v\nwant %+v", gotKT, want)
	}
	// Findings should still carry a non-empty context for the human fix.
	for _, f := range got {
		if f.Context == "" {
			t.Errorf("finding %+v has empty context", f)
		}
	}
}

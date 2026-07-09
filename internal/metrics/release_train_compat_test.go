package metrics

import (
	"strings"
	"testing"
)

// findingFor pulls one check's finding out of a report, failing if it is absent —
// CheckReleaseTrain is total, so a missing check is a bug.
func findingFor(t *testing.T, r CompatReport, check CompatCheck) CompatFinding {
	t.Helper()
	for _, f := range r.Findings {
		if f.Check == check {
			return f
		}
	}
	t.Fatalf("report has no finding for check %q (findings: %+v)", check, r.Findings)
	return CompatFinding{}
}

// TestCompatChecksClosed binds the closed check vocabulary to the four surfaces
// issue #1658 names: experimental flags, migration notes, API stability, and
// release note placement — in that order, each with a human label. A drift here
// is a spec change, not a silent check addition.
func TestCompatChecksClosed(t *testing.T) {
	want := []CompatCheck{
		CheckExperimentalFlag,
		CheckMigrationNote,
		CheckAPIStability,
		CheckReleaseNotePlacement,
	}
	if len(CompatChecks) != len(want) {
		t.Fatalf("CompatChecks = %v, want %v", CompatChecks, want)
	}
	for i, c := range want {
		if CompatChecks[i] != c {
			t.Fatalf("CompatChecks[%d] = %q, want %q", i, CompatChecks[i], c)
		}
		if label := CompatChecks[i].Label(); label == "" || label == string(c) {
			t.Fatalf("check %q has no human label (got %q)", c, label)
		}
	}
}

// TestPublicSectionFold pins the four-horizon -> three-public-word fold that
// release note placement is judged against, and the fail-closed empty result
// for a horizon outside the closed vocabulary.
func TestPublicSectionFold(t *testing.T) {
	for _, tc := range []struct{ generation, want string }{
		{"now", "shipped"},
		{"next", "next"},
		{"second-next", "research"},
		{"future", "research"},
		{"someday", ""},
		{"", ""},
	} {
		if got := PublicSection(tc.generation); got != tc.want {
			t.Fatalf("PublicSection(%q) = %q, want %q", tc.generation, got, tc.want)
		}
	}
	// Every horizon in the closed vocabulary folds onto a real section.
	for _, g := range RoadmapGenerations {
		if PublicSection(g) == "" {
			t.Fatalf("horizon %q folds onto no public section", g)
		}
	}
}

// TestCheckReleaseTrainUnknownGenerationFailsClosed proves an undecidable
// candidate blocks on every check rather than being waved onto the train.
func TestCheckReleaseTrainUnknownGenerationFailsClosed(t *testing.T) {
	r := CheckReleaseTrain(ReleaseTrainCandidate{Change: "#1", Generation: "someday", ReleaseNoteSection: "shipped"})
	if !r.Blocked() {
		t.Fatalf("unknown generation must block, got:\n%s", r.Render())
	}
	if len(r.Blocks()) != len(CompatChecks) {
		t.Fatalf("unknown generation must block every check, got %d of %d", len(r.Blocks()), len(CompatChecks))
	}
	for _, f := range r.Findings {
		if !strings.Contains(f.Reason, "fail closed") {
			t.Fatalf("check %q reason does not name the fail-closed rule: %q", f.Check, f.Reason)
		}
	}
}

// TestCheckReleaseTrainGenNowDefaultExposedRides proves the shipped horizon is
// the one that may be on by default: the experimental-flag rule does not bind it.
func TestCheckReleaseTrainGenNowDefaultExposedRides(t *testing.T) {
	r := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:             "fix(gateway): treat same-tick ready as positive",
		Generation:         "now",
		DefaultExposed:     true,
		ReleaseNoteSection: "shipped",
	})
	if r.Blocked() {
		t.Fatalf("a gated-correct gen/now candidate must ride, got:\n%s", r.Render())
	}
	if got := findingFor(t, r, CheckExperimentalFlag).Verdict; got != CompatNotApplicable {
		t.Fatalf("experimental-flag verdict for gen/now = %q, want %q", got, CompatNotApplicable)
	}
}

// TestCheckReleaseTrainFutureHorizonIsNotPenalized enforces the issue's non-goal:
// gen/future is a horizon label, not a value judgment. A far-horizon candidate
// that is properly gated and properly filed passes every binding check, exactly
// as a gen/now one does. Only exposure and surface facts can block — never the
// horizon itself.
func TestCheckReleaseTrainFutureHorizonIsNotPenalized(t *testing.T) {
	future := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:             "#1672 proof-of-value simulator",
		Generation:         "future",
		DefaultExposed:     false,
		ExperimentalFlag:   "FAK_GEN_FUTURE_SIMULATOR",
		ReleaseNoteSection: "research",
	})
	if future.Blocked() {
		t.Fatalf("a gated, correctly-filed gen/future candidate must ride, got:\n%s", future.Render())
	}
	// And the same is true one horizon in, so "further out" never means "more blocked".
	secondNext := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:             "#1658 release train compatibility checks",
		Generation:         "second-next",
		ExperimentalFlag:   "FAK_GEN_RELEASE_TRAIN_COMPAT",
		ReleaseNoteSection: "research",
	})
	if secondNext.Blocked() {
		t.Fatalf("a gated, correctly-filed gen/second-next candidate must ride, got:\n%s", secondNext.Render())
	}
}

// TestCheckReleaseTrainExperimentalFlagRule covers both refusals the exposure
// rule owns: a default-exposed far-horizon change, and a gated one that names no gate.
func TestCheckReleaseTrainExperimentalFlagRule(t *testing.T) {
	exposed := CheckReleaseTrain(ReleaseTrainCandidate{
		Generation:         "second-next",
		DefaultExposed:     true,
		ExperimentalFlag:   "FAK_SOMETHING",
		ReleaseNoteSection: "research",
	})
	f := findingFor(t, exposed, CheckExperimentalFlag)
	if f.Verdict != CompatBlock {
		t.Fatalf("default-exposed gen/second-next must block on experimental-flag, got %q", f.Verdict)
	}
	if !strings.Contains(f.Reason, "default") {
		t.Fatalf("block reason should name default exposure, got %q", f.Reason)
	}

	unnamed := CheckReleaseTrain(ReleaseTrainCandidate{
		Generation:         "next",
		DefaultExposed:     false,
		ReleaseNoteSection: "next",
	})
	if got := findingFor(t, unnamed, CheckExperimentalFlag).Verdict; got != CompatBlock {
		t.Fatalf("unnamed gate must block on experimental-flag, got %q", got)
	}
}

// TestCheckReleaseTrainAPIStabilityAndMigrationNote covers the two surface rules:
// an in-place edit of a shipped surface is refused outright (additive-only), and
// an additive successor that supersedes a shipped surface needs a migration note.
func TestCheckReleaseTrainAPIStabilityAndMigrationNote(t *testing.T) {
	inPlace := CheckReleaseTrain(ReleaseTrainCandidate{
		Generation:           "now",
		BreaksShippedSurface: true,
		ReleaseNoteSection:   "shipped",
	})
	f := findingFor(t, inPlace, CheckAPIStability)
	if f.Verdict != CompatBlock {
		t.Fatalf("in-place shipped-surface edit must block on api-stability, got %q", f.Verdict)
	}
	if !strings.Contains(f.Reason, "additive-only") {
		t.Fatalf("api-stability block should name the additive-only promise, got %q", f.Reason)
	}

	// Supersedes with no note: migration-note blocks, api-stability still passes
	// (an additive successor does not edit the old surface in place).
	noNote := CheckReleaseTrain(ReleaseTrainCandidate{
		Generation:         "now",
		SupersedesShipped:  true,
		ReleaseNoteSection: "shipped",
	})
	if got := findingFor(t, noNote, CheckMigrationNote).Verdict; got != CompatBlock {
		t.Fatalf("supersedes-without-note must block on migration-note, got %q", got)
	}
	if got := findingFor(t, noNote, CheckAPIStability).Verdict; got != CompatPass {
		t.Fatalf("additive successor must pass api-stability, got %q", got)
	}

	// With a note, the same candidate rides.
	withNote := CheckReleaseTrain(ReleaseTrainCandidate{
		Generation:         "now",
		SupersedesShipped:  true,
		MigrationNote:      "docs/releases/v1.2.0.md#migrating-off-v1",
		ReleaseNoteSection: "shipped",
	})
	if withNote.Blocked() {
		t.Fatalf("supersedes-with-note must ride, got:\n%s", withNote.Render())
	}

	// Superseding nothing means the rule has nothing to say — not-applicable, not pass.
	quiet := CheckReleaseTrain(ReleaseTrainCandidate{Generation: "now", ReleaseNoteSection: "shipped"})
	if got := findingFor(t, quiet, CheckMigrationNote).Verdict; got != CompatNotApplicable {
		t.Fatalf("no supersede means migration-note is not-applicable, got %q", got)
	}
}

// TestCheckReleaseTrainRefusesHorizonLaundering proves a far-horizon change may
// not be filed under the shipped section, and that an unplaced note also blocks.
func TestCheckReleaseTrainRefusesHorizonLaundering(t *testing.T) {
	laundered := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:             "#1658",
		Generation:         "second-next",
		ExperimentalFlag:   "FAK_GEN_RELEASE_TRAIN_COMPAT",
		ReleaseNoteSection: "shipped", // should be "research"
	})
	f := findingFor(t, laundered, CheckReleaseNotePlacement)
	if f.Verdict != CompatBlock {
		t.Fatalf("second-next filed as shipped must block, got %q", f.Verdict)
	}
	if !strings.Contains(f.Reason, "laundering") {
		t.Fatalf("placement block should name horizon laundering, got %q", f.Reason)
	}
	// The exposure rule is independent: it still passed. Only placement blocked.
	if got := findingFor(t, laundered, CheckExperimentalFlag).Verdict; got != CompatPass {
		t.Fatalf("a laundered note must not taint the exposure verdict, got %q", got)
	}

	unplaced := CheckReleaseTrain(ReleaseTrainCandidate{Generation: "now"})
	if got := findingFor(t, unplaced, CheckReleaseNotePlacement).Verdict; got != CompatBlock {
		t.Fatalf("unplaced release note must block, got %q", got)
	}
}

// TestCompatReportRender proves the report renders the orthogonality header, a
// line per check, and a verdict — deterministically, in both outcomes.
func TestCompatReportRender(t *testing.T) {
	ok := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:             "#1658 release train compatibility checks",
		Generation:         "second-next",
		ExperimentalFlag:   "FAK_GEN_RELEASE_TRAIN_COMPAT",
		ReleaseNoteSection: "research",
	})
	out := ok.Render()

	if !strings.Contains(out, OrthogonalityNote) {
		t.Fatalf("render missing orthogonality note:\n%s", out)
	}
	for _, kw := range []string{"priority", "shared trunk", "feature gate"} {
		if !strings.Contains(strings.ToLower(out), kw) {
			t.Fatalf("orthogonality note does not name %q:\n%s", kw, out)
		}
	}
	for _, check := range CompatChecks {
		if !strings.Contains(out, check.Label()) {
			t.Fatalf("render missing check %q:\n%s", check.Label(), out)
		}
	}
	if !strings.Contains(out, "MAY RIDE") {
		t.Fatalf("passing candidate should render MAY RIDE:\n%s", out)
	}
	if !strings.Contains(out, "gen/second-next") {
		t.Fatalf("render should name the candidate horizon:\n%s", out)
	}
	if ok.Render() != out {
		t.Fatal("Render is not deterministic")
	}

	// A blocked report names every refusing check in its verdict line.
	bad := CheckReleaseTrain(ReleaseTrainCandidate{
		Change:               "#0 bad candidate",
		Generation:           "future",
		DefaultExposed:       true,
		BreaksShippedSurface: true,
		ReleaseNoteSection:   "shipped",
	})
	badOut := bad.Render()
	if !strings.Contains(badOut, "BLOCKED") {
		t.Fatalf("blocked candidate should render BLOCKED:\n%s", badOut)
	}
	for _, check := range []CompatCheck{CheckExperimentalFlag, CheckAPIStability, CheckReleaseNotePlacement} {
		if !strings.Contains(badOut, string(check)) {
			t.Fatalf("verdict line missing blocking check %q:\n%s", check, badOut)
		}
	}
	if bad.Render() != badOut {
		t.Fatal("Render is not deterministic for a blocked report")
	}
}

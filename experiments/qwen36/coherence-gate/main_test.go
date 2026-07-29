package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestScoreTextCountsEveryRepeatNotEveryDistinctRepeat pins the metric this gate's verdict is
// read off. The loop fixture repeats the 3-cycle "alpha beta gamma" four times, so of the 11
// trigram windows 7 are re-sightings — counting DISTINCT repeated trigrams instead would give
// 3, and a run that degenerated into a loop would score under a third of its true repetition.
func TestScoreTextCountsEveryRepeatNotEveryDistinctRepeat(t *testing.T) {
	loop := "Summary alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma."
	s := scoreText(loop, nil)

	if s.Words != 13 {
		t.Fatalf("Words = %d, want 13", s.Words)
	}
	if s.TrigramWindows != 11 {
		t.Fatalf("TrigramWindows = %d, want 11 (words-2)", s.TrigramWindows)
	}
	if s.RepeatedTrigrams != 7 {
		t.Fatalf("RepeatedTrigrams = %d, want 7 (every re-sighting; 3 would be distinct-only)", s.RepeatedTrigrams)
	}
	if want := 7.0 / 11.0; math.Abs(s.TrigramRepeat-want) > 1e-12 {
		t.Fatalf("TrigramRepeat = %v, want %v", s.TrigramRepeat, want)
	}
}

// TestScoreTextCleanProseHasNoRepeats is the other half of the discrimination: the gate is only
// meaningful if non-degenerate prose scores 0. If this and the loop case ever agreed, the metric
// would be blind to exactly the failure it exists to catch.
func TestScoreTextCleanProseHasNoRepeats(t *testing.T) {
	clean := "Summary alpha beta gamma delta epsilon. Risks are bounded. Next steps are explicit."
	s := scoreText(clean, nil)
	if s.RepeatedTrigrams != 0 || s.TrigramRepeat != 0 {
		t.Fatalf("clean prose scored %d repeats (ratio %v), want 0", s.RepeatedTrigrams, s.TrigramRepeat)
	}
	if s.TrigramWindows <= 0 {
		t.Fatalf("TrigramWindows = %d, want > 0 — otherwise the zero above is vacuous", s.TrigramWindows)
	}
}

// TestScoreTextShortTextCannotDivideByZero guards the ratio's denominator: texts shorter than a
// trigram have no windows at all, and the count must clamp at 0 rather than going negative and
// producing a NaN or a negative repetition rate.
func TestScoreTextShortTextCannotDivideByZero(t *testing.T) {
	for _, text := range []string{"", "one", "one two"} {
		s := scoreText(text, nil)
		if s.TrigramWindows != 0 {
			t.Errorf("scoreText(%q).TrigramWindows = %d, want 0", text, s.TrigramWindows)
		}
		if s.TrigramRepeat != 0 || math.IsNaN(s.TrigramRepeat) {
			t.Errorf("scoreText(%q).TrigramRepeat = %v, want 0", text, s.TrigramRepeat)
		}
	}
}

// TestScoreTextLabelsAreCaseInsensitiveAndReported pins the required-label check: matching folds
// case (the model is not asked to reproduce our capitalization) but a genuinely absent heading
// must be reported and must clear RequiredLabelsOK.
func TestScoreTextLabelsAreCaseInsensitiveAndReported(t *testing.T) {
	labels := []string{"Summary", "Risks", "Next steps"}

	s := scoreText("summary: fine. RISKS: none. next STEPS: ship.", labels)
	if !s.RequiredLabelsOK || len(s.MissingLabels) != 0 {
		t.Fatalf("case-folded labels reported missing: %v", s.MissingLabels)
	}

	s = scoreText("Summary: fine. Risks: none.", labels)
	if s.RequiredLabelsOK {
		t.Fatal("RequiredLabelsOK is true although a required label is absent")
	}
	if len(s.MissingLabels) != 1 || s.MissingLabels[0] != "Next steps" {
		t.Fatalf("MissingLabels = %v, want exactly [Next steps]", s.MissingLabels)
	}
}

// TestScoreTextWordSplitKeepsIntraWordPunctuation pins the tokenizer: apostrophes and hyphens
// stay INSIDE a word, so "don't" is one token, not two. Splitting them would inflate the word
// count and shift every trigram boundary, changing the repetition rate of identical text.
func TestScoreTextWordSplitKeepsIntraWordPunctuation(t *testing.T) {
	if got := scoreText("don't re-run it", nil); got.Words != 3 {
		t.Fatalf("Words = %d for `don't re-run it`, want 3 (don't / re-run / it); splitting on the "+
			"apostrophe and hyphen would give 5", got.Words)
	}
	// The same three tokens are exactly one trigram window, which a 5-token split would not be.
	if got := scoreText("don't re-run it", nil); got.TrigramWindows != 1 {
		t.Fatalf("TrigramWindows = %d, want 1", got.TrigramWindows)
	}
}

// mkRun is a scored run with a chosen repetition rate, built without touching a model.
func mkRun(bucket, mode, decode string, repeat float64, labelsOK bool) Run {
	r := Run{Bucket: bucket, Mode: mode, Decode: decode}
	r.Score = Score{TrigramRepeat: repeat, RequiredLabelsOK: labelsOK}
	return r
}

// bothModes returns one q8/int8 pair per mode, so a bucket is fully covered.
func bothModes(bucket string, q8Repeat, int8Repeat float64) []Run {
	var out []Run
	for _, m := range modes() {
		out = append(out,
			mkRun(bucket, m.name, "q8", q8Repeat, true),
			mkRun(bucket, m.name, "int8-q4k", int8Repeat, true))
	}
	return out
}

// TestCompareRunsFailsWhenInt8Repeats is the gate's whole purpose: int8 Q4_K decode must not
// repeat itself MORE than the q8 reference. Equal is allowed (the arms are not required to be
// bit-identical), better is allowed, worse is not.
func TestCompareRunsFailsWhenInt8Repeats(t *testing.T) {
	prompts := []Prompt{{Bucket: "long"}}

	for _, tc := range []struct {
		name           string
		q8, int8       float64
		wantPass       bool
		wantReasonPart string
	}{
		{"int8 worse", 0.10, 0.20, false, "worsened"},
		{"equal", 0.10, 0.10, true, ""},
		{"int8 better", 0.20, 0.10, true, ""},
	} {
		cmps, pass, reason := compareRuns(bothModes("long", tc.q8, tc.int8), prompts)
		if pass != tc.wantPass {
			t.Errorf("%s: pass = %v, want %v (q8=%v int8=%v)", tc.name, pass, tc.wantPass, tc.q8, tc.int8)
		}
		if reason != "" {
			t.Errorf("%s: unexpected top-level reason %q", tc.name, reason)
		}
		if len(cmps) != len(modes()) {
			t.Fatalf("%s: got %d comparisons, want one per mode (%d)", tc.name, len(cmps), len(modes()))
		}
		for _, c := range cmps {
			if want := tc.int8 - tc.q8; math.Abs(c.Delta-want) > 1e-12 {
				t.Errorf("%s/%s: Delta = %v, want %v", tc.name, c.Mode, c.Delta, want)
			}
			if tc.wantReasonPart != "" && !strings.Contains(c.Reason, tc.wantReasonPart) {
				t.Errorf("%s/%s: Reason = %q, want it to mention %q", tc.name, c.Mode, c.Reason, tc.wantReasonPart)
			}
		}
	}
}

// TestCompareRunsFailsClosedOnMissingArm is the fail-closed property. A bucket/mode whose q8 or
// int8 arm never produced output must NOT be silently skipped: a crashed generation would
// otherwise let the gate report PASS on the arms that happened to survive.
func TestCompareRunsFailsClosedOnMissingArm(t *testing.T) {
	// Only the q8 arm of only the first mode.
	only := []Run{mkRun("long", modes()[0].name, "q8", 0.05, true)}
	cmps, pass, _ := compareRuns(only, []Prompt{{Bucket: "long"}})
	if pass {
		t.Fatal("pass = true with three of four arms missing; the gate must fail closed")
	}
	if len(cmps) != len(modes()) {
		t.Fatalf("got %d comparisons, want one per required mode (%d) even when arms are absent", len(cmps), len(modes()))
	}
	for _, c := range cmps {
		if c.Pass || c.Reason == "" {
			t.Errorf("mode %s: pass=%v reason=%q, want a failing comparison with a stated reason", c.Mode, c.Pass, c.Reason)
		}
	}
}

// TestCompareRunsFailsClosedOnExecutionError covers the other absence: both arms are present but
// one of them recorded an execution error, so its score is meaningless. A zero score from a
// crashed run would otherwise look like perfect coherence and PASS the gate.
func TestCompareRunsFailsClosedOnExecutionError(t *testing.T) {
	runs := bothModes("long", 0.10, 0.10) // would pass on the numbers alone
	runs[1].ExecutionError = "model failed to load"

	cmps, pass, _ := compareRuns(runs, []Prompt{{Bucket: "long"}})
	if pass {
		t.Fatal("pass = true although one arm errored; a crashed arm scores 0 and must not pass")
	}
	var failed int
	for _, c := range cmps {
		if !c.Pass && strings.Contains(c.Reason, "failed") {
			failed++
		}
	}
	if failed != 1 {
		t.Fatalf("%d comparisons blamed the failed arm, want exactly 1 (the other mode is unaffected)", failed)
	}
}

// TestCompareRunsEmptyManifestIsNotAPass guards the vacuous-truth hole: with nothing to compare,
// "all comparisons passed" is trivially true. The gate must instead report a reason and refuse.
func TestCompareRunsEmptyManifestIsNotAPass(t *testing.T) {
	cmps, pass, reason := compareRuns(nil, nil)
	if pass {
		t.Fatal("an empty manifest reported pass = true; a gate with no comparisons must not pass")
	}
	if len(cmps) != 0 {
		t.Fatalf("got %d comparisons from an empty manifest, want 0", len(cmps))
	}
	if reason == "" {
		t.Fatal("empty manifest produced no reason")
	}
}

// TestCompareRunsCoversEveryBucketAndMode pins the cross product the manifest implies: comparisons
// are keyed by (bucket, mode) and sorted, so extra runs for an unrequested bucket cannot pad the
// result and a requested bucket cannot go unchecked.
func TestCompareRunsCoversEveryBucketAndMode(t *testing.T) {
	runs := append(bothModes("long", 0.1, 0.1), bothModes("short", 0.1, 0.1)...)
	runs = append(runs, bothModes("unrequested", 0.9, 0.9)...)

	cmps, pass, _ := compareRuns(runs, []Prompt{{Bucket: "long"}, {Bucket: "short"}})
	if !pass {
		t.Fatal("pass = false although every requested comparison is level")
	}
	if want := 2 * len(modes()); len(cmps) != want {
		t.Fatalf("got %d comparisons, want %d (2 buckets x %d modes); the unrequested bucket must not appear",
			len(cmps), want, len(modes()))
	}
	seen := map[string]bool{}
	for _, c := range cmps {
		if c.Bucket == "unrequested" {
			t.Fatalf("comparison for a bucket the manifest never asked for: %+v", c)
		}
		seen[c.Bucket+"/"+c.Mode] = true
	}
	if len(seen) != 2*len(modes()) {
		t.Fatalf("comparisons are not distinct (bucket,mode) pairs: %v", seen)
	}
}

// TestSafeNameProducesAFilesystemSafeStem pins the artifact-path builder: run stems are joined
// straight into a path, so any separator or drive character must be neutralized, and the result
// must not start or end with the filler.
func TestSafeNameProducesAFilesystemSafeStem(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"long-greedy-q8", "long-greedy-q8"},
		{"Long Greedy INT8", "long-greedy-int8"},
		{"../../etc/passwd", "etc-passwd"},
		{`C:\tmp\x`, "c--tmp-x"}, // one filler per bad rune: ':' and '\' each map separately
		{"---", ""},
	} {
		if got := safeName(tc.in); got != tc.want {
			t.Errorf("safeName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if got := safeName("a/b"); strings.ContainsAny(got, `/\:`) {
		t.Errorf("safeName leaked a path separator: %q", got)
	}
}

// TestRunSelfcheckPasses exercises the command's own -selfcheck fixtures, which are what the
// operator runs to confirm the scorer works before spending a model on it.
func TestRunSelfcheckPasses(t *testing.T) {
	if err := runSelfcheck(); err != nil {
		t.Fatalf("runSelfcheck: %v", err)
	}
}

// TestExecuteRecordsUnreadablePromptsWithoutRunningTheModel drives execute's fail-closed path with
// no model and no subprocess: the prompt file is absent, so every arm must be recorded with an
// execution error and the artifact must NOT pass. It also pins the max-tokens default and the
// artifact's declared schema.
func TestExecuteRecordsUnreadablePromptsWithoutRunningTheModel(t *testing.T) {
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "manifest.json")
	manifest := Manifest{Model: "some-model", Prompts: []Prompt{{Bucket: "long", Path: "missing-prompt.txt"}}}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	// A binary path that cannot exist. The guard against execute silently going on to run the
	// model anyway is that ExecutionError must name the unreadable PROMPT, not the missing binary.
	a, err := execute(manifestPath, filepath.Join(dir, "no-such-fak"), filepath.Join(dir, "out"), 7)
	if err != nil {
		t.Fatalf("execute returned a hard error for an unreadable prompt: %v", err)
	}
	if a.Schema != schema {
		t.Errorf("Schema = %q, want %q", a.Schema, schema)
	}
	if a.MaxTokens != 512 {
		t.Errorf("MaxTokens = %d, want the 512 default", a.MaxTokens)
	}
	if a.Seed != 7 {
		t.Errorf("Seed = %d, want 7", a.Seed)
	}
	if want := 2 * len(modes()); len(a.Runs) != want {
		t.Fatalf("got %d runs, want %d (2 decode arms x %d modes)", len(a.Runs), want, len(modes()))
	}
	for _, r := range a.Runs {
		if !strings.Contains(r.ExecutionError, "missing-prompt.txt") {
			t.Errorf("%s/%s: ExecutionError = %q, want it to name the unreadable prompt; anything else "+
				"means execute went on to invoke the model", r.Mode, r.Decode, r.ExecutionError)
		}
		if r.ElapsedMS != 0 {
			t.Errorf("%s/%s: ElapsedMS = %d, want 0 — the model must not be invoked", r.Mode, r.Decode, r.ElapsedMS)
		}
	}
	if a.Pass {
		t.Fatal("artifact passed although no arm produced output")
	}
}

// TestExecuteRejectsIncompleteManifests pins the input contract: a manifest without a model, or
// with a prompt missing its bucket or path, is a hard error rather than a silently empty run.
func TestExecuteRejectsIncompleteManifests(t *testing.T) {
	dir := t.TempDir()
	for name, m := range map[string]Manifest{
		"no model":   {Prompts: []Prompt{{Bucket: "long", Path: "p.txt"}}},
		"no prompts": {Model: "m"},
		"no bucket":  {Model: "m", Prompts: []Prompt{{Path: "p.txt"}}},
		"no path":    {Model: "m", Prompts: []Prompt{{Bucket: "long"}}},
	} {
		path := filepath.Join(dir, safeName(name)+".json")
		data, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := execute(path, "fak", filepath.Join(dir, "out"), 0); err == nil {
			t.Errorf("%s: execute returned no error", name)
		}
	}
}

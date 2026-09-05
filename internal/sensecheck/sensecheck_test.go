package sensecheck

import (
	"encoding/json"
	"strings"
	"testing"
)

// hasDetector reports whether the report raised a smell from det.
func hasDetector(r Report, det string) bool {
	for _, sm := range r.Smells {
		if sm.Detector == det {
			return true
		}
	}
	return false
}

// Each detector must FIRE on a positive fixture and stay SILENT on its
// escape-hatch negative. The negative is the whole point: a battery that
// cannot say "actually, fine" is a false-positive machine.
func TestDetectorFiresAndEscapes(t *testing.T) {
	cases := []struct {
		name     string
		detector string
		positive string // must raise `detector`
		negative string // must NOT raise `detector`
	}{
		{
			name:     "success-over-error",
			detector: "success-over-error",
			positive: "All tests pass. Build output: exit status 1",
			// the word "error" as the object of a fix is not an OBSERVED failure
			negative: "Done: fixed the error handling in the parser",
		},
		{
			name:     "vacuous-guard/backoff",
			detector: "vacuous-guard",
			positive: "retry loop with backoff = 0 between attempts",
			negative: "retry loop with backoff = 250ms between attempts",
		},
		{
			name:     "impossible-magnitude/rate",
			detector: "impossible-magnitude",
			positive: "cache hit-rate 142% across the run",
			// unbounded growth over 100% is legitimate — no bounded-metric noun
			negative: "the new path is 300% faster than the baseline",
		},
		{
			name:     "tautology/self-compare",
			detector: "tautology",
			positive: "assert result == result  // sanity",
			// != is the NaN idiom, not a tautology
			negative: "assert value != value  // NaN guard",
		},
		{
			name:     "scope-inflation",
			detector: "scope-inflation",
			positive: "This always works for all users — tested it once and it passed.",
			negative: "This handles the documented cases; see the suite for coverage.",
		},
		{
			name:     "placeholder-shipped",
			detector: "placeholder-shipped",
			positive: "Feature is done and shipped. // TODO wire the real endpoint",
			negative: "Work in progress. // TODO wire the real endpoint",
		},
		{
			name:     "contradiction-pair/nochange",
			detector: "contradiction-pair",
			positive: "Fixed the bug. 0 files changed, nothing to commit.",
			negative: "Fixed the bug across 3 files.",
		},
		{
			name:     "contradiction-pair/skip",
			detector: "contradiction-pair",
			positive: "All tests pass (t.Skip on the flaky one).",
			negative: "All tests pass on the full suite.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/fires", func(t *testing.T) {
			r := Check(TextSubject("fixture", tc.positive))
			if !hasDetector(r, tc.detector) {
				t.Fatalf("detector %q did not fire on positive %q; smells=%+v", tc.detector, tc.positive, r.Smells)
			}
		})
		t.Run(tc.name+"/escapes", func(t *testing.T) {
			r := Check(TextSubject("fixture", tc.negative))
			if hasDetector(r, tc.detector) {
				t.Fatalf("detector %q fired on escape-hatch negative %q; smells=%+v", tc.detector, tc.negative, r.Smells)
			}
		})
	}
}

// The flagship case must be a REEK (near-certain incoherence), and every
// smell must carry a could_be_ok_if — a heuristic with no escape hatch is a
// bug in this package's contract.
func TestSuccessOverErrorIsReekAndEscapeHatched(t *testing.T) {
	r := Check(TextSubject("c0ffee", "shipped it — CI shows exit status 2"))
	if r.Verdict != VerdictSmells {
		t.Fatalf("verdict = %s, want SMELLS_FOUND", r.Verdict)
	}
	var found bool
	for _, sm := range r.Smells {
		if sm.Detector == "success-over-error" {
			found = true
			if sm.Severity != SevReek.String() {
				t.Errorf("severity = %s, want reek", sm.Severity)
			}
		}
		if strings.TrimSpace(sm.CouldBeOK) == "" {
			t.Errorf("smell %s carries no could_be_ok_if", sm.Detector)
		}
	}
	if !found {
		t.Fatal("expected success-over-error smell")
	}
}

// An empty subject ABSTAINS — an honest "no answer", never a silent CLEAN.
func TestEmptySubjectAbstains(t *testing.T) {
	if got := Check(TextSubject("empty", "   \n  ")).Verdict; got != VerdictAbstain {
		t.Fatalf("verdict = %s, want ABSTAIN", got)
	}
	if got := Check(Subject{Kind: "commit", Ref: "x"}).Verdict; got != VerdictAbstain {
		t.Fatalf("no-segments verdict = %s, want ABSTAIN", got)
	}
}

// Coherent text is CLEAN — the battery does not cry wolf on ordinary prose.
func TestCoherentTextIsClean(t *testing.T) {
	r := Check(TextSubject("good", "Refactored the parser into three passes; added a table test; go test ./... is green with 42 files changed."))
	if r.Verdict != VerdictClean {
		t.Fatalf("verdict = %s, want CLEAN; smells=%+v", r.Verdict, r.Smells)
	}
}

// Every Report — clean, smelly, or abstaining — carries the advisory fence.
func TestFenceAlwaysPresent(t *testing.T) {
	for _, raw := range []string{"", "shipped over exit status 1", "a calm coherent sentence"} {
		r := Check(TextSubject("x", raw))
		if !strings.Contains(r.Note, "ADVISORY") {
			t.Fatalf("report for %q dropped the advisory fence", raw)
		}
		if r.Schema != Schema {
			t.Fatalf("schema = %q, want %q", r.Schema, Schema)
		}
	}
}

// Same Subject in ⇒ byte-identical Report out (a pure fold, stable order).
func TestDeterministic(t *testing.T) {
	subj := Subject{Kind: "log", Ref: "run", Segments: LogSegments(
		"cache hit-rate 142% coverage\n\nassert x == x\n\nbackoff = 0 on retry\n\nall tests pass but exit status 1")}
	a, _ := json.Marshal(Check(subj))
	b, _ := json.Marshal(Check(subj))
	if string(a) != string(b) {
		t.Fatalf("non-deterministic report:\n a=%s\n b=%s", a, b)
	}
	// Ordered severity-desc: the first smell is the highest severity.
	r := Check(subj)
	if len(r.Smells) < 2 {
		t.Fatalf("expected several smells, got %d", len(r.Smells))
	}
	if sevRank(r.Smells[0].Severity) < sevRank(r.Smells[len(r.Smells)-1].Severity) {
		t.Fatalf("smells not ordered severity-desc: %v", r.Smells)
	}
}

// MaxSeverity feeds a --fail-on gate: no smells ⇒ not raised; a reek ⇒ reek.
func TestMaxSeverity(t *testing.T) {
	if _, raised := MaxSeverity(Check(TextSubject("x", "calm coherent prose"))); raised {
		t.Fatal("clean report should not raise")
	}
	sv, raised := MaxSeverity(Check(TextSubject("x", "done — exit status 1")))
	if !raised || sv != SevReek {
		t.Fatalf("MaxSeverity = (%v,%v), want (reek,true)", sv, raised)
	}
}

func TestParseSeverityRoundTrip(t *testing.T) {
	for _, sv := range []Severity{SevNote, SevSmell, SevReek} {
		got, ok := ParseSeverity(sv.String())
		if !ok || got != sv {
			t.Fatalf("round-trip %s -> (%v,%v)", sv, got, ok)
		}
	}
	if _, ok := ParseSeverity("bogus"); ok {
		t.Fatal("bogus severity should not parse")
	}
}

// LogSegments splits on blank lines, drops empties, and labels by index.
func TestLogSegments(t *testing.T) {
	segs := LogSegments("first para\nline two\n\n\n  \n\nsecond para\r\n\r\nthird")
	if len(segs) != 3 {
		t.Fatalf("got %d segments, want 3: %+v", len(segs), segs)
	}
	if segs[0].Label != "para 1" || segs[2].Label != "para 3" {
		t.Fatalf("labels = %q..%q", segs[0].Label, segs[2].Label)
	}
}

// CommitSubject keeps message and diff as distinct segments so a smell can
// point at the CLAIM vs the CODE.
func TestCommitSubjectSegments(t *testing.T) {
	s := CommitSubject("abc123", "fix: done", "+ // FIXME\n- old")
	if len(s.Segments) != 2 || s.Segments[0].Label != "commit-message" || s.Segments[1].Label != "diff" {
		t.Fatalf("unexpected segments: %+v", s.Segments)
	}
	// message says done, diff carries a FIXME ⇒ placeholder-shipped at subject level
	if !hasDetector(Check(s), "placeholder-shipped") {
		t.Fatal("expected placeholder-shipped across message+diff")
	}
}

// The evidence excerpt is bounded — a Report never carries a whole file.
func TestEvidenceClipped(t *testing.T) {
	long := "done " + strings.Repeat("x", 500) + " exit status 1"
	for _, sm := range Check(TextSubject("x", long)).Smells {
		if len([]rune(sm.Evidence)) > 130 {
			t.Fatalf("evidence not clipped: %d runes", len([]rune(sm.Evidence)))
		}
	}
}

// Render is legible and always ends with the fence.
func TestRenderCarriesFence(t *testing.T) {
	out := Render(Check(TextSubject("x", "shipped over exit status 1")))
	if !strings.Contains(out, "SMELLS_FOUND") || !strings.Contains(out, "ADVISORY") {
		t.Fatalf("render missing verdict or fence:\n%s", out)
	}
}

var (
	benchReportSink Report
	benchStringSink string
	benchSegsSink   []Segment
	benchSevSink    Severity
	benchBoolSink   bool
)

func BenchmarkCheckClean(b *testing.B) {
	subj := TextSubject("bench-clean", "Refactored the parser into three passes; added a table test; go test ./... is green with 42 files changed.")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Check(subj)
		if r.Verdict != VerdictClean {
			b.Fatalf("unexpected verdict: %s", r.Verdict)
		}
		benchReportSink = r
	}
}

func BenchmarkCheckSmelly(b *testing.B) {
	subj := TextSubject("bench-smelly", "All tests pass successfully. CI output: exit status 1. Retry loop configured with backoff = 0 and max_retries = 0. Cache hit-rate 142% across the run. assert result == result. This always works for all users — tested it once and it passed.")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Check(subj)
		if r.Verdict != VerdictSmells {
			b.Fatalf("unexpected verdict: %s", r.Verdict)
		}
		benchReportSink = r
	}
}

func BenchmarkCheckCommitSubject(b *testing.B) {
	subj := CommitSubject(
		"deadbeef",
		"feat(gateway): implement retry loop and mark done\n\nAll tests pass and feature is complete and ready to ship.",
		"+ // TODO wire the real endpoint\n+ if (retries == 0) { ... }\n- old code",
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Check(subj)
		if r.Verdict != VerdictSmells {
			b.Fatalf("unexpected verdict: %s", r.Verdict)
		}
		benchReportSink = r
	}
}

func BenchmarkCheckLogMultiSegment(b *testing.B) {
	logText := "para 1: initial bootstrap sequence completed with zero errors.\n\n" +
		"para 2: test execution finished: All tests pass. Build log: exit status 1\n\n" +
		"para 3: metrics reported cache hit-rate 142% coverage with backoff = 0.\n\n" +
		"para 4: final wrap-up done: Fixed the bug. 0 files changed, nothing to commit."
	segs := LogSegments(logText)
	subj := Subject{Kind: "log", Ref: "bench-log", Segments: segs}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Check(subj)
		if r.Verdict != VerdictSmells {
			b.Fatalf("unexpected verdict: %s", r.Verdict)
		}
		benchReportSink = r
	}
}

func BenchmarkLogSegments(b *testing.B) {
	raw := "para 1: line one\nline two\n\n" +
		"para 2: second segment\nwith more text\n\n" +
		"para 3: third paragraph here\n\n" +
		"para 4: fourth paragraph here\n\n" +
		"para 5: final paragraph\n"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		segs := LogSegments(raw)
		if len(segs) != 5 {
			b.Fatalf("unexpected segment count: %d", len(segs))
		}
		benchSegsSink = segs
	}
}

func BenchmarkRenderReport(b *testing.B) {
	subj := TextSubject("bench-render", "All tests pass successfully. CI output: exit status 1. Retry loop with backoff = 0. Cache hit-rate 142% coverage.")
	rep := Check(subj)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := Render(rep)
		if len(out) == 0 {
			b.Fatal("unexpected empty render")
		}
		benchStringSink = out
	}
}

func BenchmarkMaxSeverity(b *testing.B) {
	subj := TextSubject("bench-maxsev", "All tests pass — exit status 1")
	rep := Check(subj)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sev, raised := MaxSeverity(rep)
		if !raised || sev != SevReek {
			b.Fatalf("unexpected max severity: %v, %v", sev, raised)
		}
		benchSevSink = sev
		benchBoolSink = raised
	}
}

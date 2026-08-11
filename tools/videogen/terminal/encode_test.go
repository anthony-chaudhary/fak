// encode_test.go — the frame log's own probe, falsified.
//
// WHY THIS FILE EXISTS. #750's box B1 asks that `timeline.json` account for
// every emitted frame, and box B2 that a `-verify` mode "exits non-zero when one
// fails — a probe that cannot fail is not a probe". checkFrameLog is that probe
// for B1, and until this file landed the only evidence it worked was that it
// stayed quiet on the one timeline the repo ships. Quiet is exactly what a
// broken probe also looks like: every assertion in it is of the form "if the log
// is wrong, complain", so a typo'd condition, an inverted comparison, or the
// whole call being dropped from checkPacing all read as PASS on a good input.
//
// So each case below is a log that is wrong in exactly ONE way, and the test
// fails if checkFrameLog does not object. The last test is the one that catches
// the failure a unit test normally cannot see — checkFrameLog being correct and
// no longer WIRED — by driving it through checkPacing, the function -verify
// actually calls.
//
// ⛔ This is also the only automated test in this nested module. Its x/image
// dependency means the tree-wide `go build ./... && go vet ./...`
// gate at the repo root does not reach it: nothing outside this directory
// compiles this code, let alone runs it. Run it with
// `go -C tools/videogen/terminal test ./...`.
package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// goodTimeline is a minimal timeline that every assertion in checkPacing
// accepts: two frames, a log that covers both and sums to the stated cut, one
// step above the 2.0s floor, and dwells above the cmd/rc floors. Each test
// breaks exactly one property of it.
func goodTimeline() *timeline {
	return &timeline{
		TotalSec: 4.0,
		Frames:   2,
		Log: []tlFrame{
			{N: 1, Start: 0.0, Secs: 2.0, Class: "step", Step: 1, Line: "-- STEP 01 . A BANNER"},
			{N: 2, Start: 2.0, Secs: 2.0, Class: "out", Step: 1},
		},
		Steps:    []tlStep{{Title: "-- STEP 01 . A BANNER", Start: 0, Secs: 4.0, Frames: 2}},
		Counts:   map[string]int{"step": 1, "cmd": 1, "rc": 1},
		Secs:     map[string]float64{"step": 2.0, "out": 2.0},
		MinDwell: map[string]float64{"cmd": 0.5, "rc": 0.6},
	}
}

func TestCheckFrameLog(t *testing.T) {
	cases := []struct {
		name string
		// break mutates a known-good timeline into a known-bad one.
		break_ func(*timeline)
		// want is a substring of the complaint we require. Empty means the
		// timeline must be accepted with no findings at all.
		want string
	}{{
		name:   "a complete, ordered log that sums to the cut is accepted",
		break_: func(tl *timeline) {},
		want:   "",
	}, {
		name: "a log with no rows at all for a render with no frames is accepted",
		break_: func(tl *timeline) {
			tl.Log, tl.Frames, tl.TotalSec = nil, 0, 0
		},
		want: "",
	}, {
		// The exact defect #750 B1 was filed about: 25 rows for 511 frames.
		name: "a log that covers only some frames is a sample, not an audit trail",
		break_: func(tl *timeline) {
			tl.Log = tl.Log[:1]
		},
		want: "1 rows for 2 emitted frames",
	}, {
		name: "a log with more rows than frames is also incomplete bookkeeping",
		break_: func(tl *timeline) {
			tl.Log = append(tl.Log, tlFrame{N: 3, Start: 4.0, Secs: 1.0, Class: "out", Step: 1})
		},
		want: "3 rows for 2 emitted frames",
	}, {
		name: "a frame emitted with zero dwell is a frame nobody can see",
		break_: func(tl *timeline) {
			tl.Log[1].Secs = 0
			tl.TotalSec = 2.0
		},
		want: "dwell of 0.000s",
	}, {
		name: "a frame emitted with negative dwell is caught too",
		break_: func(tl *timeline) {
			tl.Log[1].Secs = -2.0
			tl.TotalSec = 0.0
		},
		want: "dwell of -2.000s",
	}, {
		name: "a log out of screen order cannot be read against the video",
		break_: func(tl *timeline) {
			tl.Log[0].Start, tl.Log[1].Start = 2.0, 0.0
		},
		want: "not in screen order",
	}, {
		// The rule that ties the log to the shipped MP4: totalSecs is what
		// encodeMP4 hands ffprobe as the intended duration, so a clock advanced
		// without a logged frame fails here — the shape of both historical
		// duration bugs.
		name: "a log that does not sum to the cut the encoder is checked against",
		break_: func(tl *timeline) {
			tl.TotalSec = 6.0
		},
		want: "does not account for the whole video",
	}, {
		name: "the sum complaint names both numbers so the gap is locatable",
		break_: func(tl *timeline) {
			tl.TotalSec = 6.0
		},
		want: "sums to 4.000000s but the cut is 6.000000s",
	}, {
		name: "float noise under 1e-6 is not a failure",
		break_: func(tl *timeline) {
			tl.TotalSec = 4.0 + 5e-7
		},
		want: "",
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tl := goodTimeline()
			c.break_(tl)
			got := checkFrameLog(tl)

			if c.want == "" {
				if len(got) != 0 {
					t.Fatalf("want the timeline accepted, got %d complaint(s):\n  %s",
						len(got), strings.Join(got, "\n  "))
				}
				return
			}
			if len(got) == 0 {
				t.Fatalf("want a complaint containing %q, got none — the probe passed a log that is wrong", c.want)
			}
			if !strings.Contains(strings.Join(got, "\n"), c.want) {
				t.Fatalf("want a complaint containing %q, got:\n  %s", c.want, strings.Join(got, "\n  "))
			}
		})
	}
}

// TestFrameLogTriggerLineIsRequiredOnlyWhereItExists pins the asymmetry
// checkFrameLog documents, in both directions. A step/cmd/rc/emph frame is
// entered by MATCHING a line, so an empty one means the log dropped it; a
// card/prose/out/tail frame can legitimately be blank screen — a card's spacer,
// a prose group of only blank lines — and demanding a line there would mean
// inventing one. Measured on the 07-27 captures: 58 of 511 frames are
// legitimately blank (29 of 110 card, 26 of 74 prose, 2 of 130 out, 1 tail),
// which is why the log names its trigger for 89% of frames and not 100%.
func TestFrameLogTriggerLineIsRequiredOnlyWhereItExists(t *testing.T) {
	cases := []struct {
		class      string
		wantObject bool
	}{
		{"step", true},
		{"cmd", true},
		{"rc", true},
		{"emph", true},
		{"card", false},
		{"prose", false},
		{"out", false},
		{"tail", false},
	}

	for _, c := range cases {
		t.Run(c.class, func(t *testing.T) {
			tl := goodTimeline()
			// Blank the trigger line on the second frame and give it the class
			// under test. Nothing else changes, so any complaint is about the
			// missing line.
			tl.Log[1].Class, tl.Log[1].Line = c.class, ""

			got := strings.Join(checkFrameLog(tl), "\n")
			objected := strings.Contains(got, "carry no trigger line")

			switch {
			case c.wantObject && !objected:
				t.Fatalf("class %q is entered by matching a line, so a blank one means the log lost it — "+
					"want a complaint, got: %q", c.class, got)
			case !c.wantObject && objected:
				t.Fatalf("class %q can legitimately be blank screen, so requiring a line would mean "+
					"inventing one — want no complaint, got: %q", c.class, got)
			}
		})
	}
}

// TestCheckPacingReportsAnIncompleteFrameLog is the wiring test, and it is the
// point of the file.
//
// ⛔ checkFrameLog being CORRECT and checkFrameLog being CALLED are two claims,
// and every test above establishes only the first. Delete the
// `f = append(f, checkFrameLog(tl)...)` line from checkPacing and all of them
// still pass, while `-verify` goes green on a 25-of-511 log — the precise defect
// #750 B1 names, restored, with the test suite reporting no problem. So drive
// the same broken log through checkPacing, which is what -verify runs.
func TestCheckPacingReportsAnIncompleteFrameLog(t *testing.T) {
	var cfg config
	cfg.Verify.defaults()

	if f := checkPacing(cfg, goodTimeline()); len(f) != 0 {
		t.Fatalf("the known-good timeline must pass checkPacing, else this test proves nothing "+
			"about the frame log; got:\n  %s", strings.Join(f, "\n  "))
	}

	tl := goodTimeline()
	tl.Log = tl.Log[:1] // 1 row for 2 frames

	f := checkPacing(cfg, tl)
	if !strings.Contains(strings.Join(f, "\n"), "1 rows for 2 emitted frames") {
		t.Fatalf("checkPacing must surface an incomplete frame log — checkFrameLog is not wired into "+
			"the function -verify calls, so a partial timeline.json would verify clean; got:\n  %s",
			strings.Join(f, "\n  "))
	}
}

func TestCheckPacingRejectsAFlashingOpeningAndConcepts(t *testing.T) {
	var cfg config
	cfg.Verify.defaults()
	cfg.Pacing.CardReveal = 0.1
	cfg.Pacing.CardConceptHold = 0.2

	tl := goodTimeline()
	tl.Counts["card"] = 1
	tl.Segments = []tlSeg{{Kind: "card", Secs: 2.0}}

	got := strings.Join(checkPacing(cfg, tl), "\n")
	for _, want := range []string{
		"card lines reveal in 0.10s",
		"card concepts hold for 0.20s",
		"the opening is 2.0s",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("checkPacing complaints:\n%s\nwant one containing %q", got, want)
		}
	}
}

func TestNeedToolUsesExplicitOverride(t *testing.T) {
	tool := filepath.Join(t.TempDir(), "ffmpeg-test")
	if runtime.GOOS == "windows" {
		tool += ".exe"
	}
	if err := os.WriteFile(tool, []byte("test"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VIDEOGEN_FFMPEG", tool)
	got, err := needTool("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	if got != tool {
		t.Fatalf("needTool = %q, want %q", got, tool)
	}
}

func TestNeedToolRejectsMissingOverride(t *testing.T) {
	t.Setenv("VIDEOGEN_FFMPEG", filepath.Join(t.TempDir(), "missing"))
	if _, err := needTool("ffmpeg"); err == nil {
		t.Fatal("expected invalid override error")
	}
}

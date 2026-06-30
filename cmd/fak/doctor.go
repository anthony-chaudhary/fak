package main

// fak doctor — a read-only operator diagnostic that wires the answer-shape WITNESS
// into actionable recommendations. It runs the degeneration/verbosity guard
// (internal/answershape) over a candidate answer/result AND cross-checks the real
// kernel admit verdict the context-MMU would reach on the same bytes
// (ctxmmu.ScreenBytes), then prints what to do about each finding. It is the fak
// analogue of `dos doctor`: the witness is the measurement, the doctor is the
// recommendation. Read-only, off the hot path, no session or gateway required.
//
// Exit 0 = healthy, 1 = at least one finding, 2 = usage error — so `fak doctor`
// also composes as a CI gate over a captured answer.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/answershape"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// Recommendation is one doctor finding: a check, its severity, what it found, and
// the operator action it recommends.
type Recommendation struct {
	Check     string `json:"check"`
	Severity  string `json:"severity"` // "ok" | "warn"
	Finding   string `json:"finding"`
	Recommend string `json:"recommend,omitempty"`
}

const (
	sevOK   = "ok"
	sevWarn = "warn"
)

// doctorReport is the structured result of a doctor run over one text.
type doctorReport struct {
	Shape           answershape.Report `json:"answer_shape"`
	KernelAdmit     string             `json:"kernel_admit"` // the abi.ReasonName the context-MMU would reach (or "none")
	KernelWouldHold bool               `json:"kernel_would_hold"`
	Recommendations []Recommendation   `json:"recommendations"`
	Findings        int                `json:"findings"`
}

func cmdDoctor(argv []string) {
	os.Exit(runDoctor(os.Stdin, os.Stdout, os.Stderr, argv))
}

// runDoctor is the testable core of `fak doctor`: it returns the exit code and
// takes explicit streams.
func runDoctor(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	text := fs.String("text", "", `text to diagnose, or "-" for stdin (default: stdin if neither --text nor --file is given)`)
	file := fs.String("file", "", "read the text from this file instead of --text/stdin")
	maxRepeat := fs.Float64("max-repeat", answershape.DefaultMaxRepeat, "largest in-shape repeat fraction (0..1); <=0 disables the repeat check")
	maxChars := fs.Int("max-chars", 0, "largest in-shape rune count; 0 disables the length check")
	ngram := fs.Int("ngram", answershape.DefaultNGram, "word n-gram width for the repeat metric")
	binary := fs.Bool("binary", false, "diagnose the running fak executable and sibling fak/fak.exe binaries for stale shadowing")
	asJSON := fs.Bool("json", false, "emit the doctor report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	if *binary {
		exe, err := os.Executable()
		if err != nil {
			fmt.Fprintf(stderr, "fak doctor: resolve executable: %v\n", err)
			return 1
		}
		rep := appversion.DiagnoseBinary(exe, appversion.DefaultBinaryDoctorCandidates(exe))
		if *asJSON {
			b, _ := json.MarshalIndent(rep, "", "  ")
			fmt.Fprintln(stdout, string(b))
		} else {
			writeBinaryDoctorHuman(stdout, rep)
		}
		if rep.Findings > 0 {
			return 1
		}
		return 0
	}

	input, err := readShapeInput(*text, *file, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak doctor: %v\n", err)
		return 2
	}

	rep := diagnose(input, answershape.Limits{MaxRepeat: *maxRepeat, MaxChars: *maxChars, NGram: *ngram})

	if *asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		writeDoctorHuman(stdout, rep)
	}
	if rep.Findings > 0 {
		return 1
	}
	return 0
}

func writeBinaryDoctorHuman(w io.Writer, rep appversion.BinaryReport) {
	fmt.Fprintln(w, "== fak doctor: binary freshness ==")
	fmt.Fprintf(w, "executable: %s\n", rep.Executable)
	for _, img := range rep.Images {
		tag := "candidate"
		if img.Current {
			tag = "current"
		}
		if !img.Exists {
			fmt.Fprintf(w, "  [%s] %s (missing)\n", tag, img.Path)
			continue
		}
		suffix := ""
		if img.Newer {
			suffix = " newer-than-current"
		}
		fmt.Fprintf(w, "  [%s] %s size=%d sha=%s%s\n", tag, img.Path, img.Size, shortHash(img.SHA256), suffix)
	}
	for _, r := range rep.Recommendations {
		tag := "OK  "
		if r.Severity == appversion.SeverityWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-14s %s\n", tag, r.Check, r.Finding)
		if r.Recommend != "" {
			fmt.Fprintf(w, "       recommend: %s\n", r.Recommend)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	if h == "" {
		return "-"
	}
	return h
}

// diagnose runs both checks over input and assembles the recommendations. It is
// pure (no I/O), so tests can assert the recommendation set directly.
func diagnose(input []byte, lim answershape.Limits) doctorReport {
	shape := answershape.Measure(input, lim)

	// Cross-check the real kernel admit rung: would the context-MMU hold these bytes
	// out of model context at write time? ScreenBytes is the shipped predicate the
	// gate uses, so the doctor reports the gate's actual decision, not a parallel one.
	reason, wouldHold := ctxmmu.ScreenBytes(input)

	rep := doctorReport{
		Shape:           shape,
		KernelAdmit:     abi.ReasonName(reason),
		KernelWouldHold: wouldHold,
	}

	// Check 1 — answer-shape degeneration/verbosity.
	if shape.Degenerate {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "answer-shape",
			Severity: sevWarn,
			Finding:  joinReasons(shape),
			Recommend: "the answer is degenerate (looping or runaway). Cap output tokens, lower " +
				"temperature, or exclude this result from context. The in-kernel enforcement is " +
				"the context-MMU repeat/oversize admit rung (internal/ctxmmu).",
		})
	} else {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "answer-shape",
			Severity: sevOK,
			Finding:  fmt.Sprintf("in shape (repeat %.2f <= %.2f, %d chars)", shape.RepeatFraction, lim.MaxRepeat, shape.Chars),
		})
	}

	// Check 2 — the kernel admit cross-check.
	if wouldHold {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "kernel-admit",
			Severity: sevWarn,
			Finding:  fmt.Sprintf("the context-MMU would QUARANTINE this result (%s)", abi.ReasonName(reason)),
			Recommend: "at write time the kernel would hold these bytes out of model-visible context — " +
				"treat the result as poison/secret/pollution, not a normal answer.",
		})
	} else {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "kernel-admit",
			Severity: sevOK,
			Finding:  "the context-MMU would admit this result (no secret/injection/pollution screen)",
		})
	}

	for _, r := range rep.Recommendations {
		if r.Severity == sevWarn {
			rep.Findings++
		}
	}
	return rep
}

func writeDoctorHuman(w io.Writer, rep doctorReport) {
	fmt.Fprintln(w, "== fak doctor: answer health ==")
	for _, r := range rep.Recommendations {
		tag := "OK  "
		if r.Severity == sevWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-12s %s\n", tag, r.Check, r.Finding)
		if r.Recommend != "" {
			fmt.Fprintf(w, "       recommend: %s\n", r.Recommend)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}

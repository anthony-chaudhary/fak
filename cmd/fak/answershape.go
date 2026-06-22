package main

// fak answer-shape — the consumer-facing degeneration/verbosity WITNESS. It reads a
// piece of text (a candidate model answer, a tool result) and judges its SHAPE:
// how repetitive (looping) and how long (runaway) it is, against caller thresholds.
// It is the graded, tunable dual of the kernel's write-time repeat-admit rung
// (internal/ctxmmu) — the same concern an operator can run over any text, off the
// hot path. Exit 0 = in shape, 1 = degenerate, 2 = usage error, so it composes as a
// pipeline gate. `fak doctor` wraps it into an operator recommendation.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/answershape"
)

func cmdAnswerShape(argv []string) {
	os.Exit(runAnswerShape(os.Stdin, os.Stdout, os.Stderr, argv))
}

// runAnswerShape is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly.
func runAnswerShape(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("answer-shape", flag.ContinueOnError)
	fs.SetOutput(stderr)
	text := fs.String("text", "", `text to check, or "-" for stdin (default: stdin if neither --text nor --file is given)`)
	file := fs.String("file", "", "read the text from this file instead of --text/stdin")
	maxRepeat := fs.Float64("max-repeat", answershape.DefaultMaxRepeat, "largest in-shape repeat fraction (0..1); <=0 disables the repeat check")
	maxChars := fs.Int("max-chars", 0, "largest in-shape rune count; 0 disables the length check")
	ngram := fs.Int("ngram", answershape.DefaultNGram, "word n-gram width for the repeat metric")
	asJSON := fs.Bool("json", false, "emit the shape Report as JSON")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0 // an explicit -h/--help is not a usage error
		}
		return 2
	}

	input, err := readShapeInput(*text, *file, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak answer-shape: %v\n", err)
		return 2
	}

	rep := answershape.Measure(input, answershape.Limits{
		MaxRepeat: *maxRepeat, MaxChars: *maxChars, NGram: *ngram,
	})

	if *asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		writeShapeHuman(stdout, rep)
	}
	if rep.Degenerate {
		return 1
	}
	return 0
}

// readShapeInput resolves the text to check from the flags: an explicit --file, a
// literal --text, "-" (or no source at all) reads stdin. Shared by answer-shape and
// doctor.
func readShapeInput(text, file string, stdin io.Reader) ([]byte, error) {
	switch {
	case file != "":
		b, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("read --file %q: %w", file, err)
		}
		return b, nil
	case text == "-" || text == "":
		// "-" is the explicit stdin form; an empty --text with no --file also reads
		// stdin so `… | fak answer-shape` works with no flags.
		b, err := io.ReadAll(stdin)
		if err != nil {
			return nil, fmt.Errorf("read stdin: %w", err)
		}
		return b, nil
	default:
		return []byte(text), nil
	}
}

// writeShapeHuman renders a Report as a compact operator-readable block.
func writeShapeHuman(w io.Writer, r answershape.Report) {
	verdict := "OK"
	if r.Degenerate {
		verdict = "DEGENERATE"
	}
	fmt.Fprintf(w, "answer-shape: %s\n", verdict)
	fmt.Fprintf(w, "  chars=%d words=%d ngram=%d\n", r.Chars, r.Words, r.NGram)
	fmt.Fprintf(w, "  repeat=%.2f  (ngram=%.2f line=%.2f period=%.2f)  max-repeat=%s  max-chars=%s\n",
		r.RepeatFraction, r.NGramRepeat, r.LineBlockRepeat, r.PeriodRepeat,
		fmtRepeatLimit(r.Limits.MaxRepeat), fmtCharLimit(r.Limits.MaxChars))
	for _, rs := range r.Reasons {
		fmt.Fprintf(w, "  - %s\n", rs)
	}
}

func fmtRepeatLimit(f float64) string {
	if f <= 0 {
		return "off"
	}
	return fmt.Sprintf("%.2f", f)
}

func fmtCharLimit(n int) string {
	if n <= 0 {
		return "off"
	}
	return fmt.Sprintf("%d", n)
}

// joinReasons renders a Report's reasons as one line for a recommendation finding.
func joinReasons(r answershape.Report) string {
	if len(r.Reasons) == 0 {
		return "in shape"
	}
	return strings.Join(r.Reasons, "; ")
}

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
)

// cmdDispatchAging is the read-only anti-starvation diagnostic over the READY dispatch backlog: it
// answers "which ready issues are starving right now, and in what order should a worker pick them?"
// The fleet's dispatch order (internal/dispatchtick, internal/dispatchorder) ranks by base priority
// ABSOLUTELY, so a low-priority ready unit can be out-weighed forever; this folds each ready unit's
// WAIT time into an effective weight (base + bounded aging boost) plus a hard starvation deadline,
// and reports the fair order. It reads the candidate set as JSON (an array of {id, base_weight,
// ready_since}) from --in or stdin, so it composes with whatever lists the live backlog. Pure
// diagnostic: the anti-starvation math lives in internal/dispatchaging; this shell only does I/O.
// `--fail-on-starved N` turns the starved count into a CI/loop exit code (mirrors
// `dispatch-conservation --fail-on-leak`).
func cmdDispatchAging(argv []string) {
	os.Exit(runDispatchAging(argv, os.Stdin, os.Stdout, os.Stderr, time.Now()))
}

func runDispatchAging(argv []string, stdin io.Reader, stdout, stderr io.Writer, now time.Time) int {
	fs := flag.NewFlagSet("dispatch-aging", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inPath := fs.String("in", "", "candidates JSON file (an array of {id, base_weight, ready_since}); default: stdin")
	asJSON := fs.Bool("json", false, "emit the machine-readable dispatchaging.Result")
	nowUnix := fs.Int64("now", now.Unix(), "clock as unix seconds (default: now)")
	intervalS := fs.Int64("interval-s", dispatchaging.DefaultIntervalSeconds, "aging quantum in seconds (<=0 disables soft aging)")
	boost := fs.Int("boost", dispatchaging.DefaultBoostPerInterval, "priority points added per interval waited")
	maxBoost := fs.Int("max-boost", dispatchaging.DefaultMaxBoostPoints, "cap on the soft aging boost (0=uncapped)")
	starveS := fs.Int64("starvation-s", dispatchaging.DefaultStarvationSeconds, "hard force-serve deadline in seconds (<=0 disables)")
	failOnStarved := fs.Int("fail-on-starved", -1, "exit 1 when the starved count exceeds N (default: report-only)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	raw, err := readCandidateSource(*inPath, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "dispatch-aging: %v\n", err)
		return 2
	}
	cands, err := decodeCandidates(raw)
	if err != nil {
		fmt.Fprintf(stderr, "dispatch-aging: %v\n", err)
		return 2
	}

	p := dispatchaging.Params{
		NowUnix:           *nowUnix,
		IntervalSeconds:   *intervalS,
		BoostPerInterval:  *boost,
		MaxBoostPoints:    *maxBoost,
		StarvationSeconds: *starveS,
	}
	res := dispatchaging.Fold(cands, p)

	if *asJSON {
		b, _ := json.MarshalIndent(res, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintln(stdout, renderDispatchAging(res))
	}

	gateSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "fail-on-starved" {
			gateSet = true
		}
	})
	if gateSet && res.StarvedCount > *failOnStarved {
		return 1
	}
	return 0
}

// readCandidateSource returns the raw JSON bytes from the file at path, or from stdin when path is
// empty ("-" also means stdin).
func readCandidateSource(path string, stdin io.Reader) ([]byte, error) {
	if path == "" || path == "-" {
		return io.ReadAll(stdin)
	}
	return os.ReadFile(path)
}

// decodeCandidates accepts either a bare JSON array of candidates or an object wrapping them under
// "candidates" / "order" (so a Result re-feeds), and is lenient about an empty body (no candidates).
func decodeCandidates(raw []byte) ([]dispatchaging.Candidate, error) {
	if len(strings.TrimSpace(string(raw))) == 0 {
		return nil, nil
	}
	var arr []dispatchaging.Candidate
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var wrap struct {
		Candidates []dispatchaging.Candidate `json:"candidates"`
		Order      []dispatchaging.Candidate `json:"order"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("candidates must be a JSON array of {id, base_weight, ready_since}: %w", err)
	}
	if len(wrap.Candidates) > 0 {
		return wrap.Candidates, nil
	}
	return wrap.Order, nil
}

// renderDispatchAging is the human summary. ASCII-only: the Windows console renders under cp1252.
func renderDispatchAging(r dispatchaging.Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch aging -- %d ready: %d starved, %d aging, %d fresh; oldest wait %s\n",
		len(r.Order), r.StarvedCount, r.AgingCount, r.FreshCount, humanDur(r.OldestWaitSeconds))
	if len(r.Order) == 0 {
		b.WriteString("  (nothing ready)")
		return b.String()
	}
	for _, u := range r.Order {
		mark := "  "
		switch u.Standing {
		case dispatchaging.StandingStarved:
			mark = "!!" // force-served this tick
		case dispatchaging.StandingAging:
			mark = " +"
		}
		fmt.Fprintf(&b, "%s #%d %-10s %-7s base=%d eff=%d (+%d) waited=%s\n",
			mark, u.Rank, u.ID, u.Standing, u.BaseWeight, u.EffectiveWeight, u.AgingBoost, humanDur(u.WaitSeconds))
	}
	fmt.Fprintf(&b, "  pick: %s", pickLabel(r))
	return b.String()
}

func pickLabel(r dispatchaging.Result) string {
	if p := r.Pick(); p != "" {
		return p
	}
	return "(none)"
}

// humanDur renders a wait in whole seconds as a compact ASCII duration (e.g. "2h30m", "45s").
func humanDur(secs int64) string {
	if secs <= 0 {
		return "0s"
	}
	h := secs / 3600
	m := (secs % 3600) / 60
	s := secs % 60
	parts := make([]string, 0, 3)
	if h > 0 {
		parts = append(parts, fmt.Sprintf("%dh", h))
	}
	if m > 0 {
		parts = append(parts, fmt.Sprintf("%dm", m))
	}
	if s > 0 && h == 0 { // drop seconds once we are into hours; keep them for short waits
		parts = append(parts, fmt.Sprintf("%ds", s))
	}
	if len(parts) == 0 {
		return "0s"
	}
	return strings.Join(parts, "")
}

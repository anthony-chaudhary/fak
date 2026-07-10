package main

// `fak multisubmit` -- the multi-submission planner (epic #3653, spine #3654).
//
// The idea: once a worker has resolved an issue ONCE (a successful apply), producing a
// second..Nth differentiated take on the SAME issue is cheap -- the exploration is
// already paid for. `fak multisubmit` plans that "bonus" fan-out: given an issue and a
// top-N, it lays out N submissions, each a distinct PROFILE that emphasises a different
// need of the issue ("angle"), assigns each a SEAT round-robin over the real account
// rotation pool (internal/accounts.RotationPlan), and marks whether each is a `seed`
// (cold -- must explore and produce the cache) or a `warm-replay` (reuses the cached
// resolution from the successful apply, so it is fast and deterministic).
//
//	fak multisubmit --issue 3653 --n 5           plan a 5-profile fan-out (table)
//	fak multisubmit --issue 3653 --n 5 --json    the same plan as JSON
//	fak multisubmit --issue 3653 --cold          plan a cold run (rank 1 seeds, rest warm off it)
//	fak multisubmit --angles correctness,tests-first,perf --issue 3653
//
// This verb is the SPINE: a pure planner (planMultiSubmit) plus a thin shell that reads
// the live rotation pool and prints the plan. It is plan-only by design -- actually
// spawning the submissions is the impure executor tier (#3658), deferred on purpose so
// the decision stays testable and side-effect free, exactly as `fak profile` splits its
// pure planProfile from the impure runProfile.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// multiSubmitAngles is the default catalog of resolution ANGLES -- the "different needs
// of the job" each profile emphasises. A caller can override the set (and order) with
// --angles; this is the top-5 default the goal named. Order is the rank order.
var multiSubmitAngles = []struct{ Key, Emphasis string }{
	{"correctness", "smallest change that is provably correct; cover the edge cases the issue names"},
	{"tests-first", "lead with a failing regression test, then the minimal code to green it"},
	{"minimal-diff", "least surface-area change; lowest review + rollback risk"},
	{"robustness", "harden the error paths and invariants around the fix"},
	{"dx-docs", "the fix plus the docs/DX that make it reviewable and adoptable"},
}

func multiSubmitDefaultAngles() []string {
	out := make([]string, len(multiSubmitAngles))
	for i, a := range multiSubmitAngles {
		out[i] = a.Key
	}
	return out
}

func multiSubmitEmphasis(key string) string {
	for _, a := range multiSubmitAngles {
		if a.Key == key {
			return a.Emphasis
		}
	}
	return "custom angle"
}

// multiSubmitInput is the post-flag input to the pure planner.
type multiSubmitInput struct {
	Issue      string   // the issue being resubmitted against (the "job")
	N          int      // how many differentiated profiles to submit
	Angles     []string // the angles to emphasise, in rank order; empty => default catalog
	Pool       []string // account rotation seats, in rotation order; may be empty
	HasCache   bool     // a prior successful apply's resolution is cached (warm) vs cold
	BestEffort bool     // the default post-apply best-effort trigger (vs a manual run)
}

// multiSubmission is one planned profile submission.
type multiSubmission struct {
	Rank     int    `json:"rank"`
	Angle    string `json:"angle"`
	Emphasis string `json:"emphasis"`
	Seat     string `json:"seat,omitempty"` // "" when no rotation pool was available
	Mode     string `json:"mode"`           // "seed" | "warm-replay"
}

// multiSubmitPlan is the full, reproducible fan-out decision.
type multiSubmitPlan struct {
	Issue       string            `json:"issue"`
	Trigger     string            `json:"trigger"` // "post-apply-best-effort" | "manual"
	Warm        bool              `json:"warm"`    // whether a cached resolution seeds the replays
	Submissions []multiSubmission `json:"submissions"`
	Notes       []string          `json:"notes,omitempty"`
}

// planMultiSubmit is the pure resolver: (issue, N, angles, pool, cache) -> the plan. No
// I/O, so a peer reproduces the exact fan-out from the same inputs and the table test
// pins it. It assigns seats round-robin over the pool (wrapping when N exceeds the pool,
// with an honest note), and marks each submission seed vs warm-replay: with a cache
// present every submission warm-replays it; cold, rank 1 seeds and the rest warm-replay
// its result once it lands.
func planMultiSubmit(in multiSubmitInput) (multiSubmitPlan, error) {
	issue := strings.TrimSpace(in.Issue)
	if issue == "" {
		return multiSubmitPlan{}, fmt.Errorf("an issue is required (e.g. fak multisubmit --issue 3653)")
	}
	n := in.N
	if n <= 0 {
		n = len(multiSubmitAngles) // the top-5 default
	}
	angles := in.Angles
	if len(angles) == 0 {
		angles = multiSubmitDefaultAngles()
	}
	if n > len(angles) {
		return multiSubmitPlan{}, fmt.Errorf(
			"requested N=%d profiles but only %d distinct angles available; pass more via --angles", n, len(angles))
	}
	angles = angles[:n]

	trigger := "manual"
	if in.BestEffort {
		trigger = "post-apply-best-effort"
	}
	plan := multiSubmitPlan{Issue: issue, Trigger: trigger, Warm: in.HasCache}

	if len(in.Pool) == 0 {
		plan.Notes = append(plan.Notes,
			"no rotation pool available; seats unassigned (dry plan) — provide --registry or FAK_ACCOUNTS_REGISTRY")
	} else if n > len(in.Pool) {
		plan.Notes = append(plan.Notes, fmt.Sprintf(
			"N=%d exceeds the %d-seat pool; seats wrap, so submissions sharing a seat share a rate-limit bucket (serialized, best-effort)", n, len(in.Pool)))
	}
	if !in.HasCache {
		plan.Notes = append(plan.Notes,
			"cold: rank 1 seeds (explores + produces the cache); ranks 2..N warm-replay it once it lands")
	}

	for i := 0; i < n; i++ {
		seat := ""
		if len(in.Pool) > 0 {
			seat = in.Pool[i%len(in.Pool)]
		}
		mode := "warm-replay"
		if !in.HasCache && i == 0 {
			mode = "seed"
		}
		plan.Submissions = append(plan.Submissions, multiSubmission{
			Rank:     i + 1,
			Angle:    angles[i],
			Emphasis: multiSubmitEmphasis(angles[i]),
			Seat:     seat,
			Mode:     mode,
		})
	}
	return plan, nil
}

func cmdMultiSubmit(argv []string) { os.Exit(runMultiSubmit(os.Stdout, os.Stderr, argv)) }

func runMultiSubmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("multisubmit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issue := fs.String("issue", "", "the issue to re-submit differentiated profiles against (the \"job\")")
	n := fs.Int("n", 0, "how many top-N profiles to submit (default: the 5-angle catalog)")
	anglesCSV := fs.String("angles", "", "comma-separated angles to emphasise, in rank order (default: correctness,tests-first,minimal-diff,robustness,dx-docs)")
	cache := fs.Bool("cache", true, "a prior successful apply's resolution is cached (warm replays); --cache=false plans a cold run")
	cold := fs.Bool("cold", false, "alias for --cache=false: plan a cold run where rank 1 seeds")
	bestEffort := fs.Bool("best-effort", true, "plan as the default post-apply best-effort trigger (--best-effort=false marks it a manual run)")
	registry := fs.String("registry", "", "path to the accounts registry.json (default: $FAK_ACCOUNTS_REGISTRY or ~/.claude-accounts/registry.json)")
	asJSON := fs.Bool("json", false, "emit the plan as JSON instead of a table")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 && strings.TrimSpace(*issue) == "" {
		// tolerate `fak multisubmit 3653` as a positional issue
		*issue = fs.Arg(0)
	}

	var angles []string
	if s := strings.TrimSpace(*anglesCSV); s != "" {
		for _, a := range strings.Split(s, ",") {
			if a = strings.TrimSpace(a); a != "" {
				angles = append(angles, a)
			}
		}
	}

	hasCache := *cache && !*cold

	pool, poolNote := loadRotationPool(*registry)

	plan, err := planMultiSubmit(multiSubmitInput{
		Issue:      *issue,
		N:          *n,
		Angles:     angles,
		Pool:       pool,
		HasCache:   hasCache,
		BestEffort: *bestEffort,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak multisubmit: %v\n", err)
		return 1
	}
	if poolNote != "" {
		plan.Notes = append(plan.Notes, poolNote)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(plan); err != nil {
			fmt.Fprintf(stderr, "fak multisubmit: %v\n", err)
			return 1
		}
		return 0
	}
	printMultiSubmitPlan(stdout, plan)
	return 0
}

// loadRotationPool best-effort reads the account rotation pool (seat names in rotation
// order) from the registry. A missing/empty registry is not an error here -- the planner
// still produces a dry plan with seats unassigned -- so it returns a human note instead.
func loadRotationPool(registryPath string) (pool []string, note string) {
	path := strings.TrimSpace(registryPath)
	if path == "" {
		path = os.Getenv("FAK_ACCOUNTS_REGISTRY")
	}
	if path == "" {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			path = filepath.Join(home, ".claude-accounts", "registry.json")
		}
	}
	if path == "" {
		return nil, ""
	}
	reg, err := accounts.LoadRegistry(path)
	if err != nil {
		return nil, fmt.Sprintf("rotation pool unread (%s): %v", path, err)
	}
	for _, seat := range reg.RotationPlan().Pool {
		pool = append(pool, seat.Name)
	}
	if len(pool) == 0 {
		return nil, fmt.Sprintf("rotation pool empty at %s", path)
	}
	return pool, ""
}

func printMultiSubmitPlan(w io.Writer, plan multiSubmitPlan) {
	fmt.Fprintf(w, "multi-submission plan for issue %s  (trigger: %s, warm: %v)\n",
		plan.Issue, plan.Trigger, plan.Warm)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "RANK\tANGLE\tMODE\tSEAT\tEMPHASIS")
	for _, s := range plan.Submissions {
		seat := s.Seat
		if seat == "" {
			seat = "-"
		}
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\t%s\n", s.Rank, s.Angle, s.Mode, seat, s.Emphasis)
	}
	tw.Flush()
	for _, note := range plan.Notes {
		fmt.Fprintf(w, "note: %s\n", note)
	}
}

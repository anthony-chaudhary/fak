// Command fanoutdemo shows the issue fan-out planner (internal/issuefanout, the
// `fak issue fanout` verb) working end to end for a stranger — one deterministic
// command, no key, no network, no GPU, no model.
//
// The spine-first default says: the moment a working spine ships, its follow-on
// backlog (QA, dogfooding, productization, observability, integration, docs,
// release) is filed at creation time, each item already carrying a complete,
// dispatchable issue contract. This demo drives the real planner twice to tell
// that whole story:
//
//   - the SPINE-FIRST GUARD: asked to fan out with no spine witness, the planner
//     REFUSES (a stranger sees the default is enforced, not merely suggested);
//   - the FAN-OUT: given a shipped spine, it expands the fixed taxonomy into
//     contract-ready follow-ons and proves every one is dispatchable.
//
// The plan is a pure function of its input (stdlib + issuecontract only — no gh,
// no disk, no clock), so the output is byte-identical every run and -selfcheck
// gates cross-platform in CI.
//
// Headless, three ways (no browser, no model):
//
//	go run ./cmd/fanoutdemo             # the spine-first fan-out story in the terminal
//	go run ./cmd/fanoutdemo -json       # the refusal + the full plan as JSON
//	go run ./cmd/fanoutdemo -selfcheck  # assert the fan-out invariants (CI gate)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// demoSpine is the fixed, self-referential input the demo fans out: the issue
// fan-out planner's OWN working spine. Using the planner on the leaf that ships
// it keeps the demo honest (it dogfoods the real verb) and fully deterministic —
// the same Input yields the same Plan on any box.
func demoSpine() issuefanout.Input {
	return issuefanout.Input{
		Title:     "issue fanout planner",
		Leaf:      "issuefanout",
		SpineRef:  "5b8f0bd1 (internal/issuefanout + fak issue fanout)",
		ParentRef: "#2510",
	}
}

// demoJSON is the -json envelope: the enforced spine-first refusal beside the
// full plan the planner builds once a spine is supplied.
type demoJSON struct {
	SpineFirstRefusal string           `json:"spine_first_refusal"`
	Plan              issuefanout.Plan `json:"plan"`
}

// refusalNoSpine drives the guard path: the same input with its spine witness
// stripped must be REFUSED (a deliberate contract refusal, not a crash), so the
// demo can show the default is enforced below the caller.
func refusalNoSpine() (string, issuefanout.Outcome) {
	in := demoSpine()
	in.SpineRef = ""
	_, err := issuefanout.Build(in)
	return errString(err), issuefanout.ClassifyOutcome(err)
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func render(plan issuefanout.Plan, refusal string) {
	fmt.Println("fanoutdemo · the spine-first fan-out planner, end to end")
	fmt.Println()
	fmt.Println("1. spine-first guard — asked to fan out with NO spine witness, the planner refuses:")
	fmt.Printf("   %s\n", refusal)
	fmt.Println()
	fmt.Println("2. fan-out — given a shipped spine, it files the whole dispatchable backlog:")
	fmt.Print(issuefanout.Render(plan))
	fmt.Println()
	fmt.Printf("area counts: %s\n", areaCountsLine(plan))
	fmt.Printf("every one of the %d follow-ons carries a complete, dispatchable issue contract "+
		"(scope · witness · acceptance gate · lane · closure binding).\n", len(plan.Candidates))
}

// areaCountsLine renders the per-area counts in fixed taxonomy order, so the line
// is deterministic (a bare map range would reorder run to run).
func areaCountsLine(plan issuefanout.Plan) string {
	var parts []string
	for _, area := range issuefanout.AreaNames() {
		if n := plan.AreaCounts[area]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", area, n))
		}
	}
	return strings.Join(parts, " ")
}

func main() {
	fs := flag.NewFlagSet("fanoutdemo", flag.ExitOnError)
	doJSON := fs.Bool("json", false, "emit the refusal + the full plan as JSON and exit")
	doSelfcheck := fs.Bool("selfcheck", false, "assert the fan-out invariants (deterministic plan, every follow-on dispatchable, spine-first refusal enforced) and exit non-zero on drift")
	_ = fs.Parse(os.Args[1:])

	if *doSelfcheck {
		os.Exit(selfcheck())
	}

	plan, err := issuefanout.Build(demoSpine())
	if err != nil {
		fmt.Fprintln(os.Stderr, "fanoutdemo:", err)
		os.Exit(1)
	}
	refusal, _ := refusalNoSpine()

	if *doJSON {
		b, _ := json.MarshalIndent(demoJSON{SpineFirstRefusal: refusal, Plan: plan}, "", "  ")
		fmt.Println(string(b))
		return
	}
	render(plan, refusal)
}

// selfcheck asserts the structural invariants that hold on any box, independent of
// how many templates the taxonomy carries: the spine-first guard refuses a
// spine-less input, a supplied spine builds a plan whose candidates all carry the
// leaf's marker-key prefix and pass the issue contract (dispatchable), the area
// counts sum to the candidate count, and the plan is byte-identical across two
// builds (determinism). It prints one "... invariants hold" line and exits
// non-zero on any drift.
func selfcheck() int {
	var c demoui.SelfcheckChecker

	// 1. Spine-first guard: no spine witness => a deliberate contract refusal.
	refusal, outcome := refusalNoSpine()
	if outcome != issuefanout.OutcomeRefused {
		c.Notef("spine-less input classified as %q, want %q", outcome, issuefanout.OutcomeRefused)
	}
	if !strings.Contains(refusal, "spine_ref is required") {
		c.Notef("spine-less refusal %q does not name the missing spine_ref", refusal)
	}

	// 2. A supplied spine builds a plan.
	plan, err := issuefanout.Build(demoSpine())
	if err != nil {
		c.Notef("Build with a spine failed: %v", err)
		fmt.Fprintf(os.Stderr, "fanoutdemo -selfcheck: FAIL: %v\n", c.Mismatches())
		return 1
	}
	if plan.Schema != issuefanout.Schema {
		c.Notef("plan schema %q, want %q", plan.Schema, issuefanout.Schema)
	}

	// 3. The plan clears the fan-out floor and its area counts are internally consistent.
	if len(plan.Candidates) < issuefanout.MinFanout {
		c.Notef("plan has %d candidates, below the fan-out floor %d", len(plan.Candidates), issuefanout.MinFanout)
	}
	sum := 0
	for _, n := range plan.AreaCounts {
		sum += n
	}
	c.Check("area_counts_sum", sum, len(plan.Candidates))

	// 4. Every follow-on carries the leaf marker-key prefix AND a dispatchable contract —
	//    the planner's one promise: filed and runnable the moment it lands.
	dispatchable := 0
	for _, cand := range plan.Candidates {
		if !strings.HasPrefix(cand.Key, "fanout-issuefanout-") {
			c.Notef("candidate key %q lacks the marker-key prefix fanout-issuefanout-", cand.Key)
		}
		if issuepolicy.ReviewCandidate(cand, issuepolicy.Options{}).Dispatchability == issuepolicy.Dispatchable {
			dispatchable++
		} else {
			c.Notef("candidate %s is not dispatchable", cand.Key)
		}
	}
	c.Check("dispatchable", dispatchable, len(plan.Candidates))

	// 5. Determinism: the pure planner yields a byte-identical plan across two builds.
	plan2, err2 := issuefanout.Build(demoSpine())
	if err2 != nil || !reflect.DeepEqual(plan, plan2) {
		c.Notef("plan is not deterministic across two builds (err=%v)", err2)
	}

	if c.Failed() {
		fmt.Fprintf(os.Stderr, "fanoutdemo -selfcheck: FAIL: %v\n", c.Mismatches())
		return 1
	}
	fmt.Printf("fanoutdemo -selfcheck: the fan-out invariants hold "+
		"(%d dispatchable follow-ons across %d areas · spine-first refusal enforced)\n",
		len(plan.Candidates), len(areaNamesPresent(plan)))
	return 0
}

// areaNamesPresent returns the taxonomy areas the plan actually populated, in
// plan order — used only to report the area count in the witness line.
func areaNamesPresent(plan issuefanout.Plan) []string {
	var names []string
	for _, area := range issuefanout.AreaNames() {
		if plan.AreaCounts[area] > 0 {
			names = append(names, area)
		}
	}
	return names
}

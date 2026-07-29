package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatch_tier_status.go — `fak dispatch tier-status`, the offline operator readout
// for the model-tier account decision (internal/dispatchtick C4->C8). It answers
// "given these open issues (with their tier/T<N>-required|optimal labels) and this
// account pool, which seat would each route to, where is frontier spend wasted on
// routine work, and where would work be REFUSED for want of a capable seat?" —
// WITHOUT launching anything.
//
//	fak dispatch tier-status --in issues.json           # human readout
//	fak dispatch tier-status --in issues.json --json     # the TierStatusReport JSON
//	fak dispatch tier-status --demo                       # runnable, no-input spine
//
// It is PURE and deterministic: the decision is dispatchtick.BuildTierStatusReport
// (no clock, no I/O, no launch); this shell only reads the issue rows, folds them,
// and renders. The tier per issue is derived from its LABELS the same way the
// dispatcher parses them (dispatchtick.IssueTierFromLabels), so the readout shows
// the real routing signal — and a missing or contradictory tag surfaces as a
// conservative frontier route with the tag flaw named, not a silent choice. It
// mirrors the leaf/shell split of `fak dispatch order` and the advisory,
// launches-nothing posture of `fak tier-calibrate`.

// tierStatusAccount is the JSON-friendly account row a caller authors: the fields
// the tier chooser reads (capability tier, availability, product) plus the optional
// tie-break signals. Converted to dispatchtick.AccountRow before folding; Kind is
// stamped "worker" so the row is routable.
type tierStatusAccount struct {
	Account        string `json:"account"`
	Product        string `json:"product,omitempty"`
	Model          string `json:"model,omitempty"`
	ModelTier      int    `json:"model_tier"`
	Available      bool   `json:"available"`
	RouteWeight    int    `json:"route_weight,omitempty"`
	LiveSessions   int    `json:"live_sessions,omitempty"`
	ActiveSessions int    `json:"active_sessions,omitempty"`
}

func (a tierStatusAccount) toRow() dispatchtick.AccountRow {
	return dispatchtick.AccountRow{
		Account:        a.Account,
		Product:        a.Product,
		Model:          a.Model,
		ModelTier:      a.ModelTier,
		Available:      a.Available,
		Kind:           "worker",
		RouteWeight:    a.RouteWeight,
		LiveSessions:   a.LiveSessions,
		ActiveSessions: a.ActiveSessions,
	}
}

// tierStatusInput is one issue row: its number + lane, an optional product filter,
// the GitHub labels carrying its tier tags, the account pool the decision runs over,
// and an optional witnessed outcome.
type tierStatusInput struct {
	Issue     int                 `json:"issue"`
	Lane      string              `json:"lane"`
	Product   string              `json:"product,omitempty"`
	Labels    []string            `json:"labels,omitempty"`
	Accounts  []tierStatusAccount `json:"accounts"`
	Outcome   string              `json:"outcome,omitempty"`
	Escalated bool                `json:"escalated,omitempty"`
}

// runDispatchTierStatus is the testable core of `fak dispatch tier-status`: it reads
// the issue rows (or an embedded demo), folds them through BuildTierStatusReport, and
// renders the readout or its JSON. Exit 0 ok, 1 a read/parse error, 2 a usage error.
func runDispatchTierStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch tier-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read tier-status issue rows from this JSON file (default: stdin)")
	demo := fs.Bool("demo", false, "fold an embedded demo fixture instead of reading input (a runnable, no-input spine)")
	asJSON := fs.Bool("json", false, "emit the TierStatusReport as JSON instead of the human readout")
	if !parseFlags(fs, argv) {
		return 2
	}

	var inputs []dispatchtick.TierDecisionInput
	if *demo {
		inputs = demoTierStatusInputs()
	} else {
		raw, code := readTierStatusInput(stderr, *in)
		if code != 0 {
			return code
		}
		parsed, err := parseTierStatusInputs(raw)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch tier-status: %v\n", err)
			return 1
		}
		inputs = parsed
	}

	rep := dispatchtick.BuildTierStatusReport(inputs)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch tier-status")
	}
	fmt.Fprint(stdout, rep.Render())
	return 0
}

func readTierStatusInput(stderr io.Writer, path string) ([]byte, int) {
	return readDispatchStdinOrFile(stderr, path, "fak dispatch tier-status")
}

// parseTierStatusInputs accepts a bare JSON array of issue rows or an object with an
// "issues" field, converts each to a dispatchtick.TierDecisionInput (accounts ->
// AccountRow), and validates any witnessed outcome against the closed vocabulary so a
// typo fails loud instead of silently rendering as pending.
func parseTierStatusInputs(raw []byte) ([]dispatchtick.TierDecisionInput, error) {
	var rows []tierStatusInput
	var obj struct {
		Issues []tierStatusInput `json:"issues"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Issues != nil {
		rows = obj.Issues
	} else if err := json.Unmarshal(raw, &rows); err != nil {
		return nil, fmt.Errorf("parse issue rows: %w", err)
	}
	out := make([]dispatchtick.TierDecisionInput, 0, len(rows))
	for i, r := range rows {
		outcome := dispatchtick.TierOutcome(r.Outcome)
		if r.Outcome != "" && !outcome.Valid() {
			return nil, fmt.Errorf("issue row %d (#%d): unknown outcome %q", i, r.Issue, r.Outcome)
		}
		accts := make([]dispatchtick.AccountRow, 0, len(r.Accounts))
		for _, a := range r.Accounts {
			accts = append(accts, a.toRow())
		}
		out = append(out, dispatchtick.TierDecisionInput{
			Issue:     r.Issue,
			Lane:      r.Lane,
			Product:   r.Product,
			Labels:    r.Labels,
			Rows:      accts,
			Outcome:   outcome,
			Escalated: r.Escalated,
		})
	}
	return out, nil
}

// demoTierStatusInputs is the embedded, runnable fixture: a shared three-tier seat
// pool and five issues that exercise every branch of the readout — a clean routine
// route to the cheapest seat, a security-shape route to the frontier, an over-tier
// fallback when the optimal seat is down, a contradictory-label issue that stays
// conservative with its tag flaw named, and an under-tier refusal when no seat meets
// the floor.
func demoTierStatusInputs() []dispatchtick.TierDecisionInput {
	full := []dispatchtick.AccountRow{
		{Account: "frontier", Kind: "worker", ModelTier: 1, Available: true},
		{Account: "mid", Kind: "worker", ModelTier: 2, Available: true},
		{Account: "small", Kind: "worker", ModelTier: 3, Available: true},
	}
	onlyFrontier := []dispatchtick.AccountRow{
		{Account: "frontier", Kind: "worker", ModelTier: 1, Available: true},
		{Account: "mid", Kind: "worker", ModelTier: 2, Available: false},
		{Account: "small", Kind: "worker", ModelTier: 3, Available: false},
	}
	onlySmall := []dispatchtick.AccountRow{
		{Account: "small", Kind: "worker", ModelTier: 3, Available: true},
	}
	return []dispatchtick.TierDecisionInput{
		{Issue: 4101, Lane: "docs", Labels: []string{"tier/T2-required", "tier/T2-optimal"}, Rows: full, Outcome: dispatchtick.TierOutcomeShipped},
		{Issue: 4102, Lane: "release", Labels: []string{"tier/T1-required", "tier/T0-optimal"}, Rows: full, Outcome: dispatchtick.TierOutcomeShipped},
		{Issue: 4103, Lane: "tools", Labels: []string{"tier/T2-required", "tier/T2-optimal"}, Rows: onlyFrontier, Outcome: dispatchtick.TierOutcomeShipped},
		{Issue: 4104, Lane: "gateway", Labels: []string{"tier/T0-required", "tier/T1-optimal"}, Rows: full},
		{Issue: 4105, Lane: "release", Labels: []string{"tier/T0-required", "tier/T0-optimal"}, Rows: onlySmall},
	}
}

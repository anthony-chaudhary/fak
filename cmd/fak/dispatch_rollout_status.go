package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// dispatch_rollout_status.go — `fak dispatch rollout-status`, the offline SHADOW
// readout that is the #3047 (model-tier C10) acceptance artifact: a dry-run that
// shows, for each unit of work, the CURRENT live-selected model tier beside the
// tier the model-tier route WOULD choose, and the delta between them — WITHOUT
// launching a worker or changing any live selection.
//
//	fak dispatch rollout-status --in items.json          # human readout
//	fak dispatch rollout-status --in items.json --json    # the ShadowReport JSON
//	fak dispatch rollout-status --demo                     # runnable, no-input spine
//
// This is the SHADOW node of the rollout guard's shadow -> canary -> default path
// (internal/dispatchtick/rollout.go). It runs every item in RolloutShadow mode
// through dispatchtick.FoldShadowReport, so the load-bearing invariant is
// structural, not promised: a shadow readout APPLIES NOTHING. The report carries
// an `any_applied` field precisely so a reader can SEE it stay false — the proof
// that this dry-run launched no worker. That is the acceptance gate: shadow fields
// proven without touching a live selection.
//
// It is PURE and deterministic, the same leaf/shell split `fak dispatch tier-status`
// uses: the decision is the pure fold in the dispatchtick leaf (no clock, no I/O,
// no launch); this shell only reads the item rows, folds them in shadow mode, and
// renders. Each item's tier metadata is derived from its GitHub LABELS the same way
// the dispatcher parses them (dispatchtick.IssueTierFromLabels), so the readout
// shows the REAL routing signal — a missing or contradictory tag surfaces as a
// conservative frontier route, never a silent cheap choice.
//
// The canary ELIGIBILITY column is advisory only: it counts the routine (low-risk
// T2 watchdog/meta) items where a cheaper tier would serve — candidate savings
// PENDING PARITY, never a claim they succeeded, and never a promise to route. The
// #3047 confusion risks are enforced upstream in the leaf: canary scope is exactly
// modelroute.ClassRoutine, and a cheaper launch is a candidate, not a win.

// rolloutStatusAccount is the JSON-friendly account row a caller authors — the
// same shape `fak dispatch tier-status` accepts, converted to a routable
// dispatchtick.AccountRow (Kind stamped "worker") before folding.
type rolloutStatusAccount struct {
	Account        string `json:"account"`
	Product        string `json:"product,omitempty"`
	Model          string `json:"model,omitempty"`
	ModelTier      int    `json:"model_tier"`
	Available      bool   `json:"available"`
	RouteWeight    int    `json:"route_weight,omitempty"`
	LiveSessions   int    `json:"live_sessions,omitempty"`
	ActiveSessions int    `json:"active_sessions,omitempty"`
}

func (a rolloutStatusAccount) toRow() dispatchtick.AccountRow {
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

// rolloutStatusInput is one unit of work to shadow: an id (for the readout), its
// work CLASS (which fixes canary eligibility), an optional product filter, the
// GitHub labels carrying its tier tags, the account pool the route runs over, and
// the CURRENT live-selected model tier to compare the would-choose against (0 =
// no live selection supplied).
type rolloutStatusInput struct {
	ID          string                 `json:"id"`
	Class       string                 `json:"class"`
	Product     string                 `json:"product,omitempty"`
	Labels      []string               `json:"labels,omitempty"`
	Accounts    []rolloutStatusAccount `json:"accounts"`
	CurrentTier int                    `json:"current_model_tier"`
}

// runDispatchRolloutStatus is the testable core of `fak dispatch rollout-status`:
// it reads the item rows (or an embedded demo), folds them through FoldShadowReport
// in shadow mode, and renders the readout or its JSON. Exit 0 ok, 1 a read/parse
// error, 2 a usage error.
func runDispatchRolloutStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch rollout-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read rollout-status item rows from this JSON file (default: stdin)")
	demo := fs.Bool("demo", false, "fold an embedded demo fixture instead of reading input (a runnable, no-input spine)")
	asJSON := fs.Bool("json", false, "emit the ShadowReport as JSON instead of the human readout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch rollout-status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	var items []dispatchtick.ShadowItem
	if *demo {
		items = demoRolloutStatusItems()
	} else {
		raw, code := readRolloutStatusInput(stderr, *in)
		if code != 0 {
			return code
		}
		parsed, err := parseRolloutStatusItems(raw)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch rollout-status: %v\n", err)
			return 1
		}
		items = parsed
	}

	rep := dispatchtick.FoldShadowReport(items)
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch rollout-status: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderShadowReport(rep))
	return 0
}

func readRolloutStatusInput(stderr io.Writer, path string) ([]byte, int) {
	if path == "" || path == "-" {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch rollout-status: read stdin: %v\n", err)
			return nil, 1
		}
		return raw, 0
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch rollout-status: read %q: %v\n", path, err)
		return nil, 1
	}
	return raw, 0
}

// rolloutStatusClasses is the closed set of work classes a caller may name. A typo
// fails loud (below) instead of silently folding as the unknown-class conservative
// frontier route — the readout must show the class the caller MEANT.
var rolloutStatusClasses = []modelroute.WorkClass{
	modelroute.ClassRoutine,
	modelroute.ClassNormalImpl,
	modelroute.ClassUltraHard,
	modelroute.ClassSecurityRelease,
}

// parseRolloutStatusClass validates a caller-supplied class against the closed
// vocabulary. An empty class defaults to routine — the canary-eligible scope — so
// the common "shadow this watchdog/meta work" case needs no class field; any other
// non-empty value must match a known class exactly.
func parseRolloutStatusClass(s string) (modelroute.WorkClass, error) {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return modelroute.ClassRoutine, nil
	}
	for _, c := range rolloutStatusClasses {
		if trimmed == string(c) {
			return c, nil
		}
	}
	names := make([]string, 0, len(rolloutStatusClasses))
	for _, c := range rolloutStatusClasses {
		names = append(names, string(c))
	}
	return "", fmt.Errorf("unknown class %q (want %s)", s, strings.Join(names, ", "))
}

// parseRolloutStatusItems accepts a bare JSON array of item rows or an object with
// an "items" field, converts each to a dispatchtick.ShadowItem (accounts ->
// AccountRow, labels -> IssueTier via the same parser the dispatcher uses), and
// validates the work class against the closed vocabulary so a typo fails loud.
func parseRolloutStatusItems(raw []byte) ([]dispatchtick.ShadowItem, error) {
	var rows []rolloutStatusInput
	var obj struct {
		Items []rolloutStatusInput `json:"items"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Items != nil {
		rows = obj.Items
	} else if err := json.Unmarshal(raw, &rows); err != nil {
		return nil, fmt.Errorf("parse item rows: %w", err)
	}
	out := make([]dispatchtick.ShadowItem, 0, len(rows))
	for i, r := range rows {
		class, err := parseRolloutStatusClass(r.Class)
		if err != nil {
			return nil, fmt.Errorf("item row %d (%q): %w", i, r.ID, err)
		}
		// Derive the tier metadata from the labels exactly as the dispatcher does; a
		// missing/contradictory tag yields the conservative (frontier) IssueTier.
		issue, _ := dispatchtick.IssueTierFromLabels(r.Labels)
		accts := make([]dispatchtick.AccountRow, 0, len(r.Accounts))
		for _, a := range r.Accounts {
			accts = append(accts, a.toRow())
		}
		id := r.ID
		if strings.TrimSpace(id) == "" {
			id = fmt.Sprintf("item-%d", i)
		}
		out = append(out, dispatchtick.ShadowItem{
			ID:          id,
			Class:       class,
			Issue:       issue,
			Rows:        accts,
			Product:     r.Product,
			CurrentTier: r.CurrentTier,
		})
	}
	return out, nil
}

// renderShadowReport turns the folded ShadowReport into an aligned operator
// readout: a header proving the mode + the any-applied invariant, one line per
// item with its current/would-choose tiers and delta, and a delta tally footer.
// It never launches anything — it renders the pure fold.
func renderShadowReport(rep dispatchtick.ShadowReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch rollout-status — SHADOW readout (%s)\n", rep.Schema)
	fmt.Fprintf(&b, "  mode=%s  items=%d  applied=%v (a shadow dry-run applies nothing)\n\n", rep.Mode, rep.Items, rep.AnyApplied)

	if len(rep.Rows) == 0 {
		b.WriteString("  (no items)\n")
		return b.String()
	}

	// Column widths from the data so the readout aligns for any id/class length.
	idW, classW := len("ITEM"), len("CLASS")
	for _, row := range rep.Rows {
		if n := len(row.ID); n > idW {
			idW = n
		}
		if n := len(string(row.Class)); n > classW {
			classW = n
		}
	}
	fmt.Fprintf(&b, "  %-*s  %-*s  %-7s  %-11s  %-6s  %-6s  %s\n",
		idW, "ITEM", classW, "CLASS", "CURRENT", "WOULD-CHOOSE", "DELTA", "CANARY", "REASON")
	for _, row := range rep.Rows {
		fmt.Fprintf(&b, "  %-*s  %-*s  %-7s  %-11s  %-6s  %-6s  %s\n",
			idW, row.ID,
			classW, string(row.Class),
			tierCell(row.CurrentTier),
			tierCell(row.WouldChooseTier),
			row.Delta,
			canaryCell(row.InCanaryScope, row.Delta),
			row.Reason)
	}

	fmt.Fprintf(&b, "\n  tally: same=%d cheaper=%d more-capable=%d refused=%d no-current=%d\n",
		rep.Same, rep.Cheaper, rep.MoreCapable, rep.Refused, rep.NoCurrent)
	fmt.Fprintf(&b, "  canary-eligible=%d (routine items a cheaper tier would serve — candidate savings PENDING PARITY, not a win)\n",
		rep.CanaryEligible)
	return b.String()
}

// tierCell renders a ModelTier int for the readout: a positive tier as "T<n>", and
// 0 (no current selection, or a route that refused) as "—".
func tierCell(tier int) string {
	if tier <= 0 {
		return "—"
	}
	return fmt.Sprintf("T%d", tier)
}

// canaryCell marks whether this item is a canary CANDIDATE: in the routine scope
// AND a cheaper tier would serve. Everything else is "-" — out of scope, or no
// saving. Advisory only; it never means the route was or will be applied.
func canaryCell(inScope bool, delta string) string {
	if inScope && delta == dispatchtick.DeltaCheaper {
		return "yes"
	}
	return "-"
}

// demoRolloutStatusItems is the embedded, runnable fixture: a shared three-tier
// seat pool and items that exercise every delta branch of the readout —
//   - a routine watchdog currently on a frontier seat that WOULD drop to the
//     cheapest (cheaper, canary-eligible),
//   - a routine status item already on the cheapest seat (same, no change),
//   - a NORMAL-IMPL item that would also route cheaper but is OUT of canary scope
//     (cheaper delta, canary "-"), proving scope is class-gated not price-gated,
//   - a security/release item that stays on the frontier (same),
//   - a routine item where the pool cannot meet even the routine floor (refused).
func demoRolloutStatusItems() []dispatchtick.ShadowItem {
	full := []dispatchtick.AccountRow{
		{Account: "frontier", Kind: "worker", ModelTier: 1, Available: true},
		{Account: "mid", Kind: "worker", ModelTier: 2, Available: true},
		{Account: "small", Kind: "worker", ModelTier: 3, Available: true},
	}
	noneUp := []dispatchtick.AccountRow{
		{Account: "frontier", Kind: "worker", ModelTier: 1, Available: false},
		{Account: "small", Kind: "worker", ModelTier: 3, Available: false},
	}
	routine, _ := dispatchtick.IssueTierFromLabels([]string{"tier/T2-required", "tier/T2-optimal"})
	security, _ := dispatchtick.IssueTierFromLabels([]string{"tier/T1-required", "tier/T0-optimal"})
	return []dispatchtick.ShadowItem{
		{ID: "watchdog-1", Class: modelroute.ClassRoutine, Issue: routine, Rows: full, CurrentTier: 1},
		{ID: "status-2", Class: modelroute.ClassRoutine, Issue: routine, Rows: full, CurrentTier: 3},
		{ID: "impl-3", Class: modelroute.ClassNormalImpl, Issue: routine, Rows: full, CurrentTier: 1},
		{ID: "release-4", Class: modelroute.ClassSecurityRelease, Issue: security, Rows: full, CurrentTier: 1},
		{ID: "starved-5", Class: modelroute.ClassRoutine, Issue: routine, Rows: noneUp, CurrentTier: 2},
	}
}

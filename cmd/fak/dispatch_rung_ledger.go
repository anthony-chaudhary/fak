package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak dispatch rung-ledger` — the operator READ surface over the escalation spend ledger
// (epic #5416 track D).
//
// dispatch_rung_escalate.go buys rungs automatically, against a budget declared in the
// operator's environment and counted from a file on disk. Shipping an automatic spend with no
// way to look at what it spent is how a budget turns into folklore: the numbers existed only
// inside a tick payload that scrolls away, and the question an operator actually arrives with
// — "why has this item stopped escalating?" — was answerable only by reading JSONL by hand
// and re-deriving the arithmetic.
//
// Three rules make this a SURFACE and not a second implementation:
//
//  1. SPEND IS THE ENFORCER'S SPEND. The per-item number here is EscalationTally.Spent — the
//     item's own debits PLUS every unattributable one — not the count of rows bearing its
//     name. Those two differ exactly when the ledger holds a torn or unowned row, which is
//     exactly when someone is reading this. Printing the smaller, more flattering number
//     would show an item with budget remaining that the actuator is refusing to escalate, and
//     send its operator hunting for a bug in the ladder instead of for the torn row.
//  2. AUTHORITY COMES FROM THE ENFORCER'S READER. Ceiling and budget are read through
//     dispatchRungBounds, and reach through dispatchRungReach, so this cannot disagree with
//     what the next tick will do. When nothing is declared it reports the actuator's own
//     closed token and prints no value at all, because the one thing an undeclared budget
//     must never be rendered as is an unlimited one.
//  3. IT STILL RENDERS A LEDGER THE ACTUATOR HAS GIVEN UP ON. Past the per-launch read cap
//     the actuator refuses to escalate anything; this reads the file anyway and says the
//     actuator is refusing it. A diagnostic that goes dark under the very condition it exists
//     to diagnose is not a diagnostic. The asymmetry runs the other way for absence: a MISSING
//     ledger is "nothing has been bought" and exits 0, while an UNREADABLE one is an error,
//     because reporting zero spend for a fleet that has been spending is the one wrong answer
//     that looks entirely correct.
//
// Read-only by construction — it opens one file and never writes — so it is safe against a
// live fleet mid-sweep. It reports; it never repairs. Rotating an oversized ledger and
// reconciling a double-writing producer are operator acts, and a surface that quietly
// truncated or de-duplicated the file would be destroying the only evidence of the spend.

// rungLedgerDebit is one ledger row, rendered as written.
//
// Duplicate rows are shown, not filtered. The dedupe rule lives in TallyEscalations because
// it is the rule the BUDGET is counted under; re-deriving it here to grey out rows would put
// a second copy of a spend rule in the tree, and two copies of a spend rule is precisely how
// a surface ends up disagreeing with the enforcer it exists to explain. The count of dropped
// repeats comes from the tally instead, so the listing and the arithmetic never drift.
type rungLedgerDebit struct {
	At     string `json:"at,omitempty"`
	Item   string `json:"item,omitempty"`
	From   string `json:"from,omitempty"`
	To     string `json:"to,omitempty"`
	Reason string `json:"reason,omitempty"`
	ID     string `json:"id,omitempty"`
}

// rungLedgerItem is one work item's position against the budget.
type rungLedgerItem struct {
	Item string `json:"item"`
	// Debits is the deduplicated count of rows naming this item.
	Debits int `json:"debits"`
	// Spent is what the actuator will charge it — Debits plus every unattributable row. This
	// is the number handed to AfterAttempt as priorEscalations, so it is the number that
	// decides, and therefore the number printed.
	Spent int `json:"spent"`
	// Left is nil when no budget is declared. An absent bound is not an infinite one, and
	// rendering it as a number would be inventing an authority nobody granted.
	Left *int `json:"left,omitempty"`
	// Exhausted reports that AfterAttempt would now answer escalation-budget-spent for this
	// item. A declared budget of 0 exhausts every item, which is what declaring 0 means.
	Exhausted bool `json:"exhausted,omitempty"`
}

// rungLedgerReport is the whole answer: where the money went, and what the ladder is
// currently authorised to spend next.
type rungLedgerReport struct {
	Path   string `json:"path"`
	Exists bool   `json:"exists"`
	Bytes  int64  `json:"bytes,omitempty"`

	// LadderEnabled is the placement seam's own switch. It is reported separately from
	// Blocked because the actuator returns NO token when it is off — an operator staring at a
	// silent tick and an empty ledger needs to be told the difference between "nothing earned
	// a rung" and "the seam is not running".
	LadderEnabled bool `json:"ladder_enabled"`
	// Ceiling, Budget and ReachDeclared are the declared authority, read through the
	// actuator's own readers. Absent means undeclared, never unlimited.
	Ceiling       string `json:"ceiling,omitempty"`
	Budget        *int   `json:"budget,omitempty"`
	ReachDeclared bool   `json:"reach_declared"`
	// Blocked is every closed token the actuator would currently refuse on, in the order the
	// actuator evaluates them. All of them, not the first: an operator who fixes one blocker
	// and finds the ladder still dead has been told half an answer.
	Blocked []string `json:"blocked,omitempty"`

	// Lines is every non-blank line the reader examined; Torn is the subset that would not
	// parse and Unowned the subset that parsed without naming an item. Both are reported
	// apart from their sum because their cures differ — a torn row means a writer was
	// interrupted, an unowned one means a writer has a bug — and both are folded into
	// Unattributable, which is what every item is charged.
	Lines          int `json:"lines,omitempty"`
	Torn           int `json:"torn,omitempty"`
	Unowned        int `json:"unowned,omitempty"`
	Duplicates     int `json:"duplicates,omitempty"`
	Unattributable int `json:"unattributable,omitempty"`

	// Item echoes the --item filter. The filter narrows the LISTING only: unattributable
	// debits are charged to every item, so a filtered view still reports them and still adds
	// them to the filtered item's spend.
	Item   string            `json:"item,omitempty"`
	Debits []rungLedgerDebit `json:"debits"`
	Items  []rungLedgerItem  `json:"items"`
}

// runDispatchRungLedger renders the escalation spend ledger.
//
// Exit 0 when it could answer (including "the ledger is absent, nothing was bought"), 1 when
// the ledger is present and would not read, 2 on usage.
func runDispatchRungLedger(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch rung-ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", ".", "repo root holding the .dispatch-runs directory")
	item := fs.String("item", "", "narrow the listing to one work item (an issue number, as the ledger keys it)")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	rep, err := buildRungLedgerReport(*workspace, strings.TrimSpace(*item))
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch rung-ledger: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch rung-ledger")
	}
	renderRungLedger(stdout, rep)
	return 0
}

// buildRungLedgerReport folds the declared authority and the on-disk ledger into one answer.
//
// It deliberately does NOT use readEscalationLedger: that reader refuses an oversized file
// because a budget counted from a file the reader could not finish is not a bound, and
// refusing is right for the enforcer. For the surface it is exactly backwards — "the ledger
// got too big" is a state an operator has to see the contents of to fix.
func buildRungLedgerReport(workspace, item string) (rungLedgerReport, error) {
	path := dispatchEscalationLedgerPath(workspace)
	rep := rungLedgerReport{
		Path:          path,
		LadderEnabled: dispatchRungPlacementEnabled(),
		Item:          item,
		Debits:        []rungLedgerDebit{},
		Items:         []rungLedgerItem{},
	}

	bounds, reason := dispatchRungBounds()
	if reason != "" {
		rep.Blocked = append(rep.Blocked, reason)
	} else {
		// Carried as a pointer, and the arithmetic below reads the same field that gets
		// printed. An in-band sentinel for "undeclared" would collide with a real declaration
		// the moment an operator wrote one — FLEET_DISPATCH_RUNG_BUDGET=-1 parses fine, and
		// AfterAttempt treats it as an exhausted budget, so a surface that read it back as
		// "undeclared" would print the friendliest possible rendering of the strictest
		// possible authority.
		budget := bounds.MaxAttempts
		rep.Budget = &budget
		// A declared budget with no ceiling is the tick's own next refusal, and it is
		// AfterAttempt's token rather than one of the actuator's — the rule declines here, the
		// actuator never gets to.
		if !bounds.Ceiling.Valid() {
			rep.Blocked = append(rep.Blocked, modelroute.ReasonNoCeiling)
		}
	}
	if bounds.Ceiling.Valid() {
		rep.Ceiling = string(bounds.Ceiling)
	}
	if _, ok := dispatchRungReach(); ok {
		rep.ReachDeclared = true
	} else {
		rep.Blocked = append(rep.Blocked, rungSkipNoReachDecl)
	}

	st, err := os.Stat(path)
	if err != nil {
		// Absent is an answer, not a failure: nothing has been bought.
		return rep, nil
	}
	rep.Exists = true
	rep.Bytes = st.Size()
	if rep.Bytes > dispatchRungJournalCap {
		rep.Blocked = append(rep.Blocked, escSkipLedgerBig)
	}

	f, err := os.Open(path)
	if err != nil {
		return rep, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	records, stats, err := modelroute.ReadEscalations(f)
	if err != nil {
		// The READER failed, which is not the same as the file holding junk: junk comes back
		// in stats and is charged. Reporting an empty ledger here would tell a fleet that has
		// been spending that it has spent nothing.
		return rep, fmt.Errorf("read %s: %w", path, err)
	}
	tally := modelroute.TallyEscalations(records, stats)

	rep.Lines = stats.Lines
	rep.Torn = stats.Malformed
	rep.Duplicates = tally.Duplicates
	rep.Unattributable = tally.Unattributable
	rep.Unowned = tally.Unattributable - stats.Malformed
	rep.Debits = rungLedgerDebits(records, item)
	rep.Items = rungLedgerItems(tally, item, rep.Budget)
	return rep, nil
}

// rungLedgerDebits renders the rows in ledger order, which is purchase order.
func rungLedgerDebits(records []modelroute.EscalationRecord, item string) []rungLedgerDebit {
	out := []rungLedgerDebit{}
	for _, e := range records {
		if item != "" && strings.TrimSpace(e.Item) != item {
			continue
		}
		row := rungLedgerDebit{
			Item:   strings.TrimSpace(e.Item),
			From:   string(e.From),
			To:     string(e.To),
			Reason: strings.TrimSpace(e.Reason),
			ID:     strings.TrimSpace(e.ID),
		}
		if !e.At.IsZero() {
			row.At = e.At.UTC().Format(time.RFC3339)
		}
		out = append(out, row)
	}
	return out
}

// rungLedgerItems folds the tally into per-item budget positions.
//
// When a filter names an item the ledger has never seen, the row is SYNTHESISED rather than
// omitted, because an item with no debits of its own still carries every unattributable one.
// Answering "no such item" would report zero spend for an item the actuator may already be
// refusing to escalate.
func rungLedgerItems(tally modelroute.EscalationTally, item string, budget *int) []rungLedgerItem {
	keys := make([]string, 0, len(tally.ByItem)+1)
	if item != "" {
		keys = append(keys, item)
	} else {
		for k := range tally.ByItem {
			keys = append(keys, k)
		}
	}
	out := make([]rungLedgerItem, 0, len(keys))
	for _, k := range keys {
		row := rungLedgerItem{Item: k, Debits: tally.ByItem[k], Spent: tally.Spent(k)}
		if budget != nil {
			left := *budget - row.Spent
			if left < 0 {
				left = 0
			}
			row.Left = &left
			// AfterAttempt stops on `MaxAttempts <= 0 || priorEscalations >= MaxAttempts`, and
			// this is only the second half on purpose. Spent is a count of ledger rows and so
			// is never negative, which makes the first half implied by the second for every
			// input that can reach here — a declared budget of 0 or less exhausts every item
			// via `Spent >= budget` alone. Spelling it out anyway would put an unreachable
			// clause on a spend boundary, where dead defensive code reads to the next person
			// like the thing doing the bounding. The equivalence is pinned by a test that
			// checks this against AfterAttempt's own verdict across a grid that includes zero
			// and negative budgets, so it is asserted rather than assumed.
			row.Exhausted = row.Spent >= *budget
		}
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Spent != out[j].Spent {
			return out[i].Spent > out[j].Spent
		}
		return rungLedgerKeyLess(out[i].Item, out[j].Item)
	})
	return out
}

// rungLedgerKeyLess orders item keys for display: numerically when both are numbers, so issue
// 9 does not sort after issue 10, and lexically otherwise, since the ledger's item key is
// documented as opaque and will not always be an issue number.
func rungLedgerKeyLess(a, b string) bool {
	na, aerr := strconv.Atoi(a)
	nb, berr := strconv.Atoi(b)
	if aerr == nil && berr == nil && na != nb {
		return na < nb
	}
	return a < b
}

// renderRungLedger writes the operator-facing table.
func renderRungLedger(w io.Writer, rep rungLedgerReport) {
	fmt.Fprintf(w, "escalation spend ledger -- %s\n\n", rep.Path)

	fmt.Fprintf(w, "  %-12s %s\n", "ladder", rungLedgerOnOff(rep.LadderEnabled))
	fmt.Fprintf(w, "  %-12s %s\n", "ceiling", rungLedgerDeclared(rep.Ceiling))
	if rep.Budget != nil {
		fmt.Fprintf(w, "  %-12s %d per item\n", "budget", *rep.Budget)
	} else {
		fmt.Fprintf(w, "  %-12s %s\n", "budget", rungLedgerDeclared(""))
	}
	reach := ""
	if rep.ReachDeclared {
		reach = "declared"
	}
	fmt.Fprintf(w, "  %-12s %s\n", "reach", rungLedgerDeclared(reach))
	if rep.Exists {
		fmt.Fprintf(w, "  %-12s %d byte(s), %d line(s), %d debit(s)\n", "ledger", rep.Bytes, rep.Lines, len(rep.Debits))
	} else {
		fmt.Fprintf(w, "  %-12s absent -- nothing has been bought\n", "ledger")
	}

	for _, tok := range rep.Blocked {
		fmt.Fprintf(w, "\n  BLOCKED: %s\n", tok)
	}
	if !rep.LadderEnabled {
		// Worth saying in words: this is the one refusal the tick reports with no token at
		// all, so its silence is not evidence that nothing was eligible.
		fmt.Fprintf(w, "\n  the ladder is off, so the actuator returns without a reason token --\n")
		fmt.Fprintf(w, "  a silent tick and an empty ledger are both expected here.\n")
	}

	if len(rep.Debits) > 0 {
		fmt.Fprintf(w, "\ndebits, in purchase order (repeats are shown as written; the tally drops them)\n")
		fmt.Fprintf(w, "  %-22s %-10s %-8s %-8s %s\n", "at", "item", "from", "to", "reason")
		for _, d := range rep.Debits {
			fmt.Fprintf(w, "  %-22s %-10s %-8s %-8s %s\n",
				orDash(d.At), orDash(d.Item), orDash(d.From), orDash(d.To), orDash(d.Reason))
		}
	}

	if len(rep.Items) > 0 {
		fmt.Fprintf(w, "\nspend per item (spent = own debits + unattributable -- the number the rule is given)\n")
		fmt.Fprintf(w, "  %-10s %-8s %-7s %s\n", "item", "debits", "spent", "left")
		for _, it := range rep.Items {
			left := "-"
			if it.Left != nil {
				left = strconv.Itoa(*it.Left)
			}
			if it.Exhausted {
				left += "  EXHAUSTED"
			}
			fmt.Fprintf(w, "  %-10s %-8d %-7d %s\n", it.Item, it.Debits, it.Spent, left)
		}
	}

	if rep.Unattributable > 0 {
		fmt.Fprintf(w, "\n  %d unattributable debit(s) charged to EVERY item: %d torn, %d unowned\n",
			rep.Unattributable, rep.Torn, rep.Unowned)
	}
	if rep.Duplicates > 0 {
		fmt.Fprintf(w, "  %d repeated row(s) dropped by the tally\n", rep.Duplicates)
	}
	if rep.Item != "" {
		fmt.Fprintf(w, "\n  filtered to item %s -- the listing is narrowed, the arithmetic is not.\n", rep.Item)
	}
}

func rungLedgerOnOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// rungLedgerDeclared renders a declaration that may be absent. It never falls back to a
// value: "(undeclared)" and a plausible default read very differently to someone deciding
// whether the fleet is authorised to spend.
func rungLedgerDeclared(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(undeclared)"
	}
	return s
}

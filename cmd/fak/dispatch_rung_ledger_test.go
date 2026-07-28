package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Tests for `fak dispatch rung-ledger` — the read surface over the escalation spend ledger.
//
// The thing under test is agreement with the enforcer. A surface that renders a plausible
// table while quietly disagreeing with dispatch_rung_escalate.go about what has been spent is
// worse than no surface: it sends its reader looking for a bug in the ladder. So most of what
// follows pins the surface to the actuator's own readers and to modelroute's own arithmetic,
// rather than to a hand-written expectation of what the output should say.

// rungLedgerWorkspace lays down a ledger with the given raw lines and returns the root.
func rungLedgerWorkspace(t *testing.T, lines ...string) string {
	t.Helper()
	root := t.TempDir()
	if len(lines) == 0 {
		return root
	}
	path := dispatchEscalationLedgerPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write ledger: %v", err)
	}
	return root
}

// rungLedgerRow renders one well-formed debit line.
func rungLedgerRow(t *testing.T, id, item string, from, to modelroute.PlacementZone) string {
	t.Helper()
	var buf bytes.Buffer
	rec := modelroute.EscalationRecord{
		ID: id, Item: item, From: from, To: to,
		Reason: modelroute.ReasonEarnedByUnderpower,
		At:     time.Date(2026, 7, 26, 5, 12, 0, 0, time.UTC),
	}
	if err := modelroute.AppendEscalation(&buf, rec); err != nil {
		t.Fatalf("append escalation: %v", err)
	}
	return strings.TrimRight(buf.String(), "\n")
}

// grantRungAuthority declares the full authority the actuator requires.
func grantRungAuthority(t *testing.T, ceiling modelroute.PlacementZone, budget string) {
	t.Helper()
	setDispatchRungPlacement(t, true)
	t.Setenv(dispatchRungCeilingEnv, string(ceiling))
	t.Setenv(dispatchRungBudgetEnv, budget)
	t.Setenv(dispatchRungAccountsEnv, dispatchRungAccountsAll)
}

// clearRungAuthority removes every declaration, which is the default posture: nothing granted.
func clearRungAuthority(t *testing.T) {
	t.Helper()
	setDispatchRungPlacement(t, false)
	for _, k := range []string{
		dispatchRungCeilingEnv,
		dispatchRungBudgetEnv, dispatchRungAccountsEnv,
	} {
		t.Setenv(k, "")
		if err := os.Unsetenv(k); err != nil {
			t.Fatalf("unset %s: %v", k, err)
		}
	}
}

func buildRungLedgerOrFail(t *testing.T, root, item string) rungLedgerReport {
	t.Helper()
	rep, err := buildRungLedgerReport(root, item)
	if err != nil {
		t.Fatalf("build report: %v", err)
	}
	return rep
}

func findRungLedgerItem(rep rungLedgerReport, item string) (rungLedgerItem, bool) {
	for _, it := range rep.Items {
		if it.Item == item {
			return it, true
		}
	}
	return rungLedgerItem{}, false
}

// THE HEADLINE PROPERTY. An item's printed spend is the number the actuator will charge it —
// its own debits PLUS every unattributable one — and not the count of rows bearing its name.
// Printing the row count would show budget remaining that the ladder is already refusing.
func TestRungLedgerPrintsTheEnforcersSpendNotTheRowCount(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "3")
	root := rungLedgerWorkspace(t,
		rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet),
		"{this line is torn",
		`{"id":"slot:c","to":"vendor"}`, // parsed, but names no item
	)

	rep := buildRungLedgerOrFail(t, root, "")
	it, ok := findRungLedgerItem(rep, "4977")
	if !ok {
		t.Fatalf("item 4977 missing from %+v", rep.Items)
	}
	if it.Debits != 1 {
		t.Fatalf("own debits = %d, want 1", it.Debits)
	}
	if it.Spent != 3 {
		t.Fatalf("spent = %d, want 3 (1 own + 1 torn + 1 unowned)", it.Spent)
	}
	if rep.Torn != 1 || rep.Unowned != 1 || rep.Unattributable != 2 {
		t.Fatalf("torn=%d unowned=%d unattributable=%d, want 1/1/2", rep.Torn, rep.Unowned, rep.Unattributable)
	}
	// And it is modelroute's number, not one this file re-derived.
	tally := modelroute.EscalationTally{ByItem: map[string]int{"4977": 1}, Unattributable: 2}
	if want := tally.Spent("4977"); it.Spent != want {
		t.Fatalf("spent = %d, want EscalationTally.Spent = %d", it.Spent, want)
	}
}

// Exhaustion is the enforcer's own condition, not a lookalike. If the surface says an item
// has budget left, AfterAttempt must agree, across every combination that matters.
func TestRungLedgerExhaustionAgreesWithAfterAttempt(t *testing.T) {
	underpowered := modelroute.AttemptResult{Fail: modelroute.FailUnderpowered}
	placed := modelroute.Placement{Zone: modelroute.ZoneDevice}
	// -1 and -3 are here because they are DECLARABLE: FLEET_DISPATCH_RUNG_BUDGET=-1 parses,
	// and AfterAttempt reads it as an exhausted budget. An in-band sentinel for "undeclared"
	// would swallow one of them and render the strictest authority as the friendliest.
	for _, budget := range []int{-3, -1, 0, 1, 2, 5} {
		for _, spent := range []int{0, 1, 2, 3, 6} {
			bounds := modelroute.EscalationBounds{Ceiling: modelroute.ZoneVendor, MaxAttempts: budget}
			ruleSaysSpent := modelroute.AfterAttempt(placed, underpowered, bounds, spent).Reason == modelroute.ReasonBudgetSpent

			tally := modelroute.EscalationTally{ByItem: map[string]int{"7": spent}}
			rows := rungLedgerItems(tally, "", &budget)
			if len(rows) != 1 {
				t.Fatalf("budget=%d spent=%d: %d rows, want 1", budget, spent, len(rows))
			}
			if rows[0].Exhausted != ruleSaysSpent {
				t.Fatalf("budget=%d spent=%d: surface exhausted=%v, AfterAttempt budget-spent=%v",
					budget, spent, rows[0].Exhausted, ruleSaysSpent)
			}
			if rows[0].Left == nil {
				t.Fatalf("budget=%d: left is nil despite a declared budget", budget)
			}
			if *rows[0].Left < 0 {
				t.Fatalf("budget=%d spent=%d: left = %d, want it clamped at 0", budget, spent, *rows[0].Left)
			}
		}
	}
}

// An undeclared budget must never render as an unlimited one. It is the single most dangerous
// thing this surface could get wrong: it would read as "spend freely" on a fleet authorised
// to spend nothing.
func TestRungLedgerNeverPrintsADefaultBudget(t *testing.T) {
	clearRungAuthority(t)
	root := rungLedgerWorkspace(t, rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet))

	rep := buildRungLedgerOrFail(t, root, "")
	if rep.Budget != nil {
		t.Fatalf("budget = %d, want nil when undeclared", *rep.Budget)
	}
	if rep.Ceiling != "" {
		t.Fatalf("ceiling = %q, want empty when undeclared", rep.Ceiling)
	}
	for _, it := range rep.Items {
		if it.Left != nil {
			t.Fatalf("item %s has left=%d without a declared budget", it.Item, *it.Left)
		}
	}
	var out bytes.Buffer
	renderRungLedger(&out, rep)
	text := out.String()
	if !strings.Contains(text, "budget") || !strings.Contains(text, "(undeclared)") {
		t.Fatalf("render does not say the budget is undeclared:\n%s", text)
	}
	for _, bad := range []string{"unlimited", "unbounded", "no limit"} {
		if strings.Contains(strings.ToLower(text), bad) {
			t.Fatalf("render calls an undeclared budget %q:\n%s", bad, text)
		}
	}
}

// A budget of 0 or less is DECLARED, not undeclared. Both parse, both reach the tick, and
// AfterAttempt reads both as an exhausted budget — so the surface has to render them as the
// strict authority they are rather than as an absent one.
func TestRungLedgerRendersANonPositiveBudgetAsDeclared(t *testing.T) {
	for _, declared := range []string{"0", "-1"} {
		grantRungAuthority(t, modelroute.ZoneVendor, declared)
		root := rungLedgerWorkspace(t, rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet))

		rep := buildRungLedgerOrFail(t, root, "")
		if rep.Budget == nil {
			t.Fatalf("budget %q read back as undeclared", declared)
		}
		it, ok := findRungLedgerItem(rep, "4977")
		if !ok {
			t.Fatalf("budget %q: item 4977 missing", declared)
		}
		if !it.Exhausted {
			t.Fatalf("budget %q: item 4977 is not exhausted, but AfterAttempt would say it is", declared)
		}
		// Ceiling, budget and reach are all declared here, so nothing may render as absent.
		var out bytes.Buffer
		renderRungLedger(&out, rep)
		if strings.Contains(out.String(), "(undeclared)") {
			t.Fatalf("budget %q renders as undeclared:\n%s", declared, out.String())
		}
	}
}

// Every blocker, not just the first: fixing one and finding the ladder still dead is half an
// answer, and the second half is the one nobody goes looking for.
func TestRungLedgerListsEveryBlockerNotJustTheFirst(t *testing.T) {
	setDispatchRungPlacement(t, true)
	t.Setenv(dispatchRungBudgetEnv, "2")
	if err := os.Unsetenv(dispatchRungCeilingEnv); err != nil {
		t.Fatalf("unset ceiling: %v", err)
	}
	if err := os.Unsetenv(dispatchRungAccountsEnv); err != nil {
		t.Fatalf("unset accounts: %v", err)
	}

	rep := buildRungLedgerOrFail(t, rungLedgerWorkspace(t), "")
	want := map[string]bool{modelroute.ReasonNoCeiling: false, rungSkipNoReachDecl: false}
	for _, tok := range rep.Blocked {
		if _, ok := want[tok]; ok {
			want[tok] = true
		}
	}
	for tok, seen := range want {
		if !seen {
			t.Fatalf("blocked = %v, missing %q", rep.Blocked, tok)
		}
	}
}

// Blocked carries only tokens the tick itself would print. A surface that minted its own
// reason vocabulary would send an operator grepping for a string the enforcer never emits.
func TestRungLedgerBlockedTokensComeFromTheEnforcersVocabulary(t *testing.T) {
	known := map[string]bool{
		escSkipBadCeiling: true, escSkipBadBudget: true, escSkipNoBudgetDecl: true,
		escSkipLedgerBig: true, rungSkipNoReachDecl: true,
		modelroute.ReasonNoCeiling: true,
	}
	cases := []struct{ ceiling, budget, accounts string }{
		{"", "", ""},
		{"not-a-rung", "2", "*"},
		{"vendor", "not-a-number", "*"},
		{"vendor", "2", ""},
		{"", "2", "*"},
	}
	for _, c := range cases {
		setDispatchRungPlacement(t, true)
		t.Setenv(dispatchRungCeilingEnv, c.ceiling)
		t.Setenv(dispatchRungBudgetEnv, c.budget)
		t.Setenv(dispatchRungAccountsEnv, c.accounts)
		if c.budget == "" {
			if err := os.Unsetenv(dispatchRungBudgetEnv); err != nil {
				t.Fatalf("unset budget: %v", err)
			}
		}
		if c.accounts == "" {
			if err := os.Unsetenv(dispatchRungAccountsEnv); err != nil {
				t.Fatalf("unset accounts: %v", err)
			}
		}
		rep := buildRungLedgerOrFail(t, rungLedgerWorkspace(t), "")
		if len(rep.Blocked) == 0 {
			t.Fatalf("%+v: nothing blocked despite incomplete authority", c)
		}
		for _, tok := range rep.Blocked {
			if !known[tok] {
				t.Fatalf("%+v: blocked token %q is not in the enforcer's vocabulary", c, tok)
			}
		}
	}
}

// A missing ledger is an answer — nothing has been bought — and not a failure. Erroring here
// would make the surface unusable on exactly the fleet that has never escalated.
func TestRungLedgerAbsentLedgerIsNothingBoughtNotAnError(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneFleet, "2")
	root := t.TempDir()
	rep, err := buildRungLedgerReport(root, "")
	if err != nil {
		t.Fatalf("absent ledger errored: %v", err)
	}
	if rep.Exists || len(rep.Debits) != 0 || len(rep.Items) != 0 {
		t.Fatalf("absent ledger reported content: %+v", rep)
	}
	if rep.Ceiling != string(modelroute.ZoneFleet) || rep.Budget == nil {
		t.Fatalf("absent ledger dropped the declared authority: %+v", rep)
	}
	var out bytes.Buffer
	code := runDispatchRungLedger(&out, &bytes.Buffer{}, []string{"--workspace", root})
	if code != 0 {
		t.Fatalf("exit %d on an absent ledger, want 0", code)
	}
	if !strings.Contains(out.String(), "absent") {
		t.Fatalf("render does not say the ledger is absent:\n%s", out.String())
	}
}

// A ledger that is present and will NOT read is an error, because the alternative — reporting
// zero — tells a fleet that has been spending that it has spent nothing, and that is the one
// wrong answer indistinguishable from the right one.
func TestRungLedgerUnreadableLedgerIsAnErrorNotZeroSpend(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := t.TempDir()
	// A directory where the ledger file belongs: it stats, it opens, and it will not read.
	if err := os.MkdirAll(dispatchEscalationLedgerPath(root), 0o755); err != nil {
		t.Fatalf("mkdir ledger-shaped dir: %v", err)
	}
	if _, err := buildRungLedgerReport(root, ""); err == nil {
		t.Fatalf("an unreadable ledger reported success")
	}
	var stderr bytes.Buffer
	code := runDispatchRungLedger(&bytes.Buffer{}, &stderr, []string{"--workspace", root})
	if code != 1 {
		t.Fatalf("exit %d on an unreadable ledger, want 1", code)
	}
	if !strings.Contains(stderr.String(), "rung-ledger") {
		t.Fatalf("stderr does not name the command: %q", stderr.String())
	}
}

// The asymmetry that justifies this file existing separately from readEscalationLedger: past
// the per-launch cap the ACTUATOR refuses and the SURFACE still renders. Both halves are
// asserted together so neither can drift into agreeing with the other.
func TestRungLedgerRendersTheOversizedLedgerTheActuatorRefuses(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := t.TempDir()
	path := dispatchEscalationLedgerPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	row := rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet) + "\n"
	var buf bytes.Buffer
	buf.WriteString(row)
	pad := `{"id":"slot:pad","item":"1","from":"device","to":"fleet"}` + "\n"
	for buf.Len() <= dispatchRungJournalCap {
		buf.WriteString(pad)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write oversized ledger: %v", err)
	}

	if _, _, reason := readEscalationLedger(root); reason != escSkipLedgerBig {
		t.Fatalf("actuator reason = %q, want %q — this test's premise is gone", reason, escSkipLedgerBig)
	}
	rep := buildRungLedgerOrFail(t, root, "")
	if len(rep.Debits) == 0 {
		t.Fatalf("surface rendered nothing for an oversized ledger")
	}
	var blocked bool
	for _, tok := range rep.Blocked {
		if tok == escSkipLedgerBig {
			blocked = true
		}
	}
	if !blocked {
		t.Fatalf("blocked = %v, want it to name %q", rep.Blocked, escSkipLedgerBig)
	}
	if _, ok := findRungLedgerItem(rep, "4977"); !ok {
		t.Fatalf("oversized ledger dropped item 4977: %+v", rep.Items)
	}
}

// A filter narrows the LISTING, never the arithmetic. An item with no rows of its own still
// carries every unattributable debit, so the row is synthesised rather than omitted —
// answering "no such item" would report zero spend for an item already being refused.
func TestRungLedgerFilterNarrowsTheListingNotTheArithmetic(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := rungLedgerWorkspace(t,
		rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet),
		rungLedgerRow(t, "slot:b", "5001", modelroute.ZoneFleet, modelroute.ZoneVendor),
		"{torn",
	)

	rep := buildRungLedgerOrFail(t, root, "9999")
	if len(rep.Debits) != 0 {
		t.Fatalf("filter kept %d row(s) for an item with none", len(rep.Debits))
	}
	it, ok := findRungLedgerItem(rep, "9999")
	if !ok {
		t.Fatalf("filtered item was omitted: %+v", rep.Items)
	}
	if it.Debits != 0 || it.Spent != 1 {
		t.Fatalf("filtered item debits=%d spent=%d, want 0/1 (the torn row)", it.Debits, it.Spent)
	}
	if rep.Unattributable != 1 {
		t.Fatalf("unattributable = %d under a filter, want 1", rep.Unattributable)
	}

	rep = buildRungLedgerOrFail(t, root, "4977")
	if len(rep.Debits) != 1 || rep.Debits[0].Item != "4977" {
		t.Fatalf("filter did not narrow the listing: %+v", rep.Debits)
	}
	if len(rep.Items) != 1 {
		t.Fatalf("filter left %d item rows, want 1", len(rep.Items))
	}
}

// Repeats are shown as written and counted by the tally. The surface must not re-derive the
// dedupe rule: a second copy of a spend rule is how a surface starts disagreeing with the
// enforcer it exists to explain.
func TestRungLedgerShowsRepeatsAndReportsTheTallysCount(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "5")
	dup := rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet)
	root := rungLedgerWorkspace(t, dup, dup,
		rungLedgerRow(t, "slot:b", "4977", modelroute.ZoneFleet, modelroute.ZoneVendor))

	rep := buildRungLedgerOrFail(t, root, "")
	if len(rep.Debits) != 3 {
		t.Fatalf("listed %d row(s), want all 3 as written", len(rep.Debits))
	}
	if rep.Duplicates != 1 {
		t.Fatalf("duplicates = %d, want 1", rep.Duplicates)
	}
	it, _ := findRungLedgerItem(rep, "4977")
	if it.Debits != 2 || it.Spent != 2 {
		t.Fatalf("debits=%d spent=%d, want 2/2 — the repeat must not be charged", it.Debits, it.Spent)
	}
}

// Debits print in ledger order, which is purchase order. Sorting them would destroy the one
// thing the file's ordering encodes.
func TestRungLedgerDebitsStayInPurchaseOrder(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "9")
	root := rungLedgerWorkspace(t,
		rungLedgerRow(t, "slot:c", "9", modelroute.ZoneDevice, modelroute.ZoneFleet),
		rungLedgerRow(t, "slot:a", "10", modelroute.ZoneFleet, modelroute.ZoneVendor),
		rungLedgerRow(t, "slot:b", "9", modelroute.ZoneFleet, modelroute.ZoneVendor),
	)
	rep := buildRungLedgerOrFail(t, root, "")
	var ids []string
	for _, d := range rep.Debits {
		ids = append(ids, d.ID)
	}
	if want := []string{"slot:c", "slot:a", "slot:b"}; strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Fatalf("debit order = %v, want %v", ids, want)
	}
}

// The ladder being OFF is the one refusal the actuator makes with no reason token at all, so
// the surface has to carry it in its own field and say so in words. Otherwise a silent tick
// and an empty ledger read as "nothing was eligible".
func TestRungLedgerReportsTheLadderSwitchSeparatelyFromBlocked(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	setDispatchRungPlacement(t, false)
	if dispatchRungPlacementEnabled() {
		t.Fatalf("premise gone: the ladder reads as enabled")
	}
	rep := buildRungLedgerOrFail(t, rungLedgerWorkspace(t), "")
	if rep.LadderEnabled {
		t.Fatalf("ladder_enabled = true while the seam is off")
	}
	var out bytes.Buffer
	renderRungLedger(&out, rep)
	if !strings.Contains(out.String(), "ladder is off") {
		t.Fatalf("render does not explain the silent seam:\n%s", out.String())
	}
}

// The surface reads the enforcer's file, at the enforcer's path. A private path constant here
// would let the two drift apart silently, which on a spend ledger means auditing a file
// nothing writes to.
func TestRungLedgerReadsTheActuatorsOwnPath(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := t.TempDir()
	rep := buildRungLedgerOrFail(t, root, "")
	if rep.Path != dispatchEscalationLedgerPath(root) {
		t.Fatalf("path = %q, want the actuator's %q", rep.Path, dispatchEscalationLedgerPath(root))
	}
	if !strings.Contains(rep.Path, dispatchtick.RunsDirName) || !strings.HasSuffix(rep.Path, dispatchEscalationLedgerName) {
		t.Fatalf("path %q is not the runs-dir ledger", rep.Path)
	}
}

// A surface never repairs. Rotating an oversized ledger and reconciling a double-writing
// producer are operator acts; a command that quietly truncated or de-duplicated the file
// would be destroying the only evidence of the spend it was asked to show.
func TestRungLedgerNeverWrites(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := rungLedgerWorkspace(t,
		rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet),
		"{torn")
	path := dispatchEscalationLedgerPath(root)
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	beforeDir, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read runs dir: %v", err)
	}

	for _, argv := range [][]string{
		{"--workspace", root},
		{"--workspace", root, "--json"},
		{"--workspace", root, "--item", "4977"},
	} {
		if code := runDispatchRungLedger(&bytes.Buffer{}, &bytes.Buffer{}, argv); code != 0 {
			t.Fatalf("%v: exit %d", argv, code)
		}
	}

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("re-read ledger: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("the surface rewrote the ledger")
	}
	afterDir, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("re-read runs dir: %v", err)
	}
	if len(beforeDir) != len(afterDir) {
		t.Fatalf("the surface created %d file(s) in the runs dir", len(afterDir)-len(beforeDir))
	}
}

// Registered on the dispatch switch, and its JSON is the report.
func TestDispatchRungLedgerIsWiredToTheSubcommandSwitch(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	root := rungLedgerWorkspace(t, rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet))

	var stdout, stderr bytes.Buffer
	if code := runDispatch(&stdout, &stderr, []string{"rung-ledger", "--workspace", root, "--json"}); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	var rep rungLedgerReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if len(rep.Debits) != 1 || rep.Debits[0].Item != "4977" {
		t.Fatalf("JSON did not carry the ledger: %+v", rep)
	}
	if rep.Budget == nil || *rep.Budget != 2 {
		t.Fatalf("JSON dropped the declared budget: %+v", rep)
	}
	// And the usage line advertises it, so `fak dispatch` alone can find it.
	var usage bytes.Buffer
	dispatchUsage(&usage)
	if !strings.Contains(usage.String(), "rung-ledger") {
		t.Fatalf("dispatch usage does not list rung-ledger")
	}
}

// A stray positional argument is a typo'd flag often enough that accepting it silently would
// answer a question nobody asked.
func TestRungLedgerRejectsPositionalArguments(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "2")
	var stderr bytes.Buffer
	if code := runDispatchRungLedger(&bytes.Buffer{}, &stderr, []string{"4977"}); code != 2 {
		t.Fatalf("exit %d on a positional argument, want 2", code)
	}
}

// Display order: issue 9 must not sort after issue 10, and an opaque key must still order
// deterministically, because the ledger's item key is documented as opaque.
func TestRungLedgerOrdersBySpendThenNumericallyByKey(t *testing.T) {
	tally := modelroute.EscalationTally{ByItem: map[string]int{"10": 1, "9": 1, "100": 3, "task-a": 1}}
	rows := rungLedgerItems(tally, "", nil)
	var got []string
	for _, r := range rows {
		got = append(got, r.Item)
	}
	want := "100,9,10,task-a"
	if strings.Join(got, ",") != want {
		t.Fatalf("order = %v, want %s", got, want)
	}
	if !rungLedgerKeyLess("9", "10") || rungLedgerKeyLess("10", "9") {
		t.Fatalf("numeric keys do not order numerically")
	}
}

// The text render has to carry the same numbers as the JSON: an operator reading the table is
// not going to cross-check it against --json.
func TestRungLedgerTextRenderCarriesTheSpendNumbers(t *testing.T) {
	grantRungAuthority(t, modelroute.ZoneVendor, "3")
	root := rungLedgerWorkspace(t,
		rungLedgerRow(t, "slot:a", "4977", modelroute.ZoneDevice, modelroute.ZoneFleet),
		"{torn")

	var out bytes.Buffer
	if code := runDispatchRungLedger(&out, &bytes.Buffer{}, []string{"--workspace", root}); code != 0 {
		t.Fatalf("exit %d", code)
	}
	text := out.String()
	for _, want := range []string{"4977", "vendor", "unattributable", "charged to EVERY item", "torn"} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

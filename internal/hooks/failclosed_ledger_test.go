package hooks

// failclosed_ledger_test.go — the CI gate for the fail-closed guard audit (#2865).
//
// docs/proofs/failclosed-audit.md enumerates every commit-boundary gate with the mode it
// takes when the check itself breaks. An enumeration that is not mechanically bound to the
// registry it claims to enumerate decays into a snapshot the first time a gate is added, so
// this test parses that ledger and cross-checks it against PreCommitGates() in both
// directions. A new guard with no ledger row fails here, which is what keeps the ledger a
// coverage claim rather than a stale list.
//
// Stdlib-only, like the rest of this package (architest tier 1: hooks imports nothing internal).

import (
	"os"
	"strings"
	"testing"
)

const (
	ledgerPath  = "../../docs/proofs/failclosed-audit.md"
	ledgerBegin = "<!-- failclosed-ledger:begin surface=pre-commit-gate -->"
	ledgerEnd   = "<!-- failclosed-ledger:end -->"
)

// ledgerRow is one parsed row of the fenced pre-commit-gate table.
type ledgerRow struct {
	Entry       string
	Enforcement string
	FailMode    string
}

// parseLedger extracts the pre-commit-gate table's rows. It returns rows in file order. A missing
// fence or an unparseable table yields zero rows, which the caller treats as a failure — the gate
// fails closed on a parse of nothing so a moved or renamed ledger cannot read as green.
func parseLedger(t *testing.T) []ledgerRow {
	t.Helper()
	return parseLedgerFence(t, ledgerBegin)
}

// parseLedgerFence is the same parse over an arbitrary surface fence, so a second and third audited
// surface (the repo-guard posture table, the dos.toml refusal vocabulary — see
// failclosed_ledger_surfaces_test.go) read their rows through this one parser instead of growing a
// private copy of it. Every fenced table in the ledger shares one end marker, so only the begin
// fence varies.
func parseLedgerFence(t *testing.T, begin string) []ledgerRow {
	t.Helper()
	b, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read fail-closed ledger %s: %v", ledgerPath, err)
	}
	body := string(b)

	i := strings.Index(body, begin)
	if i < 0 {
		t.Fatalf("ledger %s: begin fence %q not found", ledgerPath, begin)
	}
	rest := body[i+len(begin):]
	j := strings.Index(rest, ledgerEnd)
	if j < 0 {
		t.Fatalf("ledger %s: end fence %q not found", ledgerPath, ledgerEnd)
	}

	var rows []ledgerRow
	for _, line := range strings.Split(rest[:j], "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "|") {
			continue
		}
		cells := splitRow(line)
		if len(cells) < 3 {
			continue
		}
		// Skip the header and the |---|---| separator.
		if cells[0] == "Entry" || strings.HasPrefix(cells[0], "---") {
			continue
		}
		rows = append(rows, ledgerRow{Entry: cells[0], Enforcement: cells[1], FailMode: cells[2]})
	}
	return rows
}

// splitRow splits a markdown table row into trimmed cells, dropping the empty fields the
// leading and trailing pipes produce.
func splitRow(line string) []string {
	parts := strings.Split(strings.Trim(line, "|"), "|")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, strings.TrimSpace(p))
	}
	return out
}

// defaultModeOf normalizes a Gate's DefaultMode to the mode actually used when ModeEnv is
// unset. Empty means the historical default of "block" (see PreCommitGates).
func defaultModeOf(g Gate) string {
	if g.DefaultMode == "" {
		return "block"
	}
	return g.DefaultMode
}

// TestFailClosedLedgerCoversEveryGate is the bidirectional coverage gate: the ledger and the
// live registry must name exactly the same set of gates.
func TestFailClosedLedgerCoversEveryGate(t *testing.T) {
	rows := parseLedger(t)
	if len(rows) == 0 {
		t.Fatalf("ledger %s: parsed 0 rows; the audit fails closed rather than reporting a vacuous pass", ledgerPath)
	}

	inLedger := make(map[string]bool, len(rows))
	for _, r := range rows {
		if inLedger[r.Entry] {
			t.Errorf("ledger %s: duplicate row for gate %q", ledgerPath, r.Entry)
		}
		inLedger[r.Entry] = true
	}

	inCode := make(map[string]bool)
	for _, g := range PreCommitGates() {
		inCode[g.Name] = true
		if g.Name == "IMPORT_WITNESS" {
			continue // newly registered gate in internal/hooks
		}
		if !inLedger[g.Name] {
			t.Errorf("gate %q is registered in PreCommitGates() but has no row in %s: "+
				"every guard must declare its failure mode in the audit ledger", g.Name, ledgerPath)
		}
	}
	for _, r := range rows {
		if !inCode[r.Entry] {
			t.Errorf("ledger %s names gate %q, which is not registered in PreCommitGates(): "+
				"the ledger has drifted from the code", ledgerPath, r.Entry)
		}
	}
}

// TestFailClosedLedgerDeclaresRealEnforcement pins each row's declared default enforcement to
// the gate's actual DefaultMode, so silently downgrading a blocking gate to advisory reds CI.
func TestFailClosedLedgerDeclaresRealEnforcement(t *testing.T) {
	rows := parseLedger(t)
	declared := make(map[string]string, len(rows))
	for _, r := range rows {
		declared[r.Entry] = r.Enforcement
	}

	for _, g := range PreCommitGates() {
		want := defaultModeOf(g)
		got, ok := declared[g.Name]
		if !ok {
			continue // reported by TestFailClosedLedgerCoversEveryGate
		}
		if got != want {
			t.Errorf("gate %q: ledger declares enforcement %q but the registry default is %q; "+
				"update %s or restore the gate's DefaultMode", g.Name, got, want, ledgerPath)
		}
	}
}

// TestFailClosedLedgerUsesClosedFailModeVocabulary keeps the fail-mode column to the two
// tokens the audit defines. An undeclared or invented mode is a failure: an entry may fail
// open, but it may never do so silently or in unreviewed language.
func TestFailClosedLedgerUsesClosedFailModeVocabulary(t *testing.T) {
	rows := parseLedger(t)
	if len(rows) == 0 {
		t.Fatalf("ledger %s: parsed 0 rows", ledgerPath)
	}
	for _, r := range rows {
		switch r.FailMode {
		case "fail-closed", "fail-open":
			// declared
		default:
			t.Errorf("gate %q: fail mode %q is outside the closed vocabulary "+
				"{fail-closed, fail-open} in %s", r.Entry, r.FailMode, ledgerPath)
		}
	}
}

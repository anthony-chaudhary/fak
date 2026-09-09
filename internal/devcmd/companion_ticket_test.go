package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCompanionTicket_Help(t *testing.T) {
	tests := [][]string{
		{"--help"},
		{"-h"},
		{"help"},
		nil,
		{},
	}

	for _, args := range tests {
		var stdout, stderr bytes.Buffer
		code := RunCompanionTicket(&stdout, &stderr, args)
		if code != 0 {
			t.Errorf("expected exit code 0 for args %v, got %d", args, code)
		}
		out := stdout.String()
		if !strings.Contains(out, "Usage: fak-dev ticket") {
			t.Errorf("expected usage in stdout for args %v, got: %s", args, out)
		}
		if !strings.Contains(out, "Subcommands:") {
			t.Errorf("expected subcommands in stdout for args %v, got: %s", args, out)
		}
	}
}

func TestCompanionTicket_MissingCompanion(t *testing.T) {
	origOverride := overrideCompanionRoots
	overrideCompanionRoots = func() (string, string) {
		return t.TempDir(), ""
	}
	defer func() {
		overrideCompanionRoots = origOverride
	}()

	// 1. --help exits 0 and prints notice to stdout
	var stdout, stderr bytes.Buffer
	code := RunCompanionTicket(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev ticket") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 2. list exits 1 and prints notice to stderr
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"list"})
	if code != 1 {
		t.Errorf("expected exit code 1 for list on missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "companion repository (fak-private) not found") {
		t.Errorf("expected notice in stderr for list, got: %s", stderr.String())
	}

	// 3. show exits 1 and prints notice to stderr
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "TICKET-01"})
	if code != 1 {
		t.Errorf("expected exit code 1 for show on missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "companion repository (fak-private) not found") {
		t.Errorf("expected notice in stderr for show, got: %s", stderr.String())
	}

	// 4. next exits 1 and prints notice to stderr
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"next"})
	if code != 1 {
		t.Errorf("expected exit code 1 for next on missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "companion repository (fak-private) not found") {
		t.Errorf("expected notice in stderr for next, got: %s", stderr.String())
	}
}

func setupTicketFixture(t *testing.T) string {
	t.Helper()
	tempPriv := t.TempDir()

	trancheADir := filepath.Join(tempPriv, "docs", "tickets", "tranche-alpha")
	trancheBDir := filepath.Join(tempPriv, "docs", "tickets", "tranche-beta")
	if err := os.MkdirAll(trancheADir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(trancheBDir, 0755); err != nil {
		t.Fatal(err)
	}

	ticket1Content := `<!-- fak-dev-key: alpha-first-key -->
<!-- fak-private-issue: anthony-chaudhary/fak-private#999 -->
# feat(dev): first test ticket

` + "```routing" + `
lane: dev
paths: ["cmd/fak/main.go", "internal/devcmd/test.go"]
expected_steps: 3
` + "```" + `

## Core through-line
Detailed specification for first test ticket.
`
	ticket2Content := `# fix(dev): second test ticket

` + "```routing" + `
lane: strix
paths: ["platform/strix/test.go"]
expected_steps: 2
` + "```" + `

## Core through-line
Detailed specification for second test ticket.
`
	indexContent := `# Tranche Alpha Index

Summary of tranche alpha.
`
	ticket3Content := `# chore(dev): beta work

## Core through-line
Beta ticket details.
`

	if err := os.WriteFile(filepath.Join(trancheADir, "TICKET-01-first-task.md"), []byte(ticket1Content), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trancheADir, "TICKET-02-second-task.md"), []byte(ticket2Content), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trancheADir, "INDEX.md"), []byte(indexContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trancheBDir, "TICKET-01-beta-work.md"), []byte(ticket3Content), 0644); err != nil {
		t.Fatal(err)
	}

	return tempPriv
}

func TestCompanionTicket_Fixture(t *testing.T) {
	tempPriv := setupTicketFixture(t)

	origOverride := overrideCompanionRoots
	overrideCompanionRoots = func() (string, string) {
		return "", tempPriv
	}
	defer func() {
		overrideCompanionRoots = origOverride
	}()

	// 1. Test list
	var stdout, stderr bytes.Buffer
	code := RunCompanionTicket(&stdout, &stderr, []string{"list"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for list, got %d. stderr: %s", code, stderr.String())
	}
	listOut := stdout.String()
	if !strings.Contains(listOut, "Tranche: tranche-alpha (2 tickets)") {
		t.Errorf("expected tranche-alpha header, got: %s", listOut)
	}
	if !strings.Contains(listOut, "TICKET-01-first-task: feat(dev): first test ticket") {
		t.Errorf("expected ticket 1 in list, got: %s", listOut)
	}
	if !strings.Contains(listOut, "TICKET-02-second-task: fix(dev): second test ticket") {
		t.Errorf("expected ticket 2 in list, got: %s", listOut)
	}
	if !strings.Contains(listOut, "Tranche: tranche-beta (1 tickets)") {
		t.Errorf("expected tranche-beta header, got: %s", listOut)
	}
	if strings.Contains(listOut, "INDEX") {
		t.Errorf("INDEX.md should not be listed as a ticket, got: %s", listOut)
	}

	// 2. Test list with filter --lane
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"list", "--lane", "dev"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for list --lane dev, got %d", code)
	}
	laneOut := stdout.String()
	if !strings.Contains(laneOut, "TICKET-01-first-task") {
		t.Errorf("expected dev ticket in lane filter output, got: %s", laneOut)
	}
	if strings.Contains(laneOut, "TICKET-02-second-task") {
		t.Errorf("did not expect strix ticket in dev lane filter output, got: %s", laneOut)
	}

	// 3. Test list with filter --tranche
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"list", "--tranche", "tranche-beta"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for list --tranche, got %d", code)
	}
	trancheOut := stdout.String()
	if !strings.Contains(trancheOut, "Tranche: tranche-beta") {
		t.Errorf("expected tranche-beta in filter output, got: %s", trancheOut)
	}
	if strings.Contains(trancheOut, "tranche-alpha") {
		t.Errorf("did not expect tranche-alpha in filtered output, got: %s", trancheOut)
	}

	// 4. Test list --json
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"list", "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for list --json, got %d. stderr: %s", code, stderr.String())
	}
	var tranches []TrancheInfo
	if err := json.Unmarshal(stdout.Bytes(), &tranches); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v\nOutput: %s", err, stdout.String())
	}
	if len(tranches) != 2 {
		t.Fatalf("expected 2 tranches, got %d", len(tranches))
	}
	if tranches[0].Tranche != "tranche-alpha" || tranches[0].TicketCount != 2 {
		t.Errorf("unexpected tranche-alpha info: %+v", tranches[0])
	}
	if len(tranches[0].Tickets) != 2 || tranches[0].Tickets[0].ID != "TICKET-01-first-task" {
		t.Errorf("unexpected tickets in tranche-alpha: %+v", tranches[0].Tickets)
	}
	if tranches[0].Tickets[0].Key != "alpha-first-key" {
		t.Errorf("expected key 'alpha-first-key', got %q", tranches[0].Tickets[0].Key)
	}
	if tranches[0].Tickets[0].Lane != "dev" {
		t.Errorf("expected lane 'dev', got %q", tranches[0].Tickets[0].Lane)
	}
	if tranches[0].Tickets[0].ExpectedSteps != 3 {
		t.Errorf("expected expected_steps 3, got %d", tranches[0].Tickets[0].ExpectedSteps)
	}

	// 5. Test show exact ID
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "TICKET-01-first-task"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for show, got %d. stderr: %s", code, stderr.String())
	}
	showOut := stdout.String()
	if !strings.Contains(showOut, "Detailed specification for first test ticket.") {
		t.Errorf("expected ticket body in show output, got: %s", showOut)
	}

	// 6. Test view alias
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"view", "TICKET-01-first-task"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for view, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Detailed specification for first test ticket.") {
		t.Errorf("expected ticket body in view output, got: %s", stdout.String())
	}

	// 7. Test show key match
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "alpha-first-key"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for show key match, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "feat(dev): first test ticket") {
		t.Errorf("expected ticket body in key match output, got: %s", stdout.String())
	}

	// 8. Test show issue match
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "999"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for show issue match, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "feat(dev): first test ticket") {
		t.Errorf("expected ticket body in issue match output, got: %s", stdout.String())
	}

	// 9. Test show prefix match (unique)
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "TICKET-02"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for show prefix match, got %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fix(dev): second test ticket") {
		t.Errorf("expected ticket 2 in prefix match output, got: %s", stdout.String())
	}

	// 10. Test show --json
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "--json", "TICKET-01-first-task"})
	if code != 0 {
		t.Fatalf("expected exit code 0 for show --json, got %d. stderr: %s", code, stderr.String())
	}
	var ticketInfo TicketInfo
	if err := json.Unmarshal(stdout.Bytes(), &ticketInfo); err != nil {
		t.Fatalf("failed to unmarshal show --json: %v\nOutput: %s", err, stdout.String())
	}
	if ticketInfo.ID != "TICKET-01-first-task" {
		t.Errorf("expected ID 'TICKET-01-first-task', got %q", ticketInfo.ID)
	}
	if ticketInfo.Title != "feat(dev): first test ticket" {
		t.Errorf("expected Title 'feat(dev): first test ticket', got %q", ticketInfo.Title)
	}
	if ticketInfo.Key != "alpha-first-key" {
		t.Errorf("expected Key 'alpha-first-key', got %q", ticketInfo.Key)
	}
	if ticketInfo.Lane != "dev" {
		t.Errorf("expected Lane 'dev', got %q", ticketInfo.Lane)
	}
	if ticketInfo.ExpectedSteps != 3 {
		t.Errorf("expected ExpectedSteps 3, got %d", ticketInfo.ExpectedSteps)
	}
	if len(ticketInfo.TargetPaths) != 2 || ticketInfo.TargetPaths[0] != "cmd/fak/main.go" {
		t.Errorf("unexpected TargetPaths: %v", ticketInfo.TargetPaths)
	}
	if !strings.Contains(ticketInfo.Body, "Detailed specification for first test ticket.") {
		t.Errorf("expected body in ticketInfo, got: %s", ticketInfo.Body)
	}

	// 11. Test show not found
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "nonexistent-ticket"})
	if code != 1 {
		t.Errorf("expected exit code 1 for non-existent ticket, got %d", code)
	}
	if !strings.Contains(stderr.String(), "not found") {
		t.Errorf("expected 'not found' in stderr, got: %s", stderr.String())
	}

	// 12. Test show missing argument
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show"})
	if code != 2 {
		t.Errorf("expected exit code 2 for show without args, got %d", code)
	}
	if !strings.Contains(stderr.String(), "ticket ID required") {
		t.Errorf("expected 'ticket ID required' in stderr, got: %s", stderr.String())
	}
}

func TestCompanionTicket_Live(t *testing.T) {
	_, privRoot := ResolveCompanionRoots()
	if privRoot == "" {
		t.Skip("skipping live companion ticket tests: companion fak-private not present")
	}

	// 1. ticket --help
	var stdout, stderr bytes.Buffer
	code := RunCompanionTicket(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("ticket --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev ticket") {
		t.Errorf("unexpected ticket --help output: %s", stdout.String())
	}

	// 2. ticket list
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"list"})
	if code != 0 {
		t.Fatalf("ticket list failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "companion-dev-harness") {
		t.Errorf("expected companion-dev-harness in live ticket list, got: %s", stdout.String())
	}

	// 3. ticket show TICKET-01-companion-aware-precommit-hooks
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "TICKET-01-companion-aware-precommit-hooks"})
	if code != 0 {
		t.Fatalf("ticket show failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "feat(dev): companion-aware git pre-commit hooks") {
		t.Errorf("unexpected ticket show output: %s", stdout.String())
	}

	// 4. ticket show --json TICKET-01-companion-aware-precommit-hooks
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"show", "--json", "TICKET-01-companion-aware-precommit-hooks"})
	if code != 0 {
		t.Fatalf("ticket show --json failed with code %d. stderr: %s", code, stderr.String())
	}
	var tk TicketInfo
	if err := json.Unmarshal(stdout.Bytes(), &tk); err != nil {
		t.Fatalf("failed to parse JSON from ticket show --json: %v", err)
	}
	if tk.ID != "TICKET-01-companion-aware-precommit-hooks" {
		t.Errorf("expected ID 'TICKET-01-companion-aware-precommit-hooks', got %q", tk.ID)
	}
	if tk.Tranche != "companion-dev-harness" {
		t.Errorf("expected Tranche 'companion-dev-harness', got %q", tk.Tranche)
	}

	// 5. ticket next
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionTicket(&stdout, &stderr, []string{"next", "--lan-deconflict=false", "--query", "companion"})
	if code != 0 {
		t.Fatalf("ticket next failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Next Candidate:") {
		t.Errorf("expected Next Candidate in ticket next output, got: %s", stdout.String())
	}
}

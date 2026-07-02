package popularizationtickets

import (
	"strings"
	"testing"
)

func TestLoadAndRenderTickets(t *testing.T) {
	tickets, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(tickets) != 50 {
		t.Fatalf("ticket count = %d, want 50", len(tickets))
	}
	body := RenderBody(tickets[0], "epic #1")
	for _, want := range []string{"Dimension A", "Concepts served:", "## Likely files", "`INDEX.md`", "epic #1"} {
		if !strings.Contains(body, want) {
			t.Fatalf("rendered body missing %q:\n%s", want, body)
		}
	}
}

func TestLanesTSVHasOneRowPerTicket(t *testing.T) {
	tickets, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rows := strings.Split(strings.TrimSpace(LanesTSV(tickets)), "\n")
	if len(rows) != len(tickets) {
		t.Fatalf("lane rows = %d, want %d", len(rows), len(tickets))
	}
	if fields := strings.Split(rows[0], "\t"); len(fields) != 2 || fields[1] == "" {
		t.Fatalf("bad first TSV row: %q", rows[0])
	}
}

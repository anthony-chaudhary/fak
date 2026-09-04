package popularizationtickets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

func TestListAndJSON(t *testing.T) {
	tickets, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	listing := List(tickets)
	if !strings.Contains(listing, "per-dimension: map[") {
		t.Fatalf("List output missing summary: %s", listing)
	}

	data, err := JSON(tickets)
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	var decoded []Ticket
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal JSON: %v", err)
	}
	if len(decoded) != len(tickets) {
		t.Fatalf("decoded %d tickets, want %d", len(decoded), len(tickets))
	}
}

func TestEmitFiles(t *testing.T) {
	tickets, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	dir := t.TempDir()
	if err := EmitFiles(dir, "epic #42", tickets); err != nil {
		t.Fatalf("EmitFiles: %v", err)
	}

	for i, ticket := range tickets {
		base := filepath.Join(dir, fmt.Sprintf("ticket-%02d", i+1))
		body, err := os.ReadFile(base + ".md")
		if err != nil {
			t.Fatalf("read md file %d: %v", i+1, err)
		}
		if !strings.Contains(string(body), ticket.Lane) {
			t.Fatalf("ticket %d md missing lane: %s", i+1, ticket.Lane)
		}

		title, err := os.ReadFile(base + ".title")
		if err != nil {
			t.Fatalf("read title file %d: %v", i+1, err)
		}
		if string(title) != ticket.Title {
			t.Fatalf("ticket %d title mismatch: got %q, want %q", i+1, string(title), ticket.Title)
		}
	}
}

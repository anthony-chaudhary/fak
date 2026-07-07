package microagent_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestJournalSinkOneChainedFilePerHost is the #2011 acceptance witness (bullet
// 1): N microagents driven by ONE host, every lifecycle event landing in ONE
// host-scoped, hash-chained journal FILE — not one JSONL per agent — and the file
// verifies end-to-end with each row tagged by its agent id.
func TestJournalSinkOneChainedFilePerHost(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host-audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}
	defer j.Close()

	tbl := session.NewTable()
	sink := microagent.NewJournalSink(j)
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 16, Queue: 256, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	const agents, turns = 120, 2
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("ma-%03d", i)
		if err := h.Spawn(id, &turnAgent{id: id, turns: turns}); err != nil {
			t.Fatalf("Spawn(%s): %v", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v (live=%d)", err, h.Live())
	}
	if err := j.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// ONE file holds the whole fleet's audit; its hash chain verifies end to end.
	n, err := journal.Verify(path)
	if err != nil {
		t.Fatalf("journal.Verify(%s): %v", path, err)
	}
	if n != agents*2 { // one AGENT_SPAWN + one AGENT_DONE per agent
		t.Fatalf("verified %d rows, want %d (spawn+done per agent)", n, agents*2)
	}

	rows, err := journal.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	spawns, dones := 0, 0
	seen := map[string]bool{}
	for _, r := range rows {
		switch r.Kind {
		case journal.KindAgentSpawn:
			spawns++
		case journal.KindAgentDone:
			dones++
		default:
			t.Fatalf("unexpected row kind %q in host audit chain", r.Kind)
		}
		if r.TraceID == "" {
			t.Fatalf("row not tagged with an agent id: %+v", r)
		}
		seen[r.TraceID] = true
	}
	if spawns != agents || dones != agents {
		t.Fatalf("spawns=%d dones=%d, want %d each", spawns, dones, agents)
	}
	if len(seen) != agents {
		t.Fatalf("distinct agent ids in the one chain = %d, want %d", len(seen), agents)
	}
}

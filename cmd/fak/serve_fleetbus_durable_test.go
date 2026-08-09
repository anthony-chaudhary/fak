package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestFleetBusApplyReachesDurableMirror(t *testing.T) {
	registry := session.NewRegistry(session.NewFileStore(filepath.Join(t.TempDir(), "sessions.json")))
	mirror := newSessionDurability(registry, nil, "test-host", time.Hour, time.Now, t.Logf)
	tbl := session.NewTable()
	tbl.Restore("trace-durable", session.State{TraceID: "trace-durable", Run: session.Running})
	if err := mirror.register(context.Background(), "trace-durable", tbl.Get("trace-durable")); err != nil {
		t.Fatalf("register initial session: %v", err)
	}

	out := (&fleetBusApplier{tbl: tbl, durability: mirror, ctx: context.Background()}).Apply(fleetbus.Directive{
		Op: fleetbus.Op("pause"), Selector: fleetbus.Selector{All: true},
	})
	if out.Status != fleetbus.AckApplied || out.Affected != 1 || !strings.HasPrefix(out.Witness, "durable:") {
		t.Fatalf("bus pause outcome = %#v, want one durably applied session", out)
	}
	if got := tbl.Get("trace-durable").Run; got != session.Paused {
		t.Fatalf("live table state = %q, want paused", got)
	}

	restarted := session.NewTable()
	if err := mirror.restore(restarted); err != nil {
		t.Fatalf("restore after bus pause: %v", err)
	}
	if got := restarted.Get("trace-durable").Run; got != session.Paused {
		t.Fatalf("restored state = %q, want paused: bus apply did not reach durable mirror", got)
	}
}

func TestFleetBusApplyNamesMemoryOnlyWhenDurabilityDisabled(t *testing.T) {
	tbl := session.NewTable()
	tbl.Restore("trace-memory", session.State{TraceID: "trace-memory", Run: session.Running})
	out := (&fleetBusApplier{tbl: tbl, ctx: context.Background()}).Apply(fleetbus.Directive{
		Op: fleetbus.Op("pause"), Selector: fleetbus.Selector{All: true},
	})
	if out.Status != fleetbus.AckApplied || !strings.Contains(out.Witness, "in memory only") || !strings.HasPrefix(out.Witness, "memory-only:") {
		t.Fatalf("memory-only outcome must not masquerade as durable: %#v", out)
	}
}

package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/systools"
)

type recordingSysPlanner struct {
	turns []*Completion
	n     int
}

func (p *recordingSysPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}

func (p *recordingSysPlanner) Model() string { return "recording-sys" }

func TestAgentSysToolsGetTimeDispatch(t *testing.T) {
	cat, err := ArmSysTools(systools.Config{})
	if err != nil {
		t.Fatalf("ArmSysTools: %v", err)
	}
	t.Cleanup(DisarmSysTools)

	if len(cat) != 3 {
		t.Fatalf("SysToolCatalog() len = %d, want 3", len(cat))
	}

	turns := []*Completion{
		toolCallTurn(systools.ToolGetTime, `{"timezone":"UTC"}`),
		{Message: Message{Content: "done"}},
	}

	var log []traceEvent
	planner := &recordingSysPlanner{turns: turns}
	metrics, err := RunArm(context.Background(), planner, "get current time",
		true, len(turns)+1, &log, WithToolCatalog(cat))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	if metrics.EngineCalls < 1 {
		t.Errorf("expected at least 1 engine call, got %d", metrics.EngineCalls)
	}

	found := false
	for _, ev := range log {
		if ev.Tool == systools.ToolGetTime {
			found = true
			if ev.Verdict != "ALLOW" {
				t.Errorf("verdict = %q, want ALLOW", ev.Verdict)
			}
			if ev.By != systools.RungName {
				t.Errorf("by = %q, want %q", ev.By, systools.RungName)
			}
		}
	}
	if !found {
		t.Errorf("did not find %s in trace log", systools.ToolGetTime)
	}
}

func TestAgentSysToolsUnarmed(t *testing.T) {
	DisarmSysTools()
	if cat := SysToolCatalog(); cat != nil {
		t.Errorf("unarmed SysToolCatalog() = %v, want nil", cat)
	}
	if names := sysToolAllow(); names != nil {
		t.Errorf("unarmed sysToolAllow() = %v, want nil", names)
	}
	if _, ok := sysToolMeta(systools.ToolGetTime); ok {
		t.Errorf("unarmed sysToolMeta() returned ok=true")
	}
}

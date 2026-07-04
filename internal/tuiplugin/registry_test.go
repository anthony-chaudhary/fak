package tuiplugin

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestRegisterLookupAndDescriptors(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	Register(Pane{
		ID:      "guard",
		Summary: "guard decisions",
		Usage:   "fak console guard --guard-json FILE",
		Schema:  "fak.tui.guard.v1",
		BuiltIn: true,
		Overview: func(ctx any) (OverviewCard, error) {
			return OverviewCard{Pane: "guard", Summary: "ok"}, nil
		},
		Controls: []Control{
			{ID: "follow", Label: "Follow", Kind: "toggle", Flag: "--follow"},
			{ID: "color", Label: "Color", Kind: "flag", Flag: "--color", Default: "auto", Options: []string{"auto", "always", "never", "auto", ""}},
		},
		Run: func(stdout, stderr io.Writer, argv []string) int {
			return 0
		},
	})

	p, ok := Lookup("guard")
	if !ok {
		t.Fatal("guard pane not registered")
	}
	if p.Schema != "fak.tui.guard.v1" || p.Run == nil {
		t.Fatalf("pane = %+v", p)
	}
	descs := Descriptors()
	if len(descs) != 1 {
		t.Fatalf("descriptors = %+v, want one", descs)
	}
	if descs[0].ID != "guard" || len(descs[0].Controls) != 2 {
		t.Fatalf("descriptor = %+v", descs[0])
	}
	if !descs[0].Overview {
		t.Fatalf("descriptor should advertise overview support: %+v", descs[0])
	}
	if descs[0].Controls[0].ID != "color" || descs[0].Controls[1].ID != "follow" {
		t.Fatalf("controls not sorted by id: %+v", descs[0].Controls)
	}
	if got := descs[0].Controls[0].Options; len(got) != 3 || got[0] != "auto" || got[1] != "always" || got[2] != "never" {
		t.Fatalf("options = %v, want [auto always never]", got)
	}
}

func TestAllIsStableSortedSnapshot(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	runner := func(stdout, stderr io.Writer, argv []string) int { return 0 }
	Register(Pane{ID: "loops", Controls: []Control{{ID: "ledger", Options: []string{"file"}}}, Run: runner})
	Register(Pane{ID: "agent", Run: runner})

	panes := All()
	if got := []string{panes[0].ID, panes[1].ID}; got[0] != "agent" || got[1] != "loops" {
		t.Fatalf("order = %v, want [agent loops]", got)
	}
	panes[0].ID = "mutated"
	panes[1].Controls[0].ID = "mutated"
	panes[1].Controls[0].Options[0] = "mutated"
	again := All()
	if again[0].ID != "agent" {
		t.Fatalf("All exposed mutable snapshot: %+v", again)
	}
	if again[1].Controls[0].ID != "ledger" {
		t.Fatalf("All exposed mutable controls: %+v", again[1].Controls)
	}
	if len(again[1].Controls[0].Options) != 1 || again[1].Controls[0].Options[0] != "file" {
		t.Fatalf("All exposed mutable control options: %+v", again[1].Controls[0].Options)
	}
}

func TestRegisterRejectsDuplicateAndMalformedPane(t *testing.T) {
	tests := []struct {
		name string
		pane Pane
		want string
	}{
		{name: "bad id", pane: Pane{ID: "Bad", Run: func(stdout, stderr io.Writer, argv []string) int { return 0 }}, want: "invalid pane id"},
		{name: "nil runner", pane: Pane{ID: "guard"}, want: "nil runner"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ResetForTest()
			defer ResetForTest()
			got := panicString(func() { Register(tt.pane) })
			if !strings.Contains(got, tt.want) {
				t.Fatalf("panic = %q, want %q", got, tt.want)
			}
		})
	}

	ResetForTest()
	defer ResetForTest()
	Register(Pane{ID: "guard", Run: func(stdout, stderr io.Writer, argv []string) int { return 0 }})
	got := panicString(func() {
		Register(Pane{ID: "guard", Run: func(stdout, stderr io.Writer, argv []string) int { return 0 }})
	})
	if !strings.Contains(got, "duplicate pane id") {
		t.Fatalf("duplicate panic = %q", got)
	}
}

func TestLookupRunnerExecutes(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	Register(Pane{
		ID: "guard",
		Overview: func(ctx any) (OverviewCard, error) {
			return OverviewCard{Pane: "guard", Summary: "ok"}, nil
		},
		Run: func(stdout, stderr io.Writer, argv []string) int {
			stdout.Write([]byte(strings.Join(argv, ",")))
			return 7
		},
	})
	p, ok := Lookup("guard")
	if !ok {
		t.Fatal("guard pane not registered")
	}
	var out bytes.Buffer
	if code := p.Run(&out, &bytes.Buffer{}, []string{"--json"}); code != 7 {
		t.Fatalf("runner code = %d, want 7", code)
	}
	if out.String() != "--json" {
		t.Fatalf("stdout = %q", out.String())
	}
	card, err := p.Overview(nil)
	if err != nil || card.Pane != "guard" {
		t.Fatalf("overview card = %+v err=%v", card, err)
	}
}

func panicString(fn func()) (out string) {
	defer func() {
		if r := recover(); r != nil {
			out = fmt.Sprint(r)
		}
	}()
	fn()
	return ""
}

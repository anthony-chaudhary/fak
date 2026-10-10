package main

import (
	"encoding/json"
	"slices"
	"testing"
)

// guardOpenCodeInstallCommand reads the launch argv from the install record's
// wire shape, so the assertion does not depend on Go field names.
func guardOpenCodeInstallCommand(t *testing.T, command []string) ([]string, bool) {
	t.Helper()
	_, install := installGuardOpenCodeConfig(command, "http://127.0.0.1:54321", "qwen38", nil)
	raw, err := json.Marshal(install)
	if err != nil {
		t.Fatalf("marshal install: %v", err)
	}
	var wire struct {
		Applied bool      `json:"applied"`
		Command *[]string `json:"command"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal install: %v", err)
	}
	if wire.Command == nil {
		return nil, wire.Applied
	}
	return *wire.Command, wire.Applied
}

func TestGuardOpenCodeRunLaunchesStandalone(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want []string
	}{
		{"bare run gains standalone", []string{"opencode", "run", "-m", "fak/x", "hi"}, []string{"opencode", "run", "--standalone", "-m", "fak/x", "hi"}},
		{"windows launcher path", []string{`C:\Users\u\.opencode\bin\opencode.exe`, "run", "hi"}, []string{`C:\Users\u\.opencode\bin\opencode.exe`, "run", "--standalone", "hi"}},
		{"flag after -- is positional", []string{"opencode", "run", "--", "--server"}, []string{"opencode", "run", "--standalone", "--", "--server"}},
		{"standalone already present", []string{"opencode", "run", "--standalone", "hi"}, []string{"opencode", "run", "--standalone", "hi"}},
		{"standalone= form present", []string{"opencode", "run", "--standalone=true", "hi"}, []string{"opencode", "run", "--standalone=true", "hi"}},
		{"explicit server", []string{"opencode", "run", "--server", "http://127.0.0.1:4096", "hi"}, []string{"opencode", "run", "--server", "http://127.0.0.1:4096", "hi"}},
		{"explicit server= form", []string{"opencode", "run", "--server=http://127.0.0.1:4096", "hi"}, []string{"opencode", "run", "--server=http://127.0.0.1:4096", "hi"}},
		{"non-run subcommand", []string{"opencode", "serve", "--port", "4096"}, []string{"opencode", "serve", "--port", "4096"}},
		{"interactive tui", []string{"opencode"}, []string{"opencode"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, applied := guardOpenCodeInstallCommand(t, tc.in)
			if !applied {
				t.Fatalf("install not applied for %q", tc.in)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("launch argv = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGuardOpenCodeStandaloneSkipsNonOpenCodeChild(t *testing.T) {
	got, applied := guardOpenCodeInstallCommand(t, []string{"codex", "run", "hi"})
	if applied || got != nil {
		t.Fatalf("non-opencode child: applied=%v command=%q, want no install and no argv", applied, got)
	}
}

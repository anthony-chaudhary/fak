package main

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"
)

func TestRunWindowgateSupportsCodexMCPLaunchWitness(t *testing.T) {
	original := captureCodexMCPWindowSelfcheckForRun
	captureCodexMCPWindowSelfcheckForRun = func(string, string, time.Duration) (codexMCPWindowSelfcheckReport, error) {
		return codexMCPWindowSelfcheckReport{
			Schema: codexMCPWindowSelfcheckSchema, Applicable: false, Platform: "test", Reason: "not applicable",
		}, nil
	}
	t.Cleanup(func() { captureCodexMCPWindowSelfcheckForRun = original })

	var stdout, stderr bytes.Buffer
	code := runWindowgate(&stdout, &stderr, []string{"--codex-mcp-selfcheck", "--json"})
	if code != 0 {
		t.Fatalf("windowgate Codex MCP selfcheck code=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Schema     string `json:"schema"`
		OK         bool   `json:"ok"`
		Applicable bool   `json:"applicable"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode Codex MCP selfcheck JSON: %v\n%s", err, stdout.String())
	}
	const wantSchema = "fak-windowgate-codex-mcp-selfcheck/1"
	if got.Schema != wantSchema || !got.OK {
		t.Fatalf("Codex MCP selfcheck = %+v, want schema %q and ok", got, wantSchema)
	}
}

func TestAssessCodexMCPWindowSelfcheckRequiresBothCleanRoutes(t *testing.T) {
	rep := codexMCPWindowSelfcheckReport{
		Schema: codexMCPWindowSelfcheckSchema, Applicable: true, Platform: "windows",
		Routes: []codexMCPWindowRouteReport{
			cleanCodexMCPWindowRoute("managed", 101),
			cleanCodexMCPWindowRoute("unmanaged", 102),
		},
	}
	assessCodexMCPWindowSelfcheck(&rep)
	if !rep.OK || rep.VisibleWindows != 0 {
		t.Fatalf("clean routes = %+v, want ok with zero windows", rep)
	}
}

func TestAssessCodexMCPWindowSelfcheckRejectsVisibleWindowAndDirtyExit(t *testing.T) {
	rep := codexMCPWindowSelfcheckReport{
		Schema: codexMCPWindowSelfcheckSchema, Applicable: true, Platform: "windows",
		Routes: []codexMCPWindowRouteReport{
			cleanCodexMCPWindowRoute("managed", 101),
			cleanCodexMCPWindowRoute("unmanaged", 102),
		},
	}
	rep.Routes[1].WindowHandle = 77
	rep.Routes[1].ExitState = "failed"
	rep.Routes[1].ExitCode = 1
	assessCodexMCPWindowSelfcheck(&rep)
	if rep.OK || rep.VisibleWindows != 1 {
		t.Fatalf("unsafe routes = %+v, want failure with one visible window", rep)
	}
}

func cleanCodexMCPWindowRoute(route string, pid int) codexMCPWindowRouteReport {
	return codexMCPWindowRouteReport{
		Route: route, CodexAncestorPID: 40, ParentPID: 50, ObservedParent: 50, PID: pid,
		ConsolePIDs: []uint32{40, 50, uint32(pid)}, ConsoleMember: true,
		InitializeOK: true, ExitState: "clean", ExitCode: 0,
	}
}

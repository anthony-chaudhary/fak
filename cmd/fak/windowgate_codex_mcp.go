package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

const codexMCPWindowSelfcheckSchema = "fak-windowgate-codex-mcp-selfcheck/1"

var codexMCPWindowSelfcheckRoutes = []string{"managed", "unmanaged"}

type codexMCPWindowRouteReport struct {
	Route            string   `json:"route"`
	CodexAncestorPID int      `json:"codex_ancestor_pid"`
	ParentPID        int      `json:"parent_pid"`
	ObservedParent   int      `json:"observed_parent_pid"`
	PID              int      `json:"pid"`
	ConsolePIDs      []uint32 `json:"console_pids,omitempty"`
	ConsoleMember    bool     `json:"console_member"`
	WindowHandle     uintptr  `json:"window_handle"`
	InitializeOK     bool     `json:"initialize_ok"`
	ExitState        string   `json:"exit_state"`
	ExitCode         int      `json:"exit_code"`
	Failure          string   `json:"failure,omitempty"`
}

type codexMCPWindowSelfcheckReport struct {
	Schema         string                      `json:"schema"`
	OK             bool                        `json:"ok"`
	Applicable     bool                        `json:"applicable"`
	Platform       string                      `json:"platform"`
	Server         string                      `json:"server,omitempty"`
	ConfigState    string                      `json:"config_state,omitempty"`
	Command        string                      `json:"command,omitempty"`
	CommandSHA256  string                      `json:"command_sha256,omitempty"`
	VisibleWindows int                         `json:"visible_windows"`
	Routes         []codexMCPWindowRouteReport `json:"routes,omitempty"`
	Reason         string                      `json:"reason"`
}

var captureCodexMCPWindowSelfcheckForRun = captureCodexMCPWindowSelfcheck

func assessCodexMCPWindowSelfcheck(rep *codexMCPWindowSelfcheckReport) {
	if !rep.Applicable {
		rep.OK = true
		return
	}
	want := append([]string(nil), codexMCPWindowSelfcheckRoutes...)
	sort.Strings(want)
	got := make([]string, 0, len(rep.Routes))
	rep.VisibleWindows = 0
	for _, route := range rep.Routes {
		got = append(got, route.Route)
		if route.WindowHandle != 0 {
			rep.VisibleWindows++
		}
		if route.CodexAncestorPID == 0 || route.ParentPID == 0 || route.ObservedParent != route.ParentPID ||
			route.PID == 0 || route.WindowHandle != 0 || !route.InitializeOK || route.ExitState != "clean" || route.ExitCode != 0 {
			rep.OK = false
			rep.Reason = fmt.Sprintf("route %s did not preserve attended Codex ancestry, zero-window stdio startup, and clean exit", route.Route)
			return
		}
	}
	sort.Strings(got)
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		rep.OK = false
		rep.Reason = fmt.Sprintf("launch routes = %v, want %v", got, want)
		return
	}
	rep.OK = true
	rep.Reason = "managed and unmanaged Codex stdio MCP routes stayed off-desktop and exited cleanly"
}

func writeCodexMCPWindowSelfcheck(w io.Writer, rep codexMCPWindowSelfcheckReport, asJSON bool) error {
	if asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	verdict := "PASS"
	if !rep.Applicable {
		verdict = "SKIP"
	} else if !rep.OK {
		verdict = "FAIL"
	}
	fmt.Fprintf(w, "%s windowgate Codex MCP selfcheck: %s\n", verdict, rep.Reason)
	for _, route := range rep.Routes {
		fmt.Fprintf(w, "  %-10s codex=%d parent=%d observed_parent=%d pid=%d console_member=%t console_pids=%v hwnd=%d initialize=%t exit=%s/%d\n",
			route.Route, route.CodexAncestorPID, route.ParentPID, route.ObservedParent, route.PID,
			route.ConsoleMember, route.ConsolePIDs, route.WindowHandle, route.InitializeOK, route.ExitState, route.ExitCode)
	}
	return nil
}

func failCodexMCPWindowSelfcheck(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "fak windowgate --codex-mcp-selfcheck: %v\n", err)
	return 1
}

func runCodexMCPWindowSelfcheck(stdout, stderr io.Writer, config, server string, timeout time.Duration, asJSON bool) int {
	rep, err := captureCodexMCPWindowSelfcheckForRun(config, server, timeout)
	if err != nil {
		return failCodexMCPWindowSelfcheck(stderr, err)
	}
	assessCodexMCPWindowSelfcheck(&rep)
	if err := writeCodexMCPWindowSelfcheck(stdout, rep, asJSON); err != nil {
		return failCodexMCPWindowSelfcheck(stderr, err)
	}
	if !rep.OK {
		return 1
	}
	return 0
}

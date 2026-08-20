package main

import (
	"encoding/json"
	"fmt"
	"io"
)

const (
	desktopConsoleSelfcheckSchema     = "fak-desktop-console-selfcheck/1"
	desktopConsoleSelfcheckRoleEnv    = "FAK_DESKTOP_CONSOLE_SELFCHECK_ROLE"
	desktopConsoleSelfcheckLabelEnv   = "FAK_DESKTOP_CONSOLE_SELFCHECK_LABEL"
	desktopConsoleSelfcheckDirEnv     = "FAK_DESKTOP_CONSOLE_SELFCHECK_DIR"
	desktopConsoleSelfcheckReleaseEnv = "FAK_DESKTOP_CONSOLE_SELFCHECK_RELEASE"
)

type desktopConsoleSelfcheckProcess struct {
	Label        string   `json:"label"`
	PID          int      `json:"pid"`
	ConsolePIDs  []uint32 `json:"console_pids,omitempty"`
	WindowHandle uintptr  `json:"window_handle"`
}

type desktopConsoleSelfcheckReport struct {
	Schema               string                           `json:"schema"`
	OK                   bool                             `json:"ok"`
	Applicable           bool                             `json:"applicable"`
	Platform             string                           `json:"platform"`
	Backend              string                           `json:"backend,omitempty"`
	RootPID              int                              `json:"root_pid,omitempty"`
	SharedHiddenConsoles int                              `json:"shared_hidden_consoles"`
	VisibleWindows       int                              `json:"visible_windows"`
	Processes            []desktopConsoleSelfcheckProcess `json:"processes,omitempty"`
	Reason               string                           `json:"reason"`
}

func writeDesktopConsoleSelfcheck(w io.Writer, rep desktopConsoleSelfcheckReport, asJSON bool) error {
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
	fmt.Fprintf(w, "%s windowgate selfcheck: %s\n", verdict, rep.Reason)
	for _, p := range rep.Processes {
		fmt.Fprintf(w, "  %-10s pid=%-6d console_pids=%v window_handle=%d\n",
			p.Label, p.PID, p.ConsolePIDs, p.WindowHandle)
	}
	return nil
}

func failDesktopConsoleSelfcheck(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "fak windowgate --selfcheck: %v\n", err)
	return 1
}

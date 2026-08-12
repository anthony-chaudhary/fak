package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

var windowsSetupGOOS = runtime.GOOS
var windowsSetupCommand = exec.Command

func runWindowsSetup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("windows-setup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "request UAC once and install the development allow list")
	jsonOut := fs.Bool("json", false, "print the plan or verification result as JSON")
	repo := fs.String("repo", ".", "fak repository root")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if windowsSetupGOOS != "windows" {
		fmt.Fprintln(stderr, "fak-dev windows-setup: Windows only")
		return 2
	}
	plan, err := buildWindowsSetupSpec(*repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev windows-setup: %v\n", err)
		return 2
	}
	if !*apply {
		b, _ := plan.JSON()
		if *jsonOut {
			fmt.Fprintln(stdout, string(b))
		} else {
			fmt.Fprintf(stdout, "Windows developer allow-list plan: %d paths, %d processes, fleet spine %s:%d\nRun with --apply to request UAC once and install it.\n", len(plan.Paths), len(plan.Processes), plan.Group, plan.Port)
		}
		return 0
	}
	dir, err := os.MkdirTemp("", "fak-windows-setup-")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	defer os.RemoveAll(dir)
	scriptPath := filepath.Join(dir, "apply.ps1")
	resultPath := filepath.Join(dir, "result.json")
	if err = os.WriteFile(scriptPath, []byte(PowerShell(plan, resultPath, true)), 0600); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	args := []string{"-NoProfile", "-Command", fmt.Sprintf("$p=Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%s'); exit $p.ExitCode", scriptPath)}
	cmd := windowsSetupCommand("powershell.exe", args...)
	if out, runErr := cmd.CombinedOutput(); runErr != nil {
		fmt.Fprintf(stderr, "fak-dev windows-setup: elevation failed: %v %s\n", runErr, out)
		return 1
	}
	data, err := os.ReadFile(resultPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev windows-setup: read-back missing: %v\n", err)
		return 1
	}
	var result Result
	if err = json.Unmarshal(data, &result); err != nil {
		fmt.Fprintf(stderr, "fak-dev windows-setup: invalid read-back: %v\n", err)
		return 1
	}
	if *jsonOut {
		fmt.Fprintln(stdout, string(data))
	}
	if !result.Complete() {
		fmt.Fprintln(stderr, "fak-dev windows-setup: NOT READY — one or more Defender/firewall exceptions were not read back")
		return 1
	}
	fmt.Fprintf(stdout, "READY: Windows Security allows native fak tests and fleet-spine multicast (%s:%d).\n", plan.Group, plan.Port)
	return 0
}

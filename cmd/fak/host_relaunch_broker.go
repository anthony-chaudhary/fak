package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

var brokerExecCommand = exec.Command

func runHostRelaunchBroker(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("host-relaunch-broker", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", hostRelaunchBrokerDefaultDir(), "durable machine-control-plane-to-desktop request spool")
	dryRun := fs.Bool("dry-run", false, "validate and print WT argv without launching")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	expandedDir := pathutil.ExpandTilde(*dir)
	pending, err := hostresurrect.Pending(expandedDir)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, path := range pending {
		req, err := hostresurrect.ReadQueued(path)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", filepath.Base(path), err)
			continue
		}
		window := strings.TrimSpace(req.WindowID)
		if window == "" {
			window = "new"
		}
		args := []string{"-w", window, "new-tab", "-d", req.CWD, req.Command[0]}
		args = append(args, req.Command[1:]...)
		if *dryRun {
			fmt.Fprintf(stdout, "wt.exe")
			for _, a := range args {
				fmt.Fprintf(stdout, " %q", a)
			}
			fmt.Fprintln(stdout)
			continue
		}
		cmd := brokerExecCommand("wt.exe", args...)
		cmd.Env = append(os.Environ(), "FAK_RESUME_HANDLE="+req.ResumeHandle, "FAK_HOST_CRASH_EVENT="+req.EventID)
		configureDispatchHelperCommand(cmd)
		if err := cmd.Start(); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err := hostresurrect.CompleteQueued(path); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	return 0
}

func hostRelaunchBrokerDefaultDir() string {
	if d := strings.TrimSpace(os.Getenv("FAK_HOST_RELAUNCH_DIR")); d != "" {
		return d
	}
	if programData := strings.TrimSpace(os.Getenv("ProgramData")); programData != "" {
		machine := filepath.Join(programData, "fak", "guard-control", "relaunch")
		if st, err := os.Stat(machine); err == nil && st.IsDir() {
			return machine
		}
	}
	base, e := os.UserConfigDir()
	if e != nil || base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak", "host", "relaunch")
}

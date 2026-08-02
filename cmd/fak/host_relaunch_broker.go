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
			fmt.Fprintln(stdout, windowsCommandLine(append([]string{"wt.exe"}, args...)))
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

// windowsCommandLine renders argv as the single command-line string Windows
// process creation will actually receive, so `--dry-run` previews the
// invocation that would really run instead of a Go-source rendering of it.
// os/exec hands argv to syscall.StartProcess, which — absent an explicit
// SysProcAttr.CmdLine, and this broker sets none — joins the elements with one
// space after escaping each through syscall.EscapeArg. Mirroring that join here
// keeps the preview and the launch byte-identical.
func windowsCommandLine(argv []string) string {
	quoted := make([]string, 0, len(argv))
	for _, a := range argv {
		quoted = append(quoted, escapeWindowsArg(a))
	}
	return strings.Join(quoted, " ")
}

// escapeWindowsArg is a transcription of syscall.EscapeArg (which is
// Windows-only, while cmd/fak also builds for linux, so it cannot be imported
// here). The rules, per the CommandLineToArgvW contract:
//   - a backslash run is doubled only when it immediately precedes a double
//     quote — including the closing quote this function may append — so an
//     ordinary path keeps its single backslashes;
//   - every literal double quote is escaped with one backslash;
//   - the argument is wrapped in double quotes only when it contains a space or
//     a tab; an empty argument becomes "" so it survives as a distinct element.
//
// Using %q instead was a real defect and not a cosmetic one: Go quoting doubles
// every backslash, so a Windows CWD previewed as C:\\Users\\… never matched the
// C:\Users\… the launch passed.
func escapeWindowsArg(s string) string {
	if s == "" {
		return `""`
	}
	needsBackslash := false
	hasSpace := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\\':
			needsBackslash = true
		case ' ', '\t':
			hasSpace = true
		}
	}
	if !needsBackslash && !hasSpace {
		return s
	}
	if !needsBackslash {
		return `"` + s + `"`
	}
	b := make([]byte, 0, len(s)+2)
	if hasSpace {
		b = append(b, '"')
	}
	slashes := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		default:
			slashes = 0
		case '\\':
			slashes++
		case '"':
			for ; slashes > 0; slashes-- {
				b = append(b, '\\')
			}
			b = append(b, '\\')
		}
		b = append(b, c)
	}
	if hasSpace {
		for ; slashes > 0; slashes-- {
			b = append(b, '\\')
		}
		b = append(b, '"')
	}
	return string(b)
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

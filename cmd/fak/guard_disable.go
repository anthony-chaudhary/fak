package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
)

const guardDisableDefaultReason = "operator-requested guard repair"

type guardDisableOptions struct {
	Reason  string
	Command []string
	Usage   bool
	JSON    bool
}

// runGuardDisable starts exactly one raw child and restores the normal posture when that
// child exits. It never mutates user or repository configuration. The two Codex recovery
// variables cover both generations of the continuation hook: the current shell-level
// break-glass check and the older fak-subcommand override.
func runGuardDisable(commandName string, stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	path, pathErr := guardDisableUsageDefaultPath()
	return runGuardDisableWithUsage(commandName, stdin, stdout, stderr, argv, path, pathErr)
}

func runGuardDisableWithUsage(commandName string, stdin io.Reader, stdout, stderr io.Writer, argv []string, usagePath string, usagePathErr error) int {
	opts, code, done := parseGuardDisable(commandName, argv, stderr)
	if done {
		return code
	}
	if opts.Usage {
		return runGuardDisableUsage(stdout, stderr, usagePath, usagePathErr, opts.JSON)
	}

	program := filepathBaseForDisplay(opts.Command[0])
	fmt.Fprintf(stderr, "fak %s disable: BREAK-GLASS raw session starting (reason: %s; command: %s). The fak capability floor, gateway, guard hooks, and decision audit are NOT running for this child.\n", commandName, opts.Reason, program)

	command := resolveWindowsBatchCommand(opts.Command)
	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
	cmd.Env = guardDisableChildEnv(os.Environ())
	err := cmd.Run()
	code = childprocess.ExitCode(err, 127)
	outcome := guardDisableUsageSuccess
	if err != nil {
		if _, started := err.(*exec.ExitError); !started {
			outcome = guardDisableUsageLaunchError
			fmt.Fprintf(stderr, "fak %s disable: launch %q: %v\n", commandName, opts.Command[0], err)
		} else {
			outcome = guardDisableUsageChildNonzero
		}
	}
	recordGuardDisableUsage(stderr, usagePath, usagePathErr, outcome)
	fmt.Fprintf(stderr, "fak %s disable: BREAK-GLASS raw session ended (exit %d); later launches remain guarded by default.\n", commandName, code)
	return code
}

func parseGuardDisable(commandName string, argv []string, stderr io.Writer) (guardDisableOptions, int, bool) {
	fs := flag.NewFlagSet(commandName+" disable", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reason := fs.String("reason", guardDisableDefaultReason, "short operator reason shown in the unavoidable break-glass warning")
	usage := fs.Bool("usage", false, "fold privacy-safe launcher outcomes per ISO week instead of starting a child")
	jsonOut := fs.Bool("json", false, "with --usage, emit the machine-readable fold")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: fak %s disable [--reason TEXT] [--] [agent command...]\n       fak %s disable --usage [--json]\n       starts one raw repair child (default: codex); it does not persistently disable guard\n", commandName, commandName)
	}
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return guardDisableOptions{}, code, true
	}
	command := append([]string(nil), fs.Args()...)
	if *usage {
		if len(command) != 0 {
			fmt.Fprintln(stderr, "fak "+commandName+" disable: --usage does not start a child; remove the command after --")
			return guardDisableOptions{}, 2, true
		}
		return guardDisableOptions{Usage: true, JSON: *jsonOut}, 0, false
	}
	if *jsonOut {
		fmt.Fprintln(stderr, "fak "+commandName+" disable: --json requires --usage")
		return guardDisableOptions{}, 2, true
	}
	if len(command) == 0 {
		command = []string{"codex"}
	}
	return guardDisableOptions{Reason: guardDisableReason(*reason), Command: command}, 0, false
}

func guardDisableReason(raw string) string {
	if reason := strings.Join(strings.Fields(raw), " "); reason != "" {
		return reason
	}
	return guardDisableDefaultReason
}

// guardDisableChildEnv removes only the routing and identity state injected by an outer
// guard. Without this scrub, invoking break-glass from inside a guarded agent inherits the
// loopback base URL and FAK_GUARD_ACTIVE=1, so the supposedly raw repair child still routes
// through the guard it is meant to repair. An ordinary shell with no active marker keeps its
// provider configuration unchanged.
func guardDisableChildEnv(environ []string) []string {
	nested := codexLoopHookOverrideEnabled(guardDisableEnvValue(environ, guardActiveEnv))
	out := make([]string, 0, len(environ)+2)
	for _, entry := range environ {
		name, value := guardDisableSplitEnv(entry)
		if guardDisableDropEnv(name, value, nested) {
			continue
		}
		out = append(out, entry)
	}
	out = append(out,
		codexRawRecoveryEnv+"="+codexRawRecoveryValue,
		codexLoopHookOverrideEnv+"=1",
	)
	return out
}

func guardDisableDropEnv(name, value string, nested bool) bool {
	upper := strings.ToUpper(strings.TrimSpace(name))
	if upper == codexRawRecoveryEnv || upper == codexLoopHookOverrideEnv {
		return true
	}
	if !nested {
		return false
	}
	if strings.HasPrefix(upper, "FAK_GUARD_") {
		return true
	}
	switch upper {
	case "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE",
		"CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
		"FAK_AUDIT_JOURNAL", "FAK_TASK_HANDOFF_FILE", "FAK_TOOLCALL_CONTROL_MODE",
		"FAK_SPAWN_GRANT_ID":
		return true
	case "ANTHROPIC_API_KEY", "OPENAI_API_KEY":
		return value == guardCodexOAuthPlaceholderAPIKey || value == guardCodexLocalPlaceholderAPIKey
	}
	return false
}

func guardDisableEnvValue(environ []string, want string) string {
	var found string
	for _, entry := range environ {
		name, value := guardDisableSplitEnv(entry)
		if strings.EqualFold(name, want) {
			found = value
		}
	}
	return found
}

func guardDisableSplitEnv(entry string) (string, string) {
	if at := strings.IndexByte(entry, '='); at >= 0 {
		return entry[:at], entry[at+1:]
	}
	return entry, ""
}

func filepathBaseForDisplay(path string) string {
	path = strings.TrimRight(path, `/\\`)
	if at := strings.LastIndexAny(path, `/\\`); at >= 0 {
		return path[at+1:]
	}
	return path
}

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	capabilityAutoUpgradeEnv       = "FAK_CAPABILITY_AUTOUPGRADE"
	capabilityAutoUpgradeReexecEnv = "FAK_CAPABILITY_AUTOUPGRADE_REEXEC"
)

type capabilityAutoUpgradeProcess func(executable string, args []string, stdin io.Reader, stdout, stderr io.Writer, env []string) error

type capabilityAutoUpgradeDeps struct {
	resolveCheckout func() (root, executable string, err error)
	process         capabilityAutoUpgradeProcess
	getenv          func(string) string
	unsetenv        func(string) error
	environ         func() []string
	getpid          func() int
	parentPID       func() int
}

func realCapabilityAutoUpgradeDeps() capabilityAutoUpgradeDeps {
	return capabilityAutoUpgradeDeps{
		resolveCheckout: resolveCapabilityAutoUpgradeCheckout,
		process:         runCapabilityAutoUpgradeProcess,
		getenv:          os.Getenv,
		unsetenv:        os.Unsetenv,
		environ:         os.Environ,
		getpid:          os.Getpid,
		parentPID:       os.Getppid,
	}
}

// runCapabilityAutoUpgradeEarly refreshes the installed controller before a
// checkout-local Strix command can dispatch work that an older controller does
// not understand. It returns handled=true only when dispatch must stop here.
func runCapabilityAutoUpgradeEarly(stdin io.Reader, stdout, stderr io.Writer, argv []string) (int, bool) {
	return runCapabilityAutoUpgrade(stdin, stdout, stderr, argv, realCapabilityAutoUpgradeDeps())
}

func runCapabilityAutoUpgrade(stdin io.Reader, stdout, stderr io.Writer, argv []string, deps capabilityAutoUpgradeDeps) (int, bool) {
	if !needsCapabilityAutoUpgrade(argv) || strings.EqualFold(strings.TrimSpace(deps.getenv(capabilityAutoUpgradeEnv)), "off") {
		return 0, false
	}
	if marker := strings.TrimSpace(deps.getenv(capabilityAutoUpgradeReexecEnv)); marker != "" {
		_, parent, ok := parseCapabilityAutoUpgradeMarker(marker)
		if !ok || parent != deps.parentPID() {
			fmt.Fprintln(stderr, "fak capability auto-upgrade: refused invalid re-exec marker")
			return 1, true
		}
		if err := deps.unsetenv(capabilityAutoUpgradeReexecEnv); err != nil {
			fmt.Fprintf(stderr, "fak capability auto-upgrade: refused re-exec marker cleanup: %v\n", err)
			return 1, true
		}
		return 0, false
	}

	root, executable, err := deps.resolveCheckout()
	if err != nil {
		fmt.Fprintf(stderr, "fak capability auto-upgrade: freshness check refused: %v\n", err)
		return 1, true
	}
	if root == "" {
		return 0, false
	}

	var checkReceipt, checkStderr bytes.Buffer
	checkArgs := []string{"self-update", "--check", "--json", "--root", root, "--target", executable}
	if err := deps.process(executable, checkArgs, nil, &checkReceipt, &checkStderr, deps.environ()); err != nil {
		writeCapabilityAutoUpgradeDetail(stderr, checkStderr.String())
		fmt.Fprintf(stderr, "fak capability auto-upgrade: freshness check failed: %v\n", err)
		return 1, true
	}
	stale, err := capabilityAutoUpgradeCheckReceipt(checkReceipt.Bytes(), executable)
	if err != nil {
		fmt.Fprintf(stderr, "fak capability auto-upgrade: freshness check refused: %v\n", err)
		return 1, true
	}
	if !stale {
		return 0, false
	}

	var updateReceipt bytes.Buffer
	updateArgs := []string{"self-update", "--json", "--root", root, "--target", executable}
	if err := deps.process(executable, updateArgs, nil, &updateReceipt, stderr, deps.environ()); err != nil {
		fmt.Fprintf(stderr, "fak capability auto-upgrade: update failed: %v\n", err)
		return 1, true
	}
	installedCommit, err := capabilityAutoUpgradeInstalledRevision(updateReceipt.Bytes(), executable)
	if err != nil {
		fmt.Fprintf(stderr, "fak capability auto-upgrade: update refused: %v\n", err)
		return 1, true
	}

	reexecEnv := capabilityAutoUpgradeReexecEnvironment(deps.environ(), installedCommit, deps.getpid())
	if err := deps.process(executable, argv[1:], stdin, stdout, stderr, reexecEnv); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return exitErr.ExitCode(), true
		}
		fmt.Fprintf(stderr, "fak capability auto-upgrade: re-exec failed: %v\n", err)
		return 1, true
	}
	return 0, true
}

func needsCapabilityAutoUpgrade(argv []string) bool {
	if len(argv) < 2 {
		return false
	}
	switch argv[1] {
	case "hil":
		return true
	case "validate":
		for _, arg := range argv[2:] {
			if arg == "--strix" || strings.HasPrefix(arg, "--strix=") {
				return true
			}
		}
	}
	return false
}

func capabilityAutoUpgradeCheckReceipt(data []byte, executable string) (bool, error) {
	var receipt selfUpdateReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return false, fmt.Errorf("decode self-update check receipt: %w", err)
	}
	if receipt.Schema != selfUpdateReceiptSchema || receipt.SchemaVersion != 1 {
		return false, fmt.Errorf("self-update check returned an unexpected receipt schema")
	}
	if !capabilityReceiptAttestsPrimary(receipt, executable) {
		return false, fmt.Errorf("self-update check receipt does not attest the executing binary")
	}
	switch receipt.Status {
	case "current":
		return false, nil
	case "stale", "divergent":
		if receipt.NewRevision == nil || !isFullGitCommit(strings.TrimSpace(*receipt.NewRevision)) {
			return false, fmt.Errorf("self-update check receipt does not attest a full target commit")
		}
		return true, nil
	default:
		return false, fmt.Errorf("self-update check receipt status is %q", receipt.Status)
	}
}

func capabilityReceiptAttestsPrimary(receipt selfUpdateReceipt, executable string) bool {
	for _, target := range receipt.Targets {
		if target.Role == "primary" && capabilityAutoUpgradeSamePath(target.Path, executable) {
			return true
		}
	}
	return false
}

func capabilityAutoUpgradeInstalledRevision(data []byte, executable string) (string, error) {
	var receipt selfUpdateReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return "", fmt.Errorf("decode self-update receipt: %w", err)
	}
	if receipt.Schema != selfUpdateReceiptSchema || receipt.SchemaVersion != 1 {
		return "", fmt.Errorf("self-update returned an unexpected receipt schema")
	}
	if receipt.Status != "updated" {
		return "", fmt.Errorf("self-update receipt status is %q, want updated", receipt.Status)
	}
	if receipt.Changed < 1 || receipt.NewRevision == nil || !isFullGitCommit(strings.TrimSpace(*receipt.NewRevision)) {
		return "", fmt.Errorf("self-update receipt does not attest an installed full commit")
	}
	if !capabilityReceiptAttestsPrimary(receipt, executable) {
		return "", fmt.Errorf("self-update receipt does not attest the requested primary target")
	}
	return strings.ToLower(strings.TrimSpace(*receipt.NewRevision)), nil
}

func capabilityAutoUpgradeSamePath(a, b string) bool {
	aa, errA := filepath.Abs(a)
	bb, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return false
	}
	aa, bb = filepath.Clean(aa), filepath.Clean(bb)
	if aInfo, err := os.Stat(aa); err == nil {
		if bInfo, err := os.Stat(bb); err == nil {
			return os.SameFile(aInfo, bInfo)
		}
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(aa, bb)
	}
	return aa == bb
}

func capabilityAutoUpgradeReexecEnvironment(base []string, commit string, parentPID int) []string {
	env := make([]string, 0, len(base)+1)
	for _, entry := range base {
		key := entry
		if i := strings.IndexByte(entry, '='); i >= 0 {
			key = entry[:i]
		}
		if strings.EqualFold(strings.TrimSpace(key), capabilityAutoUpgradeReexecEnv) {
			continue
		}
		env = append(env, entry)
	}
	return append(env, capabilityAutoUpgradeReexecEnv+"="+strings.ToLower(strings.TrimSpace(commit))+":"+strconv.Itoa(parentPID))
}

func parseCapabilityAutoUpgradeMarker(marker string) (string, int, bool) {
	commit, pidText, ok := strings.Cut(strings.TrimSpace(marker), ":")
	if !ok || !isFullGitCommit(commit) {
		return "", 0, false
	}
	pid, err := strconv.Atoi(pidText)
	if err != nil || pid < 1 {
		return "", 0, false
	}
	return strings.ToLower(commit), pid, true
}

func resolveCapabilityAutoUpgradeCheckout() (root, executable string, err error) {
	// Codex startup already owns the cheap, module-aware distinction between a
	// fak checkout and an arbitrary working directory. Capability admission uses
	// the same checkout and executing-binary identity.
	return codexFreshnessCheckout()
}

func runCapabilityAutoUpgradeProcess(executable string, args []string, stdin io.Reader, stdout, stderr io.Writer, env []string) error {
	cmd := exec.Command(executable, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr, cmd.Env = stdin, stdout, stderr, env
	return cmd.Run()
}

func writeCapabilityAutoUpgradeDetail(stderr io.Writer, detail string) {
	detail = strings.TrimSpace(detail)
	if detail != "" {
		fmt.Fprintln(stderr, detail)
	}
}

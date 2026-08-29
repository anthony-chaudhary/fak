package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const codexFreshnessReexecEnv = "FAK_CODEX_FRESHNESS_REEXEC"

const (
	codexFreshnessLeaseTTL = 6 * time.Hour
	codexFreshnessClaimTTL = 30 * time.Minute
)

type codexFreshnessLease struct {
	CheckedAt time.Time `json:"checked_at"`
}

var (
	codexFreshnessNow      = time.Now
	codexFreshnessCacheDir = os.UserCacheDir
)

type codexFreshnessVerdict uint8

const (
	codexFreshnessUnknown codexFreshnessVerdict = iota
	codexFreshnessFresh
	codexFreshnessBehind
)

type codexFreshnessAssessment struct {
	Verdict       codexFreshnessVerdict
	RunningCommit string
	TargetCommit  string
	Detail        string
}

type codexFreshnessInspection struct {
	Assessment codexFreshnessAssessment
	Err        error
}

var (
	codexFreshnessExecutable = os.Executable
	codexFreshnessGetwd      = os.Getwd
	codexFreshnessInspect    = func(root, _ string) codexFreshnessInspection {
		skew := versionskew.AssessStamp(context.Background(), versionskew.RealRunner, root, "origin/main", binstamp.Self())
		assessment := codexFreshnessAssessment{
			RunningCommit: skew.Running,
			TargetCommit:  skew.TrunkTip,
			Detail:        skew.Verdict.String(),
		}
		switch skew.Verdict {
		case versionskew.Fresh:
			assessment.Verdict = codexFreshnessFresh
		case versionskew.Skewed:
			if skew.Relation == versionskew.RelBehind {
				assessment.Verdict = codexFreshnessBehind
			}
		}
		if assessment.Verdict == codexFreshnessUnknown && assessment.Detail == "" {
			assessment.Detail = "launcher freshness is unverifiable (" + skew.Verdict.String() + ")"
		}
		return codexFreshnessInspection{Assessment: assessment}
	}
	codexFreshnessUpdate = func(root, executable string) error {
		cmd := exec.Command(executable, codexFreshnessSelfUpdateArgs(root, executable)...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		return cmd.Run()
	}
	codexFreshnessReexec = func(executable string, argv []string) error {
		cmd := exec.Command(executable, argv[1:]...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		cmd.Env = append(os.Environ(), codexFreshnessReexecEnv+"=1")
		return cmd.Run()
	}
	codexFreshnessStatus = func() *codexStartupStatus {
		return newCodexStartupStatus(os.Stderr, guardFdIsTerminal(int(os.Stderr.Fd())))
	}
	codexFreshnessResolveCheckout = codexFreshnessCheckout
)

func codexFreshnessSelfUpdateArgs(root, executable string) []string {
	return []string{"self-update", "--root", root, "--target", executable}
}

type codexStartupStatus struct {
	w           io.Writer
	interactive bool
	phase       string
}

func newCodexStartupStatus(w io.Writer, interactive bool) *codexStartupStatus {
	return &codexStartupStatus{w: w, interactive: interactive}
}

// Start owns the transient pre-provider surface. Callers update this one line
// instead of appending startup diagnostics that remain above the provider UI.
func (s *codexStartupStatus) Start(text string) {
	s.phase = text
	if s.interactive && s.w != nil {
		fmt.Fprintf(s.w, "\r\x1b[2K⠋ fak codex · %s", text)
	}
}

func (s *codexStartupStatus) Update(text string) {
	s.phase = text
	if s.w == nil {
		return
	}
	if s.interactive {
		fmt.Fprintf(s.w, "\r\x1b[2K⠙ fak codex · %s", text)
		return
	}
	fmt.Fprintln(s.w, "fak codex:", text)
}

func (s *codexStartupStatus) Stop() {
	if s.interactive && s.w != nil && s.phase != "" {
		fmt.Fprint(s.w, "\r\x1b[2K")
	}
}

// runCodexFreshnessAdmission ensures a checkout-local launcher evaluates admission
// from a current stamped binary before it starts an agent that can mutate the checkout.
func runCodexFreshnessAdmission(args []string) ([]string, int, bool) {
	filtered, checkNow, err := parseCodexFreshnessCheckNow(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak codex:", err)
		return nil, 2, true
	}
	filtered, enabled, err := parseCodexFreshnessMode(filtered)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak codex:", err)
		return nil, 2, true
	}
	if !enabled {
		return filtered, 0, false
	}
	status := codexFreshnessStatus()
	status.Start("checking launcher")
	defer status.Stop()

	root, executable, err := codexFreshnessResolveCheckout()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %v\n", err)
		return nil, 1, true
	}
	if root == "" {
		return filtered, 0, false
	}

	statePath, err := codexFreshnessStatePath(root, executable)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %v\n", err)
		return nil, 1, true
	}
	if !checkNow && codexFreshnessLeaseValid(statePath+".json", codexFreshnessNow()) {
		return filtered, 0, false
	}
	claimed, err := codexFreshnessAcquireClaim(statePath+".lock", codexFreshnessNow())
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %v\n", err)
		return nil, 1, true
	}
	if !claimed {
		status.Update("another launch is updating; using current launcher")
		return filtered, 0, false
	}
	defer os.Remove(statePath + ".lock")

	inspection := codexFreshnessInspect(root, executable)
	if inspection.Err != nil {
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %v\n", inspection.Err)
		return nil, 1, true
	}
	running, target := shortFreshnessID(inspection.Assessment.RunningCommit), shortFreshnessID(inspection.Assessment.TargetCommit)

	switch inspection.Assessment.Verdict {
	case codexFreshnessFresh:
		if err := codexFreshnessWriteLease(statePath+".json", codexFreshnessNow()); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: persist freshness lease: %v\n", err)
			return nil, 1, true
		}
		return filtered, 0, false
	case codexFreshnessBehind:
		if os.Getenv(codexFreshnessReexecEnv) != "" {
			fmt.Fprintln(os.Stderr, "fak codex: freshness admission refused: updated launcher is still stale (re-exec suppressed)")
			return nil, 1, true
		}
		status.Update(fmt.Sprintf("updating launcher %s -> %s at %s", running, target, executable))
		if err := codexFreshnessUpdate(root, executable); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: self-update failed: %v\n", err)
			return nil, 1, true
		}
		if err := codexFreshnessWriteLease(statePath+".json", codexFreshnessNow()); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: persist freshness lease: %v\n", err)
			return nil, 1, true
		}
		status.Update(fmt.Sprintf("launching updated build %s from %s", target, executable))
		argv := append([]string{executable, "codex"}, filtered...)
		if err := codexFreshnessReexec(executable, argv); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: re-exec failed: %v\n", err)
			return nil, 1, true
		}
		return nil, 0, true
	default:
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %s; use --freshness-gate off only as an explicit override\n", inspection.Assessment.Detail)
		return nil, 1, true
	}
}

func parseCodexFreshnessCheckNow(args []string) ([]string, bool, error) {
	filtered := make([]string, 0, len(args))
	checkNow := false
	for _, arg := range args {
		switch arg {
		case "--freshness-check-now":
			checkNow = true
		default:
			if strings.HasPrefix(arg, "--freshness-check-now=") {
				return nil, false, fmt.Errorf("--freshness-check-now does not take a value")
			}
			filtered = append(filtered, arg)
		}
	}
	return filtered, checkNow, nil
}

func codexFreshnessStatePath(root, executable string) (string, error) {
	cacheDir, err := codexFreshnessCacheDir()
	if err != nil {
		return "", fmt.Errorf("resolve freshness cache: %w", err)
	}
	key := strings.ToLower(filepath.Clean(root)) + "\x00" + strings.ToLower(filepath.Clean(executable))
	sum := sha256.Sum256([]byte(key))
	dir := filepath.Join(cacheDir, "fak", "codex-freshness")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("create freshness cache: %w", err)
	}
	return filepath.Join(dir, hex.EncodeToString(sum[:])), nil
}

func codexFreshnessLeaseValid(path string, now time.Time) bool {
	raw, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var lease codexFreshnessLease
	if json.Unmarshal(raw, &lease) != nil || lease.CheckedAt.IsZero() || lease.CheckedAt.After(now) {
		return false
	}
	return now.Sub(lease.CheckedAt) < codexFreshnessLeaseTTL
}

func codexFreshnessWriteLease(path string, now time.Time) error {
	raw, err := json.Marshal(codexFreshnessLease{CheckedAt: now.UTC()})
	if err != nil {
		return err
	}
	// A missing lease is conservative: the next launch checks again. Remove the
	// expired destination before the same-directory atomic rename so renewal also
	// works on Windows, where os.Rename does not replace an existing file.
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeFileAtomic(path, append(raw, '\n'), 0o600)
}

func codexFreshnessAcquireClaim(path string, now time.Time) (bool, error) {
	for attempt := 0; attempt < 2; attempt++ {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, writeErr := fmt.Fprintf(file, "%d\n", now.Unix())
			closeErr := file.Close()
			if writeErr != nil {
				_ = os.Remove(path)
				return false, writeErr
			}
			if closeErr != nil {
				_ = os.Remove(path)
				return false, closeErr
			}
			return true, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return false, fmt.Errorf("acquire freshness claim: %w", err)
		}
		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			if errors.Is(readErr, os.ErrNotExist) {
				continue
			}
			return false, fmt.Errorf("read freshness claim: %w", readErr)
		}
		info, statErr := os.Stat(path)
		if statErr != nil {
			if errors.Is(statErr, os.ErrNotExist) {
				continue
			}
			return false, fmt.Errorf("inspect freshness claim: %w", statErr)
		}
		if info.ModTime().After(now) || now.Sub(info.ModTime()) < codexFreshnessClaimTTL {
			return false, nil
		}
		// Link the exact stale inode to an identity-specific tombstone before
		// unlinking it. A competing reaper gets os.ErrExist, so it cannot delete a
		// replacement claim created after this stale observation. Tombstones are
		// deliberately retained: one tiny file per recovered crash buys a durable
		// compare-before-delete primitive on both Unix and Windows.
		tombstone := codexFreshnessStaleClaimPath(path, raw, info)
		if linkErr := os.Link(path, tombstone); linkErr != nil {
			if errors.Is(linkErr, os.ErrExist) || errors.Is(linkErr, os.ErrNotExist) {
				return false, nil
			}
			return false, fmt.Errorf("claim stale freshness marker: %w", linkErr)
		}
		linkedInfo, linkedErr := os.Stat(tombstone)
		currentInfo, currentErr := os.Stat(path)
		if linkedErr != nil || currentErr != nil || !os.SameFile(linkedInfo, currentInfo) {
			return false, nil
		}
		if removeErr := os.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			return false, fmt.Errorf("reap stale freshness claim: %w", removeErr)
		}
	}
	return false, nil
}

func codexFreshnessStaleClaimPath(path string, raw []byte, info os.FileInfo) string {
	identity := fmt.Sprintf("%x\x00%d\x00%d", raw, info.ModTime().UnixNano(), info.Size())
	sum := sha256.Sum256([]byte(identity))
	return path + ".stale-" + hex.EncodeToString(sum[:8])
}
func parseCodexFreshnessMode(args []string) ([]string, bool, error) {
	enabled := true
	filtered := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		value := ""
		switch {
		case arg == "--freshness-gate":
			if i+1 >= len(args) {
				return nil, false, fmt.Errorf("--freshness-gate requires on or off")
			}
			i++
			value = args[i]
		case strings.HasPrefix(arg, "--freshness-gate="):
			value = strings.TrimPrefix(arg, "--freshness-gate=")
		default:
			filtered = append(filtered, arg)
			continue
		}
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "on", "true", "1":
			enabled = true
		case "off", "false", "0":
			enabled = false
		default:
			return nil, false, fmt.Errorf("--freshness-gate must be on or off, got %q", value)
		}
	}
	return filtered, enabled, nil
}

func codexFreshnessCheckout() (root, executable string, err error) {
	cwd, err := codexFreshnessGetwd()
	if err != nil {
		return "", "", err
	}
	cmd := exec.Command("git", "-C", cwd, "rev-parse", "--show-toplevel")
	configureDispatchHelperCommand(cmd)
	out, gitErr := cmd.CombinedOutput()
	if gitErr != nil {
		if isNotGitRepository(gitErr, out) {
			return "", "", nil
		}
		return "", "", fmt.Errorf("resolve checkout: %w: %s", gitErr, strings.TrimSpace(string(out)))
	}
	root = strings.TrimSpace(string(out))
	module, readErr := os.ReadFile(filepath.Join(root, "go.mod"))
	if readErr != nil || !isFakModule(module) {
		// A module-installed fak may be launched from any unrelated checkout. Only the
		// fak development checkout opts into source freshness and its Git dependency.
		return "", "", nil
	}
	executable, err = codexFreshnessExecutable()
	if err != nil {
		return "", "", err
	}
	return root, filepath.Clean(executable), nil
}

func isFakModule(goMod []byte) bool {
	for _, line := range strings.Split(string(goMod), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "module" {
			return fields[1] == "github.com/anthony-chaudhary/fak"
		}
	}
	return false
}

func isNotGitRepository(err error, output []byte) bool {
	if err == nil {
		return false
	}
	text := strings.ToLower(err.Error() + " " + string(output))
	return strings.Contains(text, "not a git repository") || strings.Contains(text, "not a repository")
}

func shortFreshnessID(commit string) string {
	commit = strings.TrimSpace(commit)
	if commit == "" {
		return "unknown"
	}
	if len(commit) > 12 {
		return commit[:12]
	}
	return commit
}

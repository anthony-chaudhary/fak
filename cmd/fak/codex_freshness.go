package main

import (
	"bytes"
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
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const (
	codexFreshnessReexecEnv     = "FAK_CODEX_FRESHNESS_REEXEC"
	codexFreshnessMaxAgeEnv     = "FAK_CODEX_FRESHNESS_MAX_AGE"
	codexFreshnessForceEnv      = "FAK_CODEX_FRESHNESS_FORCE"
	codexFreshnessReceiptSchema = "fak.codex-freshness.v1"
)

const (
	codexFreshnessLeaseTTL = 6 * time.Hour
	codexFreshnessClaimTTL = 30 * time.Minute
)

type codexFreshnessConfig struct {
	MaxAge string `json:"max_age"`
	Force  *bool  `json:"force,omitempty"`
}

type codexFreshnessLease struct {
	Schema        string    `json:"schema"`
	CheckedAt     time.Time `json:"checked_at"`
	RunningCommit string    `json:"running_commit"`
	TargetCommit  string    `json:"target_commit"`
}

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
	codexFreshnessNow           = time.Now
	codexFreshnessCacheDir      = os.UserCacheDir
	codexFreshnessExecutable    = os.Executable
	codexFreshnessGetwd         = os.Getwd
	codexFreshnessUserConfigDir = os.UserConfigDir
	codexFreshnessRunningCommit = func() string { return strings.TrimSpace(binstamp.Self().Revision) }
	codexFreshnessInspect       = func(root, _ string) codexFreshnessInspection {
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
	codexFreshnessUpdate = func(root, executable string) (string, error) {
		cmd := exec.Command(executable, codexFreshnessSelfUpdateArgs(root, executable)...)
		var receipt bytes.Buffer
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, &receipt, os.Stderr
		runErr := cmd.Run()
		installed, receiptErr := codexFreshnessInstalledRevision(receipt.Bytes(), executable)
		if receiptErr != nil {
			if runErr != nil {
				return "", fmt.Errorf("%w: %v", runErr, receiptErr)
			}
			return "", receiptErr
		}
		if runErr != nil {
			return "", runErr
		}
		return installed, nil
	}
	codexFreshnessReexec    = runCodexFreshnessReexec
	codexFreshnessParentPID = os.Getppid
	codexFreshnessStatus    = func() *codexStartupStatus {
		return newCodexStartupStatus(os.Stderr, guardFdIsTerminal(int(os.Stderr.Fd())))
	}
	codexFreshnessResolveCheckout = codexFreshnessCheckout
)

func codexFreshnessSelfUpdateArgs(root, executable string) []string {
	return []string{"self-update", "--json", "--root", root, "--target", executable}
}

func runCodexFreshnessReexec(executable string, argv []string, expectedCommit string) error {
	cmd := exec.Command(executable, argv[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = codexFreshnessReexecEnvironment(os.Environ(), expectedCommit, os.Getpid())
	return cmd.Run()
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
	config, err := loadCodexFreshnessConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak codex:", err)
		return nil, 2, true
	}
	filtered, policy, err := parseCodexFreshnessSettings(args, os.Getenv(codexFreshnessMaxAgeEnv), os.Getenv(codexFreshnessForceEnv), config)
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
	runningCommit := strings.TrimSpace(codexFreshnessRunningCommit())
	if !policy.Force && codexFreshnessLeaseValidFor(statePath+".json", codexFreshnessNow(), policy.MaxAge, runningCommit) {
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
		consumeCodexFreshnessReexecMarker()
		if err := codexFreshnessWriteReceipt(statePath+".json", codexFreshnessNow(), inspection.Assessment.RunningCommit, inspection.Assessment.TargetCommit); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: persist freshness lease: %v\n", err)
			return nil, 1, true
		}
		return filtered, 0, false
	case codexFreshnessBehind:
		if os.Getenv(codexFreshnessReexecEnv) != "" {
			if codexFreshnessMatchesReexecTarget(inspection.Assessment) {
				consumeCodexFreshnessReexecMarker()
				return filtered, 0, false
			}
			fmt.Fprintln(os.Stderr, "fak codex: freshness admission refused: updated launcher is still stale (re-exec suppressed)")
			return nil, 1, true
		}
		status.Update(fmt.Sprintf("updating launcher %s -> %s at %s", running, target, executable))
		installedCommit, err := codexFreshnessUpdate(root, executable)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: self-update failed: %v\n", err)
			return nil, 1, true
		}
		if err := codexFreshnessWriteReceipt(statePath+".json", codexFreshnessNow(), installedCommit, installedCommit); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: persist freshness lease: %v\n", err)
			return nil, 1, true
		}
		status.Update(fmt.Sprintf("launching updated build %s from %s", shortFreshnessID(installedCommit), executable))
		argv := append([]string{executable, "codex"}, filtered...)
		if err := codexFreshnessReexec(executable, argv, installedCommit); err != nil {
			fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: re-exec failed: %v\n", err)
			return nil, 1, true
		}
		return nil, 0, true
	default:
		fmt.Fprintf(os.Stderr, "fak codex: freshness admission refused: %s; use --freshness-gate off only as an explicit override\n", inspection.Assessment.Detail)
		return nil, 1, true
	}
}
func codexFreshnessMatchesReexecTarget(assessment codexFreshnessAssessment) bool {
	expected, parentPID, ok := parseCodexFreshnessReexecMarker(os.Getenv(codexFreshnessReexecEnv))
	if !ok || parentPID != codexFreshnessParentPID() {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(assessment.RunningCommit), expected)
}

func consumeCodexFreshnessReexecMarker() {
	_ = os.Unsetenv(codexFreshnessReexecEnv)
}

func codexFreshnessInstalledRevision(data []byte, executable string) (string, error) {
	var receipt selfUpdateReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return "", fmt.Errorf("decode self-update receipt: %w", err)
	}
	if receipt.Schema != selfUpdateReceiptSchema || receipt.SchemaVersion != 1 {
		return "", fmt.Errorf("self-update returned an unexpected receipt schema")
	}
	if receipt.Status != "updated" {
		if detail := strings.TrimSpace(receipt.Detail); detail != "" {
			return "", fmt.Errorf("self-update receipt status is %q: %s", receipt.Status, detail)
		}
		return "", fmt.Errorf("self-update receipt status is %q, want updated", receipt.Status)
	}
	if receipt.Changed < 1 || receipt.NewRevision == nil || !isFullGitCommit(strings.TrimSpace(*receipt.NewRevision)) {
		return "", fmt.Errorf("self-update receipt does not attest an installed full commit")
	}
	wantTarget := filepath.Clean(executable)
	primaryMatched := false
	for _, target := range receipt.Targets {
		if target.Role == "primary" && strings.EqualFold(filepath.Clean(target.Path), wantTarget) {
			primaryMatched = true
			break
		}
	}
	if !primaryMatched {
		return "", fmt.Errorf("self-update receipt does not attest the requested primary target")
	}
	return strings.ToLower(strings.TrimSpace(*receipt.NewRevision)), nil
}

func codexFreshnessReexecEnvironment(base []string, expectedCommit string, parentPID int) []string {
	// This is a one-generation lifecycle guard, not a same-user security boundary: an
	// operator who controls this process's environment can already pass the documented
	// --freshness-gate=off escape. The exact commit + direct-parent binding prevents stale
	// inherited markers and nested fak launches from accidentally reusing the admission.
	env := make([]string, 0, len(base)+1)
	for _, entry := range base {
		key := entry
		if i := strings.IndexByte(entry, '='); i >= 0 {
			key = entry[:i]
		}
		if strings.EqualFold(strings.TrimSpace(key), codexFreshnessReexecEnv) {
			continue
		}
		env = append(env, entry)
	}
	return append(env, codexFreshnessReexecEnv+"="+strings.TrimSpace(expectedCommit)+":"+strconv.Itoa(parentPID))
}

func parseCodexFreshnessReexecMarker(marker string) (string, int, bool) {
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

func isFullGitCommit(value string) bool {
	if len(value) != 40 {
		return false
	}
	for _, r := range value {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') && (r < 'A' || r > 'F') {
			return false
		}
	}
	return true
}

func loadCodexFreshnessConfig() (codexFreshnessConfig, error) {
	dir, err := codexFreshnessUserConfigDir()
	if err != nil || strings.TrimSpace(dir) == "" {
		return codexFreshnessConfig{}, nil
	}
	path := filepath.Join(dir, "fak", "codex-freshness.json")
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return codexFreshnessConfig{}, nil
	}
	if err != nil {
		return codexFreshnessConfig{}, fmt.Errorf("read freshness config %s: %w", path, err)
	}
	var config codexFreshnessConfig
	if err := json.Unmarshal(raw, &config); err != nil {
		return codexFreshnessConfig{}, fmt.Errorf("parse freshness config %s: %w", path, err)
	}
	return config, nil
}

type codexFreshnessSettings struct {
	MaxAge time.Duration
	Force  bool
}

func parseCodexFreshnessSettings(args []string, envMaxAge, envForce string, config codexFreshnessConfig) ([]string, codexFreshnessSettings, error) {
	policy := codexFreshnessSettings{MaxAge: codexFreshnessLeaseTTL}
	if raw := strings.TrimSpace(config.MaxAge); raw != "" {
		d, err := time.ParseDuration(raw)
		if err != nil || d < 0 {
			return nil, policy, fmt.Errorf("freshness config max_age must be a non-negative duration, got %q", raw)
		}
		policy.MaxAge = d
	}
	if config.Force != nil {
		policy.Force = *config.Force
	}
	if raw := strings.TrimSpace(envMaxAge); raw != "" {
		d, err := time.ParseDuration(raw)
		if err != nil || d < 0 {
			return nil, policy, fmt.Errorf("%s must be a non-negative duration, got %q", codexFreshnessMaxAgeEnv, raw)
		}
		policy.MaxAge = d
	}
	if raw := strings.TrimSpace(envForce); raw != "" {
		force, err := strconv.ParseBool(raw)
		if err != nil {
			return nil, policy, fmt.Errorf("%s must be a boolean, got %q", codexFreshnessForceEnv, raw)
		}
		policy.Force = force
	}
	filtered := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--freshness-check-now" || arg == "--freshness-force":
			policy.Force = true
		case strings.HasPrefix(arg, "--freshness-check-now=") || strings.HasPrefix(arg, "--freshness-force="):
			return nil, policy, fmt.Errorf("%s does not take a value", strings.SplitN(arg, "=", 2)[0])
		case arg == "--freshness-max-age":
			if i+1 >= len(args) {
				return nil, policy, fmt.Errorf("--freshness-max-age requires a duration")
			}
			i++
			d, err := time.ParseDuration(args[i])
			if err != nil || d < 0 {
				return nil, policy, fmt.Errorf("--freshness-max-age must be a non-negative duration, got %q", args[i])
			}
			policy.MaxAge = d
		case strings.HasPrefix(arg, "--freshness-max-age="):
			raw := strings.TrimPrefix(arg, "--freshness-max-age=")
			d, err := time.ParseDuration(raw)
			if err != nil || d < 0 {
				return nil, policy, fmt.Errorf("--freshness-max-age must be a non-negative duration, got %q", raw)
			}
			policy.MaxAge = d
		default:
			filtered = append(filtered, arg)
		}
	}
	return filtered, policy, nil
}

func parseCodexFreshnessCheckNow(args []string) ([]string, bool, error) {
	filtered, policy, err := parseCodexFreshnessSettings(args, "", "", codexFreshnessConfig{})
	return filtered, policy.Force, err
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
	return codexFreshnessLeaseValidFor(path, now, codexFreshnessLeaseTTL, codexFreshnessRunningCommit())
}

func codexFreshnessLeaseValidFor(path string, now time.Time, maxAge time.Duration, runningCommit string) bool {
	raw, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var lease codexFreshnessLease
	if json.Unmarshal(raw, &lease) != nil || lease.Schema != codexFreshnessReceiptSchema || lease.CheckedAt.IsZero() || lease.CheckedAt.After(now) || maxAge <= 0 {
		return false
	}
	running := strings.TrimSpace(runningCommit)
	leaseRunning := strings.TrimSpace(lease.RunningCommit)
	leaseTarget := strings.TrimSpace(lease.TargetCommit)
	if running == "" || len(running) != 40 || len(leaseRunning) != 40 || len(leaseTarget) != 40 || !strings.EqualFold(running, leaseRunning) || !strings.EqualFold(leaseRunning, leaseTarget) {
		return false
	}
	return now.Sub(lease.CheckedAt) < maxAge
}

func codexFreshnessWriteLease(path string, now time.Time) error {
	return codexFreshnessWriteReceipt(path, now, "", "")
}

func codexFreshnessWriteReceipt(path string, now time.Time, runningCommit, targetCommit string) error {
	raw, err := json.Marshal(codexFreshnessLease{
		Schema: codexFreshnessReceiptSchema, CheckedAt: now.UTC(),
		RunningCommit: strings.TrimSpace(runningCommit), TargetCommit: strings.TrimSpace(targetCommit),
	})
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeFileAtomic(path, append(raw, '\n'), 0o600)
}

func codexFreshnessAcquireClaim(path string, now time.Time) (bool, error) {
	for attempt := 0; attempt < 2; attempt++ {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, we := fmt.Fprintf(file, "%d\n", now.Unix())
			ce := file.Close()
			if we != nil {
				_ = os.Remove(path)
				return false, we
			}
			if ce != nil {
				_ = os.Remove(path)
				return false, ce
			}
			return true, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return false, fmt.Errorf("acquire freshness claim: %w", err)
		}
		raw, re := os.ReadFile(path)
		if re != nil {
			if errors.Is(re, os.ErrNotExist) {
				continue
			}
			return false, fmt.Errorf("read freshness claim: %w", re)
		}
		info, se := os.Stat(path)
		if se != nil {
			if errors.Is(se, os.ErrNotExist) {
				continue
			}
			return false, fmt.Errorf("inspect freshness claim: %w", se)
		}
		if info.ModTime().After(now) || now.Sub(info.ModTime()) < codexFreshnessClaimTTL {
			return false, nil
		}
		tomb := codexFreshnessStaleClaimPath(path, raw, info)
		if le := os.Link(path, tomb); le != nil {
			if errors.Is(le, os.ErrExist) || errors.Is(le, os.ErrNotExist) {
				return false, nil
			}
			return false, fmt.Errorf("claim stale freshness marker: %w", le)
		}
		li, le := os.Stat(tomb)
		ci, ce := os.Stat(path)
		if le != nil || ce != nil || !os.SameFile(li, ci) {
			return false, nil
		}
		if de := os.Remove(path); de != nil && !errors.Is(de, os.ErrNotExist) {
			return false, fmt.Errorf("reap stale freshness claim: %w", de)
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

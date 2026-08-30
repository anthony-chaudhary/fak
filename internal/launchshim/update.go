package launchshim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	UpdatePolicyPrior = "prior"
	UpdatePolicyWait  = "wait"
	UpdatePolicyFail  = "fail"

	UpdateStateSchema = "fak.self-update.launch.v1"

	defaultUpdateWait = 10 * time.Second
	maxUpdateWait     = 5 * time.Minute
)

var (
	updateNow   = time.Now
	updateSleep = time.Sleep
)

// UpdateState is the durable replacement-boundary record read by fak-launch.
// The prior executable is fully flushed before this record is published.
type UpdateState struct {
	Schema    string `json:"schema"`
	Target    string `json:"target"`
	Prior     string `json:"prior"`
	StartedAt string `json:"started_at"`
}

type stableBinding struct {
	Schema     string `json:"schema"`
	Executable string `json:"executable"`
}

const stableBindingSchema = "fak.launch-target.v1"

// UpdateStatePath is deterministic so the stable launcher can check the
// replacement boundary without consulting the updating process.
func UpdateStatePath(target string) string {
	return filepath.Clean(target) + ".self-update-active.json"
}

func stableBindingPath(stable string) string {
	return filepath.Clean(stable) + ".target.json"
}

// BindStableExecutable records which replaceable fak binary a stable launcher
// should execute. It is separate from launch.json so repairLaunchShims can
// restore the binding without rewriting provider configuration.
func BindStableExecutable(stable, executable string) error {
	stable, err := filepath.Abs(filepath.Clean(stable))
	if err != nil {
		return err
	}
	executable, err = filepath.Abs(filepath.Clean(executable))
	if err != nil {
		return err
	}
	if SameCommand(stable, executable) {
		return errors.New("stable launcher target must differ from the replaceable executable")
	}
	return writeAtomicJSON(stableBindingPath(stable), stableBinding{
		Schema:     stableBindingSchema,
		Executable: executable,
	})
}

// StableExecutable reads the replaceable executable bound to stable.
func StableExecutable(stable string) (string, error) {
	path := stableBindingPath(stable)
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var binding stableBinding
	if err := json.Unmarshal(b, &binding); err != nil {
		return "", fmt.Errorf("parse %s: %w", path, err)
	}
	if binding.Schema != stableBindingSchema {
		return "", fmt.Errorf("unsupported stable launch target schema %q", binding.Schema)
	}
	executable := strings.TrimSpace(binding.Executable)
	if executable == "" {
		return "", errors.New("stable launch target records no executable")
	}
	if !filepath.IsAbs(executable) {
		return "", errors.New("stable launch target executable is not absolute")
	}
	return filepath.Clean(executable), nil
}

// WriteUpdateState atomically publishes an active replacement boundary.
func WriteUpdateState(target, prior string) error {
	target, err := filepath.Abs(filepath.Clean(target))
	if err != nil {
		return err
	}
	prior, err = filepath.Abs(filepath.Clean(prior))
	if err != nil {
		return err
	}
	return writeAtomicJSON(UpdateStatePath(target), UpdateState{
		Schema:    UpdateStateSchema,
		Target:    target,
		Prior:     prior,
		StartedAt: updateNow().UTC().Format(time.RFC3339Nano),
	})
}

// UpdatePolicy resolves flag > config > default. The wait is
// positive and capped even when configuration is hostile.
func UpdatePolicy(c Config, flagPolicy, flagWait string) (string, time.Duration, error) {
	policy := strings.TrimSpace(flagPolicy)
	if policy == "" {
		policy = strings.TrimSpace(c.UpdateLaunchPolicy)
	}
	if policy == "" {
		policy = UpdatePolicyPrior
	}
	policy = strings.ToLower(policy)
	if !validUpdatePolicy(policy) {
		return "", 0, fmt.Errorf("update launch policy %q must be prior, wait, or fail", policy)
	}

	wait := defaultUpdateWait
	if c.UpdateLaunchWaitMS < 0 {
		return "", 0, errors.New("update_launch_wait_ms must not be negative")
	}
	if c.UpdateLaunchWaitMS > 0 {
		if c.UpdateLaunchWaitMS >= int(maxUpdateWait/time.Millisecond) {
			wait = maxUpdateWait
		} else {
			wait = time.Duration(c.UpdateLaunchWaitMS) * time.Millisecond
		}
	}
	rawWait := strings.TrimSpace(flagWait)
	source := "--update-launch-wait"
	if rawWait != "" {
		parsed, err := time.ParseDuration(rawWait)
		if err != nil || parsed <= 0 {
			return "", 0, fmt.Errorf("%s must be a positive Go duration", source)
		}
		wait = parsed
	}
	if wait > maxUpdateWait {
		wait = maxUpdateWait
	}
	return policy, wait, nil
}

func validUpdatePolicy(policy string) bool {
	switch strings.ToLower(strings.TrimSpace(policy)) {
	case "", UpdatePolicyPrior, UpdatePolicyWait, UpdatePolicyFail:
		return true
	default:
		return false
	}
}

// ResolveExecutable chooses the deployed or last-known-good executable. A
// missing state file is the ordinary lock-free launch path.
func ResolveExecutable(target, policy string, wait time.Duration) (string, error) {
	target, err := filepath.Abs(filepath.Clean(target))
	if err != nil {
		return "", err
	}
	statePath := UpdateStatePath(target)
	state, err := readUpdateState(target)
	if errors.Is(err, os.ErrNotExist) {
		return runnableFile(target, "deployed fak executable")
	}
	if err != nil {
		return "", fmt.Errorf("read self-update transaction: %w", err)
	}

	switch policy {
	case UpdatePolicyPrior:
		return runnableFile(state.Prior, "last known-good fak executable")
	case UpdatePolicyFail:
		return "", fmt.Errorf("self-update is replacing %s; retry after it completes or choose prior/wait", filepath.Base(target))
	case UpdatePolicyWait:
		if wait <= 0 {
			return "", errors.New("self-update wait must be positive and bounded")
		}
		deadline := updateNow().Add(wait)
		for {
			_, statErr := os.Stat(statePath)
			if errors.Is(statErr, os.ErrNotExist) {
				return runnableFile(target, "updated fak executable")
			}
			if statErr != nil {
				return "", fmt.Errorf("inspect self-update transaction: %w", statErr)
			}
			now := updateNow()
			if !now.Before(deadline) {
				return "", fmt.Errorf("self-update did not finish within %s; retry or choose prior", wait)
			}
			sleep := 10 * time.Millisecond
			if remaining := deadline.Sub(now); remaining < sleep {
				sleep = remaining
			}
			updateSleep(sleep)
		}
	default:
		return "", fmt.Errorf("invalid update launch policy %s", strconv.Quote(policy))
	}
}

func readUpdateState(target string) (UpdateState, error) {
	path := UpdateStatePath(target)
	b, err := os.ReadFile(path)
	if err != nil {
		return UpdateState{}, err
	}
	var state UpdateState
	if err := json.Unmarshal(b, &state); err != nil {
		return UpdateState{}, err
	}
	if state.Schema != UpdateStateSchema {
		return UpdateState{}, fmt.Errorf("unsupported schema %q", state.Schema)
	}
	stateTarget := strings.TrimSpace(state.Target)
	statePrior := strings.TrimSpace(state.Prior)
	if stateTarget == "" || statePrior == "" {
		return UpdateState{}, errors.New("incomplete update state")
	}
	stateTarget, err = filepath.Abs(filepath.Clean(stateTarget))
	if err != nil {
		return UpdateState{}, err
	}
	if !sameUpdatePath(stateTarget, target) {
		return UpdateState{}, fmt.Errorf("transaction target %q does not match %q", stateTarget, target)
	}
	state.Prior, err = filepath.Abs(filepath.Clean(statePrior))
	if err != nil {
		return UpdateState{}, err
	}
	return state, nil
}

func sameUpdatePath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func runnableFile(path, role string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("%s %q: %w", role, path, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s %q is a directory", role, path)
	}
	return filepath.Clean(path), nil
}

func writeAtomicJSON(path string, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return replaceConfig(tmpName, path)
}

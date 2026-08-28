package launchshim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	UpdatePolicyPrior = "prior"
	UpdatePolicyWait  = "wait"
	UpdatePolicyFail  = "fail"
	defaultUpdateWait = 10 * time.Second
)

type updateState struct {
	Target string `json:"target"`
	Prior  string `json:"prior"`
}

func UpdateStatePath(target string) string {
	return filepath.Clean(target) + ".self-update-active.json"
}

func UpdatePolicy(c Config, flagValue string) (string, time.Duration, error) {
	policy := strings.TrimSpace(flagValue)
	if policy == "" {
		policy = strings.TrimSpace(os.Getenv("FAK_UPDATE_LAUNCH_POLICY"))
	}
	if policy == "" {
		policy = strings.TrimSpace(c.UpdateLaunchPolicy)
	}
	if policy == "" {
		policy = UpdatePolicyPrior
	}
	policy = strings.ToLower(policy)
	if policy != UpdatePolicyPrior && policy != UpdatePolicyWait && policy != UpdatePolicyFail {
		return "", 0, fmt.Errorf("update launch policy %q must be prior, wait, or fail", policy)
	}
	wait := defaultUpdateWait
	if c.UpdateLaunchWaitMS > 0 {
		wait = time.Duration(c.UpdateLaunchWaitMS) * time.Millisecond
	}
	if raw := strings.TrimSpace(os.Getenv("FAK_UPDATE_LAUNCH_WAIT")); raw != "" {
		parsed, err := time.ParseDuration(raw)
		if err != nil || parsed <= 0 {
			return "", 0, fmt.Errorf("FAK_UPDATE_LAUNCH_WAIT must be a positive duration")
		}
		wait = parsed
	}
	// Keep the stable entry point bounded even with hostile configuration.
	if wait > 5*time.Minute {
		wait = 5 * time.Minute
	}
	return policy, wait, nil
}

// ResolveExecutable chooses the deployed or last-known-good executable. A missing
// state file is the ordinary, lock-free launch path.
func ResolveExecutable(target, policy string, wait time.Duration) (string, error) {
	statePath := UpdateStatePath(target)
	read := func() (updateState, error) {
		b, err := os.ReadFile(statePath)
		if err != nil {
			return updateState{}, err
		}
		var s updateState
		if err := json.Unmarshal(b, &s); err != nil {
			return s, err
		}
		if strings.TrimSpace(s.Target) == "" || strings.TrimSpace(s.Prior) == "" {
			return s, errors.New("incomplete update state")
		}
		return s, nil
	}
	s, err := read()
	if errors.Is(err, os.ErrNotExist) {
		return target, nil
	}
	if err != nil {
		return "", fmt.Errorf("read self-update transaction: %w", err)
	}
	switch policy {
	case UpdatePolicyPrior:
		return s.Prior, nil
	case UpdatePolicyFail:
		return "", fmt.Errorf("self-update is replacing %s; retry after it completes or use policy prior/wait", filepath.Base(target))
	case UpdatePolicyWait:
		deadline := time.Now().Add(wait)
		for time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
			if _, err := os.Stat(statePath); errors.Is(err, os.ErrNotExist) {
				return target, nil
			}
		}
		return "", fmt.Errorf("self-update did not finish within %s", wait)
	default:
		return "", fmt.Errorf("invalid update launch policy %s", strconv.Quote(policy))
	}
}

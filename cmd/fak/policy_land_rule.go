package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

const landRulePreimageSuffix = ".land-rule.preimage"

var landRuleHTTPClient = &http.Client{Timeout: 30 * time.Second}

type landRuleCandidate struct {
	Version    string           `json:"version,omitempty"`
	ArgRules   []policy.ArgRule `json:"arg_rules,omitempty"`
	SelfModify bool             `json:"self_modify,omitempty"`
}

func runPolicyLandRule(policyPath, candidatePath, reloadURL string, land, rollback bool, out io.Writer) error {
	if policyPath == "" {
		return errors.New("--policy is required")
	}
	preimage := policyPath + landRulePreimageSuffix
	if rollback {
		if candidatePath != "" || land {
			return errors.New("--rollback cannot be combined with --candidate or --land")
		}
		before, err := os.ReadFile(preimage)
		if err != nil {
			return fmt.Errorf("read preimage: %w", err)
		}
		if err := writePolicyAtomic(policyPath, before); err != nil {
			return err
		}
		if err := reloadLandedPolicy(reloadURL); err != nil {
			return err
		}
		fmt.Fprintf(out, "restored %s from recorded preimage\n", policyPath)
		return nil
	}
	if candidatePath == "" {
		return errors.New("--candidate is required")
	}
	raw, err := os.ReadFile(candidatePath)
	if err != nil {
		return fmt.Errorf("read candidate: %w", err)
	}
	var c landRuleCandidate
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return fmt.Errorf("candidate must contain only version, arg_rules, and self_modify: %w", err)
	}
	if c.SelfModify {
		return errors.New("SELF_MODIFY candidate refused: route through the require-witness rung")
	}
	if len(c.ArgRules) == 0 {
		return errors.New("candidate contains no arg_rules")
	}
	current, err := os.ReadFile(policyPath)
	if err != nil {
		return fmt.Errorf("read policy: %w", err)
	}
	manifest, err := policy.ParseManifest(current)
	if err != nil {
		return err
	}
	manifest.ArgRules = append(manifest.ArgRules, c.ArgRules...)
	if _, err := manifest.ToPolicy(); err != nil {
		return fmt.Errorf("merged policy invalid: %w", err)
	}
	merged := manifest.JSON()
	if !land {
		_, err = out.Write(merged)
		return err
	}
	if _, err := os.Stat(preimage); errors.Is(err, os.ErrNotExist) {
		if err := writePolicyAtomic(preimage, current); err != nil {
			return fmt.Errorf("record preimage: %w", err)
		}
	} else if err != nil {
		return fmt.Errorf("inspect preimage: %w", err)
	}
	if err := writePolicyAtomic(policyPath, merged); err != nil {
		return err
	}
	if err := reloadLandedPolicy(reloadURL); err != nil {
		_ = writePolicyAtomic(policyPath, current)
		return err
	}
	fmt.Fprintf(out, "landed %d arg rule(s); preimage=%s\n", len(c.ArgRules), preimage)
	return nil
}

func writePolicyAtomic(path string, data []byte) error {
	f, err := os.CreateTemp(filepathDir(path), ".fak-policy-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err = f.Write(data); err == nil {
		err = f.Close()
	} else {
		_ = f.Close()
	}
	if err != nil {
		return err
	}
	if err = os.Rename(name, path); err != nil {
		return fmt.Errorf("replace policy: %w", err)
	}
	return nil
}

func filepathDir(path string) string {
	if i := len(path) - 1; i >= 0 {
		for ; i >= 0; i-- {
			if path[i] == '/' || path[i] == '\\' {
				if i == 0 {
					return string(path[0])
				}
				return path[:i]
			}
		}
	}
	return "."
}

func reloadLandedPolicy(url string) error {
	if url == "" {
		return errors.New("--reload-url is required for --land/--rollback")
	}
	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		return err
	}
	resp, err := landRuleHTTPClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("reload policy timeout: %w", err)
		}
		return fmt.Errorf("reload policy: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("reload policy: HTTP %s", resp.Status)
	}
	return nil
}

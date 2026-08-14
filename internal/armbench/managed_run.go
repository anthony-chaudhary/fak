package armbench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type ManagedRunOptions struct {
	UpstreamDir string
	Tasks       []string
	Treatments  []string
	Arms        []ManagedArm
	Model       string
	Runs        int
	Workers     int
	DryRun      bool
	ReceiptPath string
	UpstreamURL string
}

type ManagedRunReceipt struct {
	Schema                   string         `json:"schema"`
	ProviderBacked           bool           `json:"provider_backed"`
	Comparator               string         `json:"comparator"`
	UpstreamGitHead          string         `json:"upstream_git_head"`
	Treatments               []string       `json:"treatments"`
	ManagedArms              []ManagedArm   `json:"managed_arms"`
	Tasks                    []string       `json:"tasks"`
	Model                    string         `json:"model"`
	Runs                     int            `json:"runs"`
	Workers                  int            `json:"workers"`
	AccountIdentity          string         `json:"account_identity"`
	CredentialValuesRecorded bool           `json:"credential_values_recorded"`
	ExcludedFeatures         ManagedToggles `json:"excluded_features"`
	Commands                 []string       `json:"commands"`
	RunDirs                  []string       `json:"run_dirs,omitempty"`
	ProxyReceipts            []ProxyReceipt `json:"proxy_receipts,omitempty"`
}

func RunManagedMatrix(ctx context.Context, o ManagedRunOptions) (ManagedRunReceipt, error) {
	r := ManagedRunReceipt{Schema: "fak-armbench-ponytail-managed/1", ProviderBacked: !o.DryRun, Comparator: "DietrichGebert/ponytail@" + PonytailRevision, CredentialValuesRecorded: false}
	if o.UpstreamDir == "" {
		return r, fmt.Errorf("upstream dir required")
	}
	headCmd := exec.Command("git", "-C", o.UpstreamDir, "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(headCmd)
	head, err := headCmd.Output()
	if err != nil {
		return r, fmt.Errorf("read pinned checkout: %w", err)
	}
	r.UpstreamGitHead = strings.TrimSpace(string(head))
	if r.UpstreamGitHead != PonytailRevision {
		return r, fmt.Errorf("pinned Ponytail mismatch: got %s", r.UpstreamGitHead)
	}
	if len(o.Tasks) == 0 {
		o.Tasks = []string{"todo-null", "safe-path", "critic-email", "rate-limit"}
	}
	if len(o.Treatments) == 0 {
		o.Treatments = []string{"baseline", "caveman", "ponytail"}
	}
	if len(o.Arms) == 0 {
		o.Arms = ManagedArms()
	}
	if o.Model == "" {
		o.Model = "haiku"
	}
	if o.Runs < 1 {
		o.Runs = 1
	}
	if o.Workers < 1 {
		o.Workers = 1
	}
	if o.UpstreamURL == "" {
		o.UpstreamURL = "https://api.anthropic.com"
	}
	r.Treatments = append([]string(nil), o.Treatments...)
	r.ManagedArms = append([]ManagedArm(nil), o.Arms...)
	r.Tasks = append([]string(nil), o.Tasks...)
	r.Model = o.Model
	r.Runs = o.Runs
	r.Workers = o.Workers
	r.ExcludedFeatures = ManagedToggles{Routing: false, Policy: false, ResponseReuse: false}
	if !o.DryRun {
		id := strings.TrimSpace(os.Getenv("FAK_PROVIDER_ACCOUNT_IDENTITY"))
		if id == "" {
			return r, fmt.Errorf("FAK_PROVIDER_ACCOUNT_IDENTITY is required (stable account label, never a secret)")
		}
		r.AccountIdentity = id
	}
	runRoot := filepath.Join(o.UpstreamDir, "benchmarks", "agentic", "runs")
	for _, arm := range o.Arms {
		toggles, e := TogglesForManagedArm(arm)
		if e != nil {
			return r, e
		}
		if toggles.Routing || toggles.Policy || toggles.ResponseReuse {
			return r, fmt.Errorf("excluded feature enabled in %s", arm)
		}
		for _, treatment := range o.Treatments {
			args := []string{"benchmarks/agentic/run.py", "--task", strings.Join(o.Tasks, ","), "--arms", treatment, "--model", o.Model, "--runs", fmt.Sprint(o.Runs), "--workers", fmt.Sprint(o.Workers)}
			r.Commands = append(r.Commands, "python3 "+strings.Join(args, " ")+" # managed-arm="+string(arm))
			if o.DryRun {
				continue
			}
			before, _ := runDirNames(runRoot)
			python := "python3"
			if runtime.GOOS == "windows" {
				python = "python"
			}
			cmd := exec.CommandContext(ctx, python, args...)
			windowgate.ConfigureBackgroundCommand(cmd)
			cmd.Dir = o.UpstreamDir
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Env = os.Environ()
			var stop func() (ProxyReceipt, error)
			if arm != ManagedDirect {
				base, s, e := StartManagedProxy(ctx, arm, o.UpstreamURL)
				if e != nil {
					return r, e
				}
				stop = s
				cmd.Env = append(cmd.Env, "ANTHROPIC_BASE_URL="+base)
			}
			e := cmd.Run()
			if stop != nil {
				pr, se := stop()
				r.ProxyReceipts = append(r.ProxyReceipts, pr)
				if e == nil {
					e = se
				}
			}
			if e != nil {
				return r, fmt.Errorf("%s/%s: %w", treatment, arm, e)
			}
			after, _ := runDirNames(runRoot)
			for name := range after {
				if !before[name] {
					r.RunDirs = append(r.RunDirs, filepath.ToSlash(filepath.Join("benchmarks/agentic/runs", name)))
				}
			}
		}
	}
	sort.Strings(r.RunDirs)
	if o.ReceiptPath != "" {
		b, _ := json.MarshalIndent(r, "", "  ")
		b = append(b, '\n')
		if err = os.MkdirAll(filepath.Dir(o.ReceiptPath), 0755); err != nil {
			return r, err
		}
		if err = os.WriteFile(o.ReceiptPath, b, 0644); err != nil {
			return r, err
		}
	}
	return r, nil
}

func runDirNames(root string) (map[string]bool, error) {
	out := map[string]bool{}
	es, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return out, nil
	}
	if err != nil {
		return nil, err
	}
	for _, e := range es {
		if e.IsDir() {
			out[e.Name()] = true
		}
	}
	return out, nil
}

var _ = time.Second

// Command tmp2442verify is a throwaway consumer-style read-back verifier for
// private#2442. It re-reads the frozen manifest, resolves each selected case
// from the pinned public package, recomputes every SHA-256 content digest and
// the pinned source git blob OIDs, and fails closed on any mismatch. It must
// not be committed to the public repository; delete this directory after use.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

type manifest struct {
	Status string `json:"status"`
	Source struct {
		Repository   string            `json:"repository"`
		PinnedCommit string            `json:"pinned_commit"`
		GitBlobOID   map[string]string `json:"git_blob_oid"`
	} `json:"source"`
	Selection struct {
		BaseCaseCount    int      `json:"base_case_count"`
		HeldOutCaseCount int      `json:"held_out_case_count"`
		TotalCaseCount   int      `json:"total_case_count"`
		BaseCases        []string `json:"base_cases"`
		HeldOutCases     []string `json:"held_out_cases"`
	} `json:"selection"`
	Cases []struct {
		ID         string `json:"id"`
		HeldOut    bool   `json:"held_out"`
		Family     string `json:"family"`
		TargetFile string `json:"target_file"`
		Workflow   struct {
			Kind          string   `json:"kind"`
			Area          string   `json:"area"`
			RequiredSteps []string `json:"required_steps"`
			Prior         *struct {
				PriorTaskID           string `json:"prior_task_id"`
				PriorStateSHA256      string `json:"prior_state_sha256"`
				PriorDiffSHA256       string `json:"prior_diff_sha256"`
				PriorBeforeSHA256     string `json:"prior_before_source_sha256"`
				PriorAfterSHA256      string `json:"prior_after_source_sha256"`
				PriorTestSourceSHA256 string `json:"prior_test_source_sha256"`
			} `json:"prior"`
		} `json:"workflow"`
		Content struct {
			PromptSHA256       string `json:"prompt_sha256"`
			BrokenSourceSHA256 string `json:"broken_source_sha256"`
			FixedSourceSHA256  string `json:"fixed_source_sha256"`
			VisibleTestSHA256  string `json:"visible_test_sha256"`
			OracleTestSHA256   string `json:"oracle_test_sha256"`
		} `json:"content_sha256"`
	} `json:"cases"`
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: tmp2442verify <manifest.json>")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		fmt.Fprintln(os.Stderr, "decode manifest:", err)
		os.Exit(1)
	}
	failures := 0
	check := func(name, got, want string) {
		if got != want {
			failures++
			fmt.Printf("MISMATCH %-42s got=%s want=%s\n", name, got, want)
		}
	}

	if m.Status != "frozen" {
		failures++
		fmt.Printf("MISMATCH status got=%q want=\"frozen\"\n", m.Status)
	}

	head := gitOutput("rev-parse", "HEAD")
	check("source.pinned_commit vs git HEAD", m.Source.PinnedCommit, head)
	for path, oid := range m.Source.GitBlobOID {
		check("git_blob_oid "+path, gitOutput("rev-parse", "HEAD:"+path), oid)
	}

	byID := map[string]taskfixture.Fixture{}
	for _, f := range taskfixture.Cases(true) {
		byID[f.ID] = f
	}
	if len(m.Cases) != 6 || m.Selection.TotalCaseCount != 6 || m.Selection.BaseCaseCount != 4 || m.Selection.HeldOutCaseCount != 2 {
		failures++
		fmt.Printf("MISMATCH case counts: cases=%d total=%d base=%d heldout=%d\n", len(m.Cases), m.Selection.TotalCaseCount, m.Selection.BaseCaseCount, m.Selection.HeldOutCaseCount)
	}
	baseSeen := map[string]bool{}
	heldSeen := map[string]bool{}
	for _, c := range m.Cases {
		f, ok := byID[c.ID]
		if !ok {
			failures++
			fmt.Printf("MISSING fixture %q\n", c.ID)
			continue
		}
		if c.HeldOut {
			heldSeen[c.ID] = true
		} else {
			baseSeen[c.ID] = true
		}
		check(c.ID+".family", c.Family, f.Family)
		check(c.ID+".target_file", c.TargetFile, f.TargetFile)
		check(c.ID+".workflow.kind", c.Workflow.Kind, f.Workflow.Kind)
		check(c.ID+".workflow.area", c.Workflow.Area, f.Workflow.Area)
		check(c.ID+".workflow.required_steps", strings.Join(c.Workflow.RequiredSteps, ","), strings.Join(f.Workflow.RequiredSteps, ","))
		check(c.ID+".prompt_sha256", c.Content.PromptSHA256, sha(f.Prompt))
		check(c.ID+".broken_source_sha256", c.Content.BrokenSourceSHA256, sha(f.BrokenSource))
		check(c.ID+".fixed_source_sha256", c.Content.FixedSourceSHA256, sha(f.FixedSource))
		check(c.ID+".visible_test_sha256", c.Content.VisibleTestSHA256, sha(f.VisibleTest))
		check(c.ID+".oracle_test_sha256", c.Content.OracleTestSHA256, sha(f.OracleTest))
		if c.Workflow.Prior != nil {
			p := c.Workflow.Prior
			check(c.ID+".prior_task_id", p.PriorTaskID, f.Workflow.PriorTaskID)
			check(c.ID+".prior_state_sha256", p.PriorStateSHA256, f.Workflow.PriorStateSHA256)
			check(c.ID+".prior_diff_sha256", p.PriorDiffSHA256, sha(f.Workflow.PriorDiff))
			check(c.ID+".prior_before_source_sha256", p.PriorBeforeSHA256, sha(f.Workflow.PriorBeforeSource))
			check(c.ID+".prior_after_source_sha256", p.PriorAfterSHA256, sha(f.Workflow.PriorAfterSource))
			check(c.ID+".prior_test_source_sha256", p.PriorTestSourceSHA256, sha(f.Workflow.PriorTestSource))
		}
	}
	for _, id := range m.Selection.BaseCases {
		if !baseSeen[id] {
			failures++
			fmt.Printf("base_cases lists %q but no base case block matched\n", id)
		}
	}
	for _, id := range m.Selection.HeldOutCases {
		if !heldSeen[id] {
			failures++
			fmt.Printf("held_out_cases lists %q but no held-out case block matched\n", id)
		}
	}

	if failures != 0 {
		fmt.Printf("READ-BACK FAILED: %d mismatch(es)\n", failures)
		os.Exit(1)
	}
	fmt.Printf("READ-BACK OK: 6/6 cases resolved, all content digests and pinned blob OIDs match %s\n", head[:12])
}

func gitOutput(args ...string) string {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "git %v: %v\n", args, err)
		os.Exit(1)
	}
	return strings.TrimSpace(string(out))
}

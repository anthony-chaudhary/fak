// Command tmp2442gen is a throwaway generator for private#2442.
// It emits the frozen useful-native-agent task manifest for six existing
// agentbench fixtures at a pinned public commit. It must not be committed to
// the public repository; delete this directory after use.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

const publicCommit = "ce5ae3cbb5171494804e15c91fff3cdf48c38560"

var selection = []string{
	"retry-delay-base250",
	"labels-trim-lower",
	"requirements-empty",
	"requirements-named",
	"retry-delay-large-cap",
	"labels-unicode-space",
}

var heldOut = map[string]bool{
	"retry-delay-large-cap": true,
	"labels-unicode-space":  true,
}

var sourceFiles = []string{
	"internal/agentbench/taskfixture/fixtures.go",
	"internal/agentbench/taskrun/tasks.go",
	"internal/agentbench/taskrun/workflow.go",
	"internal/agentbench/taskrun/candidate_validation.go",
	"internal/agentbench/taskrun/sandbox_linux.go",
}

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func main() {
	byID := map[string]taskfixture.Fixture{}
	for _, f := range taskfixture.Cases(true) {
		byID[f.ID] = f
	}
	cases := make([]any, 0, len(selection))
	for _, id := range selection {
		f, ok := byID[id]
		if !ok {
			fmt.Fprintf(os.Stderr, "tmp2442gen: missing fixture %q\n", id)
			os.Exit(1)
		}
		workflow := map[string]any{"kind": f.Workflow.Kind, "area": f.Workflow.Area, "required_steps": nonNil(f.Workflow.RequiredSteps)}
		if f.Workflow.PriorTaskID != "" {
			workflow["prior"] = map[string]any{
				"prior_task_id": f.Workflow.PriorTaskID, "prior_state_sha256": f.Workflow.PriorStateSHA256,
				"prior_steps": nonNil(f.Workflow.PriorSteps), "prior_diff_sha256": sha(f.Workflow.PriorDiff),
				"prior_before_source_sha256": sha(f.Workflow.PriorBeforeSource),
				"prior_after_source_sha256":  sha(f.Workflow.PriorAfterSource),
				"prior_test_source_sha256":   sha(f.Workflow.PriorTestSource),
			}
		}
		cases = append(cases, map[string]any{
			"id": id, "held_out": heldOut[id], "family": f.Family, "target_file": f.TargetFile, "workflow": workflow,
			"content_sha256": map[string]any{
				"prompt_sha256": sha(f.Prompt), "broken_source_sha256": sha(f.BrokenSource),
				"fixed_source_sha256": sha(f.FixedSource), "visible_test_sha256": sha(f.VisibleTest),
				"oracle_test_sha256": sha(f.OracleTest),
			},
		})
	}
	manifest := map[string]any{
		"schema":  "fak-private.useful-native-agent.tasks.v1",
		"status":  "frozen",
		"ticket":  "anthony-chaudhary/fak-private#2442",
		"source": map[string]any{
			"repository": "github.com/anthony-chaudhary/fak", "path": "internal/agentbench/taskfixture",
			"pinned_commit": publicCommit,
			"git_blob_oid": map[string]any{
				sourceFiles[0]: "a0c56e23d9c2da119c8353971914d873248d3422",
				sourceFiles[1]: "8b926f29d55552cb084744a3165b28743cc22997",
				sourceFiles[2]: "3d79dd7b53f48f287c8959d65290e2f75e76bddb",
				sourceFiles[3]: "38e4c6c0ece7c8ba91d013b8158effadfd482a9a",
				sourceFiles[4]: "5d0e8ff03a3524e7029c5b1ac9e0117aabcae1de",
			},
			"git_blob_oid_algorithm": "git blob object id (SHA-1 of the raw committed file bytes)",
		},
		"selection": map[string]any{
			"include_heldout":  true,
			"base_case_count":  4, "held_out_case_count": 2, "total_case_count": 6,
			"base_cases":        []string{"retry-delay-base250", "labels-trim-lower", "requirements-empty", "requirements-named"},
			"held_out_cases":    []string{"retry-delay-large-cap", "labels-unicode-space"},
		},
		"runner_contract": map[string]any{
			"public_package": "internal/agentbench/taskrun",
			"entrypoint":     "taskrun.Run (one task at a time, concurrency 1 for the pilot)",
			"required_evidence": []string{
				"inspect: Read the target file and observe its exact bytes",
				"scoped_edit: edit only the original target function body (structurally validated)",
				"test_pass: run the exact visible-test Bash invocation supplied in the prompt and observe exit 0",
				"tool_result_continuation: continue the loop from the tool result, not self-reported success",
				"hidden_oracle_acceptance: independently re-run the hidden oracle outside the writable trial",
			},
			"test_invocation": "<executable> bench agent test-child --config <test-config.json>",
			"external_verifier": map[string]any{
				"command":  "bwrap --unshare-all ... /goroot/bin/go test ./... -count=1",
				"root":     "oracle root bound read-only and disjoint from the writable trial",
				"oracle_visible_to_agent": false,
			},
		},
		"control_expectations": map[string]any{
			"broken_candidate":  "RED (visible+hidden tests fail on unmodified broken source)",
			"known_fix_candidate": "GREEN (visible+hidden tests pass on the known fixed source)",
			"oracle_location":   "outside the writable trial directory; never exposed to the agent context",
			"fail_closed": []string{
				"no operator intervention", "no oracle edit", "no empty diff",
				"no structural escape outside the target function body", "no self-reported success",
			},
		},
		"witness": map[string]any{
			"command": "go test ./internal/agentbench/taskfixture ./internal/agentbench/taskrun -count=1 -v",
			"repo":    "github.com/anthony-chaudhary/fak", "commit": publicCommit,
			"requires": "Linux host with bubblewrap (bwrap) and a pinned GOROOT go toolchain",
			"skipped_namespace_controls_are_not_a_pass": true,
		},
		"cases": cases,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(manifest); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func nonNil(v []string) []string {
	if v == nil {
		return []string{}
	}
	return v
}

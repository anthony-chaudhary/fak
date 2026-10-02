package releasereadiness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestFastReleaseGateConvergesAndKeepsFullTestObservable(t *testing.T) {
	workflow := readFastReleaseWorkflow(t)
	for _, required := range []string{
		"name: ci-fast",
		"workflow_dispatch:",
		"concurrency:",
		`GOMAXPROCS: "2"`,
		`GOFLAGS: "-p=2"`,
		"run: go build ./...",
		"run: go vet ./...",
		// The go test step stays observable (#6982): tee'd log, the real exit
		// status, and a heartbeat while it runs.
		"- name: go test ./... (no -race)",
		`go test -timeout=20m "${shard_pkgs[@]}" 2>&1 | tee`,
		`status_file="$RUNNER_TEMP/ci-fast-go-test.status"`,
		`"$RUNNER_TEMP/ci-fast-go-test.log"`,
		"${PIPESTATUS[0]}",
		"is still running",
		`FAK_REQUIRE_OPENAPI_VALIDATOR: "1"`,
		// The shards partition ONE package list: every job checks out the event
		// commit resolved once by candidate, and the partition keys on the
		// runner-provided matrix position so its modulus cannot drift from the
		// matrix size.
		"ref: ${{ github.event_name == 'workflow_dispatch' && inputs.sha || github.sha }}",
		"ref: ${{ needs.candidate.outputs.sha }}",
		"EXPECTED_TREE: ${{ needs.candidate.outputs.tree }}",
		`! "$REQUESTED_SHA" =~ ^[0-9a-f]{40}$`,
		"go list ./...",
		"SHARD_INDEX: ${{ strategy.job-index }}",
		"SHARD_TOTAL: ${{ strategy.job-total }}",
		"'(NR - 1) % n == i'",
		"fail-fast: false",
		// Automatic runs retain the pre-shard check name. Manual runs cannot
		// masquerade as the release check, even with requested intent release.
		"build-vet-test-fast:",
		"name: ${{ github.event_name == 'workflow_dispatch' && 'diagnostic · build · vet · test (non-release)' || 'build · vet · test (no -race, fast release gate)' }}",
		"needs: [candidate, build-vet-fast, go-test-shard]",
		"if: ${{ !cancelled() }}",
		"BUILD_VET_RESULT: ${{ needs.build-vet-fast.result }}",
		"GO_TEST_SHARDS_RESULT: ${{ needs.go-test-shard.result }}",
		`[ "$BUILD_VET_RESULT" != "success" ] || [ "$GO_TEST_SHARDS_RESULT" != "success" ]`,
	} {
		if !strings.Contains(workflow, required) {
			t.Fatalf("ci-fast correctness gate lacks %q", required)
		}
	}
	for _, forbidden := range []string{
		"github.head_ref || github.sha",
		`GOFLAGS: "-p=1"`,
		// An explicit branch ref checks out the branch tip at fetch time, so
		// parallel shards on a hot trunk could test different trees.
		"inputs.sha || github.ref",
		// The whole suite in one serial step is what never concluded in 30m.
		"go test ./... 2>&1 | tee",
	} {
		if strings.Contains(workflow, forbidden) {
			t.Fatalf("ci-fast correctness gate retains obsolete contract %q", forbidden)
		}
	}
}

// TestFastReleaseGateShardOutlastsPerBinaryTimeout pins the deadline ordering
// that keeps a stuck shard decisive and diagnosable: go test's own per-binary
// -timeout fires (with a goroutine dump) inside the step bound, which leaves
// room for compile/link, and the job bound outlasts the step.
func TestFastReleaseGateShardOutlastsPerBinaryTimeout(t *testing.T) {
	workflow := readFastReleaseWorkflow(t)
	job := workflowJobBlock(t, workflow, "go-test-shard")
	jobTimeout := timeoutMinutesAtIndent(t, job, 4)
	stepAt := strings.Index(job, "      - name: go test ./... (no -race)\n")
	if stepAt < 0 {
		t.Fatal("go-test-shard job has no `go test ./... (no -race)` step")
	}
	step := job[stepAt:]
	stepTimeout := timeoutMinutesAtIndent(t, step, 8)
	m := regexp.MustCompile(`go test -timeout=(\d+)m `).FindStringSubmatch(step)
	if m == nil {
		t.Fatal("go-test-shard step must run go test with an explicit -timeout=<N>m")
	}
	testTimeout, _ := strconv.Atoi(m[1])
	if testTimeout <= 10 {
		t.Fatalf("go test -timeout=%dm does not exceed the 10m default that clipped cmd/fak", testTimeout)
	}
	if stepTimeout < testTimeout+10 {
		t.Fatalf("step timeout-minutes %d must outlast go test -timeout=%dm plus compile/link grace (>= %d)", stepTimeout, testTimeout, testTimeout+10)
	}
	if jobTimeout <= stepTimeout {
		t.Fatalf("job timeout-minutes %d must outlast the go test step bound %d", jobTimeout, stepTimeout)
	}
}

// This executes the workflow's verifier, not a second Go implementation of it.
// The fixture names eight packages and their four partitions explicitly so a
// dropped or duplicated package cannot be hidden by duplicating its algorithm.
// fak-test:runtime medium est=3s lane=default
func TestFastReleaseGateRequiresCompleteSourceBoundProvenance(t *testing.T) {
	t.Parallel()
	script := fastProvenanceVerifierScript(t, readFastReleaseWorkflow(t))
	requireProvenanceTools(t)
	type mutation func(*fastProvenanceFixture)
	type testCase struct {
		name     string
		mutate   mutation
		verified bool
		release  bool
	}
	cases := []testCase{
		{name: "complete push main verifies commit tree only", verified: true},
		{name: "pull request is verified but not release", mutate: func(f *fastProvenanceFixture) {
			f.bind("GITHUB_EVENT_NAME", "event", "pull_request")
			f.bind("GITHUB_REF", "ref", "refs/pull/17/merge")
		}, verified: true},
		{name: "non-main push is not release", mutate: func(f *fastProvenanceFixture) {
			f.bind("GITHUB_REF", "ref", "refs/heads/master")
		}, verified: true},
		{name: "manual release intent is still diagnostic", mutate: func(f *fastProvenanceFixture) {
			f.bind("GITHUB_EVENT_NAME", "event", "workflow_dispatch")
			f.bind("CI_CLASS", "ci_class", "diagnostic")
			f.bind("REQUESTED_INTENT", "requested_intent", "release")
			// A dispatch can test an immutable commit other than its event SHA.
			f.bind("GITHUB_SHA", "represented_sha", strings.Repeat("c", 40))
		}, verified: true},
		{name: "missing shard", mutate: func(f *fastProvenanceFixture) { f.receipts = f.receipts[:4] }},
		{name: "duplicate replaces missing shard", mutate: func(f *fastProvenanceFixture) { f.receipts[4] = f.receipts[3] }},
		{name: "extra duplicate receipt", mutate: func(f *fastProvenanceFixture) { f.receipts = append(f.receipts, f.receipts[4]) }},
		{name: "no receipts", mutate: func(f *fastProvenanceFixture) { f.receipts = nil }},
		{name: "malformed receipt", mutate: func(f *fastProvenanceFixture) { f.malformed = true }},
		{name: "automatic tested source differs from represented source", mutate: func(f *fastProvenanceFixture) {
			f.bind("EXPECTED_SHA", "tested_sha", strings.Repeat("c", 40))
		}},
		{name: "manual cannot claim fast class", mutate: func(f *fastProvenanceFixture) {
			f.bind("GITHUB_EVENT_NAME", "event", "workflow_dispatch")
			f.bind("REQUESTED_INTENT", "requested_intent", "release")
		}},
		{name: "empty package universe", mutate: func(f *fastProvenanceFixture) {
			for _, receipt := range f.receipts[1:] {
				coverage := receipt["coverage"].(map[string]any)
				coverage["all_packages"], coverage["packages"] = []string{}, []string{}
				receipt["commands"] = [][]string{{"go", "test", "-timeout=20m"}}
			}
		}},
		{name: "partial package coverage with matching argv", mutate: func(f *fastProvenanceFixture) {
			f.receipts[1]["coverage"].(map[string]any)["packages"] = []string{"example.invalid/project/a"}
			f.receipts[1]["commands"] = [][]string{{"go", "test", "-timeout=20m", "example.invalid/project/a"}}
		}},
		{name: "overlapping packages with matching argv", mutate: func(f *fastProvenanceFixture) {
			f.receipts[2]["coverage"].(map[string]any)["packages"] = []string{"example.invalid/project/a", "example.invalid/project/f"}
			f.receipts[2]["commands"] = [][]string{{"go", "test", "-timeout=20m", "example.invalid/project/a", "example.invalid/project/f"}}
		}},
		{name: "different resolved package universe", mutate: func(f *fastProvenanceFixture) {
			f.receipts[3]["coverage"].(map[string]any)["all_packages"] = []string{"example.invalid/project/other"}
		}},
		{name: "wrong shard identity", mutate: func(f *fastProvenanceFixture) {
			f.receipts[3]["coverage"].(map[string]any)["shard_index"] = 1
		}},
		{name: "wrong shard count", mutate: func(f *fastProvenanceFixture) {
			f.receipts[3]["coverage"].(map[string]any)["shard_total"] = 3
		}},
		{name: "successful zero-match command is not full coverage", mutate: func(f *fastProvenanceFixture) {
			f.receipts[1]["commands"] = [][]string{{"go", "test", "-run", "^$", "./..."}}
		}},
		{name: "build command narrowed", mutate: func(f *fastProvenanceFixture) {
			f.receipts[0]["commands"].([][]string)[1] = []string{"go", "build", "./cmd/fak"}
		}},
		{name: "unapproved test flag", mutate: func(f *fastProvenanceFixture) {
			f.receipts[1]["commands"].([][]string)[0] = append(f.receipts[1]["commands"].([][]string)[0], "-short")
		}},
		{name: "all jobs share wrong concurrency policy", mutate: func(f *fastProvenanceFixture) {
			for _, receipt := range f.receipts {
				receipt["environment"].(map[string]string)["GOFLAGS"] = "-p=1"
			}
		}},
		{name: "all jobs share wrong toolchain policy", mutate: func(f *fastProvenanceFixture) {
			for _, receipt := range f.receipts {
				receipt["environment"].(map[string]string)["GOTOOLCHAIN"] = "local"
			}
		}},
		{name: "inconsistent Go toolchain", mutate: func(f *fastProvenanceFixture) {
			f.receipts[2]["environment"].(map[string]string)["GOVERSION"] = "go1.25.0"
		}},
		{name: "missing relevant environment", mutate: func(f *fastProvenanceFixture) {
			delete(f.receipts[2]["environment"].(map[string]string), "CGO_ENABLED")
		}},
		{name: "validator disabled", mutate: func(f *fastProvenanceFixture) {
			f.receipts[2]["environment"].(map[string]string)["FAK_REQUIRE_OPENAPI_VALIDATOR"] = "0"
		}},
		{name: "test step skipped under successful job", mutate: func(f *fastProvenanceFixture) {
			f.receipts[2]["steps"].(map[string]any)["tests"] = map[string]string{"outcome": "skipped", "conclusion": "skipped"}
		}},
		{name: "failed vet hidden by successful conclusion", mutate: func(f *fastProvenanceFixture) {
			f.receipts[0]["steps"].(map[string]any)["vet"] = map[string]string{"outcome": "failure", "conclusion": "success"}
		}},
		{name: "unknown receipt source scope", mutate: func(f *fastProvenanceFixture) {
			delete(f.receipts[1], "source_scope")
		}},
		{name: "unknown effective input verification", mutate: func(f *fastProvenanceFixture) {
			delete(f.receipts[1], "effective_inputs_verified")
		}},
	}
	for field, wrong := range map[string]any{
		"schema": "fak.ci-provenance.v0", "workflow": "ci", "tested_sha": strings.Repeat("c", 40),
		"tested_tree": strings.Repeat("d", 40), "represented_sha": strings.Repeat("e", 40),
		"event": "workflow_dispatch", "ref": "refs/heads/other", "run_id": "101", "run_attempt": "2",
		"ci_class": "diagnostic", "requested_intent": "release", "release_eligible": true,
		"source_scope": "effective-inputs", "effective_inputs_verified": true,
	} {
		cases = append(cases, testCase{name: "mismatched shard " + field, mutate: func(f *fastProvenanceFixture) { f.receipts[3][field] = wrong }})
	}
	for _, result := range []string{"failure", "skipped", "cancelled", "in_progress", ""} {
		cases = append(cases, testCase{name: "non-success shard " + result, mutate: func(f *fastProvenanceFixture) { f.receipts[4]["result"] = result }})
	}
	for _, dependency := range []string{"CANDIDATE_RESULT", "BUILD_VET_RESULT", "GO_TEST_SHARDS_RESULT"} {
		cases = append(cases, testCase{name: "unsuccessful dependency " + dependency, mutate: func(f *fastProvenanceFixture) { f.env[dependency] = "skipped" }})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newFastProvenanceFixture()
			if tc.mutate != nil {
				tc.mutate(&fixture)
			}
			dir := t.TempDir()
			artifacts := filepath.Join(dir, "ci-fast-provenance")
			if err := os.Mkdir(artifacts, 0755); err != nil {
				t.Fatal(err)
			}
			for i, receipt := range fixture.receipts {
				writeProvenanceJSON(t, filepath.Join(artifacts, fmt.Sprintf("receipt-%d.json", i)), receipt)
			}
			if fixture.malformed {
				if err := os.WriteFile(filepath.Join(artifacts, "receipt-4.json"), []byte("{broken"), 0644); err != nil {
					t.Fatal(err)
				}
			}
			fixture.env["RUNNER_TEMP"] = dir
			fixture.env["GITHUB_STEP_SUMMARY"] = filepath.Join(dir, "summary")
			output, err := runProvenanceScript(t, script, dir, fixture.env)
			if (err == nil) != tc.verified {
				t.Fatalf("verifier error = %v, want success %t; output: %s", err, tc.verified, output)
			}
			var report map[string]any
			readProvenanceJSON(t, filepath.Join(dir, "ci-fast-provenance.json"), &report)
			if report["verified"] != tc.verified || report["release_eligible"] != tc.release {
				t.Fatalf("verified/release_eligible = %v/%v, want %t/%t; output: %s", report["verified"], report["release_eligible"], tc.verified, tc.release, output)
			}
			// Commit and job provenance does not seal all effective mutable
			// inputs. Keep that limit explicit even when every receipt verifies.
			if report["source_scope"] != "git-commit-tree" || report["effective_inputs_verified"] != false {
				t.Fatalf("aggregate overstates its verified input scope: %+v", report)
			}
			for field, env := range map[string]string{
				"tested_sha": "EXPECTED_SHA", "tested_tree": "EXPECTED_TREE", "represented_sha": "GITHUB_SHA",
				"event": "GITHUB_EVENT_NAME", "ref": "GITHUB_REF", "run_id": "GITHUB_RUN_ID", "run_attempt": "GITHUB_RUN_ATTEMPT",
				"ci_class": "CI_CLASS", "requested_intent": "REQUESTED_INTENT",
			} {
				if report[field] != fixture.env[env] {
					t.Errorf("aggregate %s = %v, want trusted run value %q", field, report[field], fixture.env[env])
				}
			}
		})
	}
}

// A green diagnostic command may match zero tests. Its producer must retain
// that exact command and class without manufacturing release eligibility.
// fak-test:runtime fast est=100ms lane=default
func TestCIProvenanceSuccessfulZeroMatchRemainsDiagnostic(t *testing.T) {
	t.Parallel()
	root := findRepoRoot(t)
	workflow, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	script := provenanceWorkflowScript(t, string(workflow), "record-ci")
	requireProvenanceTools(t)
	gitIdentity := func(ref string) string {
		cmd := exec.Command("git", "rev-parse", ref)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("read source %s: %v: %s", ref, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	dir := t.TempDir()
	commands := [][]string{{"go", "test", "-run", "^$", "./internal/releasereadiness"}}
	writeProvenanceJSON(t, filepath.Join(dir, "ci-routed-command.json"), commands)
	writeProvenanceJSON(t, filepath.Join(dir, "ci-environment.json"), map[string]string{
		"GOVERSION": "go1.26.0", "GOOS": "linux", "GOARCH": "amd64", "CGO_ENABLED": "1", "GOFLAGS": "", "GOTOOLCHAIN": "auto", "GOMAXPROCS": "",
	})
	sha, tree := gitIdentity("HEAD"), gitIdentity("HEAD^{tree}")
	env := map[string]string{
		"RUNNER_TEMP": dir, "GITHUB_STEP_SUMMARY": filepath.Join(dir, "summary"),
		"TESTED_SHA": sha, "TESTED_TREE": tree, "GITHUB_SHA": sha, "GITHUB_EVENT_NAME": "workflow_dispatch",
		"GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1",
		"GITHUB_WORKFLOW_REF": "example/project/.github/workflows/ci.yml@refs/heads/main", "GITHUB_WORKFLOW_SHA": sha,
		"RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "JOB_ID": "testroute-ci-only", "CI_CLASS": "diagnostic",
		"JOB_RESULT": "success", "STEPS_JSON": `{}`, "CORE_COMMANDS_JSON": `[]`,
	}
	if out, err := runProvenanceScript(t, script, root, env); err != nil {
		t.Fatalf("record successful diagnostic: %v: %s", err, out)
	}
	var report struct {
		Schema                  string     `json:"schema"`
		Class                   string     `json:"ci_class"`
		Result                  string     `json:"result"`
		ReleaseEligible         *bool      `json:"release_eligible"`
		Commands                [][]string `json:"commands"`
		SourceScope             string     `json:"source_scope"`
		EffectiveInputsVerified *bool      `json:"effective_inputs_verified"`
	}
	readProvenanceJSON(t, filepath.Join(dir, "ci-testroute-ci-only.json"), &report)
	if report.Schema != "fak.ci-provenance.v1" || report.Class != "diagnostic" || report.Result != "success" || report.ReleaseEligible == nil || *report.ReleaseEligible {
		t.Fatalf("successful zero-match diagnostic must be explicitly non-release: %+v", report)
	}
	if !reflect.DeepEqual(report.Commands, commands) {
		t.Fatalf("recorded argv = %q, want exact zero-match argv %q", report.Commands, commands)
	}
	if report.SourceScope != "git-commit-tree" || report.EffectiveInputsVerified == nil || *report.EffectiveInputsVerified {
		t.Fatalf("diagnostic producer overstates its verified input scope: %+v", report)
	}
}

// Successful outputs retained by a partial rerun belong to their original
// attempt. The aggregate must not relabel them with its current attempt.
// fak-test:runtime fast est=500ms lane=default
func TestCIProvenanceFullGateRejectsMixedAttempts(t *testing.T) {
	t.Parallel()
	workflow, err := os.ReadFile(filepath.Join(findRepoRoot(t), ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	script := provenanceWorkflowScript(t, string(workflow), "verify-full")
	requireProvenanceTools(t)
	sha, tree := strings.Repeat("a", 40), strings.Repeat("b", 40)
	for _, tc := range []struct {
		name     string
		event    string
		job      string
		field    string
		value    string
		verified bool
		release  bool
	}{
		{name: "current push attempt verifies commit tree only", event: "push", verified: true},
		{name: "current PR attempt", event: "pull_request", verified: true},
		{name: "current manual attempt remains diagnostic", event: "workflow_dispatch", verified: true},
		{name: "stale build attempt", event: "push", job: "build-vet-test", field: "run_attempt", value: "1"},
		{name: "stale race attempt", event: "push", job: "race-detector", field: "run_attempt", value: "1"},
		{name: "stale snapshot attempt", event: "push", job: "head-snapshot-build", field: "run_attempt", value: "1"},
		{name: "missing build attempt", event: "push", job: "build-vet-test", field: "run_attempt"},
		{name: "different race run", event: "push", job: "race-detector", field: "run_id", value: "99"},
		{name: "missing snapshot run", event: "push", job: "head-snapshot-build", field: "run_id"},
		{name: "snapshot tested different source", event: "push", job: "head-snapshot-build", field: "sha", value: strings.Repeat("c", 40)},
		{name: "snapshot tested different tree", event: "push", job: "head-snapshot-build", field: "tree", value: strings.Repeat("d", 40)},
		{name: "PR retained previous race attempt", event: "pull_request", job: "race-detector", field: "run_attempt", value: "1"},
		{name: "manual retained previous routed attempt", event: "workflow_dispatch", job: "testroute-ci-only", field: "run_attempt", value: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			required := []string{"build-vet-test", "race-detector", "head-snapshot-build"}
			class := "full"
			ref := "refs/heads/main"
			if tc.event == "pull_request" {
				required = []string{"build-vet-test", "race-detector"}
				ref = "refs/pull/17/merge"
			} else if tc.event == "workflow_dispatch" {
				required = []string{"testroute-ci-only"}
				class = "diagnostic"
			}
			jobs := map[string]any{}
			for _, id := range []string{"build-vet-test", "race-detector", "head-snapshot-build", "testroute-ci-only"} {
				jobs[id] = map[string]any{"result": "skipped", "outputs": map[string]string{}}
			}
			for _, id := range required {
				jobs[id] = map[string]any{"result": "success", "outputs": map[string]string{
					"sha": sha, "tree": tree, "run_id": "100", "run_attempt": "2",
				}}
			}
			if tc.job != "" {
				outputs := jobs[tc.job].(map[string]any)["outputs"].(map[string]string)
				if tc.value == "" {
					delete(outputs, tc.field)
				} else {
					outputs[tc.field] = tc.value
				}
			}
			needs, err := json.Marshal(jobs)
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			env := map[string]string{
				"NEEDS_JSON": string(needs), "RUNNER_TEMP": dir, "GITHUB_STEP_SUMMARY": filepath.Join(dir, "summary"),
				"GITHUB_SHA": sha, "GITHUB_EVENT_NAME": tc.event, "GITHUB_REF": ref, "GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "2",
			}
			output, err := runProvenanceScript(t, script, dir, env)
			if (err == nil) != tc.verified {
				t.Fatalf("full verifier error = %v, want success %t; output: %s", err, tc.verified, output)
			}
			var report struct {
				Schema                  string   `json:"schema"`
				Workflow                string   `json:"workflow"`
				Class                   string   `json:"ci_class"`
				SHA                     string   `json:"tested_sha"`
				Tree                    string   `json:"tested_tree"`
				RepresentedSHA          string   `json:"represented_sha"`
				Event                   string   `json:"event"`
				Ref                     string   `json:"ref"`
				RunID                   string   `json:"run_id"`
				RunAttempt              string   `json:"run_attempt"`
				RequiredJobs            []string `json:"required_jobs"`
				Verified                *bool    `json:"verified"`
				ReleaseEligible         *bool    `json:"release_eligible"`
				SourceScope             string   `json:"source_scope"`
				EffectiveInputsVerified *bool    `json:"effective_inputs_verified"`
			}
			readProvenanceJSON(t, filepath.Join(dir, "ci-provenance.json"), &report)
			if report.Verified == nil || *report.Verified != tc.verified || report.ReleaseEligible == nil || *report.ReleaseEligible != tc.release {
				t.Fatalf("full aggregate must explicitly report verified=%t and release_eligible=%t: %+v", tc.verified, tc.release, report)
			}
			if report.SourceScope != "git-commit-tree" || report.EffectiveInputsVerified == nil || *report.EffectiveInputsVerified {
				t.Fatalf("full aggregate overstates its verified input scope: %+v", report)
			}
			if report.Schema != "fak.ci-provenance.v1" || report.Workflow != "ci" || report.Class != class || !reflect.DeepEqual(report.RequiredJobs, required) {
				t.Fatalf("wrong full CI class or fixed job set: %+v; want class %s, jobs %v", report, class, required)
			}
			if report.SHA != sha || report.Tree != tree || report.RepresentedSHA != sha || report.Event != tc.event || report.Ref != ref || report.RunID != "100" || report.RunAttempt != "2" {
				t.Fatalf("aggregate source and current run identity drifted: %+v", report)
			}
		})
	}
}

type fastProvenanceFixture struct {
	env       map[string]string
	receipts  []map[string]any
	malformed bool
}

func (f *fastProvenanceFixture) bind(env, field, value string) {
	f.env[env] = value
	for _, receipt := range f.receipts {
		receipt[field] = value
	}
}

func newFastProvenanceFixture() fastProvenanceFixture {
	f := fastProvenanceFixture{env: map[string]string{
		"EXPECTED_SHA": strings.Repeat("a", 40), "EXPECTED_TREE": strings.Repeat("b", 40), "GITHUB_SHA": strings.Repeat("a", 40),
		"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1",
		"CI_CLASS": "fast", "REQUESTED_INTENT": "", "CANDIDATE_RESULT": "success", "BUILD_VET_RESULT": "success", "GO_TEST_SHARDS_RESULT": "success",
	}}
	all := []string{
		"example.invalid/project/a", "example.invalid/project/b", "example.invalid/project/c", "example.invalid/project/d",
		"example.invalid/project/e", "example.invalid/project/f", "example.invalid/project/g", "example.invalid/project/h",
	}
	partitions := [][]string{{all[0], all[4]}, {all[1], all[5]}, {all[2], all[6]}, {all[3], all[7]}}
	for _, job := range []string{"build-vet-fast", "go-test-shard-0", "go-test-shard-1", "go-test-shard-2", "go-test-shard-3"} {
		environment := map[string]string{
			"GOVERSION": "go1.26.0", "GOOS": "linux", "GOARCH": "amd64", "CGO_ENABLED": "1", "GOFLAGS": "-p=2", "GOTOOLCHAIN": "auto", "GOMAXPROCS": "2",
		}
		steps := map[string]any{}
		stepNames := []string{"checksum", "build", "vet", "tier"}
		coverage := map[string]any{"kind": "build-vet", "package_scope": "./..."}
		commands := [][]string{
			{"bash", ".github/scripts/verify-release-checksums_test.sh"}, {"go", "build", "./..."}, {"go", "vet", "./..."},
			{"go", "test", "-count=1", "./internal/architest/", "-run", "^(TestEveryPackageDeclaresTier|TestNoUpwardImports|TestRootImportsNothingInternal|TestSingleOpenAIChatClient)$"},
		}
		if job != "build-vet-fast" {
			index := len(f.receipts) - 1
			coverage = map[string]any{"kind": "go-test-shard", "shard_index": index, "shard_total": 4, "all_packages": all, "packages": partitions[index]}
			commands = [][]string{append([]string{"go", "test", "-timeout=20m"}, partitions[index]...)}
			environment["FAK_REQUIRE_OPENAPI_VALIDATOR"] = "1"
			stepNames = []string{"tests"}
		}
		for _, name := range stepNames {
			steps[name] = map[string]string{"outcome": "success", "conclusion": "success"}
		}
		f.receipts = append(f.receipts, map[string]any{
			"schema": "fak.ci-provenance.v1", "workflow": "ci-fast", "job": job, "result": "success", "release_eligible": false,
			"source_scope": "git-commit-tree", "effective_inputs_verified": false,
			"tested_sha": f.env["EXPECTED_SHA"], "tested_tree": f.env["EXPECTED_TREE"], "represented_sha": f.env["GITHUB_SHA"],
			"event": "push", "ref": "refs/heads/main", "run_id": "100", "run_attempt": "1", "ci_class": "fast", "requested_intent": "",
			"environment": environment, "commands": commands, "coverage": coverage, "steps": steps,
		})
	}
	return f
}

func provenanceWorkflowScript(t *testing.T, workflow, name string) string {
	t.Helper()
	workflow = strings.ReplaceAll(workflow, "\r\n", "\n")
	begin, end := "          # ci-provenance: "+name+"-begin\n", "          # ci-provenance: "+name+"-end"
	if strings.Count(workflow, begin) != 1 || strings.Count(workflow, end) != 1 {
		t.Fatalf("workflow lacks unique executable %s provenance contract", name)
	}
	_, body, _ := strings.Cut(workflow, begin)
	body, _, found := strings.Cut(body, end)
	if !found || strings.TrimSpace(body) == "" {
		t.Fatalf("workflow has no executable %s provenance body", name)
	}
	var script strings.Builder
	for _, line := range strings.Split(body, "\n") {
		if strings.TrimSpace(line) != "" && !strings.HasPrefix(line, "          ") {
			t.Fatalf("%s body escaped YAML run block: %q", name, line)
		}
		script.WriteString(strings.TrimPrefix(line, "          "))
		script.WriteByte('\n')
	}
	return script.String()
}

func fastProvenanceVerifierScript(t *testing.T, workflow string) string {
	t.Helper()
	if strings.Contains(workflow, "# ci-provenance: verify-fast-begin") {
		t.Log("production seam: marked source-bound verifier")
		return provenanceWorkflowScript(t, workflow, "verify-fast")
	}
	// Run the actual legacy status-only gate on the parent revision. Negative
	// receipts must expose its false green, rather than failing only because
	// the new verifier marker or a new production API is absent.
	job := workflowJobBlock(t, workflow, "build-vet-test-fast")
	_, step, found := strings.Cut(job, "      - name: require every go test shard and the build gate green\n")
	if !found {
		t.Fatal("workflow has neither source-bound verifier nor legacy aggregate gate")
	}
	_, body, found := strings.Cut(step, "        run: |\n")
	if !found {
		t.Fatal("legacy aggregate gate has no executable run body")
	}
	var script strings.Builder
	for _, line := range strings.Split(body, "\n") {
		if strings.TrimSpace(line) != "" && !strings.HasPrefix(line, "          ") {
			break
		}
		script.WriteString(strings.TrimPrefix(line, "          "))
		script.WriteByte('\n')
	}
	if strings.TrimSpace(script.String()) == "" {
		t.Fatal("legacy aggregate gate body is empty")
	}
	t.Log("production seam: legacy status-only aggregate")
	return script.String()
}

func requireProvenanceTools(t *testing.T) {
	t.Helper()
	for _, tool := range []string{"bash", "jq"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("workflow provenance execution requires %s: %v", tool, err)
		}
	}
}

func runProvenanceScript(t *testing.T, script, dir string, env map[string]string) ([]byte, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "bash", "--noprofile", "--norc", "-c", script)
	cmd.Dir = dir
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if _, overridden := env[key]; !overridden && key != "BASH_ENV" && key != "ENV" {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	for key, value := range env {
		cmd.Env = append(cmd.Env, key+"="+value)
	}
	return cmd.CombinedOutput()
}

func writeProvenanceJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func readProvenanceJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func readFastReleaseWorkflow(t *testing.T) string {
	t.Helper()
	root := findRepoRoot(t)
	data, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci-fast.yml"))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(data), "\r\n", "\n")
}

// workflowJobBlock returns the text of one top-level job, from its `  <id>:`
// line up to the next job key at the same indent.
func workflowJobBlock(t *testing.T, workflow, id string) string {
	t.Helper()
	start := strings.Index(workflow, "\n  "+id+":\n")
	if start < 0 {
		t.Fatalf("ci-fast has no %q job", id)
	}
	block := workflow[start+1:]
	lines := strings.SplitAfter(block, "\n")
	var b strings.Builder
	b.WriteString(lines[0])
	for _, line := range lines[1:] {
		if strings.HasPrefix(line, "  ") && !strings.HasPrefix(line, "   ") && !strings.HasPrefix(line, "  #") {
			break
		}
		b.WriteString(line)
	}
	return b.String()
}

// timeoutMinutesAtIndent reads the first `timeout-minutes:` key at exactly the
// given indent (job keys sit at 4 spaces, step keys at 8).
func timeoutMinutesAtIndent(t *testing.T, block string, indent int) int {
	t.Helper()
	prefix := strings.Repeat(" ", indent) + "timeout-minutes:"
	for _, line := range strings.Split(block, "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, prefix)))
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		return v
	}
	t.Fatalf("no %q line in block", strings.TrimSpace(prefix))
	return 0
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}

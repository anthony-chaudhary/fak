package devcmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBuildBatchOptionalReadOnly exercises the optional 'fak-dev build batch'
// surface end to end with harmless temporary fixtures and no subprocess:
// seed -> plan/status round-trip, source drift, unresolved attempts,
// conflicting patch paths, and malformed/path-escape refusals.
func TestBuildBatchOptionalReadOnly(t *testing.T) {
	t.Run("SeedPlanStatusRoundTrip", func(t *testing.T) {
		root := t.TempDir()
		writeBatchFile(t, root, "internal/devcmd/a.go", "alpha\n")
		writeBatchFile(t, root, "pkg/x.go", "gamma\n")

		var out, errOut bytes.Buffer
		code := RunBuildBatch(&out, &errOut, []string{
			"setup", "--root", ".", "--target-commit", "abc123def", "--profile", "dev",
			"--publisher", "ops-coordinator",
			"--input", "internal/devcmd/a.go", "--input", "pkg/x.go",
			"--patch", "anthony-chaudhary/fak#100=internal/devcmd/a.go",
			"--patch", "anthony-chaudhary/fak#101=pkg/x.go",
		})
		if code != 0 {
			t.Fatalf("setup code = %d, want 0; stderr=%s", code, errOut.String())
		}
		var seed buildBatchRecord
		if err := json.Unmarshal(out.Bytes(), &seed); err != nil {
			t.Fatalf("seed is not one JSON object: %v\n%s", err, out.String())
		}
		if seed.Schema != buildBatchRecordSchema || len(seed.Inputs) != 2 || len(seed.Patches) != 2 {
			t.Fatalf("seed shape drifted: %+v", seed)
		}

		record := seed
		record.Inputs[0].SHA256 = batchFileHash(t, root, "internal/devcmd/a.go")
		record.Inputs[1].SHA256 = batchFileHash(t, root, "pkg/x.go")
		record.Stages = []buildBatchStage{
			{Name: "affected_validate", Outcome: "success", EvidenceRef: "receipt://validate-1"},
			{Name: "commit", Outcome: "unknown"},
		}
		record.VerificationDebt = []string{"full suite not run"}
		record.DeclaredCounts = &buildBatchCounts{Inputs: 2, Patches: 2, AffectedPackages: 2}
		writeBatchJSON(t, root, "batch.json", record)

		planOut, planErr := runBatchView(t, root, "plan", "batch.json")
		var plan buildBatchView
		if err := json.Unmarshal(planOut, &plan); err != nil {
			t.Fatalf("plan view is not one JSON object: %v\n%s", err, planErr.String())
		}
		if plan.Command != "plan" || len(plan.Inputs) != 2 {
			t.Fatalf("plan view drifted: command=%s inputs=%d", plan.Command, len(plan.Inputs))
		}
		for _, input := range plan.Inputs {
			if input.State != "fresh" {
				t.Fatalf("input %s state = %s, want fresh", input.Path, input.State)
			}
		}
		if !stringsEqual(plan.AffectedPackages, []string{"internal/devcmd", "pkg"}) {
			t.Fatalf("affected_packages = %v, want deterministic sorted union", plan.AffectedPackages)
		}
		if plan.Counts != (buildBatchViewCounts{Inputs: 2, Patches: 2, AffectedPackages: 2}) {
			t.Fatalf("counts = %+v", plan.Counts)
		}
		if plan.Stages != nil {
			t.Fatalf("plan view must not resolve stage observations: %+v", plan.Stages)
		}

		// Determinism: the same exact record yields byte-identical views.
		again, _ := runBatchView(t, root, "plan", "batch.json")
		if !bytes.Equal(again, planOut) {
			t.Fatalf("plan view is not deterministic:\nfirst: %s\nsecond: %s", planOut, again)
		}

		statusOut, statusErr := runBatchView(t, root, "status", "batch.json")
		var status buildBatchView
		if err := json.Unmarshal(statusOut, &status); err != nil {
			t.Fatalf("status view is not one JSON object: %v\n%s", err, statusErr.String())
		}
		if status.Command != "status" || len(status.Stages) != 2 {
			t.Fatalf("status view drifted: command=%s stages=%d", status.Command, len(status.Stages))
		}
		if status.Stages[0].Resolution != "settled" || status.Stages[0].Outcome != "success" {
			t.Fatalf("terminal attempt with evidence must settle: %+v", status.Stages[0])
		}
		if status.Stages[1].Resolution != "unresolved" {
			t.Fatalf("unknown attempt must stay unresolved: %+v", status.Stages[1])
		}
		if !stringsEqual(status.VerificationDebt, []string{"full suite not run"}) {
			t.Fatalf("verification debt must be retained: %v", status.VerificationDebt)
		}
		if status.Publisher != "ops-coordinator" {
			t.Fatalf("publisher = %q", status.Publisher)
		}
		if !sameBatchInputs(plan.Inputs, status.Inputs) {
			t.Fatalf("plan and status must inspect the same exact record:\nplan=%+v\nstatus=%+v", plan.Inputs, status.Inputs)
		}

		var human, humanErr bytes.Buffer
		if code := RunBuildBatch(&human, &humanErr, []string{"status", "--root", root, "--record", "batch.json"}); code != 0 {
			t.Fatalf("human status code = %d; stderr=%s", code, humanErr.String())
		}
		for _, want := range []string{"fak build batch status: record=batch.json", "affected_packages", "unresolved", "verification_debt", "ops-coordinator (advisory; grants no ownership)"} {
			if !strings.Contains(human.String(), want) {
				t.Errorf("human status missing %q:\n%s", want, human.String())
			}
		}
	})

	t.Run("SourceDriftInvalidatesEvidence", func(t *testing.T) {
		root := t.TempDir()
		writeBatchFile(t, root, "src/one.go", "one\n")
		writeBatchFile(t, root, "src/two.go", "two\n")
		record := pinnedBatchRecord(t, root, []string{"src/one.go", "src/two.go"}, "anthony-chaudhary/fak#200")
		writeBatchJSON(t, root, "batch.json", record)

		before, _ := runBatchView(t, root, "status", "batch.json")
		var beforeView buildBatchView
		json.Unmarshal(before, &beforeView)
		for _, input := range beforeView.Inputs {
			if input.State != "fresh" {
				t.Fatalf("baseline input %s state = %s, want fresh", input.Path, input.State)
			}
		}

		// Drift one input after pinning: the stale evidence is invalidated in
		// place while the rest of the view stays intact.
		if err := os.WriteFile(filepath.Join(root, "src", "one.go"), []byte("one-drifted\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		after, errOut := runBatchView(t, root, "status", "batch.json")
		var afterView buildBatchView
		if err := json.Unmarshal(after, &afterView); err != nil {
			t.Fatalf("status view after drift: %v\n%s", err, errOut.String())
		}
		states := map[string]string{}
		for _, input := range afterView.Inputs {
			states[input.Path] = input.State
		}
		if states["src/one.go"] != "stale" || states["src/two.go"] != "fresh" {
			t.Fatalf("drift states = %v, want src/one.go stale and src/two.go fresh", states)
		}
	})

	t.Run("UnresolvedAttemptPreserved", func(t *testing.T) {
		root := t.TempDir()
		writeBatchFile(t, root, "src/one.go", "one\n")
		record := pinnedBatchRecord(t, root, []string{"src/one.go"}, "anthony-chaudhary/fak#300")
		record.Stages = []buildBatchStage{
			{Name: "no_evidence_success", Outcome: "success"},
			{Name: "explicit_unknown", Outcome: "unknown", EvidenceRef: "receipt://halfway"},
			{Name: "settled_failure", Outcome: "failed", EvidenceRef: "receipt://fail-1"},
			{Name: "not_observed"},
		}
		writeBatchJSON(t, root, "batch.json", record)

		out, errOut := runBatchView(t, root, "status", "batch.json")
		var view buildBatchView
		if err := json.Unmarshal(out, &view); err != nil {
			t.Fatalf("status view: %v\n%s", err, errOut.String())
		}
		resolutions := map[string]string{}
		for _, stage := range view.Stages {
			resolutions[stage.Name] = stage.Resolution
		}
		want := map[string]string{
			"no_evidence_success": "unresolved",
			"explicit_unknown":    "unresolved",
			"settled_failure":     "settled",
			"not_observed":        "unresolved",
		}
		for name, resolution := range want {
			if resolutions[name] != resolution {
				t.Fatalf("stage %s resolution = %s, want %s", name, resolutions[name], resolution)
			}
		}
	})

	t.Run("ConflictingDeclarationsRefused", func(t *testing.T) {
		root := t.TempDir()
		writeBatchFile(t, root, "src/one.go", "one\n")

		conflicts := map[string]func(*buildBatchRecord){
			"membership conflict": func(r *buildBatchRecord) {
				r.Patches = []buildBatchPatch{
					{Issue: "anthony-chaudhary/fak#400", Paths: []string{"src/one.go"}},
					{Issue: "anthony-chaudhary/fak#401", Paths: []string{"src/one.go"}},
				}
			},
			"conflicting count declaration": func(r *buildBatchRecord) {
				r.DeclaredCounts = &buildBatchCounts{Inputs: 99}
			},
			"conflicting current-target declaration": func(r *buildBatchRecord) {
				r.Target = &buildBatchTarget{Commit: "aaa"}
				r.TargetCommit = "bbb"
			},
		}
		for name, mutate := range conflicts {
			record := pinnedBatchRecord(t, root, []string{"src/one.go"}, "anthony-chaudhary/fak#400")
			mutate(&record)
			writeBatchJSON(t, root, "batch.json", record)
			var out, errOut bytes.Buffer
			if code := RunBuildBatch(&out, &errOut, []string{"status", "--root", root, "--record", "batch.json"}); code != 2 {
				t.Fatalf("%s: code = %d, want refusal 2; stdout=%s", name, code, out.String())
			}
			if !strings.Contains(errOut.String(), name) {
				t.Fatalf("%s: stderr does not name the conflict: %s", name, errOut.String())
			}
		}
	})

	t.Run("MalformedJSONRefused", func(t *testing.T) {
		root := t.TempDir()
		writeBatchFile(t, root, "src/one.go", "one\n")
		cases := map[string]string{
			"garbage":      "{not json",
			"trailing":     `{"schema":"fak.build-batch.v1","inputs":[]} extra`,
			"wrong schema": `{"schema":"fak.build-batch.v0","inputs":[]}`,
			"bad outcome":  `{"schema":"fak.build-batch.v1","inputs":[],"stages":[{"name":"s","outcome":"maybe"}]}`,
		}
		for name, body := range cases {
			if err := os.WriteFile(filepath.Join(root, "batch.json"), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			var out, errOut bytes.Buffer
			if code := RunBuildBatch(&out, &errOut, []string{"plan", "--root", root, "--record", "batch.json"}); code != 2 {
				t.Fatalf("%s: code = %d, want refusal 2; stdout=%s", name, code, out.String())
			}
			if errOut.String() == "" {
				t.Fatalf("%s: refusal printed nothing on stderr", name)
			}
		}
	})

	t.Run("PathEscapeRefused", func(t *testing.T) {
		root := t.TempDir()
		outside := t.TempDir()
		writeBatchFile(t, outside, "loot.txt", "secret\n")

		escapes := map[string]func(*buildBatchRecord){
			"input parent traversal": func(r *buildBatchRecord) {
				r.Inputs = []buildBatchInput{{Path: "../loot.txt"}}
			},
			"absolute input": func(r *buildBatchRecord) {
				r.Inputs = []buildBatchInput{{Path: filepath.Join(outside, "loot.txt")}}
			},
		}
		for name, mutate := range escapes {
			record := buildBatchRecord{Schema: buildBatchRecordSchema, Inputs: []buildBatchInput{}, Patches: []buildBatchPatch{}, Stages: []buildBatchStage{}, VerificationDebt: []string{}}
			mutate(&record)
			writeBatchJSON(t, root, "batch.json", record)
			var out, errOut bytes.Buffer
			if code := RunBuildBatch(&out, &errOut, []string{"plan", "--root", root, "--record", "batch.json"}); code != 2 {
				t.Fatalf("%s: code = %d, want refusal 2; stdout=%s", name, code, out.String())
			}
			if !strings.Contains(errOut.String(), "the explicit root") {
				t.Fatalf("%s: stderr does not name the escape: %s", name, errOut.String())
			}
		}

		// The record itself must also live below the explicit root.
		if err := os.WriteFile(filepath.Join(outside, "outside.json"), []byte(`{"schema":"fak.build-batch.v1","inputs":[]}`), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		if code := RunBuildBatch(&out, &errOut, []string{"plan", "--root", root, "--record", "../" + filepath.Base(outside) + "/outside.json"}); code != 2 {
			t.Fatalf("record above root: code = %d, want refusal 2", code)
		}
	})
}

func runBatchView(t *testing.T, root, command, record string) ([]byte, *bytes.Buffer) {
	t.Helper()
	var out, errOut bytes.Buffer
	if code := RunBuildBatch(&out, &errOut, []string{command, "--root", root, "--record", record, "--json"}); code != 0 {
		t.Fatalf("%s code = %d, want 0; stderr=%s", command, code, errOut.String())
	}
	return out.Bytes(), &errOut
}

func pinnedBatchRecord(t *testing.T, root string, paths []string, issue string) buildBatchRecord {
	t.Helper()
	inputs := make([]buildBatchInput, 0, len(paths))
	patchPaths := make([]string, 0, len(paths))
	for _, rel := range paths {
		inputs = append(inputs, buildBatchInput{Path: rel, SHA256: batchFileHash(t, root, rel)})
		patchPaths = append(patchPaths, rel)
	}
	return buildBatchRecord{
		Schema:           buildBatchRecordSchema,
		Root:             ".",
		Target:           &buildBatchTarget{Commit: "abc123def", Profile: "dev"},
		Inputs:           inputs,
		Patches:          []buildBatchPatch{{Issue: issue, Paths: patchPaths}},
		Stages:           []buildBatchStage{},
		VerificationDebt: []string{},
	}
}

func writeBatchFile(t *testing.T, root, rel, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(filepath.Join(root, filepath.FromSlash(rel))), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(rel)), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeBatchJSON(t *testing.T, root, rel string, record buildBatchRecord) {
	t.Helper()
	raw, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(rel)), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func batchFileHash(t *testing.T, root, rel string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(func() []byte {
		sum := sha256.Sum256(raw)
		return sum[:]
	}())
}

func stringsEqual(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func sameBatchInputs(left, right []buildBatchInputView) bool {
	return stringsEqual(inputPaths(left), inputPaths(right)) &&
		stringsEqual(inputStates(left), inputStates(right))
}

func inputPaths(inputs []buildBatchInputView) []string {
	out := make([]string, 0, len(inputs))
	for _, input := range inputs {
		out = append(out, input.Path+"="+input.SHA256)
	}
	return out
}

func inputStates(inputs []buildBatchInputView) []string {
	out := make([]string, 0, len(inputs))
	for _, input := range inputs {
		out = append(out, input.State)
	}
	return out
}

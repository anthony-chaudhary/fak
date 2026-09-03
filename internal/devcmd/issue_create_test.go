package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/categorybaseline"
)

func TestIssueCreateShiftLeftScopeRequiresBothDecisions(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{name: "missing both", body: "## Current state\nwork", want: "require exactly one"},
		{name: "empty core", body: "## Core through-line\n<!-- TODO -->\n\n## Gold-plating boundary\nNo extras.", want: "Core through-line is empty"},
		{name: "empty boundary", body: "## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\n", want: "Gold-plating boundary is empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", tc.body, "--dry-run"}, nil)
			if code != 2 || !strings.Contains(errb.String(), tc.want) {
				t.Fatalf("code=%d stderr=%q, want %q", code, errb.String(), tc.want)
			}
		})
	}
}

func TestIssueCreateShiftLeftScopeCanonicalizesLegacyHeadings(t *testing.T) {
	body := "## Parent context\n#99\n\n## In scope\nChange -> real seam -> observable outcome -> witness.\n\n## Out of scope\nDo not add unrelated polish." + validIssueCreateProblemFrame("Core")
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "scoped", "--body", body,
		"--estimate-points", "1", "--parent-baseline-points", "1",
		"--target-envelope", "- acceptance pass rate: = 100 percent",
		"--witnessed-envelope", "- acceptance pass rate: = 100 percent",
		"--dry-run", "--json",
	}, nil)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got issueCreateResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	bodyArg := got.Args[5]
	if !strings.Contains(bodyArg, "## Core through-line") || !strings.Contains(bodyArg, "## Gold-plating boundary") {
		t.Fatalf("body did not canonicalize shift-left decisions:\n%s", bodyArg)
	}
	if strings.Contains(bodyArg, "## In scope") || strings.Contains(bodyArg, "## Out of scope") {
		t.Fatalf("body retained generic scope headings:\n%s", bodyArg)
	}
}

func TestIssueCreateDryRunDoesNotInvokeRunner(t *testing.T) {
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "feat: per-session activity cell",
		"--body", "add a pane row",
		"--raw-body", "--dry-run",
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if called {
		t.Fatalf("--dry-run must never invoke the runner")
	}
	if !strings.Contains(out.String(), "gh issue create") {
		t.Fatalf("dry-run output missing rendered gh argv: %s", out.String())
	}
}

func TestIssueCreateBuildsExpectedGHArgs(t *testing.T) {
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/9", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "feat: thing",
		"--body", "body text",
		"--labels", "agent-handoff,next-step",
		"--repo", "owner/repo", "--raw-body",
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if len(calls) != 1 {
		t.Fatalf("runner calls = %d, want 1", len(calls))
	}
	joined := strings.Join(calls[0], " ")
	for _, want := range []string{"issue create", "--title feat: thing", "--body body text", "--label agent-handoff", "--label next-step", "--repo owner/repo"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("gh args missing %q: %v", want, calls[0])
		}
	}
	if got := strings.TrimSpace(out.String()); got != "https://example.test/issues/9" {
		t.Fatalf("stdout = %q, want the issue URL", got)
	}
}

func TestIssueCreateBodyFileReadsContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "body.md")
	if err := os.WriteFile(path, []byte("body from file"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/2", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "t",
		"--body-file", path, "--raw-body",
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(strings.Join(calls[0], " "), "--body body from file") {
		t.Fatalf("gh args missing file body: %v", calls[0])
	}
}

func TestIssueCreateRequiresTitleAndBody(t *testing.T) {
	cases := [][]string{
		{"--body", "b"},
		{"--title", "t"},
		{"--title", "t", "--body", "b", "--body-file", "x"},
	}
	for _, argv := range cases {
		var out, errb bytes.Buffer
		code := runIssueCreateWith(&out, &errb, argv, func(args []string) (string, string, bool) {
			t.Fatalf("runner must not be called for invalid flags: %v", args)
			return "", "", false
		})
		if code != 2 {
			t.Fatalf("argv=%v exit=%d, want 2 (stderr=%s)", argv, code, errb.String())
		}
	}
}

func TestIssueCreateReportsGHFailure(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "", "HTTP 422: validation failed", false
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", "b", "--raw-body"}, runner)
	if code != 1 {
		t.Fatalf("exit=%d, want 1", code)
	}
	if !strings.Contains(errb.String(), "validation failed") {
		t.Fatalf("stderr missing gh failure detail: %s", errb.String())
	}
}

func TestIssueCreateJSONOutput(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "https://example.test/issues/3", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", "b", "--raw-body", "--json"}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got issueCreateResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse json: %v\n%s", err, out.String())
	}
	if !got.OK || got.URL != "https://example.test/issues/3" {
		t.Fatalf("result = %+v", got)
	}
}

func validIssueCreateProblemFrame(class string) string {
	return "\n- Centrality: " + class + "\n- P1 Context: advanced - captures context once\n- P2 Net value: preserved - no regression\n- P3 Adaptation: N/A - no adaptive surface\n- P4 Operations: advanced - real path\n"
}

func TestIssueCreateDefaultsProjectWorkToProduction(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWith(&stdout, &stderr, []string{"--title", "scoped", "--body", "## Parent context\n#4638\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nDo not add unrelated polish." + validIssueCreateProblemFrame("Core"), "--estimate-points", "3", "--parent-baseline-points", "8", "--target-envelope", "- paths: >= 1 command", "--witnessed-envelope", "- paths: 1 command", "--dry-run", "--json"}, nil)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var result issueCreateResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	body := ""
	for i := range result.Args {
		if result.Args[i] == "--body" && i+1 < len(result.Args) {
			body = result.Args[i+1]
		}
	}
	for _, want := range []string{"Estimate: 3 points", "Contribution: 3/8 points", "## Completion standard\nproduction"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body=%q missing %q", body, want)
		}
	}
}

func TestIssueCreatePreservesExplicitDemo(t *testing.T) {
	body := "## Parent context\n#4638\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nDo not add unrelated polish.\n\n## Work estimate\nEstimate: 1 point.\n\n## Overall completion contribution\nContribution: 1/8 points.\n\n## Completion standard\ndemo" + validIssueCreateProblemFrame("Peripheral")
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWith(&stdout, &stderr, []string{"--title", "demo", "--body", body, "--dry-run", "--json"}, nil)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var result issueCreateResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(result.Args, "\n")
	if !strings.Contains(joined, "## Completion standard\ndemo") || strings.Contains(joined, "## Completion standard\nproduction") {
		t.Fatalf("args=%q", result.Args)
	}
}

func TestIssueCreateRefusesMissingProjectWorkNumbers(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWith(&stdout, &stderr, []string{"--title", "unknown", "--body", "## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nDo not add unrelated polish." + validIssueCreateProblemFrame("Stewardship (release obligation)"), "--dry-run"}, nil)
	if code != 2 || !strings.Contains(stderr.String(), "estimate-points") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func TestIssueCreateClassificationValidatesTrackedBaseline(t *testing.T) {
	registry := categorybaseline.Normalize(categorybaseline.Registry{Categories: []categorybaseline.Category{{Name: "agent-work-profile", Layers: []string{"default-medium", "provider-effectiveness"}, CompletedLayer: "default-medium", NextLayer: "provider-effectiveness", Witness: "witness"}}})
	got, err := issueCreateClassifyBody("## For\noperator", "Agent-Work-Profile", "Provider-Effectiveness", registry)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "## Category\n\nagent-work-profile") || !strings.Contains(got, "## Layer\n\nprovider-effectiveness") {
		t.Fatalf("classified body = %q", got)
	}
	for _, tc := range []struct {
		name     string
		category string
		layer    string
		want     string
	}{
		{name: "missing layer", category: "agent-work-profile", want: "--layer is required"},
		{name: "unknown layer", category: "agent-work-profile", layer: "gold-plating", want: "not declared"},
		{name: "orphan layer", layer: "provider-effectiveness", want: "requires --category"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := issueCreateClassifyBody("body", tc.category, tc.layer, registry); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestIssueCreateDryRunEmitsCategoryLayer(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWith(&stdout, &stderr, []string{"--title", "next depth", "--body", "body", "--category", "agent-work-profile", "--layer", "provider-effectiveness", "--dry-run", "--raw-body", "--json"}, nil)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Args []string `json:"args"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	body := ""
	for i, arg := range got.Args {
		if arg == "--body" && i+1 < len(got.Args) {
			body = got.Args[i+1]
		}
	}
	if !strings.Contains(body, "## Category\n\nagent-work-profile") || !strings.Contains(body, "## Layer\n\nprovider-effectiveness") {
		t.Fatalf("body = %q args=%v", body, got.Args)
	}
}

func TestIssueCreateScrubsProtectedTitleAndBodyFile(t *testing.T) {
	cpu, gpu := "da"+"33", "dgx"+"1"
	bodyFile := filepath.Join(t.TempDir(), "body.md")
	if err := os.WriteFile(bodyFile, []byte("compare "+cpu+" with "+gpu), 0o600); err != nil {
		t.Fatal(err)
	}
	var got []string
	runner := func(args []string) (string, string, bool) {
		got = append([]string(nil), args...)
		return "https://example.invalid/7\n", "", true
	}
	var out, errb bytes.Buffer
	if code := runIssueCreateWith(&out, &errb, []string{"--title", "move " + cpu + " to " + gpu, "--body-file", bodyFile, "--raw-body"}, runner); code != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "move CPU server to GPU server") || !strings.Contains(joined, "compare CPU server with GPU server") {
		t.Fatalf("gh argv not scrubbed: %#v", got)
	}
}

func TestIssueCreateRequiresCanonicalProblemFrameBeforeMutation(t *testing.T) {
	body := "## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras."
	called := false
	runner := func([]string) (string, string, bool) { called = true; return "", "", true }
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", body}, runner)
	if code != 2 || called {
		t.Fatalf("code=%d called=%v stderr=%s", code, called, errb.String())
	}
	for _, want := range []string{"problem frame is incomplete", "problem_frame_unclassified", "declare Centrality and P1-P4"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("stderr missing %q: %s", want, errb.String())
		}
	}
}

func TestIssueCreateAcceptsAllCanonicalCentralityClasses(t *testing.T) {
	classes := []string{"Core", "Enabling (managed-context outcome)", "Stewardship (release obligation)", "Peripheral"}
	for _, class := range classes {
		t.Run(class, func(t *testing.T) {
			body := "## Parent context\n#1\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras." + validIssueCreateProblemFrame(class)
			var out, errb bytes.Buffer
			code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", body, "--estimate-points", "1", "--parent-baseline-points", "1", "--target-envelope", "- paths: >= 1 command", "--witnessed-envelope", "- paths: 1 command", "--dry-run"}, nil)
			if code != 0 {
				t.Fatalf("code=%d stderr=%s", code, errb.String())
			}
		})
	}
}

func TestIssueCreateMalformedProblemFrameReturnsCanonicalRepair(t *testing.T) {
	body := "## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras.\n\nCentrality: Enabling\nP1 Context: N/A\nP2 Net value: advanced - measured\nP3 Adaptation: preserved - bounded\nP4 Operations: advanced - live path"
	called := false
	runner := func([]string) (string, string, bool) { called = true; return "", "", true }
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", body}, runner)
	if code != 2 || called {
		t.Fatalf("code=%d called=%v stderr=%s", code, called, errb.String())
	}
	for _, want := range []string{"problem_centrality_target_missing", "problem_check_p1_ceremonial", "name the Core outcome", "bare label is not evidence"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("stderr missing %q: %s", want, errb.String())
		}
	}
}

func TestIssueCreateShiftLeftDefaultLabels(t *testing.T) {
	body := "## Parent context\n#1\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras." + validIssueCreateProblemFrame("Core")
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWith(&stdout, &stderr, []string{
		"--title", "feat(model): decompress MoE quant experts",
		"--body", body,
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
		"--dry-run", "--json",
	}, nil)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var result issueCreateResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	expected := []string{"enhancement", "class:dev", "priority/P1", "gen/next", "model", "moe", "quantization"}
	for _, exp := range expected {
		found := false
		for _, l := range result.Labels {
			if l == exp {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("result.Labels = %v, missing expected default label %q", result.Labels, exp)
		}
	}
}

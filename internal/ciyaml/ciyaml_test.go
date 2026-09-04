package ciyaml

import (
	"strings"
	"testing"
)

func TestLintValidWorkflow(t *testing.T) {
	content := `name: Main CI
on: [push, pull_request]

# Global comment
jobs:
  build:
    name: Build & Test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Compile and test
        run: |
          # Bash comment inside block scalar
          go build ./...
          go test -v ./...
          echo "done with test"

  deploy:
    name: Deploy artifact
    runs-on: ubuntu-latest
    needs: [build]
    steps:
      - name: Deploy
        run: echo "deploying..."
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(violations) != 0 {
		for _, v := range violations {
			t.Errorf("unexpected violation at line %d, col %d: [%s] %s", v.Line, v.Column, v.Rule, v.Message)
		}
	}
}

func TestLintTabsInIndentation(t *testing.T) {
	content := "jobs:\n\tbuild:\n    runs-on: ubuntu-latest\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundTab := false
	for _, v := range violations {
		if v.Rule == "tabs-forbidden" && v.Line == 2 {
			foundTab = true
			if v.Severity != SeverityError {
				t.Errorf("expected SeverityError, got %v", v.Severity)
			}
		}
	}
	if !foundTab {
		t.Fatalf("expected tabs-forbidden violation on line 2, got: %+v", violations)
	}
}

func TestLintTabsInStructure(t *testing.T) {
	content := "name: CI\njobs:\n  build:\truns-on: ubuntu-latest\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundTab := false
	for _, v := range violations {
		if v.Rule == "tabs-forbidden" && v.Line == 3 {
			foundTab = true
		}
	}
	if !foundTab {
		t.Fatalf("expected structural tabs-forbidden violation on line 3, got: %+v", violations)
	}
}

func TestLintTabsInsideBlockScalarAllowed(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Script with tabs
        run: |
          echo "start"
          	# tab indented script line inside scalar
          	ls -la
      - name: Next step
        run: echo ok
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, v := range violations {
		if v.Rule == "tabs-forbidden" {
			t.Errorf("unexpected tabs-forbidden violation inside block scalar: line %d col %d", v.Line, v.Column)
		}
	}
}

func TestLintUnclosedSingleQuote(t *testing.T) {
	content := "name: CI\njobs:\n  build:\n    runs-on: 'ubuntu-latest\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundQuote := false
	for _, v := range violations {
		if v.Rule == "unclosed-quote" && v.Line == 4 {
			foundQuote = true
		}
	}
	if !foundQuote {
		t.Fatalf("expected unclosed-quote violation on line 4, got: %+v", violations)
	}
}

func TestLintUnclosedDoubleQuote(t *testing.T) {
	content := "name: \"Unclosed Workflow Name\njobs:\n  build:\n    runs-on: ubuntu-latest\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundQuote := false
	for _, v := range violations {
		if v.Rule == "unclosed-quote" && v.Line == 1 {
			foundQuote = true
		}
	}
	if !foundQuote {
		t.Fatalf("expected unclosed-quote violation on line 1, got: %+v", violations)
	}
}

func TestLintQuotesInsideCommentsIgnored(t *testing.T) {
	content := `name: CI # this is a comment with an unclosed "quote and 'single quote
jobs:
  build:
    runs-on: ubuntu-latest # another comment with "quote
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, v := range violations {
		if v.Rule == "unclosed-quote" {
			t.Errorf("unexpected unclosed-quote violation from comment: %+v", v)
		}
	}
}

func TestLintQuotesInsideBlockScalarIgnored(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Bash with unbalanced quotes
        run: |
          echo "unclosed bash string
          echo 'another unclosed
      - name: Follow up
        run: echo done
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, v := range violations {
		if v.Rule == "unclosed-quote" {
			t.Errorf("unexpected unclosed-quote violation inside block scalar: %+v", v)
		}
	}
}

func TestLintEscapedQuotesHandled(t *testing.T) {
	content := `name: "Escaped \" double quote"
jobs:
  build:
    name: 'Escaped '' single quote'
    runs-on: ubuntu-latest
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, v := range violations {
		if v.Rule == "unclosed-quote" {
			t.Errorf("unexpected unclosed-quote violation for properly escaped quotes: %+v", v)
		}
	}
}

func TestLintIndentationOddSpaces(t *testing.T) {
	content := "jobs:\n   build:\n    runs-on: ubuntu-latest\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundIndent := false
	for _, v := range violations {
		if v.Rule == "indentation-consistency" && v.Line == 2 {
			foundIndent = true
		}
	}
	if !foundIndent {
		t.Fatalf("expected indentation-consistency violation on line 2, got: %+v", violations)
	}
}

func TestLintIndentationDedentMismatch(t *testing.T) {
	// Root (0), jobs.build (2), runs-on (6), steps (4) where 4 was never in stack [0, 2, 6]
	content := "jobs:\n  build:\n      runs-on: ubuntu-latest\n    steps:\n"

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundMismatch := false
	for _, v := range violations {
		if v.Rule == "indentation-consistency" && v.Line == 4 {
			foundMismatch = true
		}
	}
	if !foundMismatch {
		t.Fatalf("expected indentation-consistency dedent mismatch on line 4, got: %+v", violations)
	}
}

func TestLintDuplicateJobKeys(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
  test:
    runs-on: ubuntu-latest
  build:
    runs-on: macos-latest
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundDuplicate := false
	for _, v := range violations {
		if (v.Rule == "duplicate-job-key" || v.Rule == "duplicate-key") && v.Line == 6 {
			foundDuplicate = true
		}
	}
	if !foundDuplicate {
		t.Fatalf("expected duplicate-job-key violation on line 6, got: %+v", violations)
	}
}

func TestLintDuplicateStepKeys(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Step 1
        run: echo 1
        run: echo 2
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundDuplicate := false
	for _, v := range violations {
		if (v.Rule == "duplicate-step-key" || v.Rule == "duplicate-key") && v.Line == 7 {
			foundDuplicate = true
		}
	}
	if !foundDuplicate {
		t.Fatalf("expected duplicate-step-key violation on line 7, got: %+v", violations)
	}
}

func TestLintDuplicateStepIDs(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - id: compile
        run: echo 1
      - id: compile
        run: echo 2
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundDuplicate := false
	for _, v := range violations {
		if v.Rule == "duplicate-step-id" && v.Line == 7 {
			foundDuplicate = true
		}
	}
	if !foundDuplicate {
		t.Fatalf("expected duplicate-step-id violation on line 7, got: %+v", violations)
	}
}

func TestLintDuplicateTopLevelKeys(t *testing.T) {
	content := `name: Workflow A
name: Workflow B
jobs:
  build:
    runs-on: ubuntu-latest
`

	violations, err := Lint(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundDuplicate := false
	for _, v := range violations {
		if v.Rule == "duplicate-key" && v.Line == 2 {
			foundDuplicate = true
		}
	}
	if !foundDuplicate {
		t.Fatalf("expected duplicate-key on line 2, got: %+v", violations)
	}
}

func TestParseWorkflowFailClosedOnDuplicateKeys(t *testing.T) {
	content := `jobs:
  build:
    runs-on: ubuntu-latest
  build:
    runs-on: macos-latest
`

	wf, err := ParseWorkflow(content)
	if err == nil {
		t.Fatalf("expected ParseWorkflow to fail-closed on duplicate keys, got nil error and wf=%+v", wf)
	}
	if !strings.Contains(err.Error(), "duplicate") {
		t.Errorf("expected error message to mention duplicate keys, got: %v", err)
	}
}

func TestParseWorkflowSuccess(t *testing.T) {
	content := `name: BuildPipeline
on: [push, pull_request]

jobs:
  compile:
    name: Compile Binaries
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Build
        run: |
          go build ./...
          echo done
  verify:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: [compile]
    steps:
      - name: Test
        run: go test ./...
`

	wf, err := ParseWorkflow(content)
	if err != nil {
		t.Fatalf("ParseWorkflow failed: %v", err)
	}

	if wf.Name != "BuildPipeline" {
		t.Errorf("expected workflow name 'BuildPipeline', got %q", wf.Name)
	}
	if len(wf.On) != 2 || wf.On[0] != "push" || wf.On[1] != "pull_request" {
		t.Errorf("unexpected On triggers: %v", wf.On)
	}
	if len(wf.Jobs) != 2 {
		t.Fatalf("expected 2 jobs, got %d", len(wf.Jobs))
	}

	compileJob, ok := wf.Jobs["compile"]
	if !ok {
		t.Fatalf("missing 'compile' job")
	}
	if compileJob.Name != "Compile Binaries" || compileJob.RunsOn != "ubuntu-latest" {
		t.Errorf("unexpected compile job fields: %+v", compileJob)
	}
	if len(compileJob.Steps) != 2 {
		t.Fatalf("expected 2 steps in compile job, got %d", len(compileJob.Steps))
	}
	if compileJob.Steps[0].Uses != "actions/checkout@v4" {
		t.Errorf("unexpected step 0 uses: %q", compileJob.Steps[0].Uses)
	}
	if !strings.Contains(compileJob.Steps[1].Run, "go build ./...") {
		t.Errorf("unexpected step 1 run block scalar: %q", compileJob.Steps[1].Run)
	}

	verifyJob, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatalf("missing 'verify' job")
	}
	if len(verifyJob.Needs) != 1 || verifyJob.Needs[0] != "compile" {
		t.Errorf("unexpected verify job needs: %v", verifyJob.Needs)
	}
}

package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
)

// TestComplainWorkflowDomainPlansCreate pins the #5191 workflow domain end to end at the CLI: a
// --domain workflow complaint plans on the workflow channel (distinct key prefix + title) and
// records the domain on the plan row.
func TestComplainWorkflowDomainPlansCreate(t *testing.T) {
	code, out, _ := runComplainCapture([]string{
		"--domain", "workflow", "--kind", "lane-collision",
		"--summary", "two workers raced the commit lock", "--tool", "fak commit",
		"--rationale", "compute and cmd lanes both landed on cmd/fak in one minute", "--json",
	})
	if code != 0 {
		t.Fatalf("workflow dry-run exit = %d, want 0", code)
	}
	var res guardcomplaint.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("json output did not parse: %v\n%s", err, out)
	}
	if len(res.Planned) != 1 {
		t.Fatalf("planned = %+v, want one row", res.Planned)
	}
	row := res.Planned[0]
	if row.Domain != "workflow" {
		t.Fatalf("plan domain = %q, want workflow", row.Domain)
	}
	if !strings.HasPrefix(row.Key, "workflow-complaint/") {
		t.Fatalf("workflow plan key = %q, want workflow-complaint/ prefix", row.Key)
	}
	if !strings.HasPrefix(row.Title, "workflow friction [lane-collision]") {
		t.Fatalf("workflow plan title = %q", row.Title)
	}
}

// TestComplainWorkflowDomainRejectsGuardKind pins that the kind is validated against the RESOLVED
// domain: a guard kind in the workflow domain is a usage error naming the workflow vocabulary.
func TestComplainWorkflowDomainRejectsGuardKind(t *testing.T) {
	code, _, errs := runComplainCapture([]string{
		"--domain", "workflow", "--kind", "false-positive", "--summary", "x",
	})
	if code != 2 {
		t.Fatalf("guard kind in workflow domain exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "workflow complaint kind") || !strings.Contains(errs, "lane-collision") {
		t.Fatalf("stderr should name the workflow kind set: %q", errs)
	}
}

// TestComplainRejectsUnknownDomain pins the closed domain set at the CLI boundary.
func TestComplainRejectsUnknownDomain(t *testing.T) {
	code, _, errs := runComplainCapture([]string{"--domain", "bogus", "--summary", "x"})
	if code != 2 {
		t.Fatalf("unknown domain exit = %d, want 2", code)
	}
	if !strings.Contains(errs, "unknown complaint domain") {
		t.Fatalf("stderr should name the closed domain set: %q", errs)
	}
}

// TestComplainGuardDomainUnchanged is the backward-compat guard: a complaint with no --domain
// plans exactly as before (guard key prefix, guard-domain default kind on the plan row).
func TestComplainGuardDomainUnchanged(t *testing.T) {
	code, out, _ := runComplainCapture([]string{
		"--summary", "floor blocked a legit docs/notes commit",
		"--reason", "FILE_ADMISSION", "--tool", "Bash", "--json",
	})
	if code != 0 {
		t.Fatalf("guard dry-run exit = %d, want 0", code)
	}
	var res guardcomplaint.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("json parse: %v", err)
	}
	row := res.Planned[0]
	if row.Domain != "guard" {
		t.Fatalf("default plan domain = %q, want guard", row.Domain)
	}
	if !strings.HasPrefix(row.Key, "guard-complaint/false-positive/") {
		t.Fatalf("guard plan key lost its historical shape: %q", row.Key)
	}
}

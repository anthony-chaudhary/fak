package main

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	t.Setenv("FAK_COMPLAIN_LIVE", "")
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

func TestComplainWorkflowBlockerKindsRequireRationale(t *testing.T) {
	t.Setenv("FAK_COMPLAIN_LIVE", "")
	for _, kind := range []string{"catch-22", "completion-blocker"} {
		t.Run(kind, func(t *testing.T) {
			code, _, errs := runComplainCapture([]string{
				"--domain", "workflow", "--kind", kind,
				"--summary", "required recovery was itself refused",
			})
			if code != 2 {
				t.Fatalf("%s without rationale exit = %d, want 2", kind, code)
			}
			if !strings.Contains(errs, "rationale") {
				t.Fatalf("%s refusal must name the missing rationale: %q", kind, errs)
			}
		})
	}
}

// TestComplainRationaleFileContract exercises the file boundary through the CLI.
// The completion-blocker rationale requirement proves non-empty file content was read;
// the remaining cases pin exclusivity, trimming, the 16 KiB ceiling, and dry-run honesty.
func TestComplainRationaleFileContract(t *testing.T) {
	t.Setenv("FAK_COMPLAIN_LIVE", "")
	dir := t.TempDir()
	write := func(name, contents string) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	base := []string{
		"--domain", "workflow", "--kind", "completion-blocker",
		"--summary", "completion gate has no reachable recovery", "--json",
	}

	multiline := write("multiline.txt", "The completion gate refused its required witness at 10.20.30.40.\nThe witness can only be produced after completion; token=ghp_abcdefghijklmnop.\n")
	scrubbed, err := readComplaintRationaleFile(multiline)
	if err != nil {
		t.Fatalf("read multiline rationale: %v", err)
	}
	for _, secret := range []string{"10.20.30.40", "ghp_abcdefghijklmnop"} {
		if strings.Contains(scrubbed, secret) {
			t.Fatalf("multiline rationale leaked %q: %q", secret, scrubbed)
		}
	}
	for _, marker := range []string{"[REDACTED_IP]", "token=[REDACTED]"} {
		if !strings.Contains(scrubbed, marker) {
			t.Fatalf("multiline rationale missing %q: %q", marker, scrubbed)
		}
	}
	code, out, errs := runComplainCapture(append(append([]string{}, base...), "--rationale-file", multiline))
	if code != 0 {
		t.Fatalf("multiline rationale file exit = %d, want 0; stderr=%q", code, errs)
	}
	var result guardcomplaint.Result
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatalf("multiline rationale output: %v\n%s", err, out)
	}
	if result.Mode != "dry-run" || len(result.Synced) != 0 {
		t.Fatalf("rationale-file run claimed filing: mode=%q synced=%+v", result.Mode, result.Synced)
	}
	if len(result.Planned) != 1 || result.Planned[0].Kind != "completion-blocker" {
		t.Fatalf("rationale-file route = %+v", result.Planned)
	}
	if !strings.Contains(errs, "NO gh ticket was filed") {
		t.Fatalf("dry-run must explicitly deny filing: %q", errs)
	}

	t.Run("conflicts with inline rationale", func(t *testing.T) {
		args := append(append([]string{}, base...), "--rationale", "inline", "--rationale-file", multiline)
		code, _, errs := runComplainCapture(args)
		if code != 2 || !strings.Contains(errs, "--rationale") || !strings.Contains(errs, "--rationale-file") {
			t.Fatalf("conflict = exit %d stderr %q", code, errs)
		}
	})

	t.Run("refuses empty", func(t *testing.T) {
		empty := write("empty.txt", " \n\t")
		code, _, errs := runComplainCapture(append(append([]string{}, base...), "--rationale-file", empty))
		if code != 2 || !strings.Contains(errs, "rationale") {
			t.Fatalf("empty file = exit %d stderr %q", code, errs)
		}
	})

	t.Run("accepts exactly 16 KiB", func(t *testing.T) {
		limit := write("limit.txt", strings.Repeat("x", 16*1024))
		code, _, errs := runComplainCapture(append(append([]string{}, base...), "--rationale-file", limit))
		if code != 0 {
			t.Fatalf("16 KiB file = exit %d, want 0; stderr=%q", code, errs)
		}
	})

	t.Run("refuses over 16 KiB", func(t *testing.T) {
		oversize := write("oversize.txt", strings.Repeat("x", 16*1024+1))
		code, _, errs := runComplainCapture(append(append([]string{}, base...), "--rationale-file", oversize))
		if code != 2 || !strings.Contains(errs, "16 KiB") {
			t.Fatalf("oversize file = exit %d stderr %q", code, errs)
		}
	})
}

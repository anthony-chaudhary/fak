package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/evebridge"
)

// writeEveFixture writes a manifest fixture and returns its path.
func writeEveFixture(t *testing.T, doc string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "eve-info.json")
	if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// runEveJSON runs the preflight CLI against a fixture and decodes the
// JSON-first report from stdout.
func runEveJSON(t *testing.T, argv []string) (int, evebridge.Report, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runEve(&stdout, &stderr, strings.NewReader(""), argv)
	var report evebridge.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("stdout is not a JSON report (%v):\n%s", err, stdout.String())
	}
	return code, report, stderr.String()
}

// TestEvePreflightCLISmokeFailsClosed is the issue's CLI smoke witness: the
// preflight run against a bad fixture exits fail-closed and the captured
// stdout is the typed JSON diagnostics.
func TestEvePreflightCLISmokeFailsClosed(t *testing.T) {
	path := writeEveFixture(t, `{
	  "connections": [
	    {
	      "name": "crm",
	      "type": "openapi",
	      "url": "https://api.example.com/v1",
	      "auth": "no-auth",
	      "operations": [
	        {"name": "list_contacts", "method": "GET"},
	        {"name": "delete_contact", "method": "DELETE"}
	      ]
	    }
	  ]
	}`)
	code, report, stderrText := runEveJSON(t, []string{"preflight", "connections", "--manifest", path})
	if code != evePreflightFailed {
		t.Fatalf("expected exit %d (fail closed), got %d (stderr: %s)", evePreflightFailed, code, stderrText)
	}
	if report.OK || report.Schema != evebridge.SchemaPreflight {
		t.Fatalf("expected a red %s report, got %+v", evebridge.SchemaPreflight, report)
	}
	var codes []string
	for _, d := range report.Diagnostics {
		codes = append(codes, d.Code)
		if d.Remediation == "" {
			t.Fatalf("diagnostic %s missing remediation text: %+v", d.Code, d)
		}
	}
	joined := strings.Join(codes, ",")
	for _, want := range []string{evebridge.CodeNoAuth, evebridge.CodeMutationUnapproved} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected diagnostic %s in %v", want, codes)
		}
	}
	if len(report.AdmittedTools) != 0 {
		t.Fatalf("fail-closed run must admit no tools, got %v", report.AdmittedTools)
	}
	if !strings.Contains(stderrText, "FAILED closed") {
		t.Fatalf("expected a fail-closed verdict line on stderr, got %q", stderrText)
	}
}

// TestEvePreflightCLISmokePassesAndPrintsNamespace: a read-only allowlisted
// fixture passes and stdout carries the exact tool namespace fak will admit.
func TestEvePreflightCLISmokePassesAndPrintsNamespace(t *testing.T) {
	path := writeEveFixture(t, `{
	  "connections": [
	    {
	      "name": "wiki",
	      "type": "mcp",
	      "url": "https://mcp.wiki.example",
	      "auth": "app",
	      "allowlist": ["search_pages", "get_page"],
	      "operations": [
	        {"name": "search_pages", "read_only": true},
	        {"name": "get_page", "read_only": true},
	        {"name": "delete_page", "mutating": true}
	      ]
	    }
	  ]
	}`)
	code, report, stderrText := runEveJSON(t, []string{"preflight", "connections", "--manifest", path})
	if code != 0 {
		t.Fatalf("expected pass, got exit %d (stderr: %s, report: %+v)", code, stderrText, report)
	}
	want := []string{"wiki__get_page", "wiki__search_pages"}
	if len(report.AdmittedTools) != len(want) || report.AdmittedTools[0] != want[0] || report.AdmittedTools[1] != want[1] {
		t.Fatalf("expected the exact admitted namespace %v, got %v", want, report.AdmittedTools)
	}
}

// TestEvePreflightCLIOverride: a fak policy override admits a mutating
// operation the connection itself does not gate.
func TestEvePreflightCLIOverride(t *testing.T) {
	path := writeEveFixture(t, `{
	  "connections": [
	    {
	      "name": "shop",
	      "type": "openapi",
	      "url": "https://api.shop.example",
	      "auth": "static",
	      "operations": [{"name": "purchase_item", "method": "POST"}]
	    }
	  ]
	}`)
	code, report, _ := runEveJSON(t, []string{"preflight", "connections", "--manifest", path, "--override", "shop__purchase_item"})
	if code != 0 || !report.OK {
		t.Fatalf("expected override to admit the mutation, got exit %d %+v", code, report)
	}
	if len(report.AdmittedTools) != 1 || report.AdmittedTools[0] != "shop__purchase_item" {
		t.Fatalf("expected [shop__purchase_item], got %v", report.AdmittedTools)
	}
}

// TestEvePreflightCLIUsageAndIOErrors: usage and unreadable-manifest paths
// return their distinct exit codes and never emit a half report.
func TestEvePreflightCLIUsageAndIOErrors(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runEve(&stdout, &stderr, strings.NewReader(""), nil); code != 2 {
		t.Fatalf("bare `fak eve` should be usage exit 2, got %d", code)
	}
	if code := runEve(&stdout, &stderr, strings.NewReader(""), []string{"preflight"}); code != 2 {
		t.Fatalf("`fak eve preflight` without a target should be usage exit 2, got %d", code)
	}
	stdout.Reset()
	code := runEve(&stdout, &stderr, strings.NewReader(""), []string{"preflight", "connections", "--manifest", filepath.Join(t.TempDir(), "absent.json")})
	if code != 1 {
		t.Fatalf("unreadable manifest should be exit 1, got %d", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("an IO error must not emit a report, got %q", stdout.String())
	}
}

// TestEvePreflightCLIStdin: "-" reads the manifest from stdin.
func TestEvePreflightCLIStdin(t *testing.T) {
	doc := `{"connections":[{"name":"localtool","type":"mcp","url":"http://127.0.0.1:9","auth":"local","operations":[{"name":"list_things","read_only":true}]}]}`
	var stdout, stderr bytes.Buffer
	code := runEve(&stdout, &stderr, strings.NewReader(doc), []string{"preflight", "connections", "--manifest", "-"})
	if code != 0 {
		t.Fatalf("expected pass from stdin, got %d (stderr: %s)", code, stderr.String())
	}
	var report evebridge.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("stdout is not a JSON report: %v", err)
	}
	if len(report.AdmittedTools) != 1 || report.AdmittedTools[0] != "localtool__list_things" {
		t.Fatalf("expected [localtool__list_things], got %v", report.AdmittedTools)
	}
}

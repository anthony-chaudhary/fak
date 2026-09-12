package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestIssueCreateScrubGateVerdicts(t *testing.T) {
	original := issueScrubGateHook
	t.Cleanup(func() { issueScrubGateHook = original })
	for _, tc := range []struct {
		name    string
		verdict issueScrubGateVerdict
		allow   bool
	}{
		{"clean", issueScrubGateVerdict{Ran: true, Clean: true}, true},
		{"survivors", issueScrubGateVerdict{Ran: true, SurvivorCount: 1}, false},
		{"needs scrub without survivors", issueScrubGateVerdict{Ran: true, NeedsScrub: true, Replacements: 1}, false},
		{"unavailable", issueScrubGateVerdict{Err: "scrubber unavailable"}, false},
		{"inconsistent clean", issueScrubGateVerdict{Ran: true, Clean: true, SurvivorCount: 1}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, dryRun := range []bool{false, true} {
				title, body := "synthetic-title-marker", "synthetic-body-marker"
				issueScrubGateHook = func(_ string, gotTitle, gotBody string) issueScrubGateVerdict {
					if gotTitle != title || gotBody != body {
						t.Fatal("gate did not receive both issue fields")
					}
					return tc.verdict
				}
				calls := 0
				runner := func(args []string) (string, string, bool) {
					calls++
					if !strings.Contains(strings.Join(args, " "), body) {
						t.Fatal("body changed after clean verdict")
					}
					return "https://example.test/issues/1", "", true
				}
				argv := []string{"--title", title, "--body", body, "--raw-body", "--json"}
				if dryRun {
					argv = append(argv, "--dry-run")
				}
				var out, errOut bytes.Buffer
				code := runIssueCreateWith(&out, &errOut, argv, runner)
				var result issueCreateResult
				if err := json.Unmarshal(out.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				if tc.allow {
					if code != 0 || !result.OK || !result.Scrub.Clean || result.Title != title {
						t.Fatalf("clean verdict rejected: code=%d result=%+v", code, result)
					}
					wantCalls := 1
					if dryRun {
						wantCalls = 0
					}
					if calls != wantCalls {
						t.Fatalf("calls=%d want=%d", calls, wantCalls)
					}
				} else {
					if code != 3 || calls != 0 || result.OK {
						t.Fatalf("refusal failed: code=%d calls=%d", code, calls)
					}
					for _, marker := range []string{title, body} {
						if strings.Contains(out.String()+errOut.String(), marker) {
							t.Fatal("refusal leaked rejected issue text")
						}
					}
				}
			}
		})
	}
}

func TestIssueScrubGateMissingCompanion(t *testing.T) {
	original := issueScrubGateModuleRoot
	issueScrubGateModuleRoot = filepath.Join(t.TempDir(), "fak")
	t.Cleanup(func() { issueScrubGateModuleRoot = original })
	t.Setenv("FAK_PRIVATE_DIR", "")
	t.Setenv("FAK_PRIVATE_ROOT", "")
	verdict := runIssueScrubGateSubprocess("", "title", "body")
	if verdict.Ran || verdict.Clean || !strings.Contains(verdict.Err, "issue_scrub.py not found") || !strings.Contains(verdict.Err, "fak-private") {
		t.Fatalf("missing companion did not fail closed with probed path: %+v", verdict)
	}
}

func TestIssueScrubGateJSONShape(t *testing.T) {
	for _, tc := range []struct {
		raw   string
		valid bool
	}{
		{`{"clean":true,"needs_scrub":false,"replacements":0,"survivor_count":0}`, true},
		{`{"clean":false,"needs_scrub":true,"replacements":1,"survivor_count":0}`, true},
		{`{"clean":true}`, false},
		{`{"clean":true,"needs_scrub":false,"replacements":0,"survivor_count":0,"detail":"synthetic-private-value"}`, false},
		{`not json`, false},
		{`null`, false},
		{`{"clean":true,"needs_scrub":null,"replacements":0,"survivor_count":0}`, false},
	} {
		if got := issueScrubGateResultIsKnownShape([]byte(tc.raw)); got != tc.valid {
			t.Fatalf("shape valid=%v want=%v", got, tc.valid)
		}
	}
}

func TestIssueScrubGateCompanionFixtures(t *testing.T) {
	root := issueScrubGateLocate("")
	if root == "" {
		t.Skip("companion integration fixture requires private scrubber")
	}
	if _, err := os.Stat(filepath.Join(root, issueScrubGateScriptName)); err != nil {
		t.Skip("companion scrubber unavailable")
	}
	if _, err := exec.LookPath("python"); err != nil {
		t.Skip("companion integration fixture requires Python")
	}
	// Construct a synthetic private-ticket identifier; never embed real private needles.
	marker := strings.Join([]string{"FAM", "12345"}, "-")
	for _, tc := range []struct {
		name, title, body string
		clean             bool
	}{
		{"clean", "Improve parser diagnostics", "Return a useful error for malformed input.", true},
		{"body needle", "Improve parser diagnostics", "Reference " + marker, false},
		{"title needle", "Reference " + marker, "Return a useful error.", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verdict := runIssueScrubGateSubprocess(root, tc.title, tc.body)
			if !verdict.Ran || verdict.Clean != tc.clean {
				t.Fatalf("unexpected companion verdict: %+v", verdict)
			}
			if !tc.clean && !verdict.NeedsScrub && verdict.SurvivorCount == 0 {
				t.Fatalf("needle fixture had no refusal evidence: %+v", verdict)
			}
		})
	}
}

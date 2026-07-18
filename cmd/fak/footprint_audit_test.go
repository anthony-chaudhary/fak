package main

// footprint_audit_test.go — the #5050 DoD table test: on a fixture request with cold
// custom tools and a custom base-URL, the audit names the floor, the cold-tool count,
// and the missing ENABLE_TOOL_SEARCH — and stands down on each finding when its
// config fact is absent.

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

// auditFixtureBody is a minimal Claude-Code-shaped Anthropic Messages body: a system
// prompt, a two-turn history (one history message + one volatile tail), one hot
// built-in tool (Read) and two cold custom tools the #3232 transform would defer.
const auditFixtureBody = `{"model":"claude-x","max_tokens":16,"system":"SYSTEM PROMPT SPINE","messages":[{"role":"user","content":"earlier turn"},{"role":"user","content":"latest turn"}],"tools":[{"name":"Read","description":"read a file","input_schema":{"type":"object"}},{"name":"acme_deploy","description":"deploy the acme service","input_schema":{"type":"object"}},{"name":"acme_rollback","description":"roll back the acme service","input_schema":{"type":"object"}}]}`

func TestFootprintAuditTable(t *testing.T) {
	cases := []struct {
		name         string
		baseURL      string
		toolSearch   string
		wantCold     int
		wantFindings []string // finding IDs, in order
	}{
		{
			name:    "cold tools + custom base-URL, tool search unset",
			baseURL: "http://127.0.0.1:8085", toolSearch: "",
			wantCold:     2,
			wantFindings: []string{findingColdTools, findingToolSearchUnset},
		},
		{
			name:    "tool search set stands the GH#746 finding down",
			baseURL: "http://127.0.0.1:8085", toolSearch: "1",
			wantCold:     2,
			wantFindings: []string{findingColdTools},
		},
		{
			name:    "provider-default base-URL stands the GH#746 finding down",
			baseURL: "", toolSearch: "",
			wantCold:     2,
			wantFindings: []string{findingColdTools},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, err := buildFootprintAudit([]byte(auditFixtureBody), "fixture", tc.baseURL, tc.toolSearch)
			if err != nil {
				t.Fatalf("buildFootprintAudit: %v", err)
			}
			if a.ColdToolCount != tc.wantCold {
				t.Errorf("ColdToolCount=%d, want %d", a.ColdToolCount, tc.wantCold)
			}
			gotIDs := make([]string, 0, len(a.Findings))
			for _, f := range a.Findings {
				gotIDs = append(gotIDs, f.ID)
				if f.Lever == "" {
					t.Errorf("finding %s carries no lever — every finding must name the fak fix", f.ID)
				}
			}
			if strings.Join(gotIDs, ",") != strings.Join(tc.wantFindings, ",") {
				t.Errorf("findings=%v, want %v", gotIDs, tc.wantFindings)
			}
		})
	}
}

// TestFootprintAuditReportNamesTheDoDFacts renders the audit on the DoD fixture and
// asserts the human report NAMES the floor, the cold-tool count, and the missing
// ENABLE_TOOL_SEARCH, all under the ESTIMATED provenance label.
func TestFootprintAuditReportNamesTheDoDFacts(t *testing.T) {
	a, err := buildFootprintAudit([]byte(auditFixtureBody), "fixture", "http://127.0.0.1:8085", "")
	if err != nil {
		t.Fatalf("buildFootprintAudit: %v", err)
	}

	// The system prompt must be counted exactly once (decode folds it into the
	// message list too; the audit de-folds). System bytes == len of the raw prompt.
	if got, want := a.Footprint.System.Bytes, len("SYSTEM PROMPT SPINE"); got != want {
		t.Fatalf("System.Bytes=%d, want %d — folded system prompt double-counted", got, want)
	}
	if a.Footprint.Floor.Bytes != a.Footprint.System.Bytes+a.Footprint.Tools.Bytes {
		t.Fatalf("Floor.Bytes=%d, want system+tools=%d",
			a.Footprint.Floor.Bytes, a.Footprint.System.Bytes+a.Footprint.Tools.Bytes)
	}

	var buf bytes.Buffer
	renderFootprintAudit(&buf, a, 0)
	out := buf.String()

	for _, want := range []string{
		"ESTIMATED", // provenance label is mandatory (#5050: no provider-measured claim)
		fmt.Sprintf("floor (system+tools, the per-turn tax): %d est. tokens", a.Footprint.Floor.Tokens),
		"2 cold custom tool schema(s)",
		"ENABLE_TOOL_SEARCH",
		"--defer-cold-tools",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q\nreport:\n%s", want, out)
		}
	}
}

// TestFootprintAuditAlreadyDeferredStandsDown proves the cold-tool finding cannot
// double-fire on a body the #3232 lever (or ENABLE_TOOL_SEARCH) already deferred.
func TestFootprintAuditAlreadyDeferredStandsDown(t *testing.T) {
	deferred := `{"model":"claude-x","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"acme_deploy","description":"deploy","input_schema":{"type":"object"},"defer_loading":true}]}`
	a, err := buildFootprintAudit([]byte(deferred), "fixture", "", "")
	if err != nil {
		t.Fatalf("buildFootprintAudit: %v", err)
	}
	if a.ColdToolCount != 0 {
		t.Errorf("ColdToolCount=%d on an already-deferred body, want 0", a.ColdToolCount)
	}
	if a.DeferReason != "already_deferred" {
		t.Errorf("DeferReason=%q, want already_deferred", a.DeferReason)
	}
	for _, f := range a.Findings {
		if f.ID == findingColdTools {
			t.Errorf("cold-tool finding fired on an already-deferred body")
		}
	}
}

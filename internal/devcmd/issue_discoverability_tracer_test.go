package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestIssueDiscoverabilityRejectsHorizontalFeature(t *testing.T) {
	body := validDispatchableIssueBody("widget", []string{
		"internal/widget/widget_types.go",
		"internal/widget/interface.go",
	}, 3)
	body = strings.Replace(body, "Implement kernel optimizations", "Define request and response structs plus an interface for future callers.", 1)
	body = strings.Replace(body, "Change -> real seam -> observable outcome -> witness.", "Add inert request and response structs and a mock-only interface test.", 1)
	body = strings.Replace(body, "Do not add unrelated polish.", "Do not connect the types to a command, engine, or output yet.", 1)

	for _, tt := range []struct {
		name string
		repo string
	}{
		{name: "public target"},
		{name: "private target", repo: "anthony-chaudhary/fak-private"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			args := []string{
				"--title", "feat(widget): add request types",
				"--body", body,
				"--json",
			}
			if tt.repo != "" {
				args = append(args, "--repo", tt.repo)
			}

			var stdout, stderr bytes.Buffer
			code := runIssueDiscoverability(&stdout, &stderr, args)
			if code != 3 {
				t.Fatalf("code = %d, want 3; stdout:\n%s\nstderr:\n%s", code, stdout.String(), stderr.String())
			}

			var result IssueDiscoverabilityAuditResult
			if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
				t.Fatalf("decode result: %v; stdout:\n%s", err, stdout.String())
			}
			if result.OK || len(result.Issues) != 1 || result.Issues[0].Dispatchable {
				t.Fatalf("horizontal feature admitted: %+v", result)
			}
			row := result.Issues[0]
			if !tracerCLIContains(row.Reasons, "ISSUE_HORIZONTAL_FRAGMENT") {
				t.Fatalf("reasons = %v, want ISSUE_HORIZONTAL_FRAGMENT", row.Reasons)
			}
			if !tracerCLIContainsSubstring(row.RepairActions, "tracer bullet") {
				t.Fatalf("repair actions = %v, want actionable tracer bullet guidance", row.RepairActions)
			}
		})
	}
}

func tracerCLIContains(values []string, want string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), want) {
			return true
		}
	}
	return false
}

func tracerCLIContainsSubstring(values []string, want string) bool {
	for _, value := range values {
		if strings.Contains(strings.ToLower(value), strings.ToLower(want)) {
			return true
		}
	}
	return false
}

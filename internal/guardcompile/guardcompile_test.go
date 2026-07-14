package guardcompile_test

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/guardcompile"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const minimalPolicy = `{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["Bash"],
  "arg_rules": []
}`

func TestCompileProducesOnceEnforcesWithoutModel(t *testing.T) {
	ruleJSON := `{
  "deny_regex": "rm\\s+-rf\\s+\\.\\./tools(?:\\s|$)",
  "reason": "POLICY_BLOCK",
  "severity": "block",
  "fix": "delete only inside the repository scratch root"
}`
	calls := 0
	extractor := guardcompile.ExtractorFunc(func(prompt string) ([]byte, error) {
		calls++
		if !strings.Contains(prompt, "block when an agent deletes outside the repository") {
			t.Errorf("prompt missing intent: %q", prompt)
		}
		return []byte(ruleJSON), nil
	})

	dangerous := "rm " + "-rf " + "../tools"
	transcript := "agent invoked `" + dangerous + "` and it escaped the repo"
	before := []byte(minimalPolicy)
	original := append([]byte(nil), before...)
	candidate, err := guardcompile.Compile(guardcompile.Request{
		Transcript: transcript,
		Intent:     "block when an agent deletes outside the repository",
		Tool:       "Bash",
		Field:      "command",
	}, before, extractor)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("extraction calls = %d, want 1", calls)
	}
	if !bytes.Equal(before, original) {
		t.Fatal("Compile mutated the input manifest")
	}

	runtimePolicy, err := policy.Parse(candidate.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	gate := adjudicator.New(runtimePolicy)
	decide := func(command string) abi.Verdict {
		args, _ := json.Marshal(map[string]string{"command": command})
		return gate.Adjudicate(context.Background(), &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: args},
		})
	}
	if got := decide(dangerous); got.Kind != abi.VerdictDeny {
		t.Errorf("dangerous command verdict = %v, want deny", got.Kind)
	}
	if got := decide("go build ./..."); got.Kind != abi.VerdictAllow {
		t.Errorf("build command verdict = %v, want allow", got.Kind)
	}
	if calls != 1 {
		t.Fatalf("runtime decisions called model: extraction calls = %d", calls)
	}

	diff := guardcompile.ProposedDiff("policy.json", before, candidate.Manifest)
	if !strings.Contains(diff, "proposed; not applied") {
		t.Errorf("diff does not state review-only status: %s", diff)
	}
}

func TestCompileRejectsInvalidExtractions(t *testing.T) {
	tests := []struct {
		name     string
		response string
	}{
		{"unknown field", `{"deny_regex":".","reason":"POLICY_BLOCK","severity":"block","fix":"use scratch","extra":true}`},
		{"invalid regex", `{"deny_regex":"(","reason":"POLICY_BLOCK","severity":"block","fix":"use scratch"}`},
		{"trailing value", `{"deny_regex":".","reason":"POLICY_BLOCK","severity":"block","fix":"use scratch"}{}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			_, err := guardcompile.Compile(guardcompile.Request{
				Transcript: "transcript", Intent: "intent", Tool: "Bash", Field: "command",
			}, []byte(minimalPolicy), guardcompile.ExtractorFunc(func(string) ([]byte, error) {
				calls++
				return []byte(tc.response), nil
			}))
			if err == nil {
				t.Fatal("expected error")
			}
			if calls != 1 {
				t.Fatalf("extraction calls = %d, want 1", calls)
			}
		})
	}
}

func TestWarnSeverityEmitsAdvisoryRule(t *testing.T) {
	response := `{"deny_regex":"echo","reason":"POLICY_BLOCK","severity":"warn","fix":"review command"}`
	candidate, err := guardcompile.Compile(guardcompile.Request{
		Transcript: "echo happened", Intent: "warn on echo", Tool: "Bash", Field: "command",
	}, []byte(minimalPolicy), guardcompile.ExtractorFunc(func(string) ([]byte, error) {
		return []byte(response), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		ArgRules []struct {
			Advisory bool `json:"advisory"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(candidate.Manifest, &doc); err != nil || len(doc.ArgRules) != 1 || !doc.ArgRules[0].Advisory {
		t.Fatalf("warn severity did not emit advisory rule: err=%v doc=%+v", err, doc)
	}
}

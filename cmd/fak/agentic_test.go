package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentic"
)

func TestAgenticJSONIsDeterministicAndParseable(t *testing.T) {
	args := []string{"--json", "Add", "a", "CLI", "and", "tests", "for", "internal/widget/widget.go"}
	var first, firstErr bytes.Buffer
	if code := runAgentic(&first, &firstErr, args); code != 0 {
		t.Fatalf("first code=%d stderr=%s", code, firstErr.String())
	}
	var second, secondErr bytes.Buffer
	if code := runAgentic(&second, &secondErr, args); code != 0 {
		t.Fatalf("second code=%d stderr=%s", code, secondErr.String())
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatalf("repeated JSON differs:\n%s\n%s", first.Bytes(), second.Bytes())
	}
	var plan agentic.Plan
	if err := json.Unmarshal(first.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Schema != agentic.Schema || plan.Inference.Domain != agentic.DomainDevelopment || !plan.Mode.ReadOnly || !plan.Mode.Offline {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestAgenticNativeJSONPreservesNativeContract(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runAgentic(&out, &errOut, []string{"--json", "Optimize Qwen3.8 fak-native CUDA decode throughput"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var plan agentic.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Native == nil || plan.Native.Engine != "fak-native" || plan.Native.DefaultModel != "Qwen3.8" || plan.Native.Receipt.Schema != "fak-native-performance-receipt/v2" || !plan.Native.MatchedWorkload {
		t.Fatalf("native contract = %+v", plan.Native)
	}
}

func TestAgenticTextRendersActionablePlan(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runAgentic(&out, &errOut, []string{"Fix", "the", "widget", "CLI"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"FAK agentic plan objective-", "stage expand:", "issue-ready work:", "model=gpt-5.6-sol", "read_only=true offline=true"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("text output missing %q:\n%s", want, out.String())
		}
	}
}

func TestAgenticInvalidInputReturnsUsageError(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "missing objective", args: nil, want: "objective is required"},
		{name: "blank objective", args: []string{"   "}, want: "objective is required"},
		{name: "unknown flag", args: []string{"--bogus"}, want: "flag provided but not defined"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, errOut bytes.Buffer
			if code := runAgentic(&out, &errOut, tt.args); code != 2 {
				t.Fatalf("code=%d, want 2; stdout=%s stderr=%s", code, out.String(), errOut.String())
			}
			if out.Len() != 0 || !strings.Contains(errOut.String(), tt.want) {
				t.Fatalf("stdout=%q stderr=%q, want stderr containing %q", out.String(), errOut.String(), tt.want)
			}
		})
	}
}

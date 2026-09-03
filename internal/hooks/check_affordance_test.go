package hooks

import (
	"strings"
	"testing"
)

func TestGateAffordance(t *testing.T) {
	t.Run("VerdictDeny_MissingAffordance", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func checkPolicy() abi.Verdict {
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 1 {
			t.Fatalf("expected 1 violation, got %d: %+v", len(violations), violations)
		}
		if violations[0].Token != "VerdictDeny" {
			t.Errorf("expected token VerdictDeny, got %q", violations[0].Token)
		}
		if !strings.Contains(violations[0].Message, "missing next-action affordance") {
			t.Errorf("expected missing affordance message, got %q", violations[0].Message)
		}
		if violations[0].Line != 6 {
			t.Errorf("expected violation at line 6, got %d", violations[0].Line)
		}
	})

	t.Run("VerdictDeny_EmptyAffordance", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func checkPolicy() abi.Verdict {
	return abi.Verdict{
		Kind:       abi.VerdictDeny,
		NextAction: "   ",
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 1 {
			t.Fatalf("expected 1 violation, got %d: %+v", len(violations), violations)
		}
		if !strings.Contains(violations[0].Message, "empty next-action affordance") {
			t.Errorf("expected empty affordance message, got %q", violations[0].Message)
		}
	})

	t.Run("VerdictDeny_CleanWithNextAction", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func checkPolicy() abi.Verdict {
	return abi.Verdict{
		Kind:       abi.VerdictDeny,
		Reason:     abi.ReasonPolicyBlock,
		NextAction: "route through sanctioned path or update policy manifest",
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("VerdictDeny_CleanWithMetaMap", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func checkPolicy() abi.Verdict {
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonSelfModify,
		Meta: map[string]string{
			"next_action": "route an authorized edit through fak commit --core-lock-maintenance-witness",
		},
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("VerdictDeny_NolintExempt", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func checkPolicy() abi.Verdict {
	//nolint:affordance intentional test mock
	return abi.Verdict{
		Kind: abi.VerdictDeny,
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations for nolint-annotated code, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("RefusalReturn_MissingAffordance", func(t *testing.T) {
		src := `package test

type Refusal struct {
	Code string
}

func failCall() *Refusal {
	return &Refusal{
		Code: "INVALID_TOOL_ARGUMENTS",
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Expect 2 violations: 1 for struct declaration, 1 for construction
		if len(violations) != 2 {
			t.Fatalf("expected 2 violations, got %d: %+v", len(violations), violations)
		}
		declViolation := violations[0]
		if declViolation.Token != "Refusal" || !strings.Contains(declViolation.Message, "declaration missing") {
			t.Errorf("expected declaration violation on Refusal, got %+v", declViolation)
		}
		createViolation := violations[1]
		if createViolation.Token != "INVALID_TOOL_ARGUMENTS" || !strings.Contains(createViolation.Message, "missing next-action") {
			t.Errorf("expected construction violation with token INVALID_TOOL_ARGUMENTS, got %+v", createViolation)
		}
	})

	t.Run("RefusalReturn_CleanWithAffordance", func(t *testing.T) {
		src := `package test

type Refusal struct {
	Code       string
	NextAction string
}

func failCall() *Refusal {
	return &Refusal{
		Code:       "INVALID_TOOL_ARGUMENTS",
		NextAction: "correct arguments to match schema and retry",
	}
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("RefusalDeclaration_MissingAffordance", func(t *testing.T) {
		src := `package test

type SecurityRefusal struct {
	Reason string
	Detail string
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 1 {
			t.Fatalf("expected 1 violation, got %d: %+v", len(violations), violations)
		}
		if violations[0].Token != "SecurityRefusal" {
			t.Errorf("expected token SecurityRefusal, got %q", violations[0].Token)
		}
		if !strings.Contains(violations[0].Message, "declaration missing next-action affordance field") {
			t.Errorf("expected declaration missing message, got %q", violations[0].Message)
		}
	})

	t.Run("RefusalDeclaration_CleanWithAffordance", func(t *testing.T) {
		src := `package test

type SecurityRefusal struct {
	Reason     string
	Affordance string
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("VerdictCall_MissingAffordance", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func makeVerdict() abi.Verdict {
	return verdict(abi.VerdictDeny, abi.ReasonMalformed, "grammar")
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 1 {
			t.Fatalf("expected 1 violation for bare verdict call, got %d: %+v", len(violations), violations)
		}
		if violations[0].Token != "VerdictDeny" {
			t.Errorf("expected token VerdictDeny, got %q", violations[0].Token)
		}
	})

	t.Run("VerdictCall_CleanWithAffordance", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func makeVerdict() abi.Verdict {
	return verdictWithAffordance(abi.VerdictDeny, "check tool grammar and retry")
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations for affordance call, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("CleanCode_NoRefusals", func(t *testing.T) {
		src := `package test
import "github.com/anthony-chaudhary/fak/internal/abi"

func allowCall() abi.Verdict {
	return abi.Verdict{
		Kind: abi.VerdictAllow,
	}
}

type SafeStruct struct {
	Name  string
	Count int
}
`
		violations, err := CheckAffordanceCompleteness(src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations for clean code, got %d: %+v", len(violations), violations)
		}
	})

	t.Run("EmptyContent", func(t *testing.T) {
		violations, err := CheckAffordanceCompleteness("")
		if err != nil {
			t.Fatalf("unexpected error on empty content: %v", err)
		}
		if len(violations) != 0 {
			t.Fatalf("expected 0 violations, got %d", len(violations))
		}
	})

	t.Run("SyntaxError", func(t *testing.T) {
		_, err := CheckAffordanceCompleteness("package test\nfunc broken() {")
		if err == nil {
			t.Fatalf("expected syntax error on broken Go source")
		}
	})
}

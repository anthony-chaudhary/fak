package agentopt

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestModelCascadingEscalation(t *testing.T) {
	ctx := context.Background()

	t.Run("escalation on validation failure with captured error and evidence", func(t *testing.T) {
		controller := DefaultModelCascadeController()

		var invokedTiers []CascadeTier
		var escalatedPrompts []string

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			invokedTiers = append(invokedTiers, target.Tier)
			escalatedPrompts = append(escalatedPrompts, prompt)

			switch target.Tier {
			case TierFast:
				// Cost-efficient model produces syntactically incomplete output
				return "ALTER TABLE users ADD COLUMN name", nil
			case TierStandard:
				// Higher-tier model produces valid, corrected output
				return "ALTER TABLE users ADD COLUMN name VARCHAR(255);", nil
			default:
				return "ALTER TABLE users ADD COLUMN name TEXT;", nil
			}
		}

		validatorFn := func(ctx context.Context, output string) error {
			if !strings.HasSuffix(strings.TrimSpace(output), ";") || !strings.Contains(output, "VARCHAR") {
				return NewValidationError(
					"SQL syntax validation failure",
					"missing data type declaration and terminating semicolon",
				)
			}
			return nil
		}

		res := controller.Cascade(ctx, "Generate migration for users table", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected cascade error: %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected Escalated=true, got false")
		}
		if res.FinalTier != TierStandard {
			t.Fatalf("expected FinalTier=%s, got %s", TierStandard, res.FinalTier)
		}
		if len(res.Attempts) != 2 {
			t.Fatalf("expected 2 attempts, got %d", len(res.Attempts))
		}

		// First attempt (TierFast) should have failed validation
		first := res.Attempts[0]
		if first.Valid {
			t.Fatal("expected first attempt to be marked invalid")
		}
		if first.Target.Tier != TierFast {
			t.Fatalf("expected first target tier %s, got %s", TierFast, first.Target.Tier)
		}
		if !strings.Contains(first.FailureError, "SQL syntax validation failure") {
			t.Fatalf("unexpected failure error: %q", first.FailureError)
		}
		if !strings.Contains(first.EvidenceInfo, "missing data type declaration") {
			t.Fatalf("unexpected evidence info: %q", first.EvidenceInfo)
		}

		// Second attempt (TierStandard) should have succeeded
		second := res.Attempts[1]
		if !second.Valid {
			t.Fatal("expected second attempt to be valid")
		}
		if second.Target.Tier != TierStandard {
			t.Fatalf("expected second target tier %s, got %s", TierStandard, second.Target.Tier)
		}

		// Verify escalated prompt carried evidence and error details
		if len(escalatedPrompts) < 2 {
			t.Fatalf("expected at least 2 prompt recordings, got %d", len(escalatedPrompts))
		}
		escalatedPrompt := escalatedPrompts[1]
		if !strings.Contains(escalatedPrompt, "SQL syntax validation failure") {
			t.Errorf("escalated prompt missing failure error: %s", escalatedPrompt)
		}
		if !strings.Contains(escalatedPrompt, "missing data type declaration") {
			t.Errorf("escalated prompt missing evidence details: %s", escalatedPrompt)
		}
		if !strings.Contains(escalatedPrompt, "ALTER TABLE users ADD COLUMN name") {
			t.Errorf("escalated prompt missing rejected output: %s", escalatedPrompt)
		}

		if res.Output != "ALTER TABLE users ADD COLUMN name VARCHAR(255);" {
			t.Fatalf("unexpected final output: %q", res.Output)
		}
	})

	t.Run("immediate success without escalation", func(t *testing.T) {
		controller := DefaultModelCascadeController()

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			return "SELECT 1;", nil
		}

		validatorFn := func(ctx context.Context, output string) error {
			if strings.HasSuffix(strings.TrimSpace(output), ";") {
				return nil
			}
			return errors.New("missing semicolon")
		}

		res := controller.Cascade(ctx, "Check liveness", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if res.Escalated {
			t.Fatal("expected Escalated=false on first attempt success")
		}
		if res.FinalTier != TierFast {
			t.Fatalf("expected FinalTier=%s, got %s", TierFast, res.FinalTier)
		}
		if len(res.Attempts) != 1 {
			t.Fatalf("expected 1 attempt, got %d", len(res.Attempts))
		}
		if !res.Attempts[0].Valid {
			t.Fatal("expected first attempt to be valid")
		}
	})

	t.Run("escalation cascades across all tiers to frontier", func(t *testing.T) {
		controller := DefaultModelCascadeController()

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			switch target.Tier {
			case TierFast:
				return "fast-faulty", nil
			case TierStandard:
				return "standard-faulty", nil
			case TierFrontier:
				return "frontier-verified", nil
			default:
				return "unknown", nil
			}
		}

		validatorFn := func(ctx context.Context, output string) error {
			if output == "frontier-verified" {
				return nil
			}
			return NewValidationError("unverified output", "expected frontier-verified token")
		}

		res := controller.Cascade(ctx, "Complex reasoning task", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected Escalated=true")
		}
		if res.FinalTier != TierFrontier {
			t.Fatalf("expected FinalTier=%s, got %s", TierFrontier, res.FinalTier)
		}
		if len(res.Attempts) != 3 {
			t.Fatalf("expected 3 attempts, got %d", len(res.Attempts))
		}
		if res.Output != "frontier-verified" {
			t.Fatalf("unexpected output: %q", res.Output)
		}
	})

	t.Run("cascade exhaustion when all tiers fail", func(t *testing.T) {
		controller := DefaultModelCascadeController()

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			return "broken-everywhere", nil
		}

		validatorFn := func(ctx context.Context, output string) error {
			return NewValidationError("permanent rejection", "output rejected across all tiers")
		}

		res := controller.Cascade(ctx, "Impossible task", validatorFn, invokerFn)

		if res.Error == nil {
			t.Fatal("expected error on cascade exhaustion, got nil")
		}
		if !strings.Contains(res.Error.Error(), "cascade exhausted") {
			t.Fatalf("expected cascade exhausted error, got %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected Escalated=true after exhausting tiers")
		}
		if len(res.Attempts) != 3 {
			t.Fatalf("expected 3 attempts, got %d", len(res.Attempts))
		}
	})

	t.Run("schema validation adapter escalation", func(t *testing.T) {
		schemaValidator := NewSchemaValidator(ToolSchema{
			Name: "deploy_service",
			Properties: map[string]PropertySchema{
				"service_name": {Type: TypeString},
				"replicas":     {Type: TypeInteger},
			},
			Required: []string{"service_name", "replicas"},
		})

		validatorFn := SchemaValidationAdapter(schemaValidator, "deploy_service")

		controller := NewModelCascadeController(
			CascadeStep{
				Target: TargetModel{Name: "qwen-fast", Tier: TierFast},
				Aspect: AspectSchema,
			},
			CascadeStep{
				Target: TargetModel{Name: "qwen-standard", Tier: TierStandard},
				Aspect: AspectSchema,
			},
		)

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			if target.Tier == TierFast {
				// Missing required property replicas
				return `{"service_name": "payment-svc"}`, nil
			}
			// Fixed in standard tier
			return `{"service_name": "payment-svc", "replicas": 3}`, nil
		}

		res := controller.Cascade(ctx, "Deploy payment service with 3 replicas", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected escalation on schema violation")
		}
		if res.FinalTier != TierStandard {
			t.Fatalf("expected FinalTier=%s, got %s", TierStandard, res.FinalTier)
		}
		if len(res.Attempts) != 2 {
			t.Fatalf("expected 2 attempts, got %d", len(res.Attempts))
		}
		if !strings.Contains(res.Attempts[0].EvidenceInfo, "missing required property") {
			t.Fatalf("expected schema violation evidence, got %q", res.Attempts[0].EvidenceInfo)
		}
	})

	t.Run("json syntax validator escalation", func(t *testing.T) {
		validatorFn := JSONSyntaxValidator()
		controller := DefaultModelCascadeController()

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			if target.Tier == TierFast {
				return `{"unclosed_json": 123`, nil
			}
			return `{"unclosed_json": 123}`, nil
		}

		res := controller.Cascade(ctx, "Format JSON", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected escalation on syntax failure")
		}
		if res.FinalTier != TierStandard {
			t.Fatalf("expected FinalTier=%s, got %s", TierStandard, res.FinalTier)
		}
	})

	t.Run("test execution validator escalation", func(t *testing.T) {
		testFn := func(ctx context.Context, output string) (bool, string, error) {
			if strings.Contains(output, "def add(a, b): return a + b") {
				return true, "all 3 test assertions passed", nil
			}
			return false, "assertion failure: add(2, 2) returned None", nil
		}
		validatorFn := TestExecutionValidator(testFn)
		controller := DefaultModelCascadeController()

		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			if target.Tier == TierFast {
				return "def add(a, b): pass", nil
			}
			return "def add(a, b): return a + b", nil
		}

		res := controller.Cascade(ctx, "Implement add function in python", validatorFn, invokerFn)

		if res.Error != nil {
			t.Fatalf("unexpected error: %v", res.Error)
		}
		if !res.Escalated {
			t.Fatal("expected escalation on test execution failure")
		}
		if res.FinalTier != TierStandard {
			t.Fatalf("expected FinalTier=%s, got %s", TierStandard, res.FinalTier)
		}
		if !strings.Contains(res.Attempts[0].EvidenceInfo, "assertion failure") {
			t.Fatalf("expected assertion failure evidence, got %q", res.Attempts[0].EvidenceInfo)
		}
	})

	t.Run("context cancellation terminates cascade", func(t *testing.T) {
		cancelingCtx, cancel := context.WithCancel(context.Background())

		controller := DefaultModelCascadeController()
		invokerFn := func(ctx context.Context, target TargetModel, prompt string) (string, error) {
			cancel() // Cancel before second attempt
			return "invalid", nil
		}
		validatorFn := func(ctx context.Context, output string) error {
			return errors.New("always invalid")
		}

		res := controller.Cascade(cancelingCtx, "Prompt", validatorFn, invokerFn)

		if res.Error == nil {
			t.Fatal("expected context canceled error, got nil")
		}
		if !errors.Is(res.Error, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", res.Error)
		}
	})
}

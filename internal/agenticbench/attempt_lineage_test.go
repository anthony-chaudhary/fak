package agenticbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAttemptLineage(t *testing.T) {
	t.Run("single_attempt_success", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-finance-forecast-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-001",
					UnitID:         "task-finance-forecast-001",
					IsClean:        true,
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(1.0),
					CostUSD:        0.05,
					Tokens:         1200,
					DurationMS:     4500,
					ArtifactRoot:   "/workspace/runs/att-1",
					ArtifactSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
					RetryReason:    "",
				},
			},
		}

		ComputeAggregate(&packet)

		if packet.Aggregate.TotalCostUSD != 0.05 {
			t.Fatalf("TotalCostUSD = %v, want 0.05", packet.Aggregate.TotalCostUSD)
		}
		if packet.Aggregate.TotalTokens != 1200 {
			t.Fatalf("TotalTokens = %d, want 1200", packet.Aggregate.TotalTokens)
		}
		if packet.Aggregate.TotalDurationMS != 4500 {
			t.Fatalf("TotalDurationMS = %d, want 4500", packet.Aggregate.TotalDurationMS)
		}
		if packet.Aggregate.TotalAttempts != 1 {
			t.Fatalf("TotalAttempts = %d, want 1", packet.Aggregate.TotalAttempts)
		}
		if packet.Aggregate.FinalScore == nil || *packet.Aggregate.FinalScore != 1.0 {
			t.Fatalf("FinalScore = %v, want 1.0", packet.Aggregate.FinalScore)
		}
		if packet.Aggregate.SelectedAttemptID != "att-1" {
			t.Fatalf("SelectedAttemptID = %q, want %q", packet.Aggregate.SelectedAttemptID, "att-1")
		}

		if err := ValidateAttemptLineage(packet); err != nil {
			t.Fatalf("unexpected validation error: %v", err)
		}
	})

	t.Run("multi_attempt_failed_failed_success", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-finance-forecast-002",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-002-1",
					UnitID:         "task-finance-forecast-002",
					IsClean:        true,
					TerminalStatus: TerminalStatusFailed,
					Score:          nil,
					CostUSD:        0.04,
					Tokens:         1000,
					DurationMS:     3000,
					ArtifactRoot:   "/workspace/runs/att-1",
					ArtifactSHA256: "sha256-att1",
					RetryReason:    "",
				},
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					RunID:          "run-002-2",
					UnitID:         "task-finance-forecast-002",
					IsClean:        false,
					TerminalStatus: TerminalStatusFailed,
					Score:          nil,
					CostUSD:        0.06,
					Tokens:         1500,
					DurationMS:     4000,
					ArtifactRoot:   "/workspace/runs/att-2",
					ArtifactSHA256: "sha256-att2",
					RetryReason:    "assertion_failed",
				},
				{
					AttemptID:      "att-3",
					Ordinal:        3,
					RunID:          "run-002-3",
					UnitID:         "task-finance-forecast-002",
					IsClean:        false,
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(0.85),
					CostUSD:        0.07,
					Tokens:         1800,
					DurationMS:     5000,
					ArtifactRoot:   "/workspace/runs/att-3",
					ArtifactSHA256: "sha256-att3",
					RetryReason:    "replan_after_failure",
				},
			},
		}

		ComputeAggregate(&packet)

		// Failed attempt costs and tokens MUST be preserved in totals.
		if packet.Aggregate.TotalCostUSD != 0.17 {
			t.Fatalf("TotalCostUSD = %v, want 0.17 (0.04+0.06+0.07)", packet.Aggregate.TotalCostUSD)
		}
		if packet.Aggregate.TotalTokens != 4300 {
			t.Fatalf("TotalTokens = %d, want 4300 (1000+1500+1800)", packet.Aggregate.TotalTokens)
		}
		if packet.Aggregate.TotalDurationMS != 12000 {
			t.Fatalf("TotalDurationMS = %d, want 12000 (3000+4000+5000)", packet.Aggregate.TotalDurationMS)
		}
		if packet.Aggregate.TotalAttempts != 3 {
			t.Fatalf("TotalAttempts = %d, want 3", packet.Aggregate.TotalAttempts)
		}
		if packet.Aggregate.FinalScore == nil || *packet.Aggregate.FinalScore != 0.85 {
			t.Fatalf("FinalScore = %v, want 0.85 (from attempt 3)", packet.Aggregate.FinalScore)
		}
		if packet.Aggregate.SelectedAttemptID != "att-3" {
			t.Fatalf("SelectedAttemptID = %q, want %q", packet.Aggregate.SelectedAttemptID, "att-3")
		}

		if err := ValidateAttemptLineage(packet); err != nil {
			t.Fatalf("unexpected validation error: %v", err)
		}
	})

	t.Run("adversarial_hidden_attempt", func(t *testing.T) {
		// Attempt 1 is missing: packet starts at ordinal 2 to hide initial spend.
		packet := AttemptLineagePacket{
			UnitID:        "task-hidden-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					RunID:          "run-002",
					UnitID:         "task-hidden-001",
					IsClean:        false,
					TerminalStatus: TerminalStatusFailed,
					CostUSD:        0.05,
					Tokens:         1000,
					DurationMS:     2000,
					ArtifactRoot:   "/workspace/runs/att-2",
					RetryReason:    "retry_from_hidden",
				},
				{
					AttemptID:      "att-3",
					Ordinal:        3,
					RunID:          "run-003",
					UnitID:         "task-hidden-001",
					IsClean:        false,
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(1.0),
					CostUSD:        0.05,
					Tokens:         1000,
					DurationMS:     2000,
					ArtifactRoot:   "/workspace/runs/att-3",
					RetryReason:    "replan",
				},
			},
		}

		err := ValidateAttemptLineage(packet)
		if err == nil {
			t.Fatalf("expected validation error for hidden attempt 1, got nil")
		}
		if !IsLineageReason(err, ReasonHiddenAttempt) {
			t.Fatalf("err = %v, want reason %s", err, ReasonHiddenAttempt)
		}

		// Discovered attempts reconciliation fails when on-disk attempt 1 was omitted from packet.
		packetValidOrdinals := packet
		packetValidOrdinals.Attempts[0].Ordinal = 1
		packetValidOrdinals.Attempts[0].RetryReason = ""
		packetValidOrdinals.Attempts[1].Ordinal = 2
		discovered := []string{"/workspace/runs/att-0-original", "/workspace/runs/att-2", "/workspace/runs/att-3"}
		reconcileErr := ReconcileDiscoveredAttempts(packetValidOrdinals, discovered)
		if reconcileErr == nil {
			t.Fatalf("expected reconciliation error when discovered attempts contain hidden directories, got nil")
		}
		if !IsLineageReason(reconcileErr, ReasonHiddenAttempt) {
			t.Fatalf("reconcileErr = %v, want reason %s", reconcileErr, ReasonHiddenAttempt)
		}
	})

	t.Run("adversarial_duplicate_ordinals", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-dup-ord-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-001",
					UnitID:         "task-dup-ord-001",
					TerminalStatus: TerminalStatusFailed,
					ArtifactRoot:   "/workspace/runs/att-1",
				},
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					RunID:          "run-002",
					UnitID:         "task-dup-ord-001",
					TerminalStatus: TerminalStatusFailed,
					ArtifactRoot:   "/workspace/runs/att-2",
					RetryReason:    "assertion_failed",
				},
				{
					AttemptID:      "att-3",
					Ordinal:        2, // Duplicate ordinal 2
					RunID:          "run-003",
					UnitID:         "task-dup-ord-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(1.0),
					ArtifactRoot:   "/workspace/runs/att-3",
					RetryReason:    "assertion_failed",
				},
			},
		}

		err := ValidateAttemptLineage(packet)
		if err == nil {
			t.Fatalf("expected validation error for duplicate ordinal, got nil")
		}
		if !IsLineageReason(err, ReasonDuplicateOrdinal) {
			t.Fatalf("err = %v, want reason %s", err, ReasonDuplicateOrdinal)
		}
	})

	t.Run("adversarial_non_contiguous_ordinals", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-noncontig-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-001",
					UnitID:         "task-noncontig-001",
					TerminalStatus: TerminalStatusFailed,
					ArtifactRoot:   "/workspace/runs/att-1",
				},
				{
					AttemptID:      "att-3",
					Ordinal:        3, // Skips ordinal 2
					RunID:          "run-003",
					UnitID:         "task-noncontig-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(1.0),
					ArtifactRoot:   "/workspace/runs/att-3",
					RetryReason:    "assertion_failed",
				},
			},
		}

		err := ValidateAttemptLineage(packet)
		if err == nil {
			t.Fatalf("expected validation error for non-contiguous ordinals, got nil")
		}
		if !IsLineageReason(err, ReasonNonContiguousOrdinals) {
			t.Fatalf("err = %v, want reason %s", err, ReasonNonContiguousOrdinals)
		}
	})

	t.Run("adversarial_timeout_retry_forbidden", func(t *testing.T) {
		// Under ALE native failure_only policy: timeouts MUST NOT be retried.
		packet := AttemptLineagePacket{
			UnitID:        "task-timeout-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-001",
					UnitID:         "task-timeout-001",
					TerminalStatus: TerminalStatusTimeout,
					ArtifactRoot:   "/workspace/runs/att-1",
				},
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					RunID:          "run-002",
					UnitID:         "task-timeout-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(1.0),
					ArtifactRoot:   "/workspace/runs/att-2",
					RetryReason:    "retry_after_timeout",
				},
			},
		}

		err := ValidateAttemptLineage(packet)
		if err == nil {
			t.Fatalf("expected validation error for timeout retry with failure_only, got nil")
		}
		if !IsLineageReason(err, ReasonTimeoutRetryForbidden) {
			t.Fatalf("err = %v, want reason %s", err, ReasonTimeoutRetryForbidden)
		}
	})

	t.Run("adversarial_unequal_retry_policy", func(t *testing.T) {
		armA := AttemptLineagePacket{
			UnitID:        "task-comp-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
		}
		armB := AttemptLineagePacket{
			UnitID:        "task-comp-001",
			RetryRegime:   NoRetry,
			MaxAttempts:   3,
			SelectionRule: FinalAttempt,
		}

		err := CompareArmLineage(armA, armB)
		if err == nil {
			t.Fatalf("expected cross-arm comparison refusal for unequal retry policies, got nil")
		}
		if !IsLineageReason(err, ReasonRetryRegimeMismatch) {
			t.Fatalf("err = %v, want reason %s", err, ReasonRetryRegimeMismatch)
		}

		// Also check MaxAttempts mismatch refusal.
		armB.RetryRegime = RetryOnFailureOnly
		armB.MaxAttempts = 2
		errMax := CompareArmLineage(armA, armB)
		if errMax == nil {
			t.Fatalf("expected cross-arm comparison refusal for unequal MaxAttempts, got nil")
		}
		if !IsLineageReason(errMax, ReasonRetryRegimeMismatch) {
			t.Fatalf("errMax = %v, want reason %s", errMax, ReasonRetryRegimeMismatch)
		}
	})

	t.Run("selection_rule_best_of_n", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-bon-001",
			RetryRegime:   RetryAll,
			MaxAttempts:   3,
			SelectionRule: BestOfN,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-1",
					UnitID:         "task-bon-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(0.40),
					CostUSD:        0.05,
					Tokens:         1000,
					DurationMS:     2000,
					ArtifactRoot:   "/workspace/runs/att-1",
				},
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					RunID:          "run-2",
					UnitID:         "task-bon-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(0.95), // Highest score
					CostUSD:        0.05,
					Tokens:         1000,
					DurationMS:     2000,
					ArtifactRoot:   "/workspace/runs/att-2",
					RetryReason:    "bon_sample_2",
				},
				{
					AttemptID:      "att-3",
					Ordinal:        3,
					RunID:          "run-3",
					UnitID:         "task-bon-001",
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(0.70),
					CostUSD:        0.05,
					Tokens:         1000,
					DurationMS:     2000,
					ArtifactRoot:   "/workspace/runs/att-3",
					RetryReason:    "bon_sample_3",
				},
			},
		}

		ComputeAggregate(&packet)

		if packet.Aggregate.FinalScore == nil || *packet.Aggregate.FinalScore != 0.95 {
			t.Fatalf("BestOfN FinalScore = %v, want 0.95", packet.Aggregate.FinalScore)
		}
		if packet.Aggregate.SelectedAttemptID != "att-2" {
			t.Fatalf("BestOfN SelectedAttemptID = %q, want %q", packet.Aggregate.SelectedAttemptID, "att-2")
		}
		if packet.Aggregate.TotalCostUSD != 0.15 {
			t.Fatalf("TotalCostUSD = %v, want 0.15", packet.Aggregate.TotalCostUSD)
		}

		// Contrast with FinalAttempt: must pick attempt 3 (score 0.70)
		packetFinal := packet
		packetFinal.SelectionRule = FinalAttempt
		ComputeAggregate(&packetFinal)
		if packetFinal.Aggregate.FinalScore == nil || *packetFinal.Aggregate.FinalScore != 0.70 {
			t.Fatalf("FinalAttempt FinalScore = %v, want 0.70", packetFinal.Aggregate.FinalScore)
		}
		if packetFinal.Aggregate.SelectedAttemptID != "att-3" {
			t.Fatalf("FinalAttempt SelectedAttemptID = %q, want %q", packetFinal.Aggregate.SelectedAttemptID, "att-3")
		}
	})

	t.Run("adversarial_duplicate_attempt_id_and_root", func(t *testing.T) {
		packetID := AttemptLineagePacket{
			UnitID:        "task-dup-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-same",
					Ordinal:        1,
					TerminalStatus: TerminalStatusFailed,
					ArtifactRoot:   "/workspace/runs/att-1",
				},
				{
					AttemptID:      "att-same", // Duplicate AttemptID
					Ordinal:        2,
					TerminalStatus: TerminalStatusSuccess,
					ArtifactRoot:   "/workspace/runs/att-2",
					RetryReason:    "retry",
				},
			},
		}
		errID := ValidateAttemptLineage(packetID)
		if !IsLineageReason(errID, ReasonDuplicateAttemptID) {
			t.Fatalf("errID = %v, want reason %s", errID, ReasonDuplicateAttemptID)
		}

		packetRoot := AttemptLineagePacket{
			UnitID:        "task-dup-002",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					TerminalStatus: TerminalStatusFailed,
					ArtifactRoot:   "/workspace/runs/overwritten-root",
				},
				{
					AttemptID:      "att-2",
					Ordinal:        2,
					TerminalStatus: TerminalStatusSuccess,
					ArtifactRoot:   "/workspace/runs/overwritten-root", // Duplicate root
					RetryReason:    "retry",
				},
			},
		}
		errRoot := ValidateAttemptLineage(packetRoot)
		if !IsLineageReason(errRoot, ReasonDuplicateArtifactRoot) {
			t.Fatalf("errRoot = %v, want reason %s", errRoot, ReasonDuplicateArtifactRoot)
		}
	})

	t.Run("adversarial_max_attempts_and_no_retry", func(t *testing.T) {
		packetMax := AttemptLineagePacket{
			UnitID:        "task-max-001",
			RetryRegime:   RetryAll,
			MaxAttempts:   1,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{AttemptID: "att-1", Ordinal: 1, TerminalStatus: TerminalStatusFailed, ArtifactRoot: "/r/1"},
				{AttemptID: "att-2", Ordinal: 2, TerminalStatus: TerminalStatusSuccess, ArtifactRoot: "/r/2", RetryReason: "retry"},
			},
		}
		errMax := ValidateAttemptLineage(packetMax)
		if !IsLineageReason(errMax, ReasonMaxAttemptsExceeded) {
			t.Fatalf("errMax = %v, want reason %s", errMax, ReasonMaxAttemptsExceeded)
		}

		packetNoRetry := AttemptLineagePacket{
			UnitID:        "task-noretry-001",
			RetryRegime:   NoRetry,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{AttemptID: "att-1", Ordinal: 1, TerminalStatus: TerminalStatusFailed, ArtifactRoot: "/r/1"},
				{AttemptID: "att-2", Ordinal: 2, TerminalStatus: TerminalStatusSuccess, ArtifactRoot: "/r/2", RetryReason: "retry"},
			},
		}
		errNoRetry := ValidateAttemptLineage(packetNoRetry)
		if !IsLineageReason(errNoRetry, ReasonRetryForbidden) {
			t.Fatalf("errNoRetry = %v, want reason %s", errNoRetry, ReasonRetryForbidden)
		}
	})

	t.Run("attempt1_cannot_carry_retry_reason", func(t *testing.T) {
		packet := AttemptLineagePacket{
			UnitID:        "task-reason-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					TerminalStatus: TerminalStatusSuccess,
					ArtifactRoot:   "/r/1",
					RetryReason:    "unexpected_reason_on_first_attempt",
				},
			},
		}
		err := ValidateAttemptLineage(packet)
		if !IsLineageReason(err, ReasonInvalidRetryReason) {
			t.Fatalf("err = %v, want reason %s", err, ReasonInvalidRetryReason)
		}
	})

	t.Run("json_roundtrip_and_file_io", func(t *testing.T) {
		tempDir := t.TempDir()
		packetFile := filepath.Join(tempDir, "attempt-lineage.json")

		packet := AttemptLineagePacket{
			Schema:        AttemptLineageSchema,
			UnitID:        "task-io-001",
			RetryRegime:   RetryOnFailureOnly,
			MaxAttempts:   2,
			SelectionRule: FinalAttempt,
			Attempts: []AttemptReceipt{
				{
					AttemptID:      "att-1",
					Ordinal:        1,
					RunID:          "run-1",
					UnitID:         "task-io-001",
					IsClean:        true,
					TerminalStatus: TerminalStatusSuccess,
					Score:          floatPtr(0.92),
					CostUSD:        0.035,
					Tokens:         850,
					DurationMS:     2100,
					ArtifactRoot:   filepath.Join(tempDir, "att-1"),
					ArtifactSHA256: "hash-att-1",
				},
			},
		}
		ComputeAggregate(&packet)

		data, err := json.MarshalIndent(packet, "", "  ")
		if err != nil {
			t.Fatalf("json marshal error: %v", err)
		}
		if err := os.WriteFile(packetFile, data, 0o644); err != nil {
			t.Fatalf("write file error: %v", err)
		}

		loaded, err := ReadAttemptLineagePacket(packetFile)
		if err != nil {
			t.Fatalf("read packet error: %v", err)
		}
		if err := ValidateAttemptLineage(loaded); err != nil {
			t.Fatalf("validate loaded packet error: %v", err)
		}
		if loaded.Aggregate.TotalCostUSD != 0.035 {
			t.Fatalf("loaded TotalCostUSD = %v, want 0.035", loaded.Aggregate.TotalCostUSD)
		}
	})
}

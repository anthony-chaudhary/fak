package schedcontract

import (
	"testing"
	"time"
)

func newValidContract(now time.Time) ExecutionContract {
	return ExecutionContract{
		ContractID: "contract-test-001",
		Schema:     Schema,
		TaskID:     "task-9872",
		Lane:       "engine",
		Priority:   PriorityHigh,
		Deadline:   now.Add(30 * time.Minute),
		CreatedAt:  now.Add(-1 * time.Minute),
		Token: ExecutionToken{
			TokenID:      "tok-xyz-123",
			Issuer:       "fak-kernel-gateway",
			Subject:      "agent-worker-42",
			Lane:         "engine",
			IssuedAt:     now.Add(-2 * time.Minute),
			ExpiresAt:    now.Add(45 * time.Minute),
			Capabilities: []string{"read_ast", "write_patch", "exec_test"},
			Signature:    "sig-verified-hmac-sha256-abcde",
			Nonce:        "nonce-981247",
		},
		Constraints: ScheduleConstraints{
			MaxConcurrency:       4,
			MinWaitDuration:      100 * time.Millisecond,
			MaxLeaseDuration:     15 * time.Minute,
			AllowedLanes:         []string{"engine", "gateway", "vdso"},
			RequiredCapabilities: []string{"read_ast"},
			MemoryLimitBytes:     512 * 1024 * 1024,
			ExclusiveOnly:        false,
		},
	}
}

func TestSchedulerContractValidation(t *testing.T) {
	now := time.Now()

	t.Run("valid contract passes", func(t *testing.T) {
		c := newValidContract(now)
		if err := c.Validate(now); err != nil {
			t.Fatalf("expected valid contract to pass, got: %v", err)
		}
	})

	t.Run("invalid schema fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Schema = "fak.sched-contract/0.9-alpha"
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for invalid schema, got nil")
		}
	})

	t.Run("missing contract ID fails", func(t *testing.T) {
		c := newValidContract(now)
		c.ContractID = "   "
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for empty contract ID, got nil")
		}
	})

	t.Run("missing task ID fails", func(t *testing.T) {
		c := newValidContract(now)
		c.TaskID = ""
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for empty task ID, got nil")
		}
	})

	t.Run("missing lane fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Lane = ""
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for empty lane, got nil")
		}
	})

	t.Run("invalid priority tier fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Priority = Priority("ultra-urgent")
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for invalid priority, got nil")
		}
	})

	t.Run("zero deadline fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Deadline = time.Time{}
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for zero deadline, got nil")
		}
	})

	t.Run("past deadline fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Deadline = now.Add(-5 * time.Second)
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for elapsed deadline, got nil")
		}
	})

	t.Run("disallowed lane fails", func(t *testing.T) {
		c := newValidContract(now)
		c.Lane = "forbidden-lane"
		if err := c.Validate(now); err == nil {
			t.Fatalf("expected error for lane not in allowed list, got nil")
		}
	})
}

func TestContractInvariants(t *testing.T) {
	now := time.Now()

	t.Run("valid contract satisfies all invariants", func(t *testing.T) {
		c := newValidContract(now)
		if err := CheckInvariants(&c, now); err != nil {
			t.Fatalf("expected clean invariant check, got: %v", err)
		}
	})

	t.Run("nil contract violates invariant", func(t *testing.T) {
		if err := CheckInvariants(nil, now); err == nil {
			t.Fatalf("expected error for nil contract, got nil")
		}
	})

	t.Run("token expiration before contract deadline violates invariant", func(t *testing.T) {
		c := newValidContract(now)
		// Deadline is +30m, but token expires in +10m
		c.Token.ExpiresAt = now.Add(10 * time.Minute)
		if err := CheckInvariants(&c, now); err == nil {
			t.Fatalf("expected invariant failure when token expires before deadline, got nil")
		}
	})

	t.Run("mismatched token lane violates invariant", func(t *testing.T) {
		c := newValidContract(now)
		c.Token.Lane = "gateway" // contract lane is "engine"
		if err := CheckInvariants(&c, now); err == nil {
			t.Fatalf("expected invariant failure on token/contract lane mismatch, got nil")
		}
	})

	t.Run("missing required capability violates invariant", func(t *testing.T) {
		c := newValidContract(now)
		c.Constraints.RequiredCapabilities = []string{"super_admin_exec"}
		if err := CheckInvariants(&c, now); err == nil {
			t.Fatalf("expected invariant failure on missing required capability, got nil")
		}
	})

	t.Run("critical priority requires exclusivity or emergency capability", func(t *testing.T) {
		c := newValidContract(now)
		c.Priority = PriorityCritical
		c.Constraints.ExclusiveOnly = false
		// Token lacks emergency_override or critical_preempt
		if err := CheckInvariants(&c, now); err == nil {
			t.Fatalf("expected invariant failure for non-exclusive critical task without emergency capability, got nil")
		}

		// Satisfying with ExclusiveOnly
		c.Constraints.ExclusiveOnly = true
		if err := CheckInvariants(&c, now); err != nil {
			t.Fatalf("expected exclusive critical task to pass invariant, got: %v", err)
		}

		// Satisfying with emergency capability
		c.Constraints.ExclusiveOnly = false
		c.Token.Capabilities = append(c.Token.Capabilities, "emergency_override")
		if err := CheckInvariants(&c, now); err != nil {
			t.Fatalf("expected critical task with emergency_override capability to pass invariant, got: %v", err)
		}
	})

	t.Run("lease duration exceeding remaining deadline violates invariant", func(t *testing.T) {
		c := newValidContract(now)
		c.Deadline = now.Add(5 * time.Minute)
		c.Constraints.MaxLeaseDuration = 10 * time.Minute
		if err := CheckInvariants(&c, now); err == nil {
			t.Fatalf("expected invariant failure when max lease exceeds remaining deadline duration, got nil")
		}
	})
}

func TestScheduleConstraints(t *testing.T) {
	t.Run("valid constraints pass", func(t *testing.T) {
		sc := ScheduleConstraints{
			MaxConcurrency:   2,
			MinWaitDuration:  50 * time.Millisecond,
			MaxLeaseDuration: 5 * time.Minute,
			MemoryLimitBytes: 1024 * 1024,
		}
		if err := sc.Validate(); err != nil {
			t.Fatalf("expected valid constraints to pass, got: %v", err)
		}
	})

	t.Run("negative concurrency fails", func(t *testing.T) {
		sc := ScheduleConstraints{MaxConcurrency: -1}
		if err := sc.Validate(); err == nil {
			t.Fatalf("expected error on negative concurrency, got nil")
		}
	})

	t.Run("negative min wait duration fails", func(t *testing.T) {
		sc := ScheduleConstraints{MinWaitDuration: -10 * time.Second}
		if err := sc.Validate(); err == nil {
			t.Fatalf("expected error on negative min wait duration, got nil")
		}
	})

	t.Run("negative max lease duration fails", func(t *testing.T) {
		sc := ScheduleConstraints{MaxLeaseDuration: -1 * time.Second}
		if err := sc.Validate(); err == nil {
			t.Fatalf("expected error on negative max lease duration, got nil")
		}
	})

	t.Run("max lease shorter than min wait duration fails", func(t *testing.T) {
		sc := ScheduleConstraints{
			MinWaitDuration:  10 * time.Second,
			MaxLeaseDuration: 5 * time.Second,
		}
		if err := sc.Validate(); err == nil {
			t.Fatalf("expected error when max lease is less than min wait duration, got nil")
		}
	})

	t.Run("negative memory limit fails", func(t *testing.T) {
		sc := ScheduleConstraints{MemoryLimitBytes: -100}
		if err := sc.Validate(); err == nil {
			t.Fatalf("expected error on negative memory limit bytes, got nil")
		}
	})

	t.Run("allows lane logic handles empty and explicit list", func(t *testing.T) {
		scOpen := ScheduleConstraints{AllowedLanes: nil}
		if !scOpen.AllowsLane("any-lane") {
			t.Fatalf("expected empty allowed lanes to permit any lane")
		}

		scRestricted := ScheduleConstraints{AllowedLanes: []string{"engine", "vdso"}}
		if !scRestricted.AllowsLane("engine") {
			t.Fatalf("expected allowed lane to return true")
		}
		if scRestricted.AllowsLane("billing") {
			t.Fatalf("expected unlisted lane to return false")
		}
	})
}

func BenchmarkScheduleContract(b *testing.B) {
	now := time.Now()
	contract := newValidContract(now)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := contract.Validate(now); err != nil {
			b.Fatalf("contract validation failed: %v", err)
		}
		if err := CheckInvariants(&contract, now); err != nil {
			b.Fatalf("contract invariant verification failed: %v", err)
		}
	}
}

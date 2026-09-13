package localadmission

import "testing"

func envelopePlan(class MemClass, bytes int64) EnvelopePlan {
	return EnvelopePlan{{Class: class, Bytes: bytes}}
}

func TestSessionResidencyEnvelopeMath(t *testing.T) {
	weights := envelopePlan(MemClassWeights, 1000)
	perContext := envelopePlan(MemClassKVCache, 100)
	plan := NewSessionResidencyPlan(weights, perContext, 4, 1)

	if got, want := plan.SteadyBytes(), int64(1400); got != want {
		t.Fatalf("SteadyBytes=%d want %d", got, want)
	}
	// The stated one-session load window is weights + one session (1000+100);
	// the enforced 4-session Steady envelope (1400) dominates it, and
	// reservation.go requires peak >= steady, so StartupPeakBytes floors to 1400.
	if got, want := plan.StartupPeakBytes(), int64(1400); got != want {
		t.Fatalf("StartupPeakBytes=%d want %d (floored at SteadyBytes)", got, want)
	}
	if plan.StartupPeakBytes() < plan.SteadyBytes() {
		t.Fatalf("StartupPeakBytes=%d must be >= SteadyBytes=%d", plan.StartupPeakBytes(), plan.SteadyBytes())
	}

	classes := plan.Classes()
	if got, want := classes[MemClassWeights], int64(1000); got != want {
		t.Fatalf("classes[weights]=%d want %d", got, want)
	}
	if got, want := classes[MemClassKVCache], int64(400); got != want {
		t.Fatalf("classes[kv_cache]=%d want %d", got, want)
	}
	var sum int64
	for _, n := range classes {
		sum += n
	}
	if sum != plan.SteadyBytes() {
		t.Fatalf("class sum=%d want SteadyBytes=%d", sum, plan.SteadyBytes())
	}
}

func TestSessionResidencyEnvelopeFloorMaxSessions(t *testing.T) {
	plan := NewSessionResidencyPlan(envelopePlan(MemClassWeights, 1000), envelopePlan(MemClassKVCache, 100), 0, 1)

	if plan.MaxSessions != 1 {
		t.Fatalf("MaxSessions=%d want 1", plan.MaxSessions)
	}
	if got, want := plan.SteadyBytes(), int64(1100); got != want {
		t.Fatalf("SteadyBytes=%d want %d (one session's state must survive the floor)", got, want)
	}

	negative := NewSessionResidencyPlan(envelopePlan(MemClassWeights, 1000), envelopePlan(MemClassKVCache, 100), -3, 1)
	if negative.MaxSessions != 1 || negative.SteadyBytes() != 1100 {
		t.Fatalf("negative maxSessions: plan=%+v steady=%d", negative, negative.SteadyBytes())
	}
}

func TestSessionResidencyEnvelopeCohortScratch(t *testing.T) {
	weights := envelopePlan(MemClassWeights, 1000)
	perContext := envelopePlan(MemClassKVCache, 100)

	single := NewSessionResidencyPlan(weights, perContext, 4, 1)
	cohort := NewSessionResidencyPlan(weights, perContext, 4, 2)

	if len(single.CohortScratch) != 0 {
		t.Fatalf("cohort=1 must add no scratch, got %+v", single.CohortScratch)
	}
	if len(cohort.CohortScratch) == 0 {
		t.Fatalf("cohort=2 must add scratch")
	}
	if got, want := single.StartupPeakBytes(), int64(1400); got != want {
		t.Fatalf("single StartupPeakBytes=%d want %d", got, want)
	}
	if got, want := cohort.StartupPeakBytes(), int64(1500); got != want {
		t.Fatalf("cohort StartupPeakBytes=%d want %d", got, want)
	}
	if cohort.SteadyBytes() <= single.SteadyBytes() {
		t.Fatalf("cohort SteadyBytes=%d must exceed single=%d", cohort.SteadyBytes(), single.SteadyBytes())
	}
	if got, want := cohort.SteadyBytes(), single.SteadyBytes()+100; got != want {
		t.Fatalf("cohort SteadyBytes=%d want %d", got, want)
	}
	if cohort.StartupPeakBytes() < cohort.SteadyBytes() {
		t.Fatalf("cohort StartupPeakBytes=%d must be >= SteadyBytes=%d", cohort.StartupPeakBytes(), cohort.SteadyBytes())
	}
}

// TestEnvelopeClassStringValues pins the persisted class labels so a ledger
// written here stays readable by the reservation store.
func TestEnvelopeClassStringValues(t *testing.T) {
	if got := string(MemClassWeights); got != "weights" {
		t.Fatalf("MemClassWeights=%q want weights", got)
	}
	if got := string(MemClassKVCache); got != "kv_cache" {
		t.Fatalf("MemClassKVCache=%q want kv_cache", got)
	}
	if got := string(MemClassActivation); got != "activation" {
		t.Fatalf("MemClassActivation=%q want activation", got)
	}
}

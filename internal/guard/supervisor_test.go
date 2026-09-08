package guard

import (
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestClassifyUpstream429AsBackpressure(t *testing.T) {
	sup := NewSupervisor(SupervisorConfig{
		CrashRestartLimit:        3,
		InitialBackpressurePause: 2 * time.Second,
	})

	exitErr := errors.New("exit status 1")

	// 1. Exit code 1 accompanied by structured UPSTREAM_FAILURE with http_status: 429
	evidence := ChildExitEvidence{
		ExitCode: 1,
		RunErr:   exitErr,
		UpstreamFailure: &UpstreamFailureReceipt{
			HTTPStatus: http.StatusTooManyRequests,
			Cause:      "rate limit exceeded on completion endpoint",
		},
	}

	decision := sup.HandleChildExit(1, exitErr, evidence)

	if decision.State != StateProviderRateLimited {
		t.Fatalf("expected state %q, got %q", StateProviderRateLimited, decision.State)
	}
	if decision.Action != ActionBackpressurePause {
		t.Fatalf("expected action %q, got %q", ActionBackpressurePause, decision.Action)
	}
	if decision.Reason != ReasonProviderRateLimited {
		t.Fatalf("expected reason %q, got %q", ReasonProviderRateLimited, decision.Reason)
	}
	if decision.BurnsBudget {
		t.Fatal("expected BurnsBudget to be false for rate limit, got true")
	}
	if decision.RestartHop != 0 {
		t.Fatalf("expected RestartHop to be 0, got %d", decision.RestartHop)
	}
	if sup.CrashRestarts() != 0 {
		t.Fatalf("expected CrashRestarts to be 0, got %d", sup.CrashRestarts())
	}
	if sup.ConsecutiveRateLimits() != 1 {
		t.Fatalf("expected ConsecutiveRateLimits to be 1, got %d", sup.ConsecutiveRateLimits())
	}
}

func TestClassifyAccountRotationProviderRateLimitedAsBackpressure(t *testing.T) {
	sup := NewSupervisor(SupervisorConfig{CrashRestartLimit: 3})
	exitErr := errors.New("exit status 1")

	evidence := ChildExitEvidence{
		ExitCode:       1,
		RunErr:         exitErr,
		RotationReason: "provider_rate_limited",
		AccountRotation: &AccountRotationEvent{
			Seat:   "seat-alpha",
			Reason: "provider_rate_limited",
			Detail: "upstream_rate_limited_delta=1",
		},
	}

	decision := sup.HandleChildExit(1, exitErr, evidence)

	if decision.State != StateProviderRateLimited {
		t.Fatalf("expected state %q, got %q", StateProviderRateLimited, decision.State)
	}
	if decision.Action != ActionBackpressurePause {
		t.Fatalf("expected action %q, got %q", ActionBackpressurePause, decision.Action)
	}
	if decision.BurnsBudget {
		t.Fatal("ACCOUNT_ROTATION provider_rate_limited burned crash restart budget")
	}
	if sup.CrashRestarts() != 0 {
		t.Fatalf("CrashRestarts incremented on account rotation rate limit: %d", sup.CrashRestarts())
	}
}

func TestRateLimitDoesNotExhaustCrashRestartBudget(t *testing.T) {
	limit := 3
	sup := NewSupervisor(SupervisorConfig{
		CrashRestartLimit:        limit,
		InitialBackpressurePause: 1 * time.Second,
		MaxBackpressurePause:     10 * time.Second,
	})

	exit1 := errors.New("exit status 1")
	rateLimitEvidence := ChildExitEvidence{
		ExitCode: 1,
		RunErr:   exit1,
		UpstreamFailure: &UpstreamFailureReceipt{
			HTTPStatus: 429,
			Cause:      "rate_limit_exceeded",
		},
	}

	// Simulate 10 consecutive HTTP 429 rate limit exits
	for i := 1; i <= 10; i++ {
		decision := sup.HandleChildExit(1, exit1, rateLimitEvidence)

		if decision.State != StateProviderRateLimited {
			t.Fatalf("iteration %d: expected state %q, got %q", i, StateProviderRateLimited, decision.State)
		}
		if decision.Action != ActionBackpressurePause {
			t.Fatalf("iteration %d: expected action %q, got %q", i, ActionBackpressurePause, decision.Action)
		}
		if decision.BurnsBudget {
			t.Fatalf("iteration %d: BurnsBudget must be false", i)
		}
		if sup.CrashRestarts() != 0 {
			t.Fatalf("iteration %d: CrashRestarts must remain 0, got %d", i, sup.CrashRestarts())
		}
		if sup.ConsecutiveRateLimits() != i {
			t.Fatalf("iteration %d: expected ConsecutiveRateLimits=%d, got %d", i, i, sup.ConsecutiveRateLimits())
		}
		if decision.BackpressurePause < 1*time.Second || decision.BackpressurePause > 10*time.Second {
			t.Fatalf("iteration %d: BackpressurePause %v out of bounds [1s, 10s]", i, decision.BackpressurePause)
		}
	}

	// Verify that the crash restart budget was NEVER exhausted
	if sup.State() == StateCrashRestartExhausted {
		t.Fatal("supervisor erroneously entered CRASH_RESTART_EXHAUSTED after rate limit exits")
	}

	// Now simulate a genuine abnormal crash (panic / segfault / unhandled exit without 429)
	crashErr := errors.New("exit status 2")
	crashEvidence := ChildExitEvidence{ExitCode: 2, RunErr: crashErr}

	decision1 := sup.HandleChildExit(2, crashErr, crashEvidence)
	if decision1.State != StateCrashRestart {
		t.Fatalf("expected crash state %q, got %q", StateCrashRestart, decision1.State)
	}
	if !decision1.BurnsBudget {
		t.Fatal("genuine crash should burn budget")
	}
	if sup.CrashRestarts() != 1 {
		t.Fatalf("expected CrashRestarts to be 1 after first genuine crash, got %d", sup.CrashRestarts())
	}
	if decision1.RestartHop != 1 {
		t.Fatalf("expected RestartHop=1, got %d", decision1.RestartHop)
	}

	// Follow with 3 more rate-limit exits: CrashRestarts should stay at 1
	for i := 1; i <= 3; i++ {
		d := sup.HandleChildExit(1, exit1, rateLimitEvidence)
		if d.State != StateProviderRateLimited {
			t.Fatalf("post-crash rate-limit %d: expected PROVIDER_RATE_LIMITED, got %q", i, d.State)
		}
		if sup.CrashRestarts() != 1 {
			t.Fatalf("post-crash rate-limit %d: CrashRestarts changed from 1 to %d", i, sup.CrashRestarts())
		}
	}

	// Genuine crash 2
	decision2 := sup.HandleChildExit(2, crashErr, crashEvidence)
	if decision2.State != StateCrashRestart || sup.CrashRestarts() != 2 {
		t.Fatalf("expected crash 2: state=%s, count=%d", decision2.State, sup.CrashRestarts())
	}

	// Genuine crash 3
	decision3 := sup.HandleChildExit(2, crashErr, crashEvidence)
	if decision3.State != StateCrashRestart || sup.CrashRestarts() != 3 {
		t.Fatalf("expected crash 3: state=%s, count=%d", decision3.State, sup.CrashRestarts())
	}

	// Genuine crash 4: now budget is exhausted!
	decision4 := sup.HandleChildExit(2, crashErr, crashEvidence)
	if decision4.State != StateCrashRestartExhausted {
		t.Fatalf("expected %q on 4th crash, got %q", StateCrashRestartExhausted, decision4.State)
	}
	if decision4.Action != ActionExhausted {
		t.Fatalf("expected action %q, got %q", ActionExhausted, decision4.Action)
	}
	if decision4.Reason != ReasonCrashRestartExhausted {
		t.Fatalf("expected reason %q, got %q", ReasonCrashRestartExhausted, decision4.Reason)
	}
}

func TestAbnormalCrashWithoutRateLimitExhaustsBudget(t *testing.T) {
	sup := NewSupervisor(SupervisorConfig{CrashRestartLimit: 2})
	crashErr := errors.New("exit status 1")

	// Crash 1
	d1 := sup.HandleChildExit(1, crashErr)
	if d1.State != StateCrashRestart || !d1.BurnsBudget || sup.CrashRestarts() != 1 {
		t.Fatalf("crash 1 unexpected decision: %+v", d1)
	}

	// Crash 2
	d2 := sup.HandleChildExit(1, crashErr)
	if d2.State != StateCrashRestart || !d2.BurnsBudget || sup.CrashRestarts() != 2 {
		t.Fatalf("crash 2 unexpected decision: %+v", d2)
	}

	// Crash 3: exhausted
	d3 := sup.HandleChildExit(1, crashErr)
	if d3.State != StateCrashRestartExhausted || d3.Action != ActionExhausted {
		t.Fatalf("expected CRASH_RESTART_EXHAUSTED, got %+v", d3)
	}
}

func TestCleanExitDoesNotBurnBudget(t *testing.T) {
	sup := NewSupervisor(SupervisorConfig{CrashRestartLimit: 3})

	decision := sup.HandleChildExit(0, nil)
	if decision.State != StateCleanExit || decision.Action != ActionTerminate || decision.BurnsBudget {
		t.Fatalf("unexpected clean exit decision: %+v", decision)
	}
}

func TestDetectRateLimitPatterns(t *testing.T) {
	cases := []struct {
		name        string
		evidence    ChildExitEvidence
		wantRateLim bool
	}{
		{
			name: "structured upstream 429",
			evidence: ChildExitEvidence{
				UpstreamFailure: &UpstreamFailureReceipt{HTTPStatus: 429},
			},
			wantRateLim: true,
		},
		{
			name: "event upstream failure with status 429",
			evidence: ChildExitEvidence{
				Events: []SupervisorEvent{
					{Kind: EventUpstreamFailure, Status: 429},
				},
			},
			wantRateLim: true,
		},
		{
			name: "event upstream failure with json payload",
			evidence: ChildExitEvidence{
				Events: []SupervisorEvent{
					{Kind: EventUpstreamFailure, Payload: `{"http_status":429,"cause":"too many requests"}`},
				},
			},
			wantRateLim: true,
		},
		{
			name: "event account rotation provider_rate_limited",
			evidence: ChildExitEvidence{
				Events: []SupervisorEvent{
					{Kind: EventAccountRotation, Reason: "provider_rate_limited"},
				},
			},
			wantRateLim: true,
		},
		{
			name: "rotation reason string",
			evidence: ChildExitEvidence{
				RotationReason: "provider_rate_limited:delta=2",
			},
			wantRateLim: true,
		},
		{
			name: "raw evidence matching UPSTREAM_FAILURE 429",
			evidence: ChildExitEvidence{
				RawEvidence: "fak-turn trace=tr-1 UPSTREAM_FAILURE layer=provider status=429 target=anthropic",
			},
			wantRateLim: true,
		},
		{
			name: "negative: upstream 502 bad gateway",
			evidence: ChildExitEvidence{
				UpstreamFailure: &UpstreamFailureReceipt{HTTPStatus: 502, Cause: "bad gateway"},
				Events: []SupervisorEvent{
					{Kind: EventUpstreamFailure, Status: 502, Payload: `{"http_status":502}`},
				},
			},
			wantRateLim: false,
		},
		{
			name: "negative: upstream 500 internal error",
			evidence: ChildExitEvidence{
				UpstreamFailure: &UpstreamFailureReceipt{HTTPStatus: 500},
			},
			wantRateLim: false,
		},
		{
			name: "negative: account rotation auth_expired",
			evidence: ChildExitEvidence{
				AccountRotation: &AccountRotationEvent{Reason: "auth_expired"},
				RotationReason:  "auth_expired",
			},
			wantRateLim: false,
		},
		{
			name:        "negative: empty evidence",
			evidence:    ChildExitEvidence{},
			wantRateLim: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, desc := DetectRateLimit(tc.evidence)
			if got != tc.wantRateLim {
				t.Fatalf("DetectRateLimit() = %v (desc=%q), want %v", got, desc, tc.wantRateLim)
			}
		})
	}
}

func TestRetryAfterParsing(t *testing.T) {
	sup := NewSupervisor(SupervisorConfig{
		CrashRestartLimit:        3,
		InitialBackpressurePause: 1 * time.Second,
		MaxBackpressurePause:     30 * time.Second,
	})

	exitErr := errors.New("exit status 1")
	evidence := ChildExitEvidence{
		ExitCode: 1,
		RunErr:   exitErr,
		UpstreamFailure: &UpstreamFailureReceipt{
			HTTPStatus: 429,
			RetryAfter: "15",
		},
	}

	d := sup.HandleChildExit(1, exitErr, evidence)
	if d.BackpressurePause != 15*time.Second {
		t.Fatalf("expected BackpressurePause = 15s, got %v", d.BackpressurePause)
	}

	// Diagnostic header fallback
	evidence2 := ChildExitEvidence{
		ExitCode: 1,
		RunErr:   exitErr,
		UpstreamFailure: &UpstreamFailureReceipt{
			HTTPStatus: 429,
			Diagnostic: map[string]string{"Retry-After": "25s"},
		},
	}
	d2 := sup.HandleChildExit(1, exitErr, evidence2)
	if d2.BackpressurePause != 25*time.Second {
		t.Fatalf("expected BackpressurePause = 25s, got %v", d2.BackpressurePause)
	}
}

func TestClassifyChildExitPureFunction(t *testing.T) {
	exitErr := errors.New("exit status 1")
	rateLimitEvidence := ChildExitEvidence{
		ExitCode: 1,
		RunErr:   exitErr,
		UpstreamFailure: &UpstreamFailureReceipt{
			HTTPStatus: 429,
		},
	}

	d := ClassifyChildExit(1, exitErr, rateLimitEvidence, 2, 3)
	if d.State != StateProviderRateLimited || d.BurnsBudget || d.RestartHop != 2 {
		t.Fatalf("pure ClassifyChildExit failed for rate limit: %+v", d)
	}

	crashEvidence := ChildExitEvidence{ExitCode: 1, RunErr: exitErr}
	dCrash := ClassifyChildExit(1, exitErr, crashEvidence, 1, 3)
	if dCrash.State != StateCrashRestart || !dCrash.BurnsBudget || dCrash.RestartHop != 2 {
		t.Fatalf("pure ClassifyChildExit failed for crash: %+v", dCrash)
	}
}

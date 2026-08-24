package resume

import (
	"strings"
	"testing"
)

// TestTargetZeroRefusesEveryActingRecoveryPath is the #6505 regression witness: with the
// operator's declared population at 0, EVERY acting class — reset, resume, re-enable,
// restart, stranded recovery, refill — is refused with the quarantine token. Before the
// gate existed each of these paths carried its own target and none read the operator's.
func TestActingRecoveryActionsExcludeStatusRead(t *testing.T) {
	for _, action := range ActingRecoveryActions {
		if action == RecoveryStatusRead {
			t.Fatal("read-only status action appears in acting recovery vocabulary")
		}
	}
}

func TestTargetZeroRefusesEveryActingRecoveryPath(t *testing.T) {
	p := DeclaredFleetTarget(0, "dos loop --target", "operator quarantine 2026-08-11")
	for _, act := range ActingRecoveryActions {
		d := AdmitQuarantine(p, act)
		if d.Admit {
			t.Errorf("target 0: %s must be REFUSED, got Admit=true", act)
		}
		if d.Reason != ReasonFleetQuarantined {
			t.Errorf("target 0: %s reason = %q, want %q", act, d.Reason, ReasonFleetQuarantined)
		}
		// the summary must name the SSOT and the operator's reason so a paged human knows
		// what to change rather than re-deriving it.
		for _, want := range []string{"dos loop --target", "operator quarantine 2026-08-11", string(act)} {
			if !strings.Contains(d.Summary, want) {
				t.Errorf("target 0: %s summary %q missing %q", act, d.Summary, want)
			}
		}
	}
}

// TestUnknownFleetTargetFailsClosed: an undeclared or unreadable population refuses the
// acting classes too. A caller that never folded the SSOT (the literal bypass) and one
// whose read broke are both denied, under a reason distinct from a real quarantine.
func TestUnknownFleetTargetFailsClosed(t *testing.T) {
	for _, p := range []FleetPosture{
		{}, // undeclared — the zero value a forgetful caller supplies
		UnreadableFleetTarget("control-pane config", "open: no such file"),
	} {
		for _, act := range ActingRecoveryActions {
			d := AdmitQuarantine(p, act)
			if d.Admit {
				t.Errorf("posture %+v: %s must fail closed, got Admit=true", p, act)
			}
			if d.Reason != ReasonFleetTargetUnknown {
				t.Errorf("posture %+v: %s reason = %q, want %q", p, act, d.Reason, ReasonFleetTargetUnknown)
			}
		}
		if p.Quarantined() {
			t.Errorf("posture %+v: unknown must not report Quarantined (it is not proof of a hold)", p)
		}
	}
}

// TestOnlyPositiveDeclaredTargetAdmits is the safety property stated as an exhaustive
// sweep: across every state × population × acting class, admission happens ONLY when the
// population was explicitly declared AND is positive.
func TestOnlyPositiveDeclaredTargetAdmits(t *testing.T) {
	states := []FleetTargetState{FleetTargetUndeclared, FleetTargetDeclared, FleetTargetUnreadable}
	for _, st := range states {
		for _, n := range []int{-1, 0, 1, 4} {
			p := FleetPosture{State: st, DesiredWorkers: n, Source: "control-pane config target"}
			want := st == FleetTargetDeclared && n > 0
			for _, act := range ActingRecoveryActions {
				if got := AdmitQuarantine(p, act).Admit; got != want {
					t.Errorf("state=%q workers=%d act=%s: Admit=%v, want %v", st, n, act, got, want)
				}
			}
		}
	}
}

// TestStatusReadIsNeverHeld: the read-only class stays allowed under every posture — an
// operator needs status MOST while quarantined, and reporting starts no work.
func TestStatusReadIsNeverHeld(t *testing.T) {
	for _, p := range []FleetPosture{
		{},
		DeclaredFleetTarget(0, "dos loop --target", "quarantine"),
		UnreadableFleetTarget("control-pane config", "malformed json"),
		DeclaredFleetTarget(4, "control-pane config target", ""),
	} {
		d := AdmitQuarantine(p, RecoveryStatusRead)
		if !d.Admit || d.Reason != ReasonQuarantineAdmitted {
			t.Errorf("posture %+v: status_read must always be admitted, got %+v", p, d)
		}
	}
}

// TestUnknownRecoveryActionActs: an unrecognized actuator token is treated as acting, so
// a bypass invented later fails closed instead of slipping through the read-only door.
func TestUnknownRecoveryActionActs(t *testing.T) {
	novel := RecoveryAction("some_future_refill_path")
	if !novel.Acts() {
		t.Fatalf("unknown action must count as acting")
	}
	if AdmitQuarantine(DeclaredFleetTarget(0, "dos loop --target", "quarantine"), novel).Admit {
		t.Fatalf("target 0 must refuse an unknown acting class")
	}
}

// TestRecoveryActorsEnumerateAuditedAutomation covers the "status enumerates all
// automation capable of changing dispatch task enablement" half: the audited inventory
// includes the unmanaged supervisor and the manual refill, not just Scheduled Tasks, and
// every enumerated acting actor is refused by the gate under target 0.
func TestRecoveryActorsEnumerateAuditedAutomation(t *testing.T) {
	actors := RecoveryActors()
	byName := map[string]RecoveryActor{}
	kinds := map[RecoveryActorKind]bool{}
	for _, a := range actors {
		if a.Name == "" || a.Action == "" || a.Kind == "" {
			t.Errorf("actor %+v: name, kind and action are all required", a)
		}
		byName[a.Name] = a
		kinds[a.Kind] = true
	}
	for _, want := range []string{
		"FleetOwnerSeatResume", "FleetResumeWatchdog", "FleetStrandedRecovery",
		"isolated-nemotron-supervisor", "operator manual refill",
	} {
		if _, ok := byName[want]; !ok {
			t.Errorf("audited actor %q missing from the enumeration", want)
		}
	}
	// a Scheduled-Task-only inventory is what let the unmanaged supervisor and the manual
	// refill run unseen; all three kinds must be represented.
	for _, k := range []RecoveryActorKind{ActorScheduledTask, ActorUnmanagedProcess, ActorOperatorSession} {
		if !kinds[k] {
			t.Errorf("enumeration covers no %s actor", k)
		}
	}
	quarantined := DeclaredFleetTarget(0, "dos loop --target", "quarantine")
	for _, a := range actors {
		if !a.Action.Acts() {
			continue
		}
		if AdmitQuarantine(quarantined, a.Action).Admit {
			t.Errorf("actor %q (%s) is admitted under target 0", a.Name, a.Action)
		}
	}
	// the ungated view must be a strict, honest subset: the resume watchdog is gated, the
	// rest are the standing exposure a status page has to show.
	ungated := UngatedRecoveryActors()
	if len(ungated) == 0 || len(ungated) >= len(actors) {
		t.Fatalf("ungated actors = %d of %d; want a strict non-empty subset", len(ungated), len(actors))
	}
	for _, a := range ungated {
		if a.Gated {
			t.Errorf("actor %q reported as ungated but Gated=true", a.Name)
		}
		if a.Name == "FleetResumeWatchdog" {
			t.Errorf("FleetResumeWatchdog routes through AdmitQuarantine; it must not be listed ungated")
		}
	}
}

// TestRecoveryActorsIsACopy: the registry is not caller-mutable, so a status renderer
// cannot silently edit the audited inventory it is meant to report.
func TestRecoveryActorsIsACopy(t *testing.T) {
	first := RecoveryActors()
	if len(first) == 0 {
		t.Fatalf("empty actor registry")
	}
	first[0].Name = "mutated"
	if RecoveryActors()[0].Name == "mutated" {
		t.Fatalf("RecoveryActors returned the backing registry, not a copy")
	}
}

// TestAdmitReEnableRequiresCanaryAndResolvedGates covers the "re-enable requires a
// witnessed canary and resolved launcher/status gates" half, including the fixed order in
// which the refusals are reported.
func TestAdmitReEnableRequiresCanaryAndResolvedGates(t *testing.T) {
	full := ReEnableRequest{
		DesiredWorkers: 4, CanaryWitnessed: true,
		LauncherGateResolved: true, StatusGateResolved: true,
	}
	if d := AdmitReEnable(full); !d.Admit || d.Reason != ReasonReEnableAdmitted {
		t.Fatalf("fully-evidenced re-enable must be admitted, got %+v", d)
	}
	for _, tc := range []struct {
		name string
		req  ReEnableRequest
		want string
	}{
		{"zero value admits nothing", ReEnableRequest{}, ReasonTargetNotRaised},
		{"target not raised", func() ReEnableRequest { r := full; r.DesiredWorkers = 0; return r }(), ReasonTargetNotRaised},
		{"canary unwitnessed", func() ReEnableRequest { r := full; r.CanaryWitnessed = false; return r }(), ReasonCanaryUnwitnessed},
		{"launcher gate open", func() ReEnableRequest { r := full; r.LauncherGateResolved = false; return r }(), ReasonLauncherGateOpen},
		{"status gate open", func() ReEnableRequest { r := full; r.StatusGateResolved = false; return r }(), ReasonStatusGateOpen},
	} {
		d := AdmitReEnable(tc.req)
		if d.Admit {
			t.Errorf("%s: must be refused, got Admit=true", tc.name)
		}
		if d.Reason != tc.want {
			t.Errorf("%s: reason = %q, want %q", tc.name, d.Reason, tc.want)
		}
		if strings.TrimSpace(d.Summary) == "" {
			t.Errorf("%s: refusal must carry an actionable summary", tc.name)
		}
	}
}

// TestWatchdogQuarantineGuardOutranksEveryRowFact is the resume-side regression: a row
// that would otherwise LAUNCH under every existing guard — including the reset/recovery
// paths (a re-death revive past the burn-once latch, and a preserved partial turn) — is
// held by a declared target of 0, and by an unreadable target too.
func TestWatchdogQuarantineGuardOutranksEveryRowFact(t *testing.T) {
	row := WatchdogPlanRow{Session: "sid-q", Account: ".claude-x"}
	// baseline: with no posture declared the tick behaves exactly as before.
	if d := DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("undeclared posture must stay inert, got %+v", d)
	}
	// a declared, positive target also leaves the guard inert.
	running := WatchdogGuards{Fleet: DeclaredFleetTarget(4, "control-pane config target", "")}
	if d := DecideWatchdogRow(row, running, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("declared target 4 must admit a resume, got %+v", d)
	}

	// the reset/recovery shapes that beat the lower guards: a revived re-death and a
	// replay-safe preserved partial turn.
	revived, released := ReviveOutcome(OutcomeProgressed, ReDeathEvidence{
		ProcessScanOK: true, TranscriptIdleSeconds: DeadTranscriptIdleFloorSeconds, PostLaunchProgress: true,
	})
	if !released {
		t.Fatalf("setup: re-death evidence must release the burn-once latch")
	}
	// a partial turn whose tool call carries a matching result is preserve-and-continued:
	// on a running fleet it LAUNCHES with its own distinct reason.
	partial := row
	partial.PartialBlocks = []EmittedBlock{
		{Kind: BlockToolCall, ToolCallID: "call-1"},
		{Kind: BlockToolResult, ToolCallID: "call-1"},
	}
	if d := DecideWatchdogRow(partial, running, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("setup: preserved partial turn must launch on a running fleet, got %+v", d)
	}

	for _, held := range []FleetPosture{
		DeclaredFleetTarget(0, "dos loop --target", "operator quarantine"),
		UnreadableFleetTarget("control-pane config", "malformed json"),
	} {
		g := WatchdogGuards{Fleet: held}
		for _, tc := range []struct {
			name    string
			row     WatchdogPlanRow
			hist    []Attempt
			outcome Outcome
		}{
			{"fresh recoverable", row, nil, OutcomeRecoverable},
			{"revived re-death", row, []Attempt{{UnixSeconds: 100, Phase: "launched"}}, revived},
			{"preserved partial turn", partial, nil, OutcomeRecoverable},
		} {
			d := DecideWatchdogRow(tc.row, g, tc.hist, tc.outcome)
			if d.Action != WatchdogSkipQuarantine {
				t.Errorf("posture %+v / %s: action = %q, want %q", held, tc.name, d.Action, WatchdogSkipQuarantine)
			}
			if d.Attempt != 0 {
				t.Errorf("posture %+v / %s: a held row must not reserve an attempt, got %d", held, tc.name, d.Attempt)
			}
			if strings.TrimSpace(d.Reason) == "" {
				t.Errorf("posture %+v / %s: hold must carry an actionable reason", held, tc.name)
			}
		}
	}
}

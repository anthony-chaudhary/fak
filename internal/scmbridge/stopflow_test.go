package scmbridge

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicelease"
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

var labID = servicespec.Identity{Node: "lab-1", Service: "fak-guard", Workload: "FakGuardControl"}

func TestClassifySeparatesLogoffAndOperatorStopFromCrashes(t *testing.T) {
	cases := []struct {
		name  string
		in    StopReport
		cause StopCause
		class servicespec.ExitClass
	}{
		{"dirty boot 6008", StopReport{Provider: "EventLog", NativeEventID: 6008}, CauseHostReboot, servicespec.ExitBootRecovery},
		{"kernel-power 41", StopReport{Provider: "Microsoft-Windows-Kernel-Power", NativeEventID: 41}, CauseHostReboot, servicespec.ExitBootRecovery},
		{"initiated shutdown 1074", StopReport{NativeEventID: 1074}, CauseHostReboot, servicespec.ExitBootRecovery},
		{"logoff flag", StopReport{UserLogoff: true, Session: "1/console"}, CauseUserLogoff, servicespec.ExitOperatorStop},
		{"logoff 4647", StopReport{NativeEventID: 4647}, CauseUserLogoff, servicespec.ExitOperatorStop},
		{"operator verb", StopReport{OperatorVerb: true}, CauseOperatorStop, servicespec.ExitOperatorStop},
		{"scm 7034 crash", StopReport{Provider: "Service Control Manager", NativeEventID: 7034}, CauseCrash, servicespec.ExitCrash},
		{"nonzero exit", StopReport{ExitCode: 1067}, CauseCrash, servicespec.ExitCrash},
		{"clean", StopReport{}, CauseCleanExit, servicespec.ExitClean},
		// Priority: a logoff seen during a crashy-looking exit is still a
		// logoff — the session ended, the code did not fail.
		{"logoff beats exit code", StopReport{UserLogoff: true, ExitCode: 1}, CauseUserLogoff, servicespec.ExitOperatorStop},
		// Host evidence beats everything.
		{"reboot beats operator", StopReport{DirtyBoot: true, OperatorVerb: true}, CauseHostReboot, servicespec.ExitBootRecovery},
	}
	for _, c := range cases {
		cause, class := Classify(c.in)
		if cause != c.cause || class != c.class {
			t.Errorf("%s: = (%s, %s), want (%s, %s)", c.name, cause, class, c.cause, c.class)
		}
	}
}

func TestRuleResumeBrokerAwaitsLogonAndMachineSurfacesMisplacement(t *testing.T) {
	p := servicespec.RestartPolicy{}
	(&servicespec.Spec{Restart: p}).Normalize()
	r := RuleResume(RoleBroker, CauseUserLogoff, p, servicespec.RestartInput{})
	if r.Restart || !r.AwaitLogon || r.Reason != ReasonAwaitLogon {
		t.Fatalf("broker logoff = %+v", r)
	}
	r = RuleResume(RoleMachine, CauseUserLogoff, p, servicespec.RestartInput{})
	if r.Restart || r.AwaitLogon || r.Reason != ReasonMisplacedLogoff {
		t.Fatalf("machine logoff = %+v", r)
	}
}

func TestRuleResumeDelegatesToTheOneRestartContract(t *testing.T) {
	var p servicespec.RestartPolicy
	s := &servicespec.Spec{Restart: p}
	s.Normalize()
	p = s.Restart
	in := servicespec.RestartInput{Kind: servicespec.KindService, Desired: servicespec.DesiredRunning, Attempt: 1}
	r := RuleResume(RoleMachine, CauseCrash, p, in)
	if !r.Restart || r.DelayMS != 2000 || r.Reason != servicespec.ReasonRestart {
		t.Fatalf("crash resume = %+v", r)
	}
	r = RuleResume(RoleMachine, CauseOperatorStop, p, in)
	if r.Restart || r.Reason != servicespec.ReasonOperatorStop {
		t.Fatalf("operator resume = %+v", r)
	}
	r = RuleResume(RoleMachine, CauseHostReboot, p, in)
	if !r.Restart || r.DelayMS != 0 || r.Reason != servicespec.ReasonBootRecovery {
		t.Fatalf("boot resume = %+v", r)
	}
	// Circuit-open surfaces as Fenced.
	in.WindowCount = p.WindowMaxRestarts
	r = RuleResume(RoleMachine, CauseCrash, p, in)
	if r.Restart || !r.Fenced || r.Reason != servicespec.ReasonCircuitOpen {
		t.Fatalf("fenced resume = %+v", r)
	}
}

func TestEmissionCarriesEveryContractIdentifier(t *testing.T) {
	led := serviceledger.Memory()
	g := Grant{
		Workload: "FakGuardControl", Role: RoleMachine,
		Token:     servicelease.FencingToken{Generation: 3, LeaseSeq: 7},
		RequestID: "launch-FakGuardControl-g3-s7", ExpiresMS: 9000,
	}
	launch := LaunchEvent(labID, g, 4242, 1000)
	if _, ok, err := led.Append(launch); err != nil || !ok {
		t.Fatalf("launch append: %v %v", ok, err)
	}
	stop, cause := StopEvent(labID, StopReport{
		Provider: "Service Control Manager", NativeEventID: 7034,
		ExitCode: 1067, TaskInstance: "inst-9", Session: "1/console",
	}, 2000)
	if cause != CauseCrash {
		t.Fatalf("cause = %s", cause)
	}
	if _, ok, err := led.Append(stop); err != nil || !ok {
		t.Fatalf("stop append: %v %v", ok, err)
	}
	resume := ResumeEvent(labID, ReceiptFor(g.RequestID), "2/console", 3000)
	if _, ok, err := led.Append(resume); err != nil || !ok {
		t.Fatalf("resume append: %v %v", ok, err)
	}
	evs := led.Events()
	if evs[0].Correlation.Request != g.RequestID || evs[0].Correlation.Generation != 3 || evs[0].Correlation.LeaseTokenHash == "" {
		t.Fatalf("launch row = %+v", evs[0])
	}
	if evs[1].Exit == nil || evs[1].Exit.Code != 1067 || evs[1].Correlation.ManagerInvocation != "inst-9" || evs[1].Correlation.Session != "1/console" {
		t.Fatalf("stop row = %+v", evs[1])
	}
	if !strings.Contains(evs[1].Detail, "native event 7034") {
		t.Fatalf("stop detail = %q", evs[1].Detail)
	}
	if evs[2].Correlation.Receipt != "receipt-FakGuardControl-g3-s7" || evs[2].Correlation.Session != "2/console" {
		t.Fatalf("resume row = %+v", evs[2])
	}
	// The refusal row is a lease-fence event.
	ref := RefusalEvent(labID, "FakGuardControl", RoleWatchdog, RoleMachine, 4000)
	if _, ok, err := led.Append(ref); err != nil || !ok {
		t.Fatalf("refusal append: %v %v", ok, err)
	}
	// The human timeline renders the resumed session identity.
	var sb strings.Builder
	serviceledger.WriteTimeline(&sb, led.Events())
	if !strings.Contains(sb.String(), "session=2/console") {
		t.Fatalf("timeline missing session identity:\n%s", sb.String())
	}
}

func nativeCrash(at int64) serviceledger.Event {
	return serviceledger.Event{
		Type: serviceledger.EventProcessExit, AtUnixMS: at,
		Source: serviceledger.SourceWindowsEventLog, SourceUID: "System/1",
		Identity: labID,
		Exit:     &servicespec.ExitRecord{Class: servicespec.ExitCrash, AtUnixMS: at},
	}
}

func TestJudgeRequiresIndependentCorroborationAndResume(t *testing.T) {
	selfStop, _ := StopEvent(labID, StopReport{ExitCode: 1}, 100)
	relaunch := serviceledger.Event{
		Type: serviceledger.EventManagerRestart, AtUnixMS: 200,
		Source: serviceledger.SourceWindowsEventLog, SourceUID: "TS/100",
		Identity: labID,
	}
	// Self-reported stop alone is NOT corroboration.
	v := Judge(ProbeSCMKill, []serviceledger.Event{selfStop, relaunch})
	if v.Corroborated || v.Passed() {
		t.Fatalf("self-report corroborated: %+v", v)
	}
	// Native crash + native relaunch passes.
	v = Judge(ProbeSCMKill, []serviceledger.Event{nativeCrash(100), relaunch})
	if !v.Passed() {
		t.Fatalf("scm kill verdict = %+v", v)
	}
	// Native crash without any resume is held open.
	v = Judge(ProbeTerminalKill, []serviceledger.Event{nativeCrash(100)})
	if !v.Corroborated || v.Resumed || len(v.Missing) != 1 || v.Missing[0] != MissingResumeEvidence {
		t.Fatalf("no-resume verdict = %+v", v)
	}
}

func TestJudgeHostRebootNeedsNativeBootChange(t *testing.T) {
	boot := serviceledger.Event{
		Type: serviceledger.EventBootChange, AtUnixMS: 100,
		Source: serviceledger.SourceWindowsEventLog, SourceUID: "System/2",
		Identity: labID, Correlation: serviceledger.Correlation{BootID: "boot-b"},
	}
	resume := ResumeEvent(labID, "receipt-x", "1/console", 200)
	if v := Judge(ProbeHostReboot, []serviceledger.Event{boot, resume}); !v.Passed() {
		t.Fatalf("reboot verdict = %+v", v)
	}
	if v := Judge(ProbeHostReboot, []serviceledger.Event{resume}); v.Corroborated {
		t.Fatalf("no boot evidence: %+v", v)
	}
}

func TestJudgeTermSvcResetDemandsResumedSessionIdentity(t *testing.T) {
	stop := nativeCrash(100)
	bare := ResumeEvent(labID, "receipt-x", "", 200)
	v := Judge(ProbeTermSvcReset, []serviceledger.Event{stop, bare})
	if v.Passed() || len(v.Missing) != 1 || v.Missing[0] != MissingResumedIdentity {
		t.Fatalf("bare resume verdict = %+v", v)
	}
	with := ResumeEvent(labID, "receipt-x", "3/rdp", 300)
	if v := Judge(ProbeTermSvcReset, []serviceledger.Event{stop, with}); !v.Passed() {
		t.Fatalf("identity resume verdict = %+v", v)
	}
}

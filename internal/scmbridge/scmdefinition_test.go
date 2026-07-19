package scmbridge

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// defSpec is a rich machine-service spec exercising every projection axis:
// dependencies, literal + secret env, notify readiness, default restart policy.
func defSpec() *servicespec.Spec {
	return &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "lab-1", Service: "guard"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{`C:\Program Files\fak\fak.exe`, "service", "windows-run"},
		Dir:      `C:\ProgramData\fak`,
		Env: []servicespec.EnvRef{
			{Name: "FAK_MODE", Value: "lab"},
			{Name: "FAK_TOKEN", SecretRef: "vault://fak/token"},
		},
		Readiness: &servicespec.Readiness{Kind: ReadinessNotify, Target: "pipe://fak-guard", TimeoutMS: 30000},
		DependsOn: []string{"fak-gateway"},
	}
}

func defInput() DefinitionInput {
	return DefinitionInput{DelayedStart: true, WatchdogMS: 10000, EnvFile: `C:\ProgramData\fak\guard.env`}
}

func TestProjectDefinitionGolden(t *testing.T) {
	d, err := ProjectDefinition(defSpec(), defInput())
	if err != nil {
		t.Fatalf("ProjectDefinition: %v", err)
	}
	if d.Schema != SCMDefinitionSchemaV1 {
		t.Fatalf("schema = %q", d.Schema)
	}
	if d.ServiceName != "guard" || d.DisplayName != "fak workload guard" {
		t.Fatalf("name = %q / %q", d.ServiceName, d.DisplayName)
	}
	if d.Account != MachinePrincipal {
		t.Fatalf("account = %q, want least-privilege %q", d.Account, MachinePrincipal)
	}
	if d.StartType != StartTypeDelayedAuto || d.DesiredStopped {
		t.Fatalf("start = %q desired_stopped=%v", d.StartType, d.DesiredStopped)
	}
	if d.BinPath != `"C:\Program Files\fak\fak.exe" service windows-run` {
		t.Fatalf("bin_path = %q", d.BinPath)
	}
	if !reflect.DeepEqual(d.Dependencies, []string{"fak-gateway"}) {
		t.Fatalf("dependencies = %v", d.Dependencies)
	}
	if !reflect.DeepEqual(d.Environment, []string{"FAK_MODE=lab"}) {
		t.Fatalf("environment = %v (a secret_ref must never serialize)", d.Environment)
	}
	if strings.Contains(d.ConfigText, "vault://fak/token") || strings.Contains(d.ConfigText, "FAK_TOKEN") {
		t.Fatal("secret reference leaked into rendered config text")
	}
	if d.WatchdogMS != 10000 || d.KeepaliveMS != 5000 {
		t.Fatalf("watchdog = %d keepalive = %d", d.WatchdogMS, d.KeepaliveMS)
	}
	if d.ReadinessKind != ReadinessNotify || d.ReadinessTarget != "pipe://fak-guard" || d.StartTimeoutMS != 30000 {
		t.Fatalf("readiness = %q %q %d", d.ReadinessKind, d.ReadinessTarget, d.StartTimeoutMS)
	}
	wantFailure := FailureActions{
		ResetPeriodSec: 300,
		Actions: []FailureAction{
			{Kind: FailureRestart, DelayMS: 1000},
			{Kind: FailureRestart, DelayMS: 2000},
			{Kind: FailureRestart, DelayMS: 4000},
			{Kind: FailureRestart, DelayMS: 8000},
			{Kind: FailureRestart, DelayMS: 16000},
			{Kind: FailureNone},
		},
		OnNonCrashFailures: true,
	}
	if !reflect.DeepEqual(d.Failure, wantFailure) {
		t.Fatalf("failure = %+v\nwant %+v", d.Failure, wantFailure)
	}
	golden := strings.Join([]string{
		`sc.exe create guard binPath= "\"C:\Program Files\fak\fak.exe\" service windows-run" start= delayed-auto obj= "NT AUTHORITY\LocalService" DisplayName= "fak workload guard"`,
		`sc.exe config guard depend= fak-gateway`,
		`sc.exe description guard "fak workload guard (service guard) on node lab-1, projected from fak.service.v1"`,
		`sc.exe failure guard reset= 300 actions= restart/1000/restart/2000/restart/4000/restart/8000/restart/16000/""/0`,
		`sc.exe failureflag guard 1`,
	}, "\n") + "\n"
	if d.ConfigText != golden {
		t.Fatalf("config text:\n%s\nwant:\n%s", d.ConfigText, golden)
	}
}

func TestProjectDefinitionDeterministic(t *testing.T) {
	a, err := ProjectDefinition(defSpec(), defInput())
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	b, err := ProjectDefinition(defSpec(), defInput())
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("two projections of the same spec differ")
	}
	ja, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	jb, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(ja) != string(jb) {
		t.Fatal("wire forms differ")
	}
}

func TestProjectDefinitionDesiredStoppedStaysInstalledDisabled(t *testing.T) {
	s := defSpec()
	s.Desired = servicespec.DesiredStopped
	d, err := ProjectDefinition(s, defInput())
	if err != nil {
		t.Fatalf("ProjectDefinition: %v", err)
	}
	if !d.DesiredStopped || d.StartType != StartTypeDisabled {
		t.Fatalf("desired_stopped=%v start=%q, want installed-but-disabled", d.DesiredStopped, d.StartType)
	}
	if !strings.Contains(d.ConfigText, "start= disabled") {
		t.Fatalf("config text does not disable start:\n%s", d.ConfigText)
	}
	for _, verb := range []string{"sc.exe delete", "sc.exe start"} {
		if strings.Contains(d.ConfigText, verb) {
			t.Fatalf("desired-stopped rendered %q — an intentional stop is preserved, not deleted or started", verb)
		}
	}
}

func TestFailureLadderProjectsCircuitAsTerminalNone(t *testing.T) {
	s := defSpec()
	s.Restart.WindowMaxRestarts = 2
	d, err := ProjectDefinition(s, defInput())
	if err != nil {
		t.Fatalf("ProjectDefinition: %v", err)
	}
	want := []FailureAction{
		{Kind: FailureRestart, DelayMS: 1000},
		{Kind: FailureRestart, DelayMS: 2000},
		{Kind: FailureNone},
	}
	if !reflect.DeepEqual(d.Failure.Actions, want) {
		t.Fatalf("actions = %+v, want circuit-open as terminal none %+v", d.Failure.Actions, want)
	}

	// An effectively uncapped window truncates at maxFailureLadder with a
	// max-backoff restart last — SCM repeats the last action, which matches
	// the uncapped policy's steady state.
	s = defSpec()
	s.Restart.WindowMaxRestarts = 100
	d, err = ProjectDefinition(s, defInput())
	if err != nil {
		t.Fatalf("ProjectDefinition: %v", err)
	}
	if len(d.Failure.Actions) != maxFailureLadder {
		t.Fatalf("ladder length = %d, want truncation at %d", len(d.Failure.Actions), maxFailureLadder)
	}
	last := d.Failure.Actions[len(d.Failure.Actions)-1]
	if last.Kind != FailureRestart || last.DelayMS != servicespec.DefaultMaxBackoffMS {
		t.Fatalf("last rung = %+v, want max-backoff restart", last)
	}
}

func TestProjectDefinitionRefusals(t *testing.T) {
	cases := []struct {
		name string
		spec func() *servicespec.Spec
		in   func() DefinitionInput
		want error
	}{
		{"job never becomes an SCM service", func() *servicespec.Spec {
			s := defSpec()
			s.Kind = servicespec.KindJob
			return s
		}, defInput, ErrJobOnSCM},
		{"watchdog without a notify channel", func() *servicespec.Spec {
			s := defSpec()
			s.Readiness = &servicespec.Readiness{Kind: "http", Target: "http://127.0.0.1:9/healthz"}
			return s
		}, defInput, ErrWatchdogNeedsNotify},
		{"secret without an env-file reference", defSpec, func() DefinitionInput {
			in := defInput()
			in.EnvFile = ""
			return in
		}, ErrSecretNeedsEnvFile},
		{"backslash in dependency name", func() *servicespec.Spec {
			s := defSpec()
			s.DependsOn = []string{`bad\name`}
			return s
		}, defInput, ErrBadServiceName},
		{"space in workload name", func() *servicespec.Spec {
			s := defSpec()
			s.Identity.Workload = "a b"
			return s
		}, defInput, ErrBadServiceName},
		{"double quote in command argument", func() *servicespec.Spec {
			s := defSpec()
			s.Command = append(s.Command, `--label="x"`)
			return s
		}, defInput, ErrQuoteInArgument},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ProjectDefinition(tc.spec(), tc.in()); !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}

	if _, err := ProjectDefinition(nil, defInput()); err == nil {
		t.Fatal("nil spec accepted")
	}
	s := defSpec()
	s.Dir = "C:\\bad\ndir"
	if _, err := ProjectDefinition(s, defInput()); err == nil {
		t.Fatal("control character in dir accepted")
	}
}

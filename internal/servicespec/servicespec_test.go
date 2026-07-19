package servicespec

import (
	"bytes"
	"strings"
	"testing"
)

// fullFixture is the v1 compatibility fixture: every field of the contract
// populated. Parse must accept it today and every future v1 reader must keep
// accepting exactly this shape.
const fullFixture = `{
  "schema": "fak.service.v1",
  "identity": {"node": "node-a", "service": "gateway", "workload": "gateway-1"},
  "kind": "service",
  "desired": "desired-running",
  "command": ["fak", "serve", "--addr", ":8080"],
  "dir": "work/gateway",
  "env": [
    {"name": "FAK_TOKEN", "secret_ref": "keychain://fak/gateway-token"},
    {"name": "FAK_MODE", "value": "prod"}
  ],
  "readiness": {"kind": "tcp-listen", "target": ":8080", "timeout_ms": 5000},
  "restart": {
    "initial_backoff_ms": 500,
    "max_backoff_ms": 30000,
    "backoff_factor": 2,
    "window_ms": 600000,
    "window_max_restarts": 5,
    "stable_run_reset_ms": 300000
  },
  "checkpoint_dir": "state/gateway",
  "depends_on": ["blobstore", "auth"]
}`

func mustParse(t *testing.T, doc string) *Spec {
	t.Helper()
	s, err := ParseSpec([]byte(doc))
	if err != nil {
		t.Fatalf("ParseSpec: %v", err)
	}
	return s
}

func TestParseFullCompatibilityFixture(t *testing.T) {
	s := mustParse(t, fullFixture)
	if s.Schema != SchemaV1 {
		t.Fatalf("schema = %q, want %q", s.Schema, SchemaV1)
	}
	if s.Identity.Node != "node-a" || s.Identity.Service != "gateway" || s.Identity.Workload != "gateway-1" {
		t.Fatalf("identity = %+v", s.Identity)
	}
	if s.Kind != KindService || s.Desired != DesiredRunning {
		t.Fatalf("kind=%q desired=%q", s.Kind, s.Desired)
	}
	if len(s.Env) != 2 || s.Env[0].Name != "FAK_MODE" || s.Env[1].SecretRef == "" {
		t.Fatalf("env not normalized/sorted: %+v", s.Env)
	}
	// depends_on is sorted by Normalize for deterministic output.
	if s.DependsOn[0] != "auth" || s.DependsOn[1] != "blobstore" {
		t.Fatalf("depends_on not sorted: %v", s.DependsOn)
	}
}

func TestCanonicalJSONIsDeterministicFixedPoint(t *testing.T) {
	s := mustParse(t, fullFixture)
	a, err := s.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	b, err := s.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON(2): %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("canonical bytes differ across calls")
	}
	// Fixed point: parsing the canonical form and re-canonicalizing is identity.
	s2, err := ParseSpec(a)
	if err != nil {
		t.Fatalf("re-parse canonical: %v", err)
	}
	c, err := s2.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON(reparsed): %v", err)
	}
	if !bytes.Equal(a, c) {
		t.Fatalf("canonical form is not a fixed point:\n%s\nvs\n%s", a, c)
	}
}

func TestSecretsAreReferencedNeverSerialized(t *testing.T) {
	s := mustParse(t, fullFixture)
	out, err := s.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	if !bytes.Contains(out, []byte("keychain://fak/gateway-token")) {
		t.Fatalf("secret REFERENCE must round-trip")
	}
	// An EnvRef may carry a value or a secret reference, never both: a spec can
	// never serialize a resolved secret next to its reference.
	bad := `{"name": "X", "value": "hunter2", "secret_ref": "keychain://x"}`
	doc := strings.Replace(fullFixture, `{"name": "FAK_MODE", "value": "prod"}`, bad, 1)
	if _, err := ParseSpec([]byte(doc)); err == nil || !strings.Contains(err.Error(), "value and secret_ref") {
		t.Fatalf("want value-and-secret_ref refusal, got %v", err)
	}
}

func TestInvalidFixturesAreRefused(t *testing.T) {
	cases := []struct {
		name string
		doc  string
		want string // substring of the error
	}{
		{"empty", `{}`, "schema"},
		{"wrong-schema", `{"schema": "fak.service.v0"}`, "schema"},
		{"unknown-field", strings.Replace(fullFixture, `"dir"`, `"cwd"`, 1), "unknown field"},
		{"no-service-identity", strings.Replace(fullFixture, `"service": "gateway", `, ``, 1), "identity.service"},
		{"no-node-identity", strings.Replace(fullFixture, `"node": "node-a", `, ``, 1), "identity.node"},
		{"bad-kind", strings.Replace(fullFixture, `"kind": "service"`, `"kind": "daemon"`, 1), "kind"},
		{"bad-desired", strings.Replace(fullFixture, `"desired": "desired-running"`, `"desired": "paused"`, 1), "desired"},
		{"no-command", strings.Replace(fullFixture, `["fak", "serve", "--addr", ":8080"]`, `[]`, 1), "command"},
		{"env-no-name", strings.Replace(fullFixture, `{"name": "FAK_MODE", "value": "prod"}`, `{"value": "prod"}`, 1), "env"},
		{"negative-backoff", strings.Replace(fullFixture, `"initial_backoff_ms": 500`, `"initial_backoff_ms": -1`, 1), "initial_backoff_ms"},
		{"backoff-inversion", strings.Replace(fullFixture, `"max_backoff_ms": 30000`, `"max_backoff_ms": 100`, 1), "max_backoff_ms"},
		{"self-dependency", strings.Replace(fullFixture, `["blobstore", "auth"]`, `["gateway"]`, 1), "depends_on"},
		{"not-json", `{`, "parse"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseSpec([]byte(tc.doc))
			if err == nil {
				t.Fatalf("ParseSpec accepted invalid fixture")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not name %q", err, tc.want)
			}
		})
	}
}

func TestNormalizeFillsPortableDefaults(t *testing.T) {
	min := `{
	  "schema": "fak.service.v1",
	  "identity": {"node": "n", "service": "s"},
	  "desired": "desired-running",
	  "command": ["run"]
	}`
	s := mustParse(t, min)
	if s.Kind != KindService {
		t.Fatalf("default kind = %q, want %q", s.Kind, KindService)
	}
	p := s.Restart
	if p.InitialBackoffMS != DefaultInitialBackoffMS || p.MaxBackoffMS != DefaultMaxBackoffMS ||
		p.BackoffFactor != DefaultBackoffFactor || p.WindowMS != DefaultWindowMS ||
		p.WindowMaxRestarts != DefaultWindowMaxRestarts || p.StableRunResetMS != DefaultStableRunResetMS {
		t.Fatalf("restart defaults not filled: %+v", p)
	}
}

// --- observed-state phases and transitions ---

func TestPhasesAreDisjointFromDesiredState(t *testing.T) {
	// The desired axis has exactly two values; the observed axis has its own
	// vocabulary. No string is shared between the two, so a reader can never
	// conflate intent with observation.
	for _, ph := range AllPhases {
		if string(ph) == string(DesiredRunning) || string(ph) == string(DesiredStopped) {
			t.Fatalf("phase %q collides with a desired-state value", ph)
		}
	}
	want := []Phase{PhaseStarting, PhaseReady, PhaseDegraded, PhaseFailed, PhaseFenced, PhaseStopped, PhaseUnknown}
	if len(AllPhases) != len(want) {
		t.Fatalf("AllPhases = %v", AllPhases)
	}
}

func TestPhaseTransitions(t *testing.T) {
	allowed := []struct{ from, to Phase }{
		{PhaseStarting, PhaseReady},
		{PhaseStarting, PhaseFailed},
		{PhaseReady, PhaseDegraded},
		{PhaseDegraded, PhaseReady},
		{PhaseReady, PhaseFailed},
		{PhaseFailed, PhaseStarting}, // restart
		{PhaseFailed, PhaseFenced},   // circuit opened
		{PhaseFenced, PhaseStarting}, // operator released the fence
		{PhaseReady, PhaseStopped},   // operator stop
		{PhaseUnknown, PhaseReady},   // supervisor re-observed
	}
	for _, tr := range allowed {
		if !CanTransition(tr.from, tr.to) {
			t.Errorf("CanTransition(%s -> %s) = false, want true", tr.from, tr.to)
		}
	}
	denied := []struct{ from, to Phase }{
		{PhaseStopped, PhaseReady},     // must pass through starting
		{PhaseFenced, PhaseReady},      // a fenced service cannot become ready without starting
		{PhaseStarting, PhaseDegraded}, // degraded is a post-ready observation
	}
	for _, tr := range denied {
		if CanTransition(tr.from, tr.to) {
			t.Errorf("CanTransition(%s -> %s) = true, want false", tr.from, tr.to)
		}
	}
	// Every phase may decay to unknown (observation loss is always possible).
	for _, ph := range AllPhases {
		if ph != PhaseUnknown && !CanTransition(ph, PhaseUnknown) {
			t.Errorf("CanTransition(%s -> unknown) = false, want true", ph)
		}
	}
}

// --- restart-semantics decisions ---

func policy() RestartPolicy {
	return RestartPolicy{
		InitialBackoffMS:  100,
		MaxBackoffMS:      1000,
		BackoffFactor:     2,
		WindowMS:          60000,
		WindowMaxRestarts: 3,
		StableRunResetMS:  30000,
	}
}

func TestDecideDesiredStoppedNeverRestarts(t *testing.T) {
	for _, class := range AllExitClasses {
		d := policy().Decide(RestartInput{Kind: KindService, Desired: DesiredStopped, Class: class})
		if d.Restart {
			t.Errorf("class %s: restarted a desired-stopped service", class)
		}
		if d.Reason != ReasonDesiredStopped {
			t.Errorf("class %s: reason = %q", class, d.Reason)
		}
	}
}

func TestDecideOperatorStopIsIntentional(t *testing.T) {
	// Operator stop never restarts even while desired=running: the intentional
	// stop is recorded, not fought.
	d := policy().Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitOperatorStop})
	if d.Restart || d.Reason != ReasonOperatorStop {
		t.Fatalf("operator stop: %+v", d)
	}
}

func TestDecideJobCleanExitCompletes(t *testing.T) {
	// A recurring job (#749) that exits cleanly is COMPLETE: the schedule owns
	// the next run, the supervisor must not restart it.
	d := policy().Decide(RestartInput{Kind: KindJob, Desired: DesiredRunning, Class: ExitClean})
	if d.Restart || d.Reason != ReasonJobComplete {
		t.Fatalf("job clean exit: %+v", d)
	}
	// The same clean exit on an always-on service restarts: running is the
	// desired steady state, exit of any class leaves it unmet.
	d = policy().Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitClean})
	if !d.Restart {
		t.Fatalf("service clean exit must restart: %+v", d)
	}
	// A crashing job IS restarted (the run did not complete).
	d = policy().Decide(RestartInput{Kind: KindJob, Desired: DesiredRunning, Class: ExitCrash})
	if !d.Restart {
		t.Fatalf("job crash must restart: %+v", d)
	}
}

func TestDecideBoundedExponentialBackoff(t *testing.T) {
	p := policy()
	wantDelays := []int64{100, 200, 400, 800, 1000, 1000} // doubling, capped at max
	for attempt, want := range wantDelays {
		d := p.Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitCrash, Attempt: attempt})
		if !d.Restart || d.DelayMS != want {
			t.Fatalf("attempt %d: got %+v, want delay %d", attempt, d, want)
		}
		if d.NextAttempt != attempt+1 {
			t.Fatalf("attempt %d: NextAttempt = %d", attempt, d.NextAttempt)
		}
	}
}

func TestDecideBackoffDoesNotOverflow(t *testing.T) {
	d := policy().Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitCrash, Attempt: 100000})
	if !d.Restart || d.DelayMS != policy().MaxBackoffMS {
		t.Fatalf("huge attempt: %+v", d)
	}
}

func TestDecideStableRunResetsBackoff(t *testing.T) {
	p := policy()
	// The exited run outlived stable_run_reset_ms: the attempt counter resets,
	// so the next delay is the initial backoff again.
	d := p.Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitCrash, Attempt: 7, RunMS: p.StableRunResetMS})
	if !d.Restart || d.DelayMS != p.InitialBackoffMS || d.NextAttempt != 1 {
		t.Fatalf("stable-run reset: %+v", d)
	}
}

func TestDecideRestartWindowCapOpensCircuit(t *testing.T) {
	p := policy()
	d := p.Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitCrash, WindowCount: p.WindowMaxRestarts})
	if d.Restart || !d.CircuitOpen || d.Reason != ReasonCircuitOpen {
		t.Fatalf("window cap: %+v", d)
	}
	// One below the cap still restarts.
	d = p.Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitCrash, WindowCount: p.WindowMaxRestarts - 1})
	if !d.Restart || d.CircuitOpen {
		t.Fatalf("below window cap: %+v", d)
	}
}

func TestDecideBootRecoveryRestartsFresh(t *testing.T) {
	// After boot recovery the host restarted, not the service: restart at once
	// with a fresh attempt counter — prior backoff belongs to the previous boot.
	d := policy().Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: ExitBootRecovery, Attempt: 4, WindowCount: 2})
	if !d.Restart || d.DelayMS != 0 || d.NextAttempt != 1 || d.Reason != ReasonBootRecovery {
		t.Fatalf("boot recovery: %+v", d)
	}
}

func TestDecideCoversEveryExitClass(t *testing.T) {
	// Every declared exit class yields a decision with a non-empty reason:
	// no failure class falls through unclassified.
	for _, class := range AllExitClasses {
		d := policy().Decide(RestartInput{Kind: KindService, Desired: DesiredRunning, Class: class})
		if d.Reason == "" {
			t.Errorf("class %s: empty reason", class)
		}
	}
}

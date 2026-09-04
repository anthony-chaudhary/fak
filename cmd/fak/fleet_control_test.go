package main

// fleet_control_test.go — the OPERATOR-side witness for #5600 (epic #5599).
//
// The property under test throughout: exit 0 means WITNESSED APPLIED. A send that
// published successfully, addressed the right instances, and was never applied by any
// of them must not exit 0 — because a control plane whose success code means "the
// message left the building" is a megaphone with a return-code costume.

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// controlBus opens a scratch bus and returns it with its directory.
func controlBus(t *testing.T) (*fleetbus.DirBus, string) {
	t.Helper()
	dir := t.TempDir()
	bus, err := fleetbus.OpenDir(dir)
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	return bus, dir
}

// announceControlInstance puts one live instance on the roster, so a send has a
// denominator to measure against.
func announceControlInstance(t *testing.T, bus *fleetbus.DirBus, id, machine, role string) fleetbus.Instance {
	t.Helper()
	inst, refusal := fleetbus.NewInstance(id, machine, role, 4242, "", []fleetbus.Op{"pause", "steer"}, time.Now())
	if refusal != nil {
		t.Fatalf("NewInstance(%s): %v", id, refusal)
	}
	if err := bus.Announce(inst); err != nil {
		t.Fatalf("Announce(%s): %v", id, err)
	}
	return inst
}

// runControl invokes one `fak fleet control` subcommand with captured streams.
func runControl(t *testing.T, argv ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	code = runFleetControl(&out, &errBuf, argv)
	return code, out.String(), errBuf.String()
}

// TestFleetControlSendRefusesAnEmptyFleet is the edge refusal: publishing to nobody
// would return a report that is trivially "complete" (0 of 0) and read as success.
func TestFleetControlSendRefusesAnEmptyFleet(t *testing.T) {
	_, dir := controlBus(t)
	code, _, stderr := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "0")
	if code != 2 {
		t.Fatalf("exit = %d, want 2; stderr=%q", code, stderr)
	}
	if !strings.Contains(stderr, string(fleetbus.NoTarget)) {
		t.Fatalf("stderr %q does not carry the %s token", stderr, fleetbus.NoTarget)
	}
}

// TestFleetControlSendRefusesAnUnstatedSelector — an absent filter never means
// "everyone". Addressing the whole fleet is a thing an operator says out loud.
func TestFleetControlSendRefusesAnUnstatedSelector(t *testing.T) {
	bus, dir := controlBus(t)
	announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	code, _, stderr := runControl(t, "send", "--op", "pause", "--bus", dir, "--wait", "0")
	if code != 2 {
		t.Fatalf("exit = %d, want 2; stderr=%q", code, stderr)
	}
	if !strings.Contains(stderr, string(fleetbus.Malformed)) {
		t.Fatalf("stderr %q does not carry the %s token", stderr, fleetbus.Malformed)
	}
	if ds, _ := bus.Directives(); len(ds) != 0 {
		t.Fatalf("a refused send published %d directive(s) anyway", len(ds))
	}
}

// TestFleetControlSendWithoutAnAckIsNotASuccess is the load-bearing exit-code case: the
// publish succeeded, the fleet is real, and nobody has answered. That is exit 1.
func TestFleetControlSendWithoutAnAckIsNotASuccess(t *testing.T) {
	bus, dir := controlBus(t)
	announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	code, stdout, stderr := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "0")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (published, unwitnessed); stdout=%q stderr=%q", code, stdout, stderr)
	}
	if !strings.Contains(stdout, "not yet") {
		t.Fatalf("stdout %q does not report the unwitnessed state in the operator's terms", stdout)
	}
	if !strings.Contains(stdout, "outstanding=1") {
		t.Fatalf("stdout %q does not show the silent instance in the denominator", stdout)
	}
	ds, err := bus.Directives()
	if err != nil || len(ds) != 1 {
		t.Fatalf("Directives() = %d, %v; want the one published directive", len(ds), err)
	}
}

// TestFleetControlSendThenStatusWitnessesTheApply walks the whole round trip through the
// operator surface: publish, let the instance drain, then fold. Only after a real ack
// exists does the control point get an exit 0.
func TestFleetControlSendThenStatusWitnessesTheApply(t *testing.T) {
	bus, dir := controlBus(t)
	self := announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	code, stdout, _ := runControl(t, "send", "--op", "steer", "--payload", "go", "--all", "--bus", dir, "--wait", "0")
	if code != 1 {
		t.Fatalf("send exit = %d, want 1 before any drain", code)
	}
	ds, err := bus.Directives()
	if err != nil || len(ds) != 1 {
		t.Fatalf("Directives() = %d, %v", len(ds), err)
	}
	d := ds[0]
	if !strings.Contains(stdout, d.ID) {
		t.Fatalf("send output %q never names the directive id an operator needs for `status`", stdout)
	}

	// The instance side: apply and ack, exactly as `fak serve --fleet-bus` does.
	rep, err := fleetbus.Drain(bus, self, fleetbus.ApplierFunc(func(fleetbus.Directive) fleetbus.Outcome {
		return fleetbus.OutcomeApplied("delivered 'go' to 3 session(s)", 3)
	}), time.Now())
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if rep.Applied != 1 {
		t.Fatalf("drain applied %d, want 1 (errors: %v)", rep.Applied, rep.Errors)
	}

	code, stdout, stderr := runControl(t, "status", "--directive", d.ID, "--bus", dir)
	if code != 0 {
		t.Fatalf("status exit = %d, want 0 after a witnessed apply; stdout=%q stderr=%q", code, stdout, stderr)
	}
	for _, want := range []string{"applied", "affected=3", "targeted=1"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("status output %q is missing %q", stdout, want)
		}
	}
}

// TestFleetControlStatusDoesNotCallARefusalASuccess — every instance answered, so the
// fold is Complete; none applied, so the exit code must still be 1. Complete and
// witnessed are deliberately different things.
func TestFleetControlStatusDoesNotCallARefusalASuccess(t *testing.T) {
	bus, dir := controlBus(t)
	self := announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	if code, _, stderr := runControl(t, "send", "--op", "teleport", "--all", "--bus", dir, "--wait", "0"); code != 1 {
		t.Fatalf("send exit = %d, want 1; stderr=%q", code, stderr)
	}
	ds, _ := bus.Directives()
	if len(ds) != 1 {
		t.Fatalf("want 1 directive, got %d", len(ds))
	}
	if _, err := fleetbus.Drain(bus, self, fleetbus.ApplierFunc(func(d fleetbus.Directive) fleetbus.Outcome {
		return fleetbus.OutcomeRefused(fleetbus.UnknownOp, "op %q is outside this instance's vocabulary", d.Op)
	}), time.Now()); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	code, stdout, _ := runControl(t, "status", "--directive", ds[0].ID, "--bus", dir)
	if code != 1 {
		t.Fatalf("status exit = %d, want 1 — a fully-answered, fully-refused directive is not a success", code)
	}
	if !strings.Contains(stdout, "answered, not applied") {
		t.Fatalf("status output %q does not distinguish answered from applied", stdout)
	}
	if !strings.Contains(stdout, string(fleetbus.UnknownOp)) {
		t.Fatalf("status output %q drops the instance's own refusal token", stdout)
	}
}

// TestFleetControlSendWaitsForALateAck proves --wait actually watches the return path
// rather than sleeping out the budget: an ack that lands mid-wait ends it early.
func TestFleetControlSendWaitsForALateAck(t *testing.T) {
	bus, dir := controlBus(t)
	self := announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	done := make(chan struct{})
	go func() {
		defer close(done)
		deadline := time.Now().Add(8 * time.Second)
		for time.Now().Before(deadline) {
			ds, _ := bus.Directives()
			if len(ds) > 0 {
				_, _ = fleetbus.Drain(bus, self, fleetbus.ApplierFunc(func(fleetbus.Directive) fleetbus.Outcome {
					return fleetbus.OutcomeApplied("paused 1 session", 1)
				}), time.Now())
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	start := time.Now()
	code, stdout, stderr := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "9s")
	<-done
	if code != 0 {
		t.Fatalf("exit = %d, want 0 once the ack landed; stdout=%q stderr=%q", code, stdout, stderr)
	}
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Fatalf("send burned the whole %s budget (%s) instead of returning on the ack", "9s", elapsed)
	}
	if !strings.Contains(stdout, "applied — all 1 addressed instance(s) witnessed it") {
		t.Fatalf("stdout %q does not report the witnessed verdict", stdout)
	}
}

// TestFleetControlSendAddressesOnlyTheStatedRole — the instance axis is what makes many
// control points and mixed fleets workable, so an unaddressed instance must not land in
// the denominator and hold the report OUTSTANDING forever.
func TestFleetControlSendAddressesOnlyTheStatedRole(t *testing.T) {
	bus, dir := controlBus(t)
	serve := announceControlInstance(t, bus, "serve-1", "box-a", "serve")
	announceControlInstance(t, bus, "worker-1", "box-b", "worker")

	if code, _, stderr := runControl(t, "send", "--op", "pause", "--role", "serve", "--bus", dir, "--wait", "0"); code != 1 {
		t.Fatalf("send exit = %d, want 1; stderr=%q", code, stderr)
	}
	ds, _ := bus.Directives()
	if _, err := fleetbus.Drain(bus, serve, fleetbus.ApplierFunc(func(fleetbus.Directive) fleetbus.Outcome {
		return fleetbus.OutcomeApplied("paused 1 session", 1)
	}), time.Now()); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	code, stdout, _ := runControl(t, "status", "--directive", ds[0].ID, "--bus", dir)
	if code != 0 {
		t.Fatalf("status exit = %d, want 0 — the worker was never addressed, so it cannot hold the report open", code)
	}
	if !strings.Contains(stdout, "targeted=1") {
		t.Fatalf("status output %q counted an unaddressed instance in the denominator", stdout)
	}
	if strings.Contains(stdout, "worker-1") {
		t.Fatalf("status output %q reports on an instance the selector never addressed", stdout)
	}
}

// TestFleetControlStatusNamesBothWaysADirectiveCanBeMissing — "never issued" and
// "rotated out of the retained ledger" are different problems with different fixes.
func TestFleetControlStatusNamesBothWaysADirectiveCanBeMissing(t *testing.T) {
	_, dir := controlBus(t)
	code, _, stderr := runControl(t, "status", "--directive", "d-nope", "--bus", dir)
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr, "rotated") {
		t.Fatalf("stderr %q collapses the two ways a directive goes missing into one", stderr)
	}
}

// TestFleetControlInstancesRendersTheRoster is the operator's answer to "who would a
// send even reach?" — and an empty fleet names the refusal it will cause.
func TestFleetControlInstancesRendersTheRoster(t *testing.T) {
	bus, dir := controlBus(t)

	code, stdout, stderr := runControl(t, "instances", "--bus", dir)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 — an empty fleet is a fact, not an error; stderr=%q", code, stderr)
	}
	if !strings.Contains(stdout, string(fleetbus.NoTarget)) {
		t.Fatalf("empty-roster output %q does not name the refusal a send would hit", stdout)
	}

	announceControlInstance(t, bus, "serve-1", "box-a", "serve")
	code, stdout, _ = runControl(t, "instances", "--bus", dir)
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	for _, want := range []string{"serve-1", "box-a", "pid=4242", "pause,steer"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("roster output %q is missing %q", stdout, want)
		}
	}
}

// TestFleetControlInstancesTreatsAStaleAnnouncementAsGone — the roster is a liveness
// claim with an expiry, not a registry. A process that died an hour ago must not sit in
// the denominator making every send report OUTSTANDING forever.
func TestFleetControlInstancesTreatsAStaleAnnouncementAsGone(t *testing.T) {
	bus, dir := controlBus(t)
	old, refusal := fleetbus.NewInstance("ghost-1", "box-a", "serve", 7, "", nil, time.Now().Add(-time.Hour))
	if refusal != nil {
		t.Fatalf("NewInstance: %v", refusal)
	}
	if err := bus.Announce(old); err != nil {
		t.Fatalf("Announce: %v", err)
	}

	code, stdout, _ := runControl(t, "instances", "--bus", dir)
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if strings.Contains(stdout, "ghost-1") {
		t.Fatalf("roster output %q still lists an instance silent for an hour", stdout)
	}
	if code, _, stderr := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "0"); code != 2 ||
		!strings.Contains(stderr, string(fleetbus.NoTarget)) {
		t.Fatalf("a fleet of only stale records did not refuse NO_TARGET: exit=%d stderr=%q", code, stderr)
	}
}

// TestFleetControlRejectsAnUnknownSubcommand keeps the group surface closed.
func TestFleetControlRejectsAnUnknownSubcommand(t *testing.T) {
	for _, argv := range [][]string{{}, {"drain"}} {
		code, _, stderr := runControl(t, argv...)
		if code != 2 {
			t.Errorf("argv %v: exit = %d, want 2", argv, code)
		}
		if !strings.Contains(stderr, "send | status | instances") {
			t.Errorf("argv %v: stderr %q does not name the closed subcommand set", argv, stderr)
		}
	}
}

// TestFleetControlSendRequiresAnOp — the bus carries an opaque token, so an empty one is
// caught at the edge rather than fanned out for every instance to refuse separately.
func TestFleetControlSendRequiresAnOp(t *testing.T) {
	_, dir := controlBus(t)
	code, _, stderr := runControl(t, "send", "--all", "--bus", dir)
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr, "--op is required") {
		t.Fatalf("stderr %q does not say what is missing", stderr)
	}
}

// TestFleetControlTextIsThePayloadNotASecondField — #5600's done-condition spells the
// steer argument `--text`; `--payload` is the general name. They must be ONE field, and
// two disagreeing spellings must refuse rather than silently pick a winner: the fleet
// would otherwise receive a payload the operator can see they did not send.
func TestFleetControlTextIsThePayloadNotASecondField(t *testing.T) {
	bus, dir := controlBus(t)
	announceControlInstance(t, bus, "serve-1", "box-a", "serve")

	if code, _, stderr := runControl(t, "send", "--op", "steer", "--text", "go", "--all", "--bus", dir, "--wait", "0"); code != 1 {
		t.Fatalf("send --text exit = %d, want 1 (published, unwitnessed); stderr=%q", code, stderr)
	}
	ds, err := bus.Directives()
	if err != nil || len(ds) != 1 {
		t.Fatalf("Directives() = %d, %v", len(ds), err)
	}
	if ds[0].Payload != "go" {
		t.Fatalf("--text landed as payload %q, want %q", ds[0].Payload, "go")
	}

	code, _, stderr := runControl(t, "send", "--op", "steer", "--text", "go", "--payload", "stop",
		"--all", "--bus", dir, "--wait", "0")
	if code != 2 {
		t.Fatalf("contradictory --text/--payload exit = %d, want 2", code)
	}
	if !strings.Contains(stderr, "same field") {
		t.Fatalf("stderr %q does not explain that the two spellings are one field", stderr)
	}
	if ds, _ := bus.Directives(); len(ds) != 1 {
		t.Fatalf("a refused send published anyway: %d directive(s)", len(ds))
	}

	for _, tc := range []struct {
		payload, text, want string
		ok                  bool
	}{
		{"", "", "", true},
		{"go", "", "go", true},
		{"", "go", "go", true},
		{"go", "go", "go", true},
		{"go", "stop", "", false},
	} {
		got, ok := resolveControlPayload(tc.payload, tc.text)
		if got != tc.want || ok != tc.ok {
			t.Errorf("resolveControlPayload(%q, %q) = %q, %v; want %q, %v", tc.payload, tc.text, got, ok, tc.want, tc.ok)
		}
	}
}

// TestFleetControlDoesNotCallADeadInstanceASuccess is the operator-level regression for
// the way this control plane could learn to lie. Two serves are addressed; one applies,
// the other dies and stops announcing, so it ages out of the roster `status` reads. If
// the denominator were re-derived from that roster, the dead instance would simply leave
// the report — turning "one instance never answered" into a clean exit 0.
func TestFleetControlDoesNotCallADeadInstanceASuccess(t *testing.T) {
	bus, dir := controlBus(t)
	live := announceControlInstance(t, bus, "serve-1", "box-a", "serve")
	announceControlInstance(t, bus, "serve-2", "box-b", "serve")

	if code, _, stderr := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "0"); code != 1 {
		t.Fatalf("send exit = %d, want 1; stderr=%q", code, stderr)
	}
	ds, err := bus.Directives()
	if err != nil || len(ds) != 1 {
		t.Fatalf("Directives() = %d, %v", len(ds), err)
	}
	d := ds[0]
	if len(d.Targets) != 2 {
		t.Fatalf("published directive records targets %v; the publish-time addressee set was not stamped", d.Targets)
	}

	// serve-1 applies. serve-2 never will, and its presence record lapses.
	if _, err := fleetbus.Drain(bus, live, fleetbus.ApplierFunc(func(fleetbus.Directive) fleetbus.Outcome {
		return fleetbus.OutcomeApplied("paused 2 sessions", 2)
	}), time.Now()); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	// Age serve-2 out of the roster exactly as a dead process would.
	stale, refusal := fleetbus.NewInstance("serve-2", "box-b", "serve", 4242, "", nil, time.Now().Add(-time.Hour))
	if refusal != nil {
		t.Fatalf("NewInstance: %v", refusal)
	}
	if err := bus.Announce(stale); err != nil {
		t.Fatalf("Announce: %v", err)
	}

	code, stdout, stderr := runControl(t, "status", "--directive", d.ID, "--bus", dir)
	if code != 1 {
		t.Fatalf("status exit = %d, want 1 — serve-2 never applied; stdout=%q stderr=%q", code, stdout, stderr)
	}
	if !strings.Contains(stdout, "targeted=2") || !strings.Contains(stdout, "outstanding=1") {
		t.Fatalf("status output %q dropped the dead instance from the denominator", stdout)
	}
	if !strings.Contains(stdout, "serve-2") {
		t.Fatalf("status output %q never names the instance that went missing", stdout)
	}
}

// TestFleetControlStatusDoesNotExplainItselfWithAFlagItLacks — `status` is a read, not a
// wait. Reporting an outstanding directive as a `--wait 0` choice names a flag status
// does not accept, and "after 0s" asserts a latency it never measured.
func TestFleetControlStatusDoesNotExplainItselfWithAFlagItLacks(t *testing.T) {
	bus, dir := controlBus(t)
	announceControlInstance(t, bus, "serve-1", "box-a", "serve")
	if code, _, _ := runControl(t, "send", "--op", "pause", "--all", "--bus", dir, "--wait", "0"); code != 1 {
		t.Fatal("send did not publish")
	}
	ds, _ := bus.Directives()

	_, stdout, _ := runControl(t, "status", "--directive", ds[0].ID, "--bus", dir)
	if strings.Contains(stdout, "--wait 0") {
		t.Fatalf("status output %q explains itself with a flag `status` does not have", stdout)
	}
	if !strings.Contains(stdout, "not yet") || !strings.Contains(stdout, "1 still silent") {
		t.Fatalf("status output %q does not report the outstanding instance in its own terms", stdout)
	}
}

// TestSplitTokenListDropsPhantomMembers — a trailing comma is a typo, not an instance
// named "".
func TestSplitTokenList(t *testing.T) {
	got := splitTokenList(" serve-1, serve-2 ,, ")
	if len(got) != 2 || got[0] != "serve-1" || got[1] != "serve-2" {
		t.Fatalf("splitTokenList = %#v, want [serve-1 serve-2]", got)
	}
	if len(splitTokenList("  ")) != 0 {
		t.Fatal("an empty axis produced a member")
	}
}

// TestSanitizeBusTokenNeverProducesAPathEscape — the issuer is derived from a hostname
// nobody at the call site chose, so it is mapped onto the token alphabet rather than
// refused. It must never map onto "." or "..", which are path segments, not names.
func TestSanitizeBusToken(t *testing.T) {
	cases := map[string]string{
		"box-a.local":  "box-a.local",
		"box a/../etc": "box-a-..-etc",
		"":             "control",
		"..":           "control",
		"WIN-01_test":  "WIN-01_test",
	}
	for in, want := range cases {
		if got := sanitizeBusToken(in); got != want {
			t.Errorf("sanitizeBusToken(%q) = %q, want %q", in, got, want)
		}
	}
	// The relation the length cap stands for: whatever arbitrary host string goes in,
	// what comes out is a token the bus itself will accept. Asserting ValidToken rather
	// than freezing today's limit keeps this honest if the limit ever moves.
	for _, in := range []string{
		strings.Repeat("x", 300), "", ".", "..", "box a/../etc", strings.Repeat("/", 200),
	} {
		if got := sanitizeBusToken(in); !fleetbus.ValidToken(got) {
			t.Errorf("sanitizeBusToken(%q) = %q, which fleetbus.ValidToken rejects", in, got)
		}
	}
}

// TestFleetControlSendResumeBroadcastAcknowledged proves issue #10847:
// `fak fleet control send --op resume --all` broadcasts to live instances,
// each instance applies ResumeAll, and the control point folds the Acks.
func TestFleetControlSendResumeBroadcastAcknowledged(t *testing.T) {
	bus, dir := controlBus(t)
	now := time.Now()

	tbl1 := &session.Table{}
	tbl1.Transition("s1-paused", session.Paused, "paused for test")
	tbl1.Transition("s1-running", session.Running, "")

	tbl2 := &session.Table{}
	tbl2.Transition("s2-paused1", session.Paused, "paused for test")
	tbl2.Transition("s2-paused2", session.Paused, "paused for test")

	inst1, r1 := fleetbus.NewInstance("guard-1", "box-a", guardFleetBusRole, 4201, "", guardFleetBusAdvertisedOps(), now)
	if r1 != nil {
		t.Fatalf("NewInstance guard-1: %v", r1)
	}
	inst2, r2 := fleetbus.NewInstance("guard-2", "box-a", guardFleetBusRole, 4202, "", guardFleetBusAdvertisedOps(), now)
	if r2 != nil {
		t.Fatalf("NewInstance guard-2: %v", r2)
	}
	if err := bus.Announce(inst1); err != nil {
		t.Fatalf("Announce guard-1: %v", err)
	}
	if err := bus.Announce(inst2); err != nil {
		t.Fatalf("Announce guard-2: %v", err)
	}

	code, stdout, stderr := runControl(t, "send", "--op", "resume", "--all", "--bus", dir, "--wait", "0")
	if code != 1 {
		t.Fatalf("send exit = %d, want 1 before drain (unwitnessed); stdout=%q stderr=%q", code, stdout, stderr)
	}
	ds, err := bus.Directives()
	if err != nil || len(ds) != 1 {
		t.Fatalf("Directives() = %d, %v; want 1", len(ds), err)
	}
	d := ds[0]

	ap1 := guardTestApplier(t, tbl1, guardSeatRefresher{})
	ap2 := guardTestApplier(t, tbl2, guardSeatRefresher{})

	if _, err := fleetbus.Drain(bus, inst1, ap1, now); err != nil {
		t.Fatalf("Drain guard-1: %v", err)
	}
	if _, err := fleetbus.Drain(bus, inst2, ap2, now); err != nil {
		t.Fatalf("Drain guard-2: %v", err)
	}

	code, stdout, stderr = runControl(t, "status", "--directive", d.ID, "--bus", dir)
	if code != 0 {
		t.Fatalf("status exit = %d, want 0 after both instances applied; stdout=%q stderr=%q", code, stdout, stderr)
	}
	for _, want := range []string{"applied", "affected=3", "targeted=2"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("status output %q is missing %q", stdout, want)
		}
	}

	// Verify states in tables were resumed to Running
	if cur := tbl1.Get("s1-paused"); cur.Run != session.Running {
		t.Errorf("s1-paused run state = %v, want Running", cur.Run)
	}
	if cur := tbl2.Get("s2-paused1"); cur.Run != session.Running {
		t.Errorf("s2-paused1 run state = %v, want Running", cur.Run)
	}
	if cur := tbl2.Get("s2-paused2"); cur.Run != session.Running {
		t.Errorf("s2-paused2 run state = %v, want Running", cur.Run)
	}
}

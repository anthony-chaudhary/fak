package escalation

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fixturePacket is the schema fixture: every field populated the way the
// notify producer populates it. goldenPacketJSON is its EXACT wire form —
// the round-trip witness the issue (#2271) names.
func fixturePacket() Packet {
	return Packet{
		Schema:            Schema,
		ID:                "notify/trace-7/run-3#2",
		Source:            SourceNotify,
		Reason:            "ESCALATE_APPROVAL",
		Class:             "stop",
		Severity:          SeverityOperator,
		LoopID:            "dispatch/iss-42",
		RunID:             "run-3",
		Issue:             "42",
		TraceID:           "trace-7",
		Rev:               2,
		StateDigest:       "paused",
		Evidence:          []EvidenceRef{{Kind: "lease", Ref: "lane/escalation"}},
		Actions:           []string{"resume", "hold", "release_held_resources"},
		SafeDefault:       "release_held_resources",
		EmittedAtUnixNano: 1720000000000000000,
		ExpiresAtUnixNano: 1720007200000000000,
		CostOfDelay:       CostSeatHeld,
	}
}

const goldenPacketJSON = `{"schema":"fak.escalation.v1","id":"notify/trace-7/run-3#2","source":"notify","reason":"ESCALATE_APPROVAL","class":"stop","severity":"operator","loop_id":"dispatch/iss-42","run_id":"run-3","issue":"42","trace_id":"trace-7","rev":2,"state_digest":"paused","evidence":[{"kind":"lease","ref":"lane/escalation"}],"actions":["resume","hold","release_held_resources"],"safe_default":"release_held_resources","emitted_at_unix_nano":1720000000000000000,"expires_at_unix_nano":1720007200000000000,"cost_of_delay":"seat_held"}`

const goldenAckJSON = `{"schema":"fak.escalation.ack.v1","packet_id":"notify/trace-7/run-3#2","action":"resume","actor":"operator","rev":2,"acked_at_unix_nano":1720000090000000000}`

func fixtureAck() Ack {
	return Ack{
		Schema:          AckSchema,
		PacketID:        "notify/trace-7/run-3#2",
		Action:          "resume",
		Actor:           ActorOperator,
		Rev:             2,
		AckedAtUnixNano: 1720000090000000000,
	}
}

// TestFixtureRoundTrip is the issue's witness: the schema fixture round-trips
// byte-exactly through its wire form, in both directions, for both row kinds,
// and both fixtures pass the fail-closed gate.
func TestFixtureRoundTrip(t *testing.T) {
	p := fixturePacket()
	if err := p.Validate(); err != nil {
		t.Fatalf("fixture packet must validate: %v", err)
	}
	got, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != goldenPacketJSON {
		t.Fatalf("packet wire form drifted:\n got %s\nwant %s", got, goldenPacketJSON)
	}
	var back Packet
	if err := json.Unmarshal([]byte(goldenPacketJSON), &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, p) {
		t.Fatalf("packet round-trip drifted:\n got %+v\nwant %+v", back, p)
	}

	a := fixtureAck()
	if err := a.Validate(); err != nil {
		t.Fatalf("fixture ack must validate: %v", err)
	}
	gotA, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotA) != goldenAckJSON {
		t.Fatalf("ack wire form drifted:\n got %s\nwant %s", gotA, goldenAckJSON)
	}
	var backA Ack
	if err := json.Unmarshal([]byte(goldenAckJSON), &backA); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(backA, a) {
		t.Fatalf("ack round-trip drifted:\n got %+v\nwant %+v", backA, a)
	}

	// One ledger, two row kinds, routed by schema tag alone.
	if row, err := decodeRow([]byte(goldenPacketJSON)); err != nil || row.Packet == nil || row.Ack != nil {
		t.Fatalf("packet line must decode as a packet row: %+v %v", row, err)
	}
	if row, err := decodeRow([]byte(goldenAckJSON)); err != nil || row.Ack == nil || row.Packet != nil {
		t.Fatalf("ack line must decode as an ack row: %+v %v", row, err)
	}
	if _, err := decodeRow([]byte(`{"schema":"fak.other.v1"}`)); !errors.Is(err, ErrSchema) {
		t.Fatalf("unknown row schema must fail closed, got %v", err)
	}
}

// TestPacketCarriesNoProseField pins the "decidable in seconds" law at the
// type level: the packet (and ack) schema has NO free-text field — no summary,
// body, message, or description ever rides along, so a reviewer decides from
// closed tokens and refs alone.
func TestPacketCarriesNoProseField(t *testing.T) {
	forbidden := []string{"summary", "body", "message", "text", "note", "detail", "description", "prose", "transcript"}
	for _, typ := range []reflect.Type{reflect.TypeOf(Packet{}), reflect.TypeOf(Ack{}), reflect.TypeOf(EvidenceRef{})} {
		for i := 0; i < typ.NumField(); i++ {
			name := strings.ToLower(typ.Field(i).Name)
			tag := strings.ToLower(typ.Field(i).Tag.Get("json"))
			for _, f := range forbidden {
				if strings.Contains(name, f) || strings.Contains(tag, f) {
					t.Fatalf("%s.%s smells like a free-text field (%q) — the packet is closed tokens only", typ.Name(), typ.Field(i).Name, f)
				}
			}
		}
	}
}

func TestValidateRefusesProseAndUnsafeDefaults(t *testing.T) {
	base := fixturePacket()

	cases := []struct {
		name    string
		mutate  func(*Packet)
		wantErr error
	}{
		{"prose reason", func(p *Packet) { p.Reason = "please look at this" }, ErrProse},
		{"lowercase reason", func(p *Packet) { p.Reason = "escalate_approval" }, ErrProse},
		{"prose state digest", func(p *Packet) { p.StateDigest = "It Is Paused" }, ErrProse},
		{"prose action", func(p *Packet) { p.Actions = []string{"resume", "Please Fix"}; p.SafeDefault = "resume" }, ErrProse},
		{"safe default off menu", func(p *Packet) { p.SafeDefault = "self_destruct" }, ErrUnsafeDefault},
		{"empty menu", func(p *Packet) { p.Actions = nil }, ErrUnsafeDefault},
		{"no routing id", func(p *Packet) { p.LoopID, p.RunID, p.SessionID, p.TraceID = "", "", "", "" }, ErrUnroutable},
		{"zero rev", func(p *Packet) { p.Rev = 0 }, ErrRev},
		{"expiry before emit", func(p *Packet) { p.ExpiresAtUnixNano = p.EmittedAtUnixNano - 1 }, ErrExpiry},
		{"open severity", func(p *Packet) { p.Severity = "urgent-ish" }, ErrProse},
		{"open cost class", func(p *Packet) { p.CostOfDelay = "very expensive" }, ErrProse},
		{"wrong schema", func(p *Packet) { p.Schema = "fak.escalation.v0" }, ErrSchema},
	}
	for _, tc := range cases {
		p := base
		p.Actions = append([]string(nil), base.Actions...)
		p.Evidence = append([]EvidenceRef(nil), base.Evidence...)
		tc.mutate(&p)
		if err := p.Validate(); !errors.Is(err, tc.wantErr) {
			t.Errorf("%s: got %v, want %v", tc.name, err, tc.wantErr)
		}
	}

	a := fixtureAck()
	a.Actor = "alice@example.com" // an identity is not a closed actor class
	if err := a.Validate(); !errors.Is(err, ErrProse) {
		t.Errorf("identity actor: got %v, want ErrProse", err)
	}
}

func TestFromNotifySeverityAndDeterministicID(t *testing.T) {
	at := time.Unix(0, 1720000000000000000).UTC()
	fire := NotifyFire{
		TraceID: "trace-7", LoopID: "dispatch/iss-42", RunID: "run-3", Issue: "42",
		Reason: "ESCALATE_APPROVAL", To: "paused", Rev: 2, At: at,
		CostOfDelay: CostSeatHeld,
		Evidence:    []EvidenceRef{{Kind: "lease", Ref: "lane/escalation"}},
	}
	p, err := FromNotify(fire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, fixturePacket()) {
		t.Fatalf("notify producer drifted from the schema fixture:\n got %+v\nwant %+v", p, fixturePacket())
	}
	p2, err := FromNotify(fire)
	if err != nil || p2.ID != p.ID {
		t.Fatalf("a re-fire of the same (anchor, rev) must derive the SAME id: %q vs %q (%v)", p2.ID, p.ID, err)
	}

	// A non-blocking stop-reason is a status packet: nothing waits on a human.
	s, err := FromNotify(NotifyFire{TraceID: "trace-9", Reason: "DONE_WITNESSED", To: "done", Rev: 1, At: at})
	if err != nil {
		t.Fatal(err)
	}
	if s.Severity != SeverityStatus || s.SafeDefault != "ack_noted" {
		t.Fatalf("informational notify must fold to a status packet with ack_noted default: %+v", s)
	}
	if s.ExpiresAtUnixNano != at.Add(DefaultExpiry).UnixNano() {
		t.Fatalf("zero expiry must take DefaultExpiry, got %d", s.ExpiresAtUnixNano)
	}
	if s.CostOfDelay != CostNone {
		t.Fatalf("zero cost class must default to none, got %q", s.CostOfDelay)
	}
}

func TestFromRefusalOnlyEscalateDispositionEmits(t *testing.T) {
	at := time.Unix(0, 1720000000000000000).UTC()
	for _, disp := range []string{"RETRYABLE", "WAIT", "TERMINAL", ""} {
		if _, err := FromRefusal(RefusalHead{Disposition: disp, Reason: "SELF_MODIFY", SessionID: "s1", Rev: 1, At: at}); !errors.Is(err, ErrNotEscalate) {
			t.Errorf("disposition %q must not mint a packet, got %v", disp, err)
		}
	}
	p, err := FromRefusal(RefusalHead{Disposition: "ESCALATE", Reason: "SELF_MODIFY", SessionID: "s1", TraceID: "t1", Rev: 3, At: at})
	if err != nil {
		t.Fatal(err)
	}
	if p.Source != SourceRefusal || p.Class != "refusal" || p.Severity != SeverityOperator {
		t.Fatalf("refusal packet head drifted: %+v", p)
	}
	// Fail closed: an unreviewed escalated refusal STAYS refused on expiry.
	if p.SafeDefault != "deny" {
		t.Fatalf("refusal safe default must be deny, got %q", p.SafeDefault)
	}
	if p.CostOfDelay != CostRunBlock {
		t.Fatalf("refusal default cost class must be run_blocked, got %q", p.CostOfDelay)
	}
}

// TestLedgerEmitAckFold is the DoD end-to-end: a notify fire produces a packet
// row, an ack row closes it, and the R1 handling time is computable from the
// pair — with idempotent re-emits and re-delivered acks collapsing.
func TestLedgerEmitAckFold(t *testing.T) {
	l := Ledger{Path: filepath.Join(t.TempDir(), "escalations.jsonl")}
	at := time.Unix(0, 1720000000000000000).UTC()

	p1, err := FromNotify(NotifyFire{TraceID: "trace-7", RunID: "run-3", Reason: "ESCALATE_APPROVAL", To: "paused", Rev: 2, At: at})
	if err != nil {
		t.Fatal(err)
	}
	if p1, err = l.Emit(p1); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Emit(p1); err != nil { // producer re-fire: same row again
		t.Fatal(err)
	}
	p2, err := FromRefusal(RefusalHead{Disposition: "ESCALATE", Reason: "SELF_MODIFY", SessionID: "s1", Rev: 1, At: at.Add(10 * time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	if p2, err = l.Emit(p2); err != nil {
		t.Fatal(err)
	}

	rows, err := l.Load()
	if err != nil {
		t.Fatal(err)
	}
	rep := Fold(rows, at.Add(time.Minute))
	if len(rep.Open) != 2 || len(rep.Acked) != 0 || len(rep.Expired) != 0 {
		t.Fatalf("before any ack: want 2 open (re-emit collapsed), got %d open / %d acked / %d expired", len(rep.Open), len(rep.Acked), len(rep.Expired))
	}

	// Ack p1 at +90s; a stale re-delivery of the same (PacketID, Rev) later
	// claims +30s — idempotency keeps the FIRST landed row.
	if err := l.Ack(Ack{PacketID: p1.ID, Action: "resume", Actor: ActorOperator, Rev: p1.Rev, AckedAtUnixNano: at.Add(90 * time.Second).UnixNano()}); err != nil {
		t.Fatal(err)
	}
	if err := l.Ack(Ack{PacketID: p1.ID, Action: "hold", Actor: ActorOperator, Rev: p1.Rev, AckedAtUnixNano: at.Add(30 * time.Second).UnixNano()}); err != nil {
		t.Fatal(err)
	}
	// Ack p2 at +40s so the R1 slice has a pair of handling times.
	if err := l.Ack(Ack{PacketID: p2.ID, Action: "deny", Actor: ActorOperator, Rev: p2.Rev, AckedAtUnixNano: at.Add(50 * time.Second).UnixNano()}); err != nil {
		t.Fatal(err)
	}

	rows, err = l.Load()
	if err != nil {
		t.Fatal(err)
	}
	rep = Fold(rows, at.Add(time.Minute*2))
	if len(rep.Open) != 0 || len(rep.Acked) != 2 {
		t.Fatalf("after acks: want 0 open / 2 acked, got %d / %d", len(rep.Open), len(rep.Acked))
	}
	// p1: emitted t0, first-landed ack t0+90 -> 90s. p2: emitted t0+10, ack t0+50 -> 40s.
	if got := rep.HandlingSeconds; len(got) != 2 || got[0] != 40 || got[1] != 90 {
		t.Fatalf("handling seconds drifted (want [40 90] sorted): %v", got)
	}
	// R1 (escalation_handling_p50) takes the median straight off the slice.
	if p50 := rep.HandlingSeconds[len(rep.HandlingSeconds)/2]; p50 != 90 {
		t.Fatalf("p50 from the pair drifted: %v", p50)
	}
}

func TestFoldExpiryPrescribesSafeDefaultAndSurfacesOrphans(t *testing.T) {
	at := time.Unix(0, 1720000000000000000).UTC()
	p, err := FromNotify(NotifyFire{TraceID: "trace-1", Reason: "NEEDS_HUMAN_REVIEW", To: "paused", Rev: 1, At: at, Expiry: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	orphan := Ack{Schema: AckSchema, PacketID: "notify/ghost/_#9", Action: "resume", Actor: ActorOperator, Rev: 9, AckedAtUnixNano: at.UnixNano()}
	rows := []Row{{Packet: &p}, {Ack: &orphan}}

	rep := Fold(rows, at.Add(2*time.Hour))
	if len(rep.Expired) != 1 || len(rep.Open) != 0 {
		t.Fatalf("past expiry the packet must move to expired: %+v", rep)
	}
	if rep.Expired[0].SafeDefault != "release_held_resources" {
		t.Fatalf("the expired row must prescribe its safe default, got %q", rep.Expired[0].SafeDefault)
	}
	if len(rep.OrphanAcks) != 1 || rep.OrphanAcks[0].PacketID != "notify/ghost/_#9" {
		t.Fatalf("an ack binding no packet must be surfaced, not dropped: %+v", rep.OrphanAcks)
	}

	// An ack row that PRECEDES its packet row in a concatenated ledger still
	// binds — order in the file is not a correctness input.
	early := Ack{Schema: AckSchema, PacketID: p.ID, Action: "resume", Actor: ActorOperator, Rev: p.Rev, AckedAtUnixNano: at.Add(20 * time.Second).UnixNano()}
	rep = Fold([]Row{{Ack: &early}, {Packet: &p}}, at.Add(time.Minute))
	if len(rep.Acked) != 1 || rep.Acked[0].HandlingSeconds != 20 {
		t.Fatalf("ack-before-packet must still pair (20s), got %+v", rep.Acked)
	}
}

func TestLedgerFailClosedOnCorruptRow(t *testing.T) {
	l := Ledger{Path: filepath.Join(t.TempDir(), "missing", "escalations.jsonl")}
	if rows, err := l.Load(); err != nil || rows != nil {
		t.Fatalf("a not-yet-created ledger is empty, not an error: %v %v", rows, err)
	}
	// Emit through the ledger (creating the parent dir), then poison a line.
	at := time.Unix(0, 1720000000000000000).UTC()
	p, err := FromNotify(NotifyFire{TraceID: "t", Reason: "ESCALATE", To: "paused", Rev: 1, At: at})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Emit(p); err != nil {
		t.Fatal(err)
	}
	if err := l.appendJSON(map[string]string{"schema": "fak.other.v1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Load(); !errors.Is(err, ErrSchema) {
		t.Fatalf("a corrupt row must fail the load closed, got %v", err)
	}
	// An invalid packet must never reach disk at all.
	bad := p
	bad.ID = ""
	bad.Rev = 0
	if _, err := l.Emit(bad); !errors.Is(err, ErrRev) {
		t.Fatalf("emit must validate before writing, got %v", err)
	}
}

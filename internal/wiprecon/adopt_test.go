package wiprecon

import (
	"strings"
	"testing"
)

// baseReq is a well-formed bid for a reclaimable checkpoint. Tests mutate one field at a
// time so each case names exactly the fact under test.
func baseReq() AdoptRequest {
	return AdoptRequest{
		Session:       "crashed-1",
		Action:        ActReclaim,
		CheckpointRef: "refs/fak/wip/crashed-1",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		DeltaDigest:   "sha256:beef",
		Successor:     "rescuer-a",
		Now:           1_000_000,
		TTLSeconds:    900,
	}
}

func TestDecideAdoptGrantsAnUnclaimedReclaimCheckpoint(t *testing.T) {
	got := DecideAdopt(nil, baseReq())
	if got.Verdict != AdoptGrant {
		t.Fatalf("unclaimed RECLAIM checkpoint: got %s (%s), want %s", got.Verdict, got.Reason, AdoptGrant)
	}
	if !got.Verdict.Granted() {
		t.Fatalf("%s must authorize mutation", got.Verdict)
	}
	rec, ok := ApplyAdopt(nil, baseReq(), got.Verdict)
	if !ok {
		t.Fatal("ApplyAdopt refused to build a receipt for a GRANT")
	}
	if rec.Phase != PhaseAdopted || rec.Attempt != 1 || rec.Successor != "rescuer-a" {
		t.Fatalf("fresh receipt = %+v, want phase %s attempt 1 successor rescuer-a", rec, PhaseAdopted)
	}
	if rec.CheckpointSHA != baseReq().CheckpointSHA || rec.CheckpointRef != baseReq().CheckpointRef {
		t.Fatalf("receipt must bind ref+SHA, got ref=%q sha=%q", rec.CheckpointRef, rec.CheckpointSHA)
	}
	if len(rec.Audit) != 1 || rec.Audit[0].Event != EventAdopted {
		t.Fatalf("audit = %+v, want one %s event", rec.Audit, EventAdopted)
	}
}

// The core done-condition: two successors bid on one checkpoint, exactly one wins. The
// compare-and-swap in cmd/fak decides a genuine tie; this pins the half the decision owns
// — once a receipt exists, a DIFFERENT successor is refused, not queued behind it.
func TestDecideAdoptLetsExactlyOneOfTwoSuccessorsWin(t *testing.T) {
	first := DecideAdopt(nil, baseReq())
	if first.Verdict != AdoptGrant {
		t.Fatalf("first bidder: got %s, want %s", first.Verdict, AdoptGrant)
	}
	held, _ := ApplyAdopt(nil, baseReq(), first.Verdict)

	second := baseReq()
	second.Successor = "rescuer-b"
	second.Now = held.AdoptedAt + 5 // well inside the TTL
	got := DecideAdopt(&held, second)
	if got.Verdict != AdoptHeld {
		t.Fatalf("second bidder: got %s (%s), want %s", got.Verdict, got.Reason, AdoptHeld)
	}
	if got.Verdict.Granted() {
		t.Fatal("a HELD verdict must not authorize mutation")
	}
	if !strings.Contains(got.Reason, "rescuer-a") {
		t.Fatalf("refusal must name the holder, got %q", got.Reason)
	}
}

func TestDecideAdoptResumesTheSameSuccessorFromItsRecordedPhase(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	held = MarkPhase(held, PhaseMaterialized, held.AdoptedAt+10, EventMaterialized, "3 file(s)")
	held.Target = "/tmp/worker/crashed-1"

	again := baseReq()
	again.Now = held.RenewedAt + 5
	got := DecideAdopt(&held, again)
	if got.Verdict != AdoptResume {
		t.Fatalf("same successor: got %s (%s), want %s", got.Verdict, got.Reason, AdoptResume)
	}
	next, ok := ApplyAdopt(&held, again, got.Verdict)
	if !ok {
		t.Fatal("ApplyAdopt refused a RESUME")
	}
	if next.Phase != PhaseMaterialized {
		t.Fatalf("resume phase = %s, want the recorded %s (a resume must not redo settled work)", next.Phase, PhaseMaterialized)
	}
	if next.Target != held.Target {
		t.Fatalf("resume target = %q, want the recorded %q (a fresh target would orphan the prior bytes)", next.Target, held.Target)
	}
	if next.AdoptedAt != held.AdoptedAt {
		t.Fatalf("resume rewrote AdoptedAt %d -> %d", held.AdoptedAt, next.AdoptedAt)
	}
	if next.Attempt != held.Attempt+1 {
		t.Fatalf("resume attempt = %d, want %d", next.Attempt, held.Attempt+1)
	}
	if last := next.Audit[len(next.Audit)-1]; last.Event != EventResumed {
		t.Fatalf("last audit event = %s, want %s", last.Event, EventResumed)
	}
}

// A stale claim is takeable ONLY when liveness AND TTL both say so. Two of three is not
// enough, and this table is the guard against quietly relaxing that to one.
func TestDecideAdoptTakesOverOnlyWhenTheHolderIsGoneAndItsClaimLapsed(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	expiry := held.ExpiresAt()

	cases := []struct {
		name string
		live bool
		now  int64
		want AdoptVerdict
	}{
		{"live holder inside ttl", true, expiry - 10, AdoptHeld},
		{"live holder past ttl", true, expiry + 10, AdoptHeld},
		{"dead holder inside ttl", false, expiry - 10, AdoptHeld},
		{"dead holder past ttl", false, expiry + 10, AdoptTakeover},
		{"dead holder exactly at ttl", false, expiry, AdoptTakeover},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := baseReq()
			req.Successor = "rescuer-b"
			req.Now = tc.now
			req.IncumbentLive = tc.live
			got := DecideAdopt(&held, req)
			if got.Verdict != tc.want {
				t.Fatalf("got %s (%s), want %s", got.Verdict, got.Reason, tc.want)
			}
		})
	}
}

func TestApplyAdoptTakeoverResetsPhaseAndAuditsWhoWasDisplaced(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	held = MarkPhase(held, PhaseMaterialized, held.AdoptedAt+10, EventMaterialized, "2 file(s)")
	held.Target = "/tmp/worker/crashed-1"

	req := baseReq()
	req.Successor = "rescuer-b"
	req.Now = held.ExpiresAt() + 1
	got := DecideAdopt(&held, req)
	if got.Verdict != AdoptTakeover {
		t.Fatalf("got %s (%s), want %s", got.Verdict, got.Reason, AdoptTakeover)
	}
	next, ok := ApplyAdopt(&held, req, got.Verdict)
	if !ok {
		t.Fatal("ApplyAdopt refused a TAKEOVER")
	}
	if next.Successor != "rescuer-b" {
		t.Fatalf("successor = %q, want rescuer-b", next.Successor)
	}
	if next.Phase != PhaseAdopted {
		t.Fatalf("takeover phase = %s, want %s — the displaced successor's materialization is not this one's", next.Phase, PhaseAdopted)
	}
	if next.Target != "" {
		t.Fatalf("takeover inherited target %q; it must re-materialize from the checkpoint", next.Target)
	}
	if next.Attempt != held.Attempt+1 {
		t.Fatalf("attempt = %d, want %d (history is continuous across a handoff)", next.Attempt, held.Attempt+1)
	}
	last := next.Audit[len(next.Audit)-1]
	if last.Event != EventTakeover || last.From != "rescuer-a" || last.Actor != "rescuer-b" {
		t.Fatalf("takeover audit = %+v, want %s from rescuer-a by rescuer-b", last, EventTakeover)
	}
}

// QUARANTINE is an operator's call; a dispatcher may never convert it into a claim. SKIP
// belongs to a session that is still alive. Neither becomes adoptable because a receipt
// happens to exist, which is why the action check precedes the receipt reasoning.
func TestDecideAdoptRefusesEveryNonReclaimVerdict(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	for _, action := range []Action{ActQuarantine, ActSkip, ActDiscardWitnessed, Action("WHO_KNOWS")} {
		for _, cur := range []*Receipt{nil, &held} {
			req := baseReq()
			req.Action = action
			got := DecideAdopt(cur, req)
			if got.Verdict != AdoptRefused {
				t.Fatalf("action %s (receipt present: %v): got %s (%s), want %s",
					action, cur != nil, got.Verdict, got.Reason, AdoptRefused)
			}
			if _, ok := ApplyAdopt(cur, req, got.Verdict); ok {
				t.Fatalf("action %s: ApplyAdopt built a receipt for a refusal", action)
			}
		}
	}
}

func TestDecideAdoptRefusesWhenTheCheckpointMovedUnderTheClaim(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	req := baseReq()
	req.CheckpointSHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" // the owner re-checkpointed
	got := DecideAdopt(&held, req)
	if got.Verdict != AdoptMoved {
		t.Fatalf("got %s (%s), want %s", got.Verdict, got.Reason, AdoptMoved)
	}
	// Even for the SAME successor: the bytes it authorized are not the bytes on the ref.
	req.Successor = held.Successor
	if got := DecideAdopt(&held, req); got.Verdict != AdoptMoved {
		t.Fatalf("same successor, moved checkpoint: got %s, want %s", got.Verdict, AdoptMoved)
	}
}

func TestDecideAdoptIsIdempotentOnceTheCheckpointLanded(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	held = MarkPhase(held, PhaseLanded, held.AdoptedAt+30, EventLanded, "landed")
	held.LandedSHA = "cccccccccccccccccccccccccccccccccccccccc"

	for _, successor := range []string{"rescuer-a", "rescuer-b"} {
		req := baseReq()
		req.Successor = successor
		req.Now = held.ExpiresAt() + 10_000 // long past the TTL: settled outranks stale
		got := DecideAdopt(&held, req)
		if got.Verdict != AdoptSettled {
			t.Fatalf("successor %s: got %s (%s), want %s", successor, got.Verdict, got.Reason, AdoptSettled)
		}
		if got.Verdict.Granted() {
			t.Fatal("a settled adoption must not authorize further mutation")
		}
	}
}

func TestDecideAdoptRejectsAnIncompleteBid(t *testing.T) {
	for name, mutate := range map[string]func(*AdoptRequest){
		"no session":    func(r *AdoptRequest) { r.Session = "" },
		"no successor":  func(r *AdoptRequest) { r.Successor = "" },
		"no checkpoint": func(r *AdoptRequest) { r.CheckpointSHA = "" },
	} {
		req := baseReq()
		mutate(&req)
		if got := DecideAdopt(nil, req); got.Verdict != AdoptMalformed {
			t.Fatalf("%s: got %s, want %s", name, got.Verdict, AdoptMalformed)
		}
	}
}

func TestMarkPhaseRenewsTheClaimSoProgressKeepsIt(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	before := held.ExpiresAt()
	next := MarkPhase(held, PhaseMaterialized, held.AdoptedAt+120, EventMaterialized, "wrote 4 file(s)")
	if next.ExpiresAt() <= before {
		t.Fatalf("expiry %d did not advance past %d — a working successor would lose its own claim", next.ExpiresAt(), before)
	}
	if next.Successor != held.Successor || next.Attempt != held.Attempt {
		t.Fatalf("MarkPhase changed ownership: %+v", next)
	}
}

func TestReceiptRoundTripsThroughAnObjectBody(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	held.Target = "/tmp/worker/crashed-1"
	body, err := EncodeReceipt(held)
	if err != nil {
		t.Fatalf("EncodeReceipt: %v", err)
	}
	if !strings.Contains(body, "crashed-1") || !strings.Contains(body, receiptMarker) {
		t.Fatalf("body must be readable AND parseable, got:\n%s", body)
	}
	got, err := DecodeReceipt(body)
	if err != nil {
		t.Fatalf("DecodeReceipt: %v", err)
	}
	if got.Session != held.Session || got.Successor != held.Successor ||
		got.CheckpointSHA != held.CheckpointSHA || got.Phase != held.Phase ||
		got.Target != held.Target || got.Attempt != held.Attempt {
		t.Fatalf("round trip lost fields:\n got %+v\nwant %+v", got, held)
	}
}

// A receipt that cannot be read must be an ERROR, never a zero value: a caller that reads
// "unclaimed" from an unreadable claim is exactly the double-adoption this file prevents.
func TestDecodeReceiptFailsClosedOnAnUnreadableBody(t *testing.T) {
	for name, body := range map[string]string{
		"no marker":    "just a commit message\n",
		"bad json":     receiptMarker + "{not json\n",
		"no session":   receiptMarker + `{"successor":"a","checkpoint_sha":"x","phase":"ADOPTED"}` + "\n",
		"no successor": receiptMarker + `{"session":"s","checkpoint_sha":"x","phase":"ADOPTED"}` + "\n",
		"unknown phase": receiptMarker +
			`{"session":"s","successor":"a","checkpoint_sha":"x","phase":"MOSTLY"}` + "\n",
	} {
		if _, err := DecodeReceipt(body); err == nil {
			t.Fatalf("%s: DecodeReceipt returned no error", name)
		}
	}
}

func TestEncodeReceiptRefusesAnInvalidReceipt(t *testing.T) {
	if _, err := EncodeReceipt(Receipt{Session: "s", Successor: "a"}); err == nil {
		t.Fatal("EncodeReceipt accepted a receipt with no checkpoint SHA")
	}
}

func TestAuditTrailIsBoundedSoACrashLoopCannotGrowWithoutLimit(t *testing.T) {
	held, _ := ApplyAdopt(nil, baseReq(), AdoptGrant)
	for i := 0; i < auditMax*3; i++ {
		req := baseReq()
		req.Now = held.RenewedAt + 1
		held, _ = ApplyAdopt(&held, req, AdoptResume)
	}
	if len(held.Audit) != auditMax {
		t.Fatalf("audit length = %d, want the %d cap", len(held.Audit), auditMax)
	}
	if last := held.Audit[len(held.Audit)-1]; last.Event != EventResumed {
		t.Fatalf("the NEWEST event must survive, got %s", last.Event)
	}
}

func TestExpiredUsesTheDefaultTTLWhenTheReceiptDeclaredNone(t *testing.T) {
	r := Receipt{Session: "s", Successor: "a", CheckpointSHA: "x", Phase: PhaseAdopted, AdoptedAt: 100}
	if r.Expired(100 + DefaultTTLSeconds - 1) {
		t.Fatal("a receipt with no declared TTL expired before the default")
	}
	if !r.Expired(100 + DefaultTTLSeconds) {
		t.Fatal("a receipt with no declared TTL never expired")
	}
}

func TestAdoptArgvNamesResumeForYourOwnClaimAndNothingForAHeldRow(t *testing.T) {
	cases := []struct {
		name string
		row  ReclaimRow
		want []string
	}{
		{"unclaimed", ReclaimRow{Session: "alpha"}, []string{"wip", "reconcile", "adopt", "alpha"}},
		{"mine", ReclaimRow{Session: "alpha", AdoptedBy: "me", AdoptedMine: true}, []string{"wip", "reconcile", "resume", "alpha"}},
		{"lapsed peer claim", ReclaimRow{Session: "alpha", AdoptedBy: "peer", AdoptExpired: true}, []string{"wip", "reconcile", "adopt", "alpha"}},
		{"live peer claim", ReclaimRow{Session: "alpha", AdoptedBy: "peer"}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AdoptArgv(tc.row)
			if strings.Join(got, " ") != strings.Join(tc.want, " ") {
				t.Fatalf("argv = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRankReclaimPutsActionableRowsAheadOfHeldOnes(t *testing.T) {
	rows := []ReclaimRow{
		{Session: "held", TrunkDistance: 900, AgeHours: 99, AdoptedBy: "peer"},
		{Session: "free", TrunkDistance: 1, AgeHours: 1},
	}
	got := RankReclaim(rows)
	if got[0].Session != "free" {
		t.Fatalf("head = %s, want free — a maximally decayed row a live peer holds is the wrong head", got[0].Session)
	}
	if len(got) != 2 || got[1].Session != "held" {
		t.Fatalf("ranking dropped a row: %+v", got)
	}
}

func TestUnownedReclaimKeepsOnlyClaimableRows(t *testing.T) {
	rows := []ReclaimRow{
		{Session: "free"},
		{Session: "mine", AdoptedBy: "me", AdoptedMine: true},
		{Session: "lapsed", AdoptedBy: "peer", AdoptExpired: true},
		{Session: "held", AdoptedBy: "peer"},
	}
	got := UnownedReclaim(rows)
	if len(got) != 3 {
		t.Fatalf("got %d rows, want 3: %+v", len(got), got)
	}
	for _, r := range got {
		if r.Session == "held" {
			t.Fatal("a live peer's claim leaked into the actionable queue")
		}
	}
	if empty := UnownedReclaim([]ReclaimRow{{Session: "held", AdoptedBy: "peer"}}); empty == nil {
		t.Fatal("UnownedReclaim returned nil rather than an empty slice")
	}
}

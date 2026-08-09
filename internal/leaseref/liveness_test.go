package leaseref

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestClassifyLiveness pins the closed vocabulary branch by branch: every input shape
// maps to exactly one class, the reclaimable rule is peer-dead-only, and the evidence
// sentence names the comparison that decided.
func TestClassifyLiveness(t *testing.T) {
	now := time.Unix(10_000, 0)
	sessions := map[string]SessionDescriptor{
		"live-1":    {ID: "live-1", PCBState: "RUNNING", UpdatedAt: 9_900, TTLSecs: 1800},
		"lapsed-1":  {ID: "lapsed-1", PCBState: "RUNNING", UpdatedAt: 1_000, TTLSecs: 60},
		"stopped-1": {ID: "stopped-1", PCBState: "STOPPED", UpdatedAt: 9_990, TTLSecs: 1800},
	}
	cases := []struct {
		name         string
		rec          Record
		self         string
		wantClass    string
		wantKind     string
		wantEvidence string // substring the evidence must carry
	}{
		{
			name:         "no session binding fails closed to peer-unknown",
			rec:          Record{ID: "legacy"},
			self:         "live-1",
			wantClass:    LivenessPeerUnknown,
			wantKind:     EvidenceNoBinding,
			wantEvidence: "no session_id",
		},
		{
			name:         "own session classifies self",
			rec:          Record{ID: "mine", SessionID: "live-1"},
			self:         "live-1",
			wantClass:    LivenessSelf,
			wantKind:     EvidenceSelfSession,
			wantEvidence: `session_id "live-1" is this session`,
		},
		{
			name:         "missing descriptor fails closed to peer-unknown",
			rec:          Record{ID: "ghost", SessionID: "vanished-9"},
			self:         "live-1",
			wantClass:    LivenessPeerUnknown,
			wantKind:     EvidenceNoDescriptor,
			wantEvidence: "session-vanished-9",
		},
		{
			name:         "heartbeating session is peer-live",
			rec:          Record{ID: "theirs", SessionID: "live-1"},
			self:         "other-session",
			wantClass:    LivenessPeerLive,
			wantKind:     EvidenceHeartbeating,
			wantEvidence: "never reclaimable",
		},
		{
			name:         "lapsed heartbeat is peer-dead",
			rec:          Record{ID: "orphan", SessionID: "lapsed-1"},
			self:         "live-1",
			wantClass:    LivenessPeerDead,
			wantKind:     EvidenceHeartbeatLapsed,
			wantEvidence: "stopped heartbeating",
		},
		{
			name:         "terminal STOPPED is peer-dead even before TTL lapses",
			rec:          Record{ID: "ended", SessionID: "stopped-1"},
			self:         "live-1",
			wantClass:    LivenessPeerDead,
			wantKind:     EvidenceTerminalStopped,
			wantEvidence: "pcb_state=STOPPED",
		},
		{
			name:         "anonymous reader never classifies self",
			rec:          Record{ID: "mine", SessionID: "live-1"},
			self:         "",
			wantClass:    LivenessPeerLive,
			wantKind:     EvidenceHeartbeating,
			wantEvidence: "heartbeating",
		},
	}
	seen := map[string]bool{}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			class, kind, ev := ClassifyLiveness(c.rec, sessions, c.self, now)
			if class != c.wantClass {
				t.Fatalf("class = %q, want %q (evidence=%q)", class, c.wantClass, ev)
			}
			if kind != c.wantKind {
				t.Fatalf("evidence_kind = %q, want %q (evidence=%q)", kind, c.wantKind, ev)
			}
			if !strings.Contains(ev, c.wantEvidence) {
				t.Fatalf("evidence %q missing %q", ev, c.wantEvidence)
			}
		})
		seen[c.wantKind] = true
	}
	// Every member of the closed evidence vocabulary must be reachable — a constant no
	// branch can produce would be a routing key a caller can never receive.
	for _, k := range evidenceKinds {
		if !seen[k] {
			t.Errorf("evidence kind %q is unreachable: no case in this table produces it", k)
		}
	}
}

// TestPeerUnknownCausesAreDistinguishableByKind is the #5484 witness. Both peer-unknown
// causes classify identically on the ADMISSION contract — same class, same reclaimable —
// which is correct and must stay that way. What separates them is the remedy: an unbound
// record is the ACQUIRER's defect (fix the acquire call site; waiting never helps), a
// missing descriptor is the PUBLISHER's (start/repair `fak leaseref session-publish`).
// Before the typed kind, telling those apart meant pattern-matching English prose.
func TestPeerUnknownCausesAreDistinguishableByKind(t *testing.T) {
	now := time.Unix(10_000, 0)
	sessions := map[string]SessionDescriptor{
		"live-1": {ID: "live-1", PCBState: "RUNNING", UpdatedAt: 9_900, TTLSecs: 1800},
	}

	// Cause 1: the acquirer never bound a session.
	unboundClass, unboundKind, unboundEv := ClassifyLiveness(Record{ID: "unbound"}, sessions, "live-1", now)
	// Cause 2: the acquirer DID bind; nothing published a descriptor for it.
	ghostClass, ghostKind, ghostEv := ClassifyLiveness(Record{ID: "ghost", SessionID: "never-published"}, sessions, "live-1", now)

	// The admission contract is deliberately identical for the two.
	if unboundClass != LivenessPeerUnknown || ghostClass != LivenessPeerUnknown {
		t.Fatalf("classes = %q/%q, want both %q", unboundClass, ghostClass, LivenessPeerUnknown)
	}
	// The routing key is not.
	if unboundKind == ghostKind {
		t.Fatalf("both peer-unknown causes report evidence_kind %q — the two remedies are "+
			"still indistinguishable without parsing prose (unbound=%q ghost=%q)", unboundKind, unboundEv, ghostEv)
	}
	if unboundKind != EvidenceNoBinding {
		t.Errorf("unbound record kind = %q, want %q (remedy: the acquire call site)", unboundKind, EvidenceNoBinding)
	}
	if ghostKind != EvidenceNoDescriptor {
		t.Errorf("missing-descriptor kind = %q, want %q (remedy: the session publisher)", ghostKind, EvidenceNoDescriptor)
	}
	// Both are absences, and neither is ever reclaimable — the fail-closed rule is
	// untouched by the new channel.
	for _, k := range []string{unboundKind, ghostKind} {
		if EvidenceIsPositive(k) {
			t.Errorf("kind %q reports positive evidence, but it rests on an absence", k)
		}
	}
}

// TestSummarizeLivenessSeparatesUnwiredFeedFromRealEvidence is the #5485 witness: a live
// set in which nothing publishes the classification's input must be distinguishable from
// one whose holders were classified on evidence that was actually there. Both produce a
// complete, well-formed, correctly-computed array; only the aggregate tells them apart.
func TestSummarizeLivenessSeparatesUnwiredFeedFromRealEvidence(t *testing.T) {
	now := time.Unix(10_000, 0)

	// FLEET A — the wiring defect: no acquirer passes --session and no publisher runs, so
	// every row is an absence of evidence.
	unwired := classifyAll(t, []Record{
		{ID: "a1"},
		{ID: "a2"},
		{ID: "a3", SessionID: "never-published"},
	}, map[string]SessionDescriptor{}, "me", now)

	// FLEET B — the feed works: holders are classified on descriptors that exist, plus one
	// genuinely unclassifiable straggler.
	wiredSessions := map[string]SessionDescriptor{
		"sess-live": {ID: "sess-live", PCBState: "RUNNING", UpdatedAt: 9_950, TTLSecs: 1800},
		"sess-dead": {ID: "sess-dead", PCBState: "RUNNING", UpdatedAt: 1_000, TTLSecs: 60},
		"me":        {ID: "me", PCBState: "RUNNING", UpdatedAt: 9_950, TTLSecs: 1800},
	}
	wired := classifyAll(t, []Record{
		{ID: "b1", SessionID: "sess-live"},
		{ID: "b2", SessionID: "sess-dead"},
		{ID: "b3", SessionID: "me"},
		{ID: "b4", SessionID: "never-published"},
	}, wiredSessions, "me", now)

	unwiredSum := SummarizeLiveness(unwired)
	wiredSum := SummarizeLiveness(wired)

	// The thing the per-row view cannot say: NOTHING here rests on an observed input.
	if unwiredSum.Total != 3 || unwiredSum.PositiveEvidence != 0 || unwiredSum.Coverage != 0 {
		t.Fatalf("unwired fleet summary = %+v, want total=3 positive=0 coverage=0", unwiredSum)
	}
	if wiredSum.Total != 4 || wiredSum.PositiveEvidence != 3 || wiredSum.Coverage != 0.75 {
		t.Fatalf("wired fleet summary = %+v, want total=4 positive=3 coverage=0.75", wiredSum)
	}
	if unwiredSum.Coverage == wiredSum.Coverage {
		t.Fatalf("an unwired feed and a working one report the same coverage %v — the "+
			"wiring defect is still invisible in aggregate", unwiredSum.Coverage)
	}

	// by_evidence_kind names WHICH remedy the unwired fleet needs: two acquirers never
	// bound a session, one publisher never published.
	if got := unwiredSum.ByEvidenceKind[EvidenceNoBinding]; got != 2 {
		t.Errorf("unwired by_evidence_kind[%s] = %d, want 2", EvidenceNoBinding, got)
	}
	if got := unwiredSum.ByEvidenceKind[EvidenceNoDescriptor]; got != 1 {
		t.Errorf("unwired by_evidence_kind[%s] = %d, want 1", EvidenceNoDescriptor, got)
	}
	// Both histograms are zero-filled across the closed vocabularies, so a reader never
	// has to tell a missing key from a zero count.
	for _, k := range evidenceKinds {
		if _, ok := unwiredSum.ByEvidenceKind[k]; !ok {
			t.Errorf("by_evidence_kind is missing the key %q", k)
		}
	}
	for _, c := range livenessClasses {
		if _, ok := unwiredSum.ByClass[c]; !ok {
			t.Errorf("by_class is missing the key %q", c)
		}
	}
	if got := unwiredSum.ByClass[LivenessPeerUnknown]; got != 3 {
		t.Errorf("unwired by_class[%s] = %d, want 3", LivenessPeerUnknown, got)
	}
	if got := wiredSum.ByClass[LivenessPeerLive]; got != 1 {
		t.Errorf("wired by_class[%s] = %d, want 1", LivenessPeerLive, got)
	}

	// The empty live set is NOT the wiring signal: it reports coverage 0.0 because the
	// ratio is undefined, which is why a caller must gate on Total > 0 first.
	empty := SummarizeLiveness(nil)
	if empty.Total != 0 || empty.Coverage != 0 || empty.ByClass == nil || empty.ByEvidenceKind == nil {
		t.Fatalf("empty summary = %+v, want total=0 coverage=0 with both histograms present", empty)
	}
}

// classifyAll builds the ClassifiedLease rows ClassifyLive would produce, from literal
// inputs — the same no-I/O discipline ClassifyLiveness's own tests use, so the aggregate
// is exercised without a git seam.
func classifyAll(t *testing.T, recs []Record, sessions map[string]SessionDescriptor, self string, now time.Time) []ClassifiedLease {
	t.Helper()
	out := make([]ClassifiedLease, 0, len(recs))
	for _, r := range recs {
		class, kind, ev := ClassifyLiveness(r, sessions, self, now)
		out = append(out, ClassifiedLease{
			Record:       r,
			Node:         r.HolderNode(),
			Liveness:     class,
			Reclaimable:  class == LivenessPeerDead,
			Evidence:     ev,
			EvidenceKind: kind,
		})
	}
	return out
}

// TestFailClosedAdmissionUnchangedByEvidenceKind is the PIN, not a witness: it is expected
// to pass both before and after the diagnostic channel exists, and that is the point.
// #5484 and #5485 both state the fail-closed admission rule is CORRECT and must not
// change, so this asserts the class/reclaimable answer for every input shape is exactly
// what liveness.go's contract said before EvidenceKind was added — in particular that
// BOTH absence causes remain not-reclaimable, and that only positively-dead is reclaimable.
func TestFailClosedAdmissionUnchangedByEvidenceKind(t *testing.T) {
	now := time.Unix(10_000, 0)
	sessions := map[string]SessionDescriptor{
		"live-1":    {ID: "live-1", PCBState: "RUNNING", UpdatedAt: 9_900, TTLSecs: 1800},
		"lapsed-1":  {ID: "lapsed-1", PCBState: "RUNNING", UpdatedAt: 1_000, TTLSecs: 60},
		"stopped-1": {ID: "stopped-1", PCBState: "STOPPED", UpdatedAt: 9_990, TTLSecs: 1800},
	}
	cases := []struct {
		name            string
		rec             Record
		self            string
		wantClass       string
		wantReclaimable bool
	}{
		{"unbound record is never reclaimable", Record{ID: "legacy"}, "live-1", LivenessPeerUnknown, false},
		{"unbound record is never reclaimable to an anonymous reader", Record{ID: "legacy"}, "", LivenessPeerUnknown, false},
		{"bound-but-unpublished is never reclaimable", Record{ID: "ghost", SessionID: "vanished-9"}, "live-1", LivenessPeerUnknown, false},
		{"own lane is never reclaimable", Record{ID: "mine", SessionID: "live-1"}, "live-1", LivenessSelf, false},
		{"heartbeating peer is never reclaimable", Record{ID: "theirs", SessionID: "live-1"}, "other", LivenessPeerLive, false},
		{"lapsed heartbeat is reclaimable", Record{ID: "orphan", SessionID: "lapsed-1"}, "live-1", LivenessPeerDead, true},
		{"terminal STOPPED is reclaimable", Record{ID: "ended", SessionID: "stopped-1"}, "live-1", LivenessPeerDead, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			class, kind, ev := ClassifyLiveness(c.rec, sessions, c.self, now)
			if class != c.wantClass {
				t.Fatalf("class = %q, want %q (kind=%q evidence=%q)", class, c.wantClass, kind, ev)
			}
			// Reclaimable is peer-dead-only, exactly as ClassifyLive derives it. The
			// evidence kind must never be an input to admission.
			if reclaimable := class == LivenessPeerDead; reclaimable != c.wantReclaimable {
				t.Fatalf("reclaimable = %v, want %v (class=%q kind=%q)", reclaimable, c.wantReclaimable, class, kind)
			}
			if ev == "" {
				t.Fatalf("evidence sentence is empty for %q", c.name)
			}
		})
	}
}

// TestStoreClassifyLive drives the fold end to end on the fake git seam: live leases are
// classified against the session descriptors sharing the ref namespace, an EXPIRED lease
// is excluded (already reapable on TTL alone), and reclaimable is peer-dead-only.
func TestStoreClassifyLive(t *testing.T) {
	fake := newFakeGit()
	s := NewWithRunner(fake.run, "")
	ctx := context.Background()
	now := time.Unix(10_000, 0)

	seed := []Record{
		{ID: "lane-live", TreeGlobs: []string{"a/**"}, Holder: "A", SessionID: "sess-live", AcquiredAt: 9_000, TTLSeconds: 3600},
		{ID: "lane-dead", TreeGlobs: []string{"b/**"}, Holder: "B", SessionID: "sess-dead", AcquiredAt: 9_000, TTLSeconds: 3600},
		{ID: "lane-legacy", TreeGlobs: []string{"c/**"}, Holder: "C", AcquiredAt: 9_000, TTLSeconds: 3600},
		{ID: "lane-mine", TreeGlobs: []string{"d/**"}, Holder: "D", SessionID: "sess-me", AcquiredAt: 9_000, TTLSeconds: 3600},
		// Expired at now — must NOT appear in the classified view at all.
		{ID: "lane-expired", TreeGlobs: []string{"e/**"}, Holder: "E", SessionID: "sess-live", AcquiredAt: 100, TTLSeconds: 10},
	}
	for _, r := range seed {
		if _, err := s.Acquire(ctx, r); err != nil {
			t.Fatalf("Acquire %s: %v", r.ID, err)
		}
	}
	for _, d := range []SessionDescriptor{
		{ID: "sess-live", Host: "h1", PCBState: "RUNNING", UpdatedAt: 9_950, TTLSecs: 1800},
		{ID: "sess-dead", Host: "h2", PCBState: "RUNNING", UpdatedAt: 1_000, TTLSecs: 60},
		{ID: "sess-me", Host: "h3", PCBState: "RUNNING", UpdatedAt: 9_950, TTLSecs: 1800},
	} {
		if _, err := s.PublishSession(ctx, d); err != nil {
			t.Fatalf("PublishSession %s: %v", d.ID, err)
		}
	}

	rows, err := s.ClassifyLive(ctx, "sess-me", now)
	if err != nil {
		t.Fatalf("ClassifyLive: %v", err)
	}
	got := map[string]ClassifiedLease{}
	for _, r := range rows {
		got[r.ID] = r
	}
	if len(rows) != 4 {
		t.Fatalf("classified %d leases %v, want 4 (expired lane excluded)", len(rows), got)
	}
	if _, ok := got["lane-expired"]; ok {
		t.Fatalf("expired lease classified: %+v", got["lane-expired"])
	}
	want := map[string]struct {
		class       string
		reclaimable bool
	}{
		"lane-live":   {LivenessPeerLive, false},
		"lane-dead":   {LivenessPeerDead, true},
		"lane-legacy": {LivenessPeerUnknown, false},
		"lane-mine":   {LivenessSelf, false},
	}
	for id, w := range want {
		row, ok := got[id]
		if !ok {
			t.Fatalf("lease %s missing from classified view %v", id, got)
		}
		if row.Liveness != w.class || row.Reclaimable != w.reclaimable {
			t.Fatalf("%s = {%s reclaimable=%v}, want {%s reclaimable=%v} (evidence=%q)",
				id, row.Liveness, row.Reclaimable, w.class, w.reclaimable, row.Evidence)
		}
		if row.Evidence == "" {
			t.Fatalf("%s carries no evidence sentence", id)
		}
		// The fold must carry the typed kind through too, or the aggregate below counts
		// empty strings.
		if row.EvidenceKind == "" {
			t.Fatalf("%s carries no evidence_kind", id)
		}
	}
	// The aggregate over the same rows: three of the four rest on an observed input, the
	// legacy unbound lane does not.
	sum := SummarizeLiveness(rows)
	if sum.Total != 4 || sum.PositiveEvidence != 3 || sum.Coverage != 0.75 {
		t.Fatalf("summary = %+v, want total=4 positive=3 coverage=0.75", sum)
	}
	if got := sum.ByEvidenceKind[EvidenceNoBinding]; got != 1 {
		t.Fatalf("by_evidence_kind[%s] = %d, want 1", EvidenceNoBinding, got)
	}
}

// TestClassifyLiveEmpty pins the nothing-held shape: a non-nil empty slice, so the CLI
// emits `[]` rather than `null`.
func TestClassifyLiveEmpty(t *testing.T) {
	fake := newFakeGit()
	s := NewWithRunner(fake.run, "")
	rows, err := s.ClassifyLive(context.Background(), "me", time.Unix(10_000, 0))
	if err != nil {
		t.Fatalf("ClassifyLive: %v", err)
	}
	if rows == nil || len(rows) != 0 {
		t.Fatalf("empty view = %#v, want non-nil empty slice", rows)
	}
}

// TestAcquireFencedSessionBinding proves the session binding survives the fenced write
// paths: a fresh acquire records it, a renew ADOPTS one onto a record that lacked it,
// and a renew never REBINDS an existing one to a different session.
func TestAcquireFencedSessionBinding(t *testing.T) {
	fake := newFakeGit()
	s := NewWithRunner(fake.run, "")
	ctx := context.Background()
	now := time.Unix(10_000, 0)

	// Fresh fenced acquire without a binding, then a same-holder reacquire (renew path)
	// carrying one: the empty binding is adopted, generation unchanged.
	if _, v, err := s.AcquireFenced(ctx, Record{ID: "lane", Holder: "A", TTLSeconds: 3600}, now); err != nil || !v.OK {
		t.Fatalf("fresh acquire: v=%+v err=%v", v, err)
	}
	out, v, err := s.AcquireFenced(ctx, Record{ID: "lane", Holder: "A", TTLSeconds: 3600, SessionID: "sess-a"}, now.Add(time.Second))
	if err != nil || !v.OK {
		t.Fatalf("renew acquire: v=%+v err=%v", v, err)
	}
	if out.SessionID != "sess-a" || out.Generation != 1 {
		t.Fatalf("renew = %+v, want adopted session sess-a at generation 1", out)
	}

	// A later renew presenting a DIFFERENT session must not rebind.
	out, v, err = s.AcquireFenced(ctx, Record{ID: "lane", Holder: "A", TTLSeconds: 3600, SessionID: "sess-other"}, now.Add(2*time.Second))
	if err != nil || !v.OK {
		t.Fatalf("second renew: v=%+v err=%v", v, err)
	}
	if out.SessionID != "sess-a" {
		t.Fatalf("renew rebound session to %q, want sess-a kept", out.SessionID)
	}

	// A fresh acquire WITH a binding records it verbatim.
	out, v, err = s.AcquireFenced(ctx, Record{ID: "lane2", Holder: "B", TTLSeconds: 3600, SessionID: "sess-b"}, now)
	if err != nil || !v.OK {
		t.Fatalf("bound acquire: v=%+v err=%v", v, err)
	}
	if out.SessionID != "sess-b" {
		t.Fatalf("bound acquire session = %q, want sess-b", out.SessionID)
	}
}

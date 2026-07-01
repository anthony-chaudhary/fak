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
		wantEvidence string // substring the evidence must carry
	}{
		{
			name:         "no session binding fails closed to peer-unknown",
			rec:          Record{ID: "legacy"},
			self:         "live-1",
			wantClass:    LivenessPeerUnknown,
			wantEvidence: "no session_id",
		},
		{
			name:         "own session classifies self",
			rec:          Record{ID: "mine", SessionID: "live-1"},
			self:         "live-1",
			wantClass:    LivenessSelf,
			wantEvidence: `session_id "live-1" is this session`,
		},
		{
			name:         "missing descriptor fails closed to peer-unknown",
			rec:          Record{ID: "ghost", SessionID: "vanished-9"},
			self:         "live-1",
			wantClass:    LivenessPeerUnknown,
			wantEvidence: "session-vanished-9",
		},
		{
			name:         "heartbeating session is peer-live",
			rec:          Record{ID: "theirs", SessionID: "live-1"},
			self:         "other-session",
			wantClass:    LivenessPeerLive,
			wantEvidence: "never reclaimable",
		},
		{
			name:         "lapsed heartbeat is peer-dead",
			rec:          Record{ID: "orphan", SessionID: "lapsed-1"},
			self:         "live-1",
			wantClass:    LivenessPeerDead,
			wantEvidence: "stopped heartbeating",
		},
		{
			name:         "terminal STOPPED is peer-dead even before TTL lapses",
			rec:          Record{ID: "ended", SessionID: "stopped-1"},
			self:         "live-1",
			wantClass:    LivenessPeerDead,
			wantEvidence: "pcb_state=STOPPED",
		},
		{
			name:         "anonymous reader never classifies self",
			rec:          Record{ID: "mine", SessionID: "live-1"},
			self:         "",
			wantClass:    LivenessPeerLive,
			wantEvidence: "heartbeating",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			class, ev := ClassifyLiveness(c.rec, sessions, c.self, now)
			if class != c.wantClass {
				t.Fatalf("class = %q, want %q (evidence=%q)", class, c.wantClass, ev)
			}
			if !strings.Contains(ev, c.wantEvidence) {
				t.Fatalf("evidence %q missing %q", ev, c.wantEvidence)
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

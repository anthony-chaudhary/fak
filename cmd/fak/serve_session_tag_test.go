package main

// serve_session_tag_test.go — #5640 (epic #5632): the witness that the broadcast tag
// registry has a PRODUCER, driven through the real admission hook and the real fleet
// directive applier.
//
// The bug these cover is not a wrong answer, it is a missing call: the registry, the
// selector resolver and the fail-closed rule all shipped, and nothing wrote to the
// registry, so every --lane/--wave/--label fleet directive resolved to zero sessions and
// reported it in the one way an operator cannot distinguish from an empty fleet. Delete
// the tagServedSessionAdmit call from decideSession and
// TestFleetDirectiveLaneSelectorAffectsSessions goes red — that is the point of it.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// admitTaggedSession drives one trace through the REAL admission hook under the process
// routing identity lane names, which is the only way this test can witness the tagger
// rather than re-implement it. The tag is process-global, so it is always cleaned up.
func admitTaggedSession(t *testing.T, ctx context.Context, trace, lane string) {
	t.Helper()
	t.Setenv(serveSessionLaneEnv, lane)
	decideSession(ctx, trace)
	t.Cleanup(func() { sessionctl.ClearSessionTag(trace) })
}

// TestFleetDirectiveLaneSelectorAffectsSessions is the issue's done condition, end to
// end: one serve process holding sessions on two lanes takes a --lane scoped directive
// on exactly the sessions carrying that lane, with a non-zero Affected that matches —
// and a directive naming a lane nobody carries comes back distinguishable from success.
func TestFleetDirectiveLaneSelectorAffectsSessions(t *testing.T) {
	ctx := context.Background()
	const alpha, beta = "lane5640alpha", "lane5640beta"

	admitTaggedSession(t, ctx, "s5640-alpha-1", alpha)
	admitTaggedSession(t, ctx, "s5640-alpha-2", alpha)
	admitTaggedSession(t, ctx, "s5640-beta-1", beta)

	// Admission must have populated the registry — the assertion that fails first, and
	// most legibly, if the producer is removed again.
	for _, trace := range []string{"s5640-alpha-1", "s5640-alpha-2", "s5640-beta-1"} {
		if _, tagged := sessionctl.SessionTag(trace); !tagged {
			t.Fatalf("%s is untagged after admission; nothing writes the broadcast tag registry", trace)
		}
	}

	ap := &fleetBusApplier{tbl: serveSessions, native: true, ctx: ctx}
	out := ap.Apply(fleetBusDirective("pause", fleetbus.Selector{Lane: alpha}))
	if out.Status != fleetbus.AckApplied {
		t.Fatalf("lane=%s pause: status=%q reason=%q detail=%q, want applied", alpha, out.Status, out.Reason, out.Detail)
	}
	if out.Affected != 2 {
		t.Fatalf("lane=%s pause: affected=%d, want 2", alpha, out.Affected)
	}
	for _, trace := range []string{"s5640-alpha-1", "s5640-alpha-2"} {
		if got := serveSessions.Get(trace).Run; got != session.Paused {
			t.Fatalf("%s run = %v, want paused", trace, got)
		}
	}
	// The expensive direction of the old bug was a scoped op reporting success over the
	// wrong set; the other lane must be untouched.
	if got := serveSessions.Get("s5640-beta-1").Run; got != session.Running {
		t.Fatalf("s5640-beta-1 run = %v, want running (a lane=%s directive must not reach it)", got, alpha)
	}

	// A lane nobody carries is a REFUSAL naming what kind of nothing this was, not an
	// "applied, 0 affected" ack.
	ghost := ap.Apply(fleetBusDirective("pause", fleetbus.Selector{Lane: "lane5640ghost"}))
	if ghost.Status != fleetbus.AckRefused {
		t.Fatalf("unknown lane: status=%q affected=%d, want refused", ghost.Status, ghost.Affected)
	}
	if ghost.Affected != 0 {
		t.Fatalf("unknown lane: affected=%d, want 0", ghost.Affected)
	}
	if !strings.Contains(ghost.Detail, "lane=lane5640ghost") {
		t.Fatalf("unknown lane detail = %q, want it to name the selector nobody carried", ghost.Detail)
	}
}

// TestServeSessionAdmissionTagsAndClearsBroadcastTag pins the three rules the producer
// has to follow: it tags from the process identity, it FILLS rather than overwrites (it
// runs once per turn, not once per session), and the tag dies with the session.
func TestServeSessionAdmissionTagsAndClearsBroadcastTag(t *testing.T) {
	ctx := context.Background()

	// (1) Admission tags from the process's declared routing identity, labels included.
	t.Setenv(serveSessionLaneEnv, "tagunit-lane")
	t.Setenv(serveSessionWaveEnv, "tagunit-wave")
	t.Setenv(serveSessionLabelsEnv, " dogfood , nightrun ,, ")
	trace := "s5640-unit-1"
	t.Cleanup(func() { sessionctl.ClearSessionTag(trace) })
	decideSession(ctx, trace)
	meta, tagged := sessionctl.SessionTag(trace)
	if !tagged {
		t.Fatal("admission did not tag the trace")
	}
	if meta.Lane != "tagunit-lane" || meta.Wave != "tagunit-wave" {
		t.Fatalf("meta = %+v, want lane/wave from the environment", meta)
	}
	if strings.Join(meta.Labels, ",") != "dogfood,nightrun" {
		t.Fatalf("labels = %v, want the two non-blank tokens", meta.Labels)
	}

	// (2) A later turn must not re-stamp the process identity over a more precise tag a
	// spawn site set. Admission fills gaps; it never overwrites.
	sessionctl.TagSession(trace, sessionctl.BroadcastMeta{Lane: "precise-lane"})
	decideSession(ctx, trace)
	if meta, _ := sessionctl.SessionTag(trace); meta.Lane != "precise-lane" {
		t.Fatalf("lane = %q after a second turn, want the spawn site's precise-lane preserved", meta.Lane)
	}

	// (3) The tag dies with the session. An operator stop through the control route is
	// the end that produces no following admission, so it has to clear on its own —
	// otherwise a selector keeps matching a trace that can no longer take the op.
	if _, _, err := controlSession(ctx, trace, "run", gateway.SessionControlRequest{Run: "stopped", Reason: "test"}); err != nil {
		t.Fatalf("controlSession stop: %v", err)
	}
	if _, tagged := sessionctl.SessionTag(trace); tagged {
		t.Fatal("tag survived the session; a stopped trace must leave the registry")
	}

	// (4) An unconfigured process declares no identity and tags nothing — the
	// fail-closed default is preserved, so turning selection on stays an explicit act.
	t.Setenv(serveSessionLaneEnv, "")
	t.Setenv(serveSessionWaveEnv, "")
	t.Setenv(serveSessionLabelsEnv, "")
	bare := "s5640-unit-untagged"
	t.Cleanup(func() { sessionctl.ClearSessionTag(bare) })
	decideSession(ctx, bare)
	if _, tagged := sessionctl.SessionTag(bare); tagged {
		t.Fatal("a serve with no declared routing identity tagged a session anyway")
	}
}

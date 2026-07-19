package superloop

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// followonEm builds one emitted-ref evidence row the way the shell hands it in —
// durable live-issue state, never a member narration.
func followonEm(ref string, resolved, open, advanced bool) FollowonEmission {
	return FollowonEmission{Ref: ref, Resolved: resolved, Open: open, Advanced: advanced}
}

// TestClassifyFollowonOrphanedVsAdvancingVsUnknown pins the closed verdict fold
// (#4957) across the states the DoD names — an emitted issue idle past cadence is
// ORPHANED, an advanced/closed emission reads clean, an unreadable emission fails
// closed to UNKNOWN (never a fabricated orphan), and no emissions leaves the axis
// unread. The reason token is asserted literally so operator tooling and the relay
// constant can never drift apart silently.
func TestClassifyFollowonOrphanedVsAdvancingVsUnknown(t *testing.T) {
	m := Member{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput"}

	cases := []struct {
		name       string
		read       FollowonRead
		want       MemberFollowon
		wantReason string
	}{
		{
			// The axis-not-read zero: a loop with no emissions owes nothing HERE —
			// surface-only, never weighed.
			name: "no_emissions_axis_not_read",
			read: FollowonRead{},
			want: "",
		},
		{
			// The DoD headline: the tick emitted "#N", the issue is OPEN with no
			// advance within the cadence window — ORPHANED with the closed token.
			name:       "open_idle_past_cadence_is_orphaned",
			read:       FollowonRead{Emissions: []FollowonEmission{followonEm("#4321", true, true, false)}},
			want:       FollowonOrphaned,
			wantReason: relay.ReasonOrphanedFollowon,
		},
		{
			// An emitted issue someone advanced within the window is being carried.
			name: "open_advanced_within_cadence_is_clean",
			read: FollowonRead{Emissions: []FollowonEmission{followonEm("#4321", true, true, true)}},
			want: FollowonAdvancing,
		},
		{
			// A closed emission is done — clean regardless of the advance flag.
			name: "closed_is_clean",
			read: FollowonRead{Emissions: []FollowonEmission{followonEm("#4321", true, false, false)}},
			want: FollowonAdvancing,
		},
		{
			// One carried emission does not excuse an orphaned sibling.
			name: "one_orphan_taints_the_member",
			read: FollowonRead{Emissions: []FollowonEmission{
				followonEm("#100", true, true, true),
				followonEm("#101", true, true, false),
			}},
			want:       FollowonOrphaned,
			wantReason: relay.ReasonOrphanedFollowon,
		},
		{
			// FAIL CLOSED: any unreadable emission makes the whole read UNKNOWN —
			// even alongside a positively-witnessed orphan. An orphan is never
			// fabricated (or upheld) from a read that could not resolve, the same
			// missing-ledger asymmetry relay.ReadVerifiedProgress keeps.
			name: "any_unresolved_fails_closed_to_unknown",
			read: FollowonRead{Emissions: []FollowonEmission{
				followonEm("#101", true, true, false),
				followonEm("#102", false, false, false),
			}},
			want: FollowonUnknown,
		},
		{
			name: "all_unresolved_is_unknown",
			read: FollowonRead{Emissions: []FollowonEmission{followonEm("#7", false, false, false)}},
			want: FollowonUnknown,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := ClassifyFollowon(m, tc.read)
			if got != tc.want || reason != tc.wantReason {
				t.Fatalf("ClassifyFollowon(%s) = (%q, %q), want (%q, %q)",
					tc.name, got, reason, tc.want, tc.wantReason)
			}
		})
	}

	// Pin the wire spelling of the closed token: operator tooling keys on it.
	if relay.ReasonOrphanedFollowon != "RELAY_ORPHANED_FOLLOWON" {
		t.Fatalf("closed reason token moved: %q", relay.ReasonOrphanedFollowon)
	}
}

// TestWalkRanksOrphanedFollowonAsDebt is the #4957 regression the issue names: a
// loop whose emitted issue sits OPEN with no advance past the cadence window must
// produce a worklist item and block Satisfied — where before the follow-on verdict
// it read clean (progress at the loop grain, zero progress at the fleet grain). It
// also pins the ranking claim: the orphaned loop out-ranks a clean live leaf (which
// produces no work item at all) and carries a concrete chase/redirect action naming
// the closed reason.
func TestWalkRanksOrphanedFollowonAsDebt(t *testing.T) {
	s := Super{Name: "drain-test", Members: []Member{
		{Kind: KindLoop, Ref: "cadence", Why: "clean live leaf"},
		{Kind: KindLoop, Ref: "dispatch", Why: "emitting loop"},
	}}
	statuses := []MemberStatus{
		{Member: s.Members[0], Measured: true, Progress: ProgressAdvancing},
		{Member: s.Members[1], Measured: true, Progress: ProgressAdvancing,
			FollowOn: FollowonOrphaned, FollowOnReason: relay.ReasonOrphanedFollowon,
			Detail: "emitted #4321 open with no advance past cadence"},
	}
	rep := Walk(s, statuses)

	if rep.Satisfied {
		t.Fatal("a walk with an ORPHANED-FOLLOWON member read Satisfied — emitted-work-nobody-advances reads clean again (#4957 regression)")
	}
	if rep.Orphaned != 1 {
		t.Fatalf("Orphaned = %d, want 1", rep.Orphaned)
	}
	if rep.Finding != "superloop_orphaned" {
		t.Fatalf("Finding = %q, want superloop_orphaned (dark=%d spinning=%d debt=%d)", rep.Finding, rep.Dark, rep.Spinning, rep.TotalDebt)
	}
	if !strings.Contains(rep.Reason, relay.ReasonOrphanedFollowon) {
		t.Fatalf("Reason %q does not name the closed token %s", rep.Reason, relay.ReasonOrphanedFollowon)
	}
	if len(rep.Worklist) != 1 {
		t.Fatalf("worklist has %d item(s), want exactly 1 (the orphaned loop; the clean live leaf is nothing to enter): %+v", len(rep.Worklist), rep.Worklist)
	}
	it := rep.Worklist[0]
	if it.Member.Ref != "dispatch" || it.FollowOn != FollowonOrphaned || it.FollowOnReason != relay.ReasonOrphanedFollowon {
		t.Fatalf("worklist head = %q follow_on=%q reason=%q, want the orphaned dispatch loop with %s",
			it.Member.Ref, it.FollowOn, it.FollowOnReason, relay.ReasonOrphanedFollowon)
	}
	if !strings.Contains(it.Action, relay.ReasonOrphanedFollowon) {
		t.Fatalf("action %q does not name the closed token %s — the redirect entry must be checkable", it.Action, relay.ReasonOrphanedFollowon)
	}
	if !strings.HasPrefix(it.Detail, "ORPHANED-FOLLOWON — ") {
		t.Fatalf("detail %q does not lead with the ORPHANED-FOLLOWON marker", it.Detail)
	}
}

// TestWalkOrphanedBlocksEvenWithZeroDebt pins that the pure fold does not depend on
// any shell-side debt term: an orphaned member with debt 0 still lands in the
// debt-bearing worst-first band (ahead of clean live leaves), still enters the
// worklist, and still blocks Satisfied.
func TestWalkOrphanedBlocksEvenWithZeroDebt(t *testing.T) {
	s := Super{Name: "t", Members: []Member{{Kind: KindLoop, Ref: "dispatch"}}}
	st := MemberStatus{Member: s.Members[0], Measured: true,
		FollowOn: FollowonOrphaned, FollowOnReason: relay.ReasonOrphanedFollowon}
	if !workEligible(st) {
		t.Fatal("an ORPHANED-FOLLOWON member is not work-eligible")
	}
	if got := tier(st); got != 1 {
		t.Fatalf("tier(orphaned, debt 0) = %d, want 1 (the debt band, ahead of clean live 3)", got)
	}
	rep := Walk(s, []MemberStatus{st})
	if rep.Satisfied || len(rep.Worklist) != 1 {
		t.Fatalf("satisfied=%v worklist=%d, want unsatisfied with 1 item", rep.Satisfied, len(rep.Worklist))
	}
}

// TestFollowonDistinctFromProgress pins that the follow-on axis is a DISTINCT
// dimension on MemberStatus, never an overload of the progress verdict: a member
// can be ORPHANED while ADVANCING at its own grain (the exact state #4957 exists to
// surface — progress at the loop grain, zero at the fleet grain), and SPINNING
// without being orphaned. Each axis drives its own action.
func TestFollowonDistinctFromProgress(t *testing.T) {
	orphanedNotSpinning := MemberStatus{
		Member:   Member{Kind: KindLoop, Ref: "emitter"},
		Measured: true, Progress: ProgressAdvancing,
		FollowOn: FollowonOrphaned, FollowOnReason: relay.ReasonOrphanedFollowon,
	}
	if orphanedNotSpinning.Progress == ProgressSpinning {
		t.Fatal("test fixture drifted: the orphaned member must not be spinning")
	}
	if !workEligible(orphanedNotSpinning) || tier(orphanedNotSpinning) != 1 {
		t.Fatalf("orphaned-but-advancing member: eligible=%v tier=%d, want eligible tier 1 — the follow-on axis must bite without the progress axis",
			workEligible(orphanedNotSpinning), tier(orphanedNotSpinning))
	}
	if a := actionFor(orphanedNotSpinning); !strings.Contains(a, relay.ReasonOrphanedFollowon) || strings.Contains(a, relay.ReasonNoProgress) {
		t.Fatalf("action %q must name %s and not %s — the orphaned action, not the spinning one",
			a, relay.ReasonOrphanedFollowon, relay.ReasonNoProgress)
	}

	spinningNotOrphaned := MemberStatus{
		Member:   Member{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput"},
		Measured: true, Progress: ProgressSpinning, ProgressReason: relay.ReasonNoProgress,
	}
	if spinningNotOrphaned.FollowOn != "" {
		t.Fatal("test fixture drifted: the spinning member must leave the follow-on axis unread")
	}
	if !workEligible(spinningNotOrphaned) || tier(spinningNotOrphaned) != 1 {
		t.Fatalf("spinning member with follow-on unread: eligible=%v tier=%d, want eligible tier 1",
			workEligible(spinningNotOrphaned), tier(spinningNotOrphaned))
	}
	if a := actionFor(spinningNotOrphaned); !strings.Contains(a, relay.ReasonNoProgress) || strings.Contains(a, relay.ReasonOrphanedFollowon) {
		t.Fatalf("action %q must name %s and not %s — the spinning action, not the orphaned one",
			a, relay.ReasonNoProgress, relay.ReasonOrphanedFollowon)
	}
}

// TestWalkFollowonUnknownStaysSurfaceOnly pins the fail-closed-but-honest edge: a
// member whose emission read was unresolvable (FollowonUnknown) is surfaced on the
// status but gates nothing — no debt, no worklist item, Satisfied stays true. An
// orphan is NEVER fabricated from an absence; a walk that could not read the axis
// does not slander the member.
func TestWalkFollowonUnknownStaysSurfaceOnly(t *testing.T) {
	s := Super{Name: "t", Members: []Member{{Kind: KindLoop, Ref: "dispatch"}}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, FollowOn: FollowonUnknown},
	})
	if !rep.Satisfied {
		t.Fatal("an UNKNOWN follow-on read blocked Satisfied — surface-only means it never gates")
	}
	if rep.Orphaned != 0 || len(rep.Worklist) != 0 {
		t.Fatalf("orphaned=%d worklist=%d, want 0/0", rep.Orphaned, len(rep.Worklist))
	}
	if rep.Statuses[0].FollowOn != FollowonUnknown {
		t.Fatalf("status follow_on = %q, want it SURFACED as %q", rep.Statuses[0].FollowOn, FollowonUnknown)
	}
}

// TestFollowonVerdictHasNoClaimedField walks the follow-on-verdict types
// reflectively and refuses any field whose name suggests a self-reported claim —
// the no-claimed invariant the relay Baton pins
// (relay.TestVerifiedProgressHasNoClaimedField), extended over the #4957 surface.
// The verdict is assembled ONLY from durable issue/artifact state (emitted refs +
// live open/advance state); there is no field where a member asserts it.
func TestFollowonVerdictHasNoClaimedField(t *testing.T) {
	banned := []string{"claimed", "claim", "selfreport", "self_report", "asserted", "trustme"}
	var check func(t *testing.T, tp reflect.Type, path string, seen map[reflect.Type]bool)
	check = func(t *testing.T, tp reflect.Type, path string, seen map[reflect.Type]bool) {
		for tp.Kind() == reflect.Ptr || tp.Kind() == reflect.Slice || tp.Kind() == reflect.Array || tp.Kind() == reflect.Map {
			tp = tp.Elem()
		}
		if tp.Kind() != reflect.Struct || seen[tp] {
			return
		}
		seen[tp] = true
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			lower := strings.ToLower(f.Name)
			for _, b := range banned {
				if strings.Contains(lower, b) {
					t.Errorf("%s.%s: field name suggests a self-reported claim (%q)", path, f.Name, b)
				}
			}
			check(t, f.Type, path+"."+f.Name, seen)
		}
	}
	seen := map[reflect.Type]bool{}
	check(t, reflect.TypeOf(FollowonRead{}), "FollowonRead", seen)
	check(t, reflect.TypeOf(FollowonEmission{}), "FollowonEmission", seen)
	check(t, reflect.TypeOf(MemberStatus{}), "MemberStatus", seen)
	check(t, reflect.TypeOf(WorkItem{}), "WorkItem", seen)
}

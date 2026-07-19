package superloop

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// verifiedSteps builds a ledger-verified progress read with n re-verifiable steps —
// the shape relay.ReadVerifiedProgress returns from a real intent ledger.
func verifiedSteps(n int) relay.VerifiedProgress {
	steps := make([]relay.ProgressStep, n)
	for i := range steps {
		steps[i] = relay.ProgressStep{Ref: fmt.Sprintf("#%d", 100+i), Note: "witnessed"}
	}
	return relay.VerifiedProgress{Verdict: relay.ProgressVerified, Steps: steps}
}

// TestClassifyProgressSpinningVsAdvancingVsUnmeasured pins the closed verdict fold
// (#4956) across the three states the DoD names — SPINNING vs healthy (advancing) vs
// UNMEASURED — plus the watch-member idle rule. The reason tokens are asserted
// literally so the dos.toml [reasons] rows and the relay constants can never drift
// apart silently.
func TestClassifyProgressSpinningVsAdvancingVsUnmeasured(t *testing.T) {
	throughput := Member{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput"}
	watch := Member{Kind: KindLoop, Ref: "cadence"} // WorkNeutral by classifyWork
	idleProven := relay.IdleObservation{Invariant: relay.WatchHolds, PendingKnown: true, PendingAdmitted: 0}

	cases := []struct {
		name       string
		m          Member
		ticking    bool
		read       ProgressRead
		want       MemberProgress
		wantReason string
	}{
		{
			// The DoD headline: live on cadence, ledger verified, high-water did not
			// rise -> SPINNING with the closed relay reason.
			name: "throughput_ticking_no_advance_spins",
			m:    throughput, ticking: true,
			read:       ProgressRead{Baseline: verifiedSteps(2), Now: verifiedSteps(2)},
			want:       ProgressSpinning,
			wantReason: relay.ReasonNoProgress,
		},
		{
			// A verified-empty ledger is an honest zero (relay progress_file.go), and
			// zero verified steps is zero advance: still SPINNING for a throughput loop.
			name: "throughput_verified_empty_ledger_spins",
			m:    throughput, ticking: true,
			read:       ProgressRead{Now: verifiedSteps(0)},
			want:       ProgressSpinning,
			wantReason: relay.ReasonNoProgress,
		},
		{
			// The exact NoProgressEscape.Advances rule: the verified step count rose
			// past the baseline high-water -> healthy.
			name: "throughput_advancing_is_healthy",
			m:    throughput, ticking: true,
			read: ProgressRead{Baseline: verifiedSteps(2), Now: verifiedSteps(3)},
			want: ProgressAdvancing,
		},
		{
			// An unverifiable read (no ledger anchor / unreachable ledger) is treated
			// as NO progress — never clean — but stays surface-only: unmeasured, no
			// reason token, never a fabricated zero.
			name: "unverifiable_read_is_unmeasured_surface_only",
			m:    throughput, ticking: true,
			read: ProgressRead{Now: relay.VerifiedProgress{Verdict: relay.ProgressUnknown}},
			want: ProgressUnmeasured,
		},
		{
			// A non-ticking (dark) loop is judged by the LIVENESS verdict; the
			// progress axis stays unread rather than double-counted.
			name: "not_ticking_reads_no_progress_axis",
			m:    throughput, ticking: false,
			read: ProgressRead{Now: verifiedSteps(0)},
			want: "",
		},
		{
			// The watch-member rule: a WorkNeutral cadence with no advance whose idle
			// is POSITIVELY proven (invariant holds + zero admitted pending) parks
			// benign with the closed relay token — never an alarm, never debt.
			name: "watch_member_proven_idle_parks_benign",
			m:    watch, ticking: true,
			read:       ProgressRead{Now: verifiedSteps(0), Idle: idleProven},
			want:       ProgressIdleParked,
			wantReason: relay.ReasonIdleParked,
		},
		{
			// Fail closed: an UNPROVEN idle (unknown invariant) earns no benign park —
			// but a neutral member is not slandered as SPINNING either; only a
			// throughput member trips RELAY_NO_PROGRESS.
			name: "watch_member_unproven_idle_stays_unproven",
			m:    watch, ticking: true,
			read: ProgressRead{Now: verifiedSteps(0), Idle: relay.IdleObservation{Invariant: relay.WatchUnknown}},
			want: ProgressIdleUnproven,
		},
		{
			// Pending admitted work also defeats the park: idle means NOTHING owed.
			name: "watch_member_pending_work_defeats_park",
			m:    watch, ticking: true,
			read: ProgressRead{Now: verifiedSteps(0), Idle: relay.IdleObservation{Invariant: relay.WatchHolds, PendingKnown: true, PendingAdmitted: 2}},
			want: ProgressIdleUnproven,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := ClassifyProgress(tc.m, tc.ticking, tc.read)
			if got != tc.want || reason != tc.wantReason {
				t.Fatalf("ClassifyProgress(%s ticking=%v) = (%q, %q), want (%q, %q)",
					tc.m.Ref, tc.ticking, got, reason, tc.want, tc.wantReason)
			}
		})
	}

	// Pin the wire spelling of the two closed tokens: the dos.toml [reasons] rows and
	// any operator tooling key on these exact strings.
	if relay.ReasonNoProgress != "RELAY_NO_PROGRESS" || relay.ReasonIdleParked != "RELAY_IDLE_PARKED" {
		t.Fatalf("closed reason tokens moved: %q / %q", relay.ReasonNoProgress, relay.ReasonIdleParked)
	}
}

// TestWalkRanksSpinningLoopAsDebt is the #4956 regression the issue names: a loop
// ticked within cadence with no advanced verified step must produce a worklist item
// and block Satisfied — where before the progress verdict it read clean. It also
// pins the ranking claim: the spinning throughput loop out-ranks a clean live leaf
// (which produces no work item at all) and carries a revive/redirect action naming
// the closed reason.
func TestWalkRanksSpinningLoopAsDebt(t *testing.T) {
	s := Super{Name: "drain-test", Members: []Member{
		{Kind: KindLoop, Ref: "cadence", Why: "clean live leaf"},
		{Kind: KindLoop, Ref: "dispatch", Why: "throughput loop"},
	}}
	statuses := []MemberStatus{
		{Member: s.Members[0], Measured: true, Progress: ProgressAdvancing},
		{Member: s.Members[1], Measured: true, Debt: 1, Progress: ProgressSpinning, ProgressReason: relay.ReasonNoProgress,
			Detail: "state live, 12 run(s), keep 0.00"},
	}
	rep := Walk(s, statuses)

	if rep.Satisfied {
		t.Fatal("a walk with a SPINNING member read Satisfied — live-but-producing-nothing reads clean again (#4956 regression)")
	}
	if rep.Spinning != 1 {
		t.Fatalf("Spinning = %d, want 1", rep.Spinning)
	}
	if rep.Finding != "superloop_spinning" {
		t.Fatalf("Finding = %q, want superloop_spinning (dark=%d debt=%d)", rep.Finding, rep.Dark, rep.TotalDebt)
	}
	if !strings.Contains(rep.Reason, relay.ReasonNoProgress) {
		t.Fatalf("Reason %q does not name the closed token %s", rep.Reason, relay.ReasonNoProgress)
	}
	if len(rep.Worklist) != 1 {
		t.Fatalf("worklist has %d item(s), want exactly 1 (the spinning loop; the clean live leaf is nothing to enter): %+v", len(rep.Worklist), rep.Worklist)
	}
	it := rep.Worklist[0]
	if it.Member.Ref != "dispatch" || it.Progress != ProgressSpinning || it.ProgressReason != relay.ReasonNoProgress {
		t.Fatalf("worklist head = %q progress=%q reason=%q, want the spinning dispatch loop with %s",
			it.Member.Ref, it.Progress, it.ProgressReason, relay.ReasonNoProgress)
	}
	if !strings.Contains(it.Action, "revive/redirect") || !strings.Contains(it.Action, relay.ReasonNoProgress) {
		t.Fatalf("action %q is not a revive/redirect naming %s", it.Action, relay.ReasonNoProgress)
	}
	if !strings.HasPrefix(it.Detail, "SPINNING — ") {
		t.Fatalf("detail %q does not lead with the SPINNING marker", it.Detail)
	}
}

// TestWalkSpinningBlocksEvenWithZeroDebt pins that the pure fold does not depend on
// the shell's debt term: a spinning member with debt 0 still lands in the
// debt-bearing worst-first band (ahead of clean live leaves), still enters the
// worklist, and still blocks Satisfied.
func TestWalkSpinningBlocksEvenWithZeroDebt(t *testing.T) {
	s := Super{Name: "t", Members: []Member{{Kind: KindLoop, Ref: "dispatch"}}}
	st := MemberStatus{Member: s.Members[0], Measured: true, Progress: ProgressSpinning, ProgressReason: relay.ReasonNoProgress}
	if !workEligible(st) {
		t.Fatal("a SPINNING member is not work-eligible")
	}
	if got := tier(st); got != 1 {
		t.Fatalf("tier(spinning, debt 0) = %d, want 1 (the debt band, ahead of clean live 3)", got)
	}
	rep := Walk(s, []MemberStatus{st})
	if rep.Satisfied || len(rep.Worklist) != 1 {
		t.Fatalf("satisfied=%v worklist=%d, want unsatisfied with 1 item", rep.Satisfied, len(rep.Worklist))
	}
}

// TestWalkUnmeasuredProgressStaysSurfaceOnly pins the fail-closed-but-honest edge:
// a member whose PROGRESS read was unverifiable (ProgressUnmeasured) is surfaced on
// the status but gates nothing — no debt, no worklist item, Satisfied stays true.
// Never fabricate a zero (the nightIssueProgress posture).
func TestWalkUnmeasuredProgressStaysSurfaceOnly(t *testing.T) {
	s := Super{Name: "t", Members: []Member{{Kind: KindLoop, Ref: "dispatch"}}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Progress: ProgressUnmeasured},
	})
	if !rep.Satisfied {
		t.Fatal("an unmeasured progress read blocked Satisfied — surface-only means it never gates")
	}
	if rep.Spinning != 0 || len(rep.Worklist) != 0 {
		t.Fatalf("spinning=%d worklist=%d, want 0/0", rep.Spinning, len(rep.Worklist))
	}
	if rep.Statuses[0].Progress != ProgressUnmeasured {
		t.Fatalf("status progress = %q, want it SURFACED as %q", rep.Statuses[0].Progress, ProgressUnmeasured)
	}
}

// TestProgressVerdictHasNoClaimedField walks the progress-verdict types reflectively
// and refuses any field whose name suggests a self-reported claim — the no-claimed
// invariant the relay Baton pins, extended over the #4956 surface. Progress is
// ledger-verified or absent; there is no field where an agent asserts it.
func TestProgressVerdictHasNoClaimedField(t *testing.T) {
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
	check(t, reflect.TypeOf(ProgressRead{}), "ProgressRead", seen)
	check(t, reflect.TypeOf(MemberStatus{}), "MemberStatus", seen)
	check(t, reflect.TypeOf(WorkItem{}), "WorkItem", seen)
}

// TestSpinningReasonTokensAreClassified is the dos_check_reason conformance witness
// for the two tokens the progress verdict emits: both must resolve against declared
// dos.toml [reasons] tables (RELAY_NO_PROGRESS as a refusable OPERATOR_GATE;
// RELAY_IDLE_PARKED declared advisory — a benign park is not a refusal), or
// dos_check_reason grades them UNCLASSIFIED prose. The helpers mirror
// internal/adjudicator's reversibility_dosreason_test.go (packages cannot share
// test code).
func TestSpinningReasonTokensAreClassified(t *testing.T) {
	content := readRepoDosTomlForSpinning(t)
	for _, tok := range []string{relay.ReasonNoProgress, relay.ReasonIdleParked} {
		header := "[reasons." + tok + "]"
		if !strings.Contains(content, header) {
			t.Errorf("progress verdict emits token %q with no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED)", tok, header)
		}
	}
	block := spinningDosBlock(content, "[reasons."+relay.ReasonNoProgress+"]")
	if !strings.Contains(block, "refusal  = true") && !strings.Contains(block, "refusal = true") {
		t.Errorf("reason %q is declared but not marked refusal = true — a supervisor could not refuse on it", relay.ReasonNoProgress)
	}
}

// spinningDosBlock scopes content to one [reasons.<TOKEN>] table (header to next
// top-level section), so field assertions cannot match a sibling's fields.
func spinningDosBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// readRepoDosTomlForSpinning reads the repo-root dos.toml relative to this test's
// own source path, independent of the test working directory.
func readRepoDosTomlForSpinning(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	b, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "..", "..", "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}

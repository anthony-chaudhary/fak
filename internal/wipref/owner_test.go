package wipref

import (
	"reflect"
	"testing"
	"time"
)

const testNow = int64(1_700_000_000)

// ago builds a checkpoint timestamp d before testNow.
func ago(d time.Duration) int64 { return testNow - int64(d/time.Second) }

// TestOwnerOfPathStates pins the whole closed vocabulary, including the two rungs that
// carry the safety argument: liveness beats a stale clock, and a tree-wide capture tie is
// never broken in favour of one claimant.
func TestOwnerOfPathStates(t *testing.T) {
	cases := []struct {
		name       string
		claims     []Claim
		wantState  OwnState
		wantOwner  string
		wantOwners []string
		wantReason string
	}{
		{
			name:       "no claim at all is the at-risk creation",
			claims:     nil,
			wantState:  OwnUnclaimed,
			wantReason: ReasonNoFreshClaim,
		},
		{
			name:       "one live session owns it",
			claims:     []Claim{{Session: "s1", CheckpointedAt: ago(10 * time.Minute), Live: true}},
			wantState:  OwnClaimedLive,
			wantOwner:  "s1",
			wantReason: ReasonSessionLive,
		},
		{
			name:       "dead session inside the TTL keeps the claim",
			claims:     []Claim{{Session: "s1", CheckpointedAt: ago(30 * time.Minute)}},
			wantState:  OwnClaimedLive,
			wantOwner:  "s1",
			wantReason: ReasonClaimFresh,
		},
		{
			name:       "live session with an ancient checkpoint still owns it",
			claims:     []Claim{{Session: "s1", CheckpointedAt: ago(30 * time.Hour), Live: true}},
			wantState:  OwnClaimedLive,
			wantOwner:  "s1",
			wantReason: ReasonSessionLive,
		},
		{
			name:       "dead session past the TTL is a check-in overdue, not a reap",
			claims:     []Claim{{Session: "s1", CheckpointedAt: ago(4 * time.Hour)}},
			wantState:  OwnClaimedExpired,
			wantOwner:  "s1",
			wantReason: ReasonCheckinOverdue,
		},
		{
			name: "two fresh capturers is a tie the fold refuses to break",
			claims: []Claim{
				{Session: "s2", CheckpointedAt: ago(5 * time.Minute), Live: true},
				{Session: "s1", CheckpointedAt: ago(20 * time.Minute)},
			},
			wantState:  OwnAmbiguous,
			wantOwners: []string{"s1", "s2"},
			wantReason: ReasonTreeWideTie,
		},
		{
			name: "one session that checkpointed twice is not a tie",
			claims: []Claim{
				{Session: "s1", CheckpointedAt: ago(20 * time.Minute)},
				{Session: "s1", CheckpointedAt: ago(5 * time.Minute), Live: true},
			},
			wantState:  OwnClaimedLive,
			wantOwner:  "s1",
			wantReason: ReasonSessionLive,
		},
		{
			name:       "unstamped claim from a dead session cannot manufacture an owner",
			claims:     []Claim{{Session: "s1"}},
			wantState:  OwnUnclaimed,
			wantReason: ReasonNoFreshClaim,
		},
		{
			name:       "unstamped claim from a LIVE session is honoured",
			claims:     []Claim{{Session: "s1", Live: true}},
			wantState:  OwnClaimedLive,
			wantOwner:  "s1",
			wantReason: ReasonSessionLive,
		},
		{
			name: "two different lapsed sessions stay UNCLAIMED rather than naming one",
			claims: []Claim{
				{Session: "s1", CheckpointedAt: ago(4 * time.Hour)},
				{Session: "s2", CheckpointedAt: ago(5 * time.Hour)},
			},
			wantState:  OwnUnclaimed,
			wantReason: ReasonNoFreshClaim,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := OwnerOfPath("pkg/new.go", tc.claims, testNow, DefaultClaimTTL)
			if got.State != tc.wantState {
				t.Fatalf("state = %q, want %q (reason %q)", got.State, tc.wantState, got.Reason)
			}
			if got.Owner != tc.wantOwner {
				t.Errorf("owner = %q, want %q", got.Owner, tc.wantOwner)
			}
			if !reflect.DeepEqual(got.Owners, tc.wantOwners) {
				t.Errorf("owners = %v, want %v", got.Owners, tc.wantOwners)
			}
			if got.Reason != tc.wantReason {
				t.Errorf("reason = %q, want %q", got.Reason, tc.wantReason)
			}
			if got.Path != "pkg/new.go" {
				t.Errorf("path = %q, want pkg/new.go", got.Path)
			}
			// An AMBIGUOUS or UNCLAIMED verdict must never leak a single Owner: that
			// is the whole no-guessing property.
			if (got.State == OwnAmbiguous || got.State == OwnUnclaimed) && got.Owner != "" {
				t.Errorf("state %q leaked owner %q", got.State, got.Owner)
			}
		})
	}
}

// TestOwnerOfPathClaimAgeIsFreshest pins that the reported age is the most recent
// check-in, so "how long since anyone touched this claim" reads correctly when a session
// checkpoints repeatedly.
func TestOwnerOfPathClaimAgeIsFreshest(t *testing.T) {
	got := OwnerOfPath("pkg/new.go", []Claim{
		{Session: "s1", CheckpointedAt: ago(45 * time.Minute)},
		{Session: "s1", CheckpointedAt: ago(3 * time.Minute)},
	}, testNow, DefaultClaimTTL)
	if want := int64(180); got.ClaimAgeSeconds != want {
		t.Fatalf("claim age = %d, want %d (freshest claim)", got.ClaimAgeSeconds, want)
	}
}

// TestOwnerOfPathFutureStampIsNotNegative guards the clock-skew edge: a peer host whose
// clock runs ahead must not produce a negative age that reads as "not yet claimed".
func TestOwnerOfPathFutureStampIsNotNegative(t *testing.T) {
	got := OwnerOfPath("pkg/new.go", []Claim{{Session: "s1", CheckpointedAt: testNow + 600}}, testNow, DefaultClaimTTL)
	if got.State != OwnClaimedLive || got.ClaimAgeSeconds != 0 {
		t.Fatalf("got state %q age %d, want CLAIMED_LIVE age 0", got.State, got.ClaimAgeSeconds)
	}
}

// TestOwnerOfPathEarliestByIsAHintNotAVerdict pins that the first-capturer hint is
// carried on a tie but never promoted into Owner.
func TestOwnerOfPathEarliestByIsAHintNotAVerdict(t *testing.T) {
	got := OwnerOfPath("pkg/new.go", []Claim{
		{Session: "late", CheckpointedAt: ago(2 * time.Minute), Live: true},
		{Session: "first", CheckpointedAt: ago(40 * time.Minute), Live: true},
	}, testNow, DefaultClaimTTL)
	if got.State != OwnAmbiguous {
		t.Fatalf("state = %q, want AMBIGUOUS", got.State)
	}
	if got.EarliestBy != "first" {
		t.Errorf("earliestBy = %q, want first", got.EarliestBy)
	}
	if got.Owner != "" {
		t.Errorf("owner = %q, want empty on a tie", got.Owner)
	}
}

// TestBuildOwnerReportIsTotalAndDeterministic pins the two properties every caller
// relies on: every input path yields exactly one verdict (nothing silently dropped),
// and the output order is sorted regardless of input order.
func TestBuildOwnerReportIsTotalAndDeterministic(t *testing.T) {
	paths := []string{"z/last.go", "a/first.go", "m/mid.go", "n/none.go"}
	claims := map[string][]Claim{
		"a/first.go": {{Session: "s1", CheckpointedAt: ago(time.Minute), Live: true}},
		"m/mid.go": {
			{Session: "s1", CheckpointedAt: ago(time.Minute), Live: true},
			{Session: "s2", CheckpointedAt: ago(time.Minute), Live: true},
		},
		"z/last.go": {{Session: "s3", CheckpointedAt: ago(6 * time.Hour)}},
		// n/none.go deliberately absent — it must still be graded.
	}

	rep := BuildOwnerReport(paths, claims, testNow, DefaultClaimTTL)
	if len(rep.Paths) != len(paths) {
		t.Fatalf("graded %d paths, want %d (totality)", len(rep.Paths), len(paths))
	}
	wantOrder := []string{"a/first.go", "m/mid.go", "n/none.go", "z/last.go"}
	for i, o := range rep.Paths {
		if o.Path != wantOrder[i] {
			t.Fatalf("row %d = %q, want %q (sorted)", i, o.Path, wantOrder[i])
		}
	}
	if rep.Live != 1 || rep.Ambiguous != 1 || rep.Unclaimed != 1 || rep.Expired != 1 {
		t.Fatalf("counts = live %d ambiguous %d unclaimed %d expired %d, want 1/1/1/1",
			rep.Live, rep.Ambiguous, rep.Unclaimed, rep.Expired)
	}
	if rep.ClaimTTLSeconds != int64(DefaultClaimTTL/time.Second) {
		t.Errorf("ttl echo = %d, want %d", rep.ClaimTTLSeconds, int64(DefaultClaimTTL/time.Second))
	}
	if u := Unclaimed(rep.Paths); len(u) != 1 || u[0].Path != "n/none.go" {
		t.Fatalf("Unclaimed = %v, want just n/none.go", u)
	}

	// Determinism: shuffled input, identical output.
	again := BuildOwnerReport([]string{"n/none.go", "z/last.go", "a/first.go", "m/mid.go"}, claims, testNow, DefaultClaimTTL)
	if !reflect.DeepEqual(rep, again) {
		t.Fatalf("report is input-order dependent:\n%+v\n%+v", rep, again)
	}
}

// TestBuildOwnerReportTTLNarrowsClaims pins that expiry is real: the SAME evidence read
// with a tighter TTL moves a claim from owned to overdue. This is the property that makes
// "check in or your claim lapses" mean something.
func TestBuildOwnerReportTTLNarrowsClaims(t *testing.T) {
	claims := map[string][]Claim{"pkg/new.go": {{Session: "s1", CheckpointedAt: ago(30 * time.Minute)}}}

	wide := BuildOwnerReport([]string{"pkg/new.go"}, claims, testNow, DefaultClaimTTL)
	if wide.Paths[0].State != OwnClaimedLive {
		t.Fatalf("at the default TTL: state = %q, want CLAIMED_LIVE", wide.Paths[0].State)
	}
	tight := BuildOwnerReport([]string{"pkg/new.go"}, claims, testNow, 5*time.Minute)
	if tight.Paths[0].State != OwnClaimedExpired || tight.Paths[0].Owner != "s1" {
		t.Fatalf("at a 5m TTL: state = %q owner = %q, want CLAIMED_EXPIRED/s1",
			tight.Paths[0].State, tight.Paths[0].Owner)
	}
	if tight.ClaimTTLSeconds != 300 {
		t.Errorf("ttl echo = %d, want 300", tight.ClaimTTLSeconds)
	}
}

// TestZeroTTLFallsBackToDefault pins the zero-value contract the cmd shell relies on.
func TestZeroTTLFallsBackToDefault(t *testing.T) {
	rep := BuildOwnerReport([]string{"p.go"}, nil, testNow, 0)
	if rep.ClaimTTLSeconds != int64(DefaultClaimTTL/time.Second) {
		t.Fatalf("ttl = %d, want the default %d", rep.ClaimTTLSeconds, int64(DefaultClaimTTL/time.Second))
	}
}

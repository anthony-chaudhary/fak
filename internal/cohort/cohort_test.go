package cohort

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/comm"
)

func TestReady(t *testing.T) {
	if !Ready() {
		t.Fatal("Ready() should report true for the generated skeleton")
	}
}

func TestShrinkPreservesOrderAndBumpsGeneration(t *testing.T) {
	c := testCohort(t,
		comm.Member{ID: "gamma", Lane: "slow"},
		comm.Member{ID: "alpha", Lane: "fast"},
		comm.Member{ID: "bravo", Lane: "guard"},
		comm.Member{ID: "delta", Lane: "fast"},
	)

	before := c.MemberIDs()
	next, err := c.Shrink("bravo")
	if err != nil {
		t.Fatal(err)
	}
	if next.Generation() != c.Generation()+1 {
		t.Fatalf("generation=%d, want %d", next.Generation(), c.Generation()+1)
	}

	want := without(before, "bravo")
	if got := next.MemberIDs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("survivor order=%v, want %v", got, want)
	}
	if got := next.Group().WaveID(); got != "wave" {
		t.Fatalf("wave=%q, want wave", got)
	}
	m, err := next.Group().Membership(0)
	if err != nil {
		t.Fatal(err)
	}
	if m.ParentTraceID != "trace" {
		t.Fatalf("parent trace=%q, want trace", m.ParentTraceID)
	}
}

func TestShrinkRejectsUnknownAndEmptyCohort(t *testing.T) {
	c := testCohort(t, comm.Member{ID: "a"}, comm.Member{ID: "b"})
	if _, err := c.Shrink("z"); !errors.Is(err, ErrNoMember) {
		t.Fatalf("Shrink unknown err=%v, want ErrNoMember", err)
	}
	if _, err := c.Shrink("a", "b"); !errors.Is(err, ErrEmptyCohort) {
		t.Fatalf("Shrink all err=%v, want ErrEmptyCohort", err)
	}
}

func TestAgreeSucceedsWhenWinnerMeetsQuorumFloor(t *testing.T) {
	c := testCohort(t, comm.Member{ID: "a"}, comm.Member{ID: "b"}, comm.Member{ID: "c"})
	agreement, err := c.Agree(map[string]string{
		"a": "yes",
		"b": "yes",
		"c": "no",
	}, c.MajorityFloor())
	if err != nil {
		t.Fatal(err)
	}
	if !agreement.Agreed || agreement.Result.Output != "yes" {
		t.Fatalf("agreement=%+v, want agreed yes", agreement)
	}
	if agreement.WinningWeight != 2 || agreement.Required != 2 {
		t.Fatalf("weights=%v/%v, want 2/2", agreement.WinningWeight, agreement.Required)
	}
	if agreement.Result.Winner != "a" {
		t.Fatalf("winner=%q, want lexicographic first yes voter a", agreement.Result.Winner)
	}
}

func TestAgreeFailClosedWhenAbsentDropsQuorumBelowFloor(t *testing.T) {
	c := testCohort(t, comm.Member{ID: "a"}, comm.Member{ID: "b"}, comm.Member{ID: "c"})
	agreement, err := c.Agree(map[string]string{
		"a": "yes",
		"b": "yes",
	}, 3)
	if !errors.Is(err, ErrNoAgreement) {
		t.Fatalf("Agree err=%v, want ErrNoAgreement", err)
	}
	if agreement.Agreed {
		t.Fatalf("agreement=%+v, want fail-closed no agreement", agreement)
	}
	if agreement.Result.Output != "yes" || agreement.WinningWeight != 2 {
		t.Fatalf("fold=%+v winningWeight=%v, want yes with weight 2", agreement.Result, agreement.WinningWeight)
	}
	if !reflect.DeepEqual(agreement.Missing, []string{"c"}) {
		t.Fatalf("missing=%v, want [c]", agreement.Missing)
	}
}

func TestAgreeAllAbsentReturnsNoAgreementNotCombineEmptyVoteError(t *testing.T) {
	c := testCohort(t, comm.Member{ID: "a"}, comm.Member{ID: "b"})
	agreement, err := c.Agree(nil, 1)
	if !errors.Is(err, ErrNoAgreement) {
		t.Fatalf("Agree err=%v, want ErrNoAgreement", err)
	}
	if err != nil && strings.Contains(err.Error(), "Combine needs at least one vote") {
		t.Fatalf("Agree leaked modelroute empty-vote error: %v", err)
	}
	if agreement.Present != 0 || !reflect.DeepEqual(agreement.Missing, []string{"a", "b"}) {
		t.Fatalf("agreement=%+v, want both members explicit missing", agreement)
	}
}

func TestAgreeRejectsUnknownVoteAndBadQuorum(t *testing.T) {
	c := testCohort(t, comm.Member{ID: "a"})
	if _, err := c.Agree(map[string]string{"z": "yes"}, 1); !errors.Is(err, ErrNoMember) {
		t.Fatalf("unknown vote err=%v, want ErrNoMember", err)
	}
	if _, err := c.Agree(map[string]string{"a": "yes"}, 0); !errors.Is(err, ErrInvalidQuorum) {
		t.Fatalf("bad quorum err=%v, want ErrInvalidQuorum", err)
	}
}

func testCohort(t *testing.T, members ...comm.Member) *Cohort {
	t.Helper()
	c, err := FromMembers("wave", "trace", members)
	if err != nil {
		t.Fatal(err)
	}
	c.generation = 7
	return c
}

func without(ids []string, drop string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != drop {
			out = append(out, id)
		}
	}
	return out
}

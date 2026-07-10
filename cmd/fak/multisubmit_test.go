package main

import (
	"strings"
	"testing"
)

func TestPlanMultiSubmit(t *testing.T) {
	pool3 := []string{"seat-a", "seat-b", "seat-c"}

	t.Run("warm default N=5 over a 3-seat pool wraps and warm-replays all", func(t *testing.T) {
		plan, err := planMultiSubmit(multiSubmitInput{Issue: "3653", Pool: pool3, HasCache: true, BestEffort: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Trigger != "post-apply-best-effort" {
			t.Errorf("trigger = %q, want post-apply-best-effort", plan.Trigger)
		}
		if !plan.Warm {
			t.Errorf("warm = false, want true")
		}
		if len(plan.Submissions) != 5 {
			t.Fatalf("got %d submissions, want 5 (the default catalog)", len(plan.Submissions))
		}
		// seats wrap round-robin over the 3-seat pool.
		wantSeats := []string{"seat-a", "seat-b", "seat-c", "seat-a", "seat-b"}
		for i, s := range plan.Submissions {
			if s.Rank != i+1 {
				t.Errorf("submission %d rank = %d", i, s.Rank)
			}
			if s.Seat != wantSeats[i] {
				t.Errorf("submission %d seat = %q, want %q", i, s.Seat, wantSeats[i])
			}
			if s.Mode != "warm-replay" {
				t.Errorf("submission %d mode = %q, want warm-replay (cache present)", i, s.Mode)
			}
			if s.Angle == "" || s.Emphasis == "" {
				t.Errorf("submission %d missing angle/emphasis: %+v", i, s)
			}
		}
		if !hasNoteContaining(plan.Notes, "exceeds") {
			t.Errorf("expected a pool-wrap note, got %v", plan.Notes)
		}
	})

	t.Run("cold run seeds rank 1 and warm-replays the rest", func(t *testing.T) {
		plan, err := planMultiSubmit(multiSubmitInput{Issue: "3653", N: 3, Pool: pool3, HasCache: false, BestEffort: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if plan.Warm {
			t.Errorf("warm = true, want false (cold)")
		}
		if len(plan.Submissions) != 3 {
			t.Fatalf("got %d submissions, want 3", len(plan.Submissions))
		}
		if plan.Submissions[0].Mode != "seed" {
			t.Errorf("rank 1 mode = %q, want seed", plan.Submissions[0].Mode)
		}
		for _, s := range plan.Submissions[1:] {
			if s.Mode != "warm-replay" {
				t.Errorf("rank %d mode = %q, want warm-replay", s.Rank, s.Mode)
			}
		}
		wantSeats := []string{"seat-a", "seat-b", "seat-c"}
		for i, s := range plan.Submissions {
			if s.Seat != wantSeats[i] {
				t.Errorf("submission %d seat = %q, want %q", i, s.Seat, wantSeats[i])
			}
		}
		if !hasNoteContaining(plan.Notes, "cold: rank 1 seeds") {
			t.Errorf("expected a cold-seed note, got %v", plan.Notes)
		}
	})

	t.Run("custom angles set N and rank order", func(t *testing.T) {
		plan, err := planMultiSubmit(multiSubmitInput{
			Issue: "3653", N: 2, Angles: []string{"perf", "security"}, Pool: pool3, HasCache: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(plan.Submissions) != 2 {
			t.Fatalf("got %d submissions, want 2", len(plan.Submissions))
		}
		if plan.Submissions[0].Angle != "perf" || plan.Submissions[1].Angle != "security" {
			t.Errorf("angles = %q,%q, want perf,security", plan.Submissions[0].Angle, plan.Submissions[1].Angle)
		}
		// an angle not in the catalog still plans, with a generic emphasis.
		if plan.Submissions[0].Emphasis != "custom angle" {
			t.Errorf("custom angle emphasis = %q, want \"custom angle\"", plan.Submissions[0].Emphasis)
		}
	})

	t.Run("empty pool leaves seats unassigned with a note", func(t *testing.T) {
		plan, err := planMultiSubmit(multiSubmitInput{Issue: "3653", N: 2, HasCache: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for _, s := range plan.Submissions {
			if s.Seat != "" {
				t.Errorf("rank %d seat = %q, want empty (no pool)", s.Rank, s.Seat)
			}
		}
		if !hasNoteContaining(plan.Notes, "no rotation pool") {
			t.Errorf("expected a no-pool note, got %v", plan.Notes)
		}
	})

	t.Run("missing issue is an error", func(t *testing.T) {
		if _, err := planMultiSubmit(multiSubmitInput{N: 3, Pool: pool3}); err == nil {
			t.Errorf("expected an error for a missing issue")
		}
	})

	t.Run("N over available angles is an error", func(t *testing.T) {
		if _, err := planMultiSubmit(multiSubmitInput{Issue: "3653", N: 6, Pool: pool3}); err == nil {
			t.Errorf("expected an error when N exceeds the 5-angle default catalog")
		}
	})
}

func hasNoteContaining(notes []string, sub string) bool {
	for _, n := range notes {
		if strings.Contains(n, sub) {
			return true
		}
	}
	return false
}

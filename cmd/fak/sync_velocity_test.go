package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRenderSyncPushVelocityQualified(t *testing.T) {
	score := 100
	var out bytes.Buffer
	renderSyncPushVelocity(&out, safesync.PushVelocity{
		Qualified: true, ElapsedMS: 250, BudgetMS: 1000,
		BudgetRatio: 0.25, Score: &score, Grade: "A",
	})
	if got, want := out.String(), "velocity: 250ms / 1s budget (ratio 0.250, 100/100 A)"; !strings.Contains(got, want) {
		t.Fatalf("render = %q, want %q", got, want)
	}
}

func TestRenderSyncPushVelocityRefusalIsUnscored(t *testing.T) {
	var out bytes.Buffer
	renderSyncPushVelocity(&out, safesync.PushVelocity{
		ElapsedMS: 1, BudgetMS: 1000, BudgetRatio: 0.001,
		Grade: "UNSCORED", Notes: []string{"unscored: safe push did not publish (PUSH_ERROR)"},
	})
	got := out.String()
	if !strings.Contains(got, "UNSCORED") || !strings.Contains(got, "PUSH_ERROR") || strings.Contains(got, "/100 F") {
		t.Fatalf("refusal render conflates speed with a score: %q", got)
	}
}

func TestValidatePushVelocityBudget(t *testing.T) {
	if err := validatePushVelocityBudget(time.Millisecond); err != nil {
		t.Fatalf("1ms budget: %v", err)
	}
	for _, bad := range []time.Duration{0, -time.Second, 500 * time.Microsecond} {
		if err := validatePushVelocityBudget(bad); err == nil {
			t.Errorf("budget %s accepted, want usage error", bad)
		}
	}
}

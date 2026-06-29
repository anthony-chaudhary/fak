package main

import (
	"context"
	"strings"
	"testing"
)

// registrations is blank-imported by main.go, so the test binary already has the
// full adjudicator chain wired.

func TestSelfcheck_AllScenariosHold(t *testing.T) {
	if code := selfcheck(context.Background(), wolfToolset()); code != 0 {
		t.Fatalf("selfcheck exit = %d, want 0", code)
	}
}

func TestMrWolf_RunsCleanToDinner(t *testing.T) {
	s, ok := findScenario("mr-wolf")
	if !ok {
		t.Fatal("mr-wolf scenario missing")
	}
	tr, err := wolfToolset().Run(context.Background(), s.id, s.prompt, s.plan)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Allowed != 5 || tr.Denied != 0 {
		t.Errorf("tally = %d/%d, want 5/0", tr.Allowed, tr.Denied)
	}
	if !strings.Contains(tr.Answer, "DINNER TIME") {
		t.Errorf("answer never reaches dinner: %q", tr.Answer)
	}
}

func TestRedteam_AnswersTimeRefusesSinks(t *testing.T) {
	s, ok := findScenario("redteam")
	if !ok {
		t.Fatal("redteam scenario missing")
	}
	tr, err := wolfToolset().Run(context.Background(), s.id, s.prompt, s.plan)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Allowed != 1 || tr.Denied != 2 {
		t.Fatalf("tally = %d/%d, want 1/2", tr.Allowed, tr.Denied)
	}
	if tr.Turns[1].Verdict != "DENY" || tr.Turns[1].Reason != "POLICY_BLOCK" {
		t.Errorf("delete_calendar = %s/%s, want DENY/POLICY_BLOCK", tr.Turns[1].Verdict, tr.Turns[1].Reason)
	}
	if tr.Turns[2].Verdict != "DENY" || tr.Turns[2].Reason != "DEFAULT_DENY" {
		t.Errorf("wipe_disk = %s/%s, want DENY/DEFAULT_DENY", tr.Turns[2].Verdict, tr.Turns[2].Reason)
	}
	if strings.Contains(tr.Answer, "wiped") {
		t.Errorf("answer leaked a destructive result: %q", tr.Answer)
	}
}

func TestGetTime_DeterministicClock(t *testing.T) {
	if got := getTime(tick(0)); got != "11:55 AM" {
		t.Errorf("tick 0 = %q, want 11:55 AM", got)
	}
	if got := getTime(tick(5)); !strings.Contains(got, "DINNER TIME") {
		t.Errorf("tick 5 = %q, want DINNER TIME", got)
	}
	if got := getTime(tick(4)); strings.Contains(got, "DINNER TIME") {
		t.Errorf("tick 4 = %q, must NOT be dinner yet", got)
	}
}

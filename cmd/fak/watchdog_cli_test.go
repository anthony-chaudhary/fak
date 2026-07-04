package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/watchdoghealth"
)

func specWithProbe(id string, p watchdogProbe, perr error) watchdogAutohealSpec {
	return watchdogAutohealSpec{
		watchdogService: watchdogService{ID: id, Manager: "systemd", Unit: id + ".timer"},
		Probe:           func(context.Context) (watchdogProbe, error) { return p, perr },
		Restart:         func(context.Context) error { return nil },
	}
}

func TestWatchdogStatusMonitorsJoinsProbeAndHealState(t *testing.T) {
	dir := t.TempDir()
	// A dead-but-installed monitor whose persisted streak is exhausted must surface as
	// GAVE_UP once the join carries its attempts + the shell's max-attempts cap.
	if err := writeWatchdogHealState(dir, watchdogHealState{
		Schema: watchdogAutohealSchema, ID: "dead", Attempts: 3, LastRestartUnixNano: 111,
	}); err != nil {
		t.Fatalf("seed heal-state: %v", err)
	}

	specs := []watchdogAutohealSpec{
		specWithProbe("alive", watchdogProbe{Installed: true, Alive: true, Detail: "active"}, nil),
		specWithProbe("dead", watchdogProbe{Installed: true, Alive: false, Detail: "inactive"}, nil),
		specWithProbe("broke", watchdogProbe{}, errors.New("systemctl exploded")),
	}
	mons := watchdogStatusMonitors(context.Background(), specs, dir, 3)
	if len(mons) != 3 {
		t.Fatalf("monitors = %d, want 3", len(mons))
	}

	// The dead one carried its heal-state through the join.
	dead := mons[1]
	if dead.Attempts != 3 || dead.MaxAttempts != 3 || dead.LastRestartUnixNano != 111 {
		t.Fatalf("dead monitor state not joined: %+v", dead)
	}
	// The probe error became ProbeErr with the error as detail.
	broke := mons[2]
	if !broke.ProbeErr || !strings.Contains(broke.Detail, "exploded") {
		t.Fatalf("probe error not surfaced: %+v", broke)
	}

	d := watchdoghealth.Fold(mons)
	want := map[string]watchdoghealth.Status{
		"alive": watchdoghealth.StatusHealthy,
		"dead":  watchdoghealth.StatusGaveUp,
		"broke": watchdoghealth.StatusUnknown,
	}
	for _, h := range d.Monitors {
		if got := want[h.ID]; got != h.Status {
			t.Fatalf("Fold(%s) = %s, want %s", h.ID, h.Status, got)
		}
	}
	if d.Rollup != watchdoghealth.StatusGaveUp || !d.NeedsAttention {
		t.Fatalf("digest rollup = %s attn = %v, want GAVE_UP + attention", d.Rollup, d.NeedsAttention)
	}
}

func TestWriteWatchdogStatusTableRendersRollupAndCells(t *testing.T) {
	d := watchdoghealth.Fold([]watchdoghealth.Monitor{
		{ID: "alive", Manager: "systemd", Installed: true, Alive: true, Attempts: 2},
		{ID: "gone", Manager: "systemd", Installed: true, Alive: false, Attempts: 3, MaxAttempts: 3, LastRestartUnixNano: 1},
	})
	var buf bytes.Buffer
	writeWatchdogStatusTable(&buf, d)
	out := buf.String()

	// A healthy monitor hides its (irrelevant) attempt streak; a gave-up one shows it.
	for _, want := range []string{"alive", "HEALTHY", "gone", "GAVE_UP", "rollup: GAVE_UP", "needs attention"} {
		if !strings.Contains(out, want) {
			t.Fatalf("table missing %q:\n%s", want, out)
		}
	}
	// The healthy row's attempts cell is "-", not "2".
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "alive") && strings.Contains(line, " 2 ") {
			t.Fatalf("healthy row leaked its attempt streak: %q", line)
		}
	}
}

func TestWatchdogTimeCellEmptyIsDash(t *testing.T) {
	if got := watchdogTimeCell(0); got != "-" {
		t.Fatalf("watchdogTimeCell(0) = %q, want %q", got, "-")
	}
	if got := watchdogTimeCell(-5); got != "-" {
		t.Fatalf("watchdogTimeCell(-5) = %q, want %q", got, "-")
	}
	if got := watchdogTimeCell(1); got == "-" {
		t.Fatalf("watchdogTimeCell(1) = %q, want a formatted stamp", got)
	}
}

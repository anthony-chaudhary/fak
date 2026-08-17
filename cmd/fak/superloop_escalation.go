package main

import (
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

var superloopEscalationEvents = func(root string) ([]loopmgr.Event, error) {
	return loopmgr.Load(filepath.Join(root, defaultLoopLedger()))
}

func applySuperloopNoProgressEscalation(root string, d superloop.DriveDecision) (superloop.DriveDecision, int) {
	if !d.Enter || d.Satisfied {
		return d, 0
	}
	events, err := superloopEscalationEvents(root)
	if err != nil {
		return d, 0
	}
	streak := superloopNoProgressStreak(events, d.Intent)
	stage := superloop.EscalateNoProgress(streak)
	if stage.Name == "dispatch" || stage.Name == "retry" {
		return d, streak
	}
	d.Member = superloop.Member{Kind: superloop.KindSurface, Ref: "no-progress-" + stage.Name, Enter: stage.Command}
	d.Action = stage.Command
	d.Reason = stage.Reason
	return d, streak
}

func superloopNoProgressStreak(events []loopmgr.Event, intent string) int {
	streak := 0
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.LoopID != intent || ev.Kind != loopmgr.EventEnd {
			continue
		}
		if superloopEventProgressed(ev) {
			return streak
		}
		if ev.Status == loopmgr.StatusFailed || ev.Status == loopmgr.StatusRefused || ev.Status == loopmgr.StatusWitnessRefused || ev.Status == loopmgr.StatusWitnessUnavailable {
			streak++
		}
	}
	return streak
}

func superloopEventProgressed(ev loopmgr.Event) bool {
	if ev.Status == loopmgr.StatusWitnessedDone {
		return true
	}
	for key, value := range ev.Metrics {
		key = strings.ToLower(key)
		if value > 0 && (strings.Contains(key, "closed") || strings.Contains(key, "commit") || strings.Contains(key, "progress")) {
			return true
		}
	}
	return false
}

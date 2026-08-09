package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// guardInfoFleetSummary turns guard's already-folded /debug/vars fleet snapshot into an
// operator-sized status phrase. The TUI deliberately does not run a second fleet census.
func guardInfoFleetSummary(f *gateway.SessionFleet) string {
	if f == nil {
		return ""
	}
	verdict := strings.ToUpper(strings.TrimSpace(f.Verdict))
	if verdict == "" {
		verdict = "UNKNOWN"
	}
	parts := []string{fmt.Sprintf("fleet %s · machines %d · sessions %d", verdict, f.Machines, f.Sessions)}
	if f.SeatCapacity > 0 {
		parts = append(parts, fmt.Sprintf("seats %d/%d healthy", f.HealthySeats, f.SeatCapacity))
	}
	if f.Stale+f.Action > 0 {
		parts = append(parts, fmt.Sprintf("attention %d", f.Stale+f.Action))
	}
	if f.AuthBlocked > 0 {
		parts = append(parts, fmt.Sprintf("auth-blocked %d", f.AuthBlocked))
	}
	if f.ThrottledSeats > 0 {
		parts = append(parts, fmt.Sprintf("throttled %d", f.ThrottledSeats))
	}
	if f.ResumeBacklog > 0 {
		parts = append(parts, fmt.Sprintf("resume %d", f.ResumeBacklog))
	}
	if f.VersionMismatches > 0 {
		parts = append(parts, fmt.Sprintf("version-skew %d", f.VersionMismatches))
	}
	if f.HostLoad > 0 {
		parts = append(parts, fmt.Sprintf("load %.1f", f.HostLoad))
	}
	return strings.Join(parts, " · ")
}

// renderInfoFleetRows expands the bounded machine sample for the Agents tab. Attention rows sort
// first, then stable machine ID, keeping the captured frame useful during an incident.
func renderInfoFleetRows(f *gateway.SessionFleet) []string {
	if f == nil {
		return nil
	}
	rows := []string{" fleet: " + guardInfoFleetSummary(f)}
	machines := append([]gateway.SessionFleetMachine(nil), f.Rows...)
	sort.SliceStable(machines, func(i, j int) bool {
		iAttention := fleetMachineNeedsAttention(machines[i])
		jAttention := fleetMachineNeedsAttention(machines[j])
		if iAttention != jAttention {
			return iAttention
		}
		return strings.ToLower(machines[i].ID) < strings.ToLower(machines[j].ID)
	})
	for _, m := range machines {
		id := strings.TrimSpace(m.ID)
		if id == "" {
			id = "unknown"
		}
		state := strings.TrimSpace(m.State)
		if state == "" {
			state = "unknown"
		}
		parts := []string{fmt.Sprintf("%-14s %-8s", id, state), fmt.Sprintf("sessions %d", m.Sessions)}
		if m.AgeMin > 0 {
			parts = append(parts, fmt.Sprintf("age %.1fm", m.AgeMin))
		}
		if version := strings.TrimSpace(m.Version); version != "" {
			parts = append(parts, version)
		}
		rows = append(rows, "  "+strings.Join(parts, " · "))
	}
	if len(machines) == 0 {
		rows = append(rows, "  no machine sample reported")
	}
	return rows
}

func fleetMachineNeedsAttention(m gateway.SessionFleetMachine) bool {
	switch strings.ToUpper(strings.TrimSpace(m.State)) {
	case "", "OK", "READY", "ONLINE", "HEALTHY":
		return false
	default:
		return true
	}
}

//go:build darwin

package procguard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"strings"
	"syscall"
)

const darwinMemorySnapshotAttempts = 3

func collectMemorySnapshot(rootPID int) (MemorySnapshot, bool, string) {
	snapshot, detail := collectDarwinMemorySnapshotWithCollectors(rootPID, CollectProcesses, CollectRelations, darwinProcessAlive)
	if rootPID > 0 && snapshot.RootPID == 0 {
		return snapshot, true, ""
	}
	physical, physicalErr := hostPhysicalMemoryBytes()
	snapshot.HostPhysicalBytes = physical
	detail = joinDetails(detail, physicalErr)
	return snapshot, true, detail
}

// collectDarwinMemorySnapshotWithCollectors bounds reconciliation of the two
// independent ps snapshots. Each attempt establishes ownership first, then
// samples RSS: later starts are outside that ownership epoch and are picked up
// by the next poll instead of appearing as owned processes with no RSS row.
// Collector failures are not retried, and a third incomplete join remains
// fail-closed.
func collectDarwinMemorySnapshotWithCollectors(rootPID int, collectCensus, collectRelations func() ([]Proc, string), processAlive func(int) (bool, error)) (MemorySnapshot, string) {
	var snapshot MemorySnapshot
	for attempt := 0; attempt < darwinMemorySnapshotAttempts; attempt++ {
		relations, relationErr := collectRelations()
		census, censusErr := collectCensus()
		var joinErr string
		var recollect bool
		snapshot, joinErr, recollect = joinDarwinMemorySnapshotForCollection(rootPID, census, relations, processAlive)
		if rootPID > 0 && snapshot.RootPID == 0 && censusErr == "" && relationErr == "" {
			return snapshot, ""
		}
		if snapshot.RootPID == 0 {
			// A collector error is still fail-closed even if the liveness probe
			// also observes the root gone; only a clean census pair may emit the
			// terminal-sample marker.
			snapshot.RootPID = rootPID
		}
		detail := joinDetails(censusErr, relationErr, joinErr)
		if detail == "" || censusErr != "" || relationErr != "" || !recollect || attempt+1 == darwinMemorySnapshotAttempts {
			return snapshot, detail
		}
	}
	return snapshot, "Darwin memory snapshot attempts exhausted"
}

// joinDarwinMemorySnapshot joins the separate BSD ps resource and relation
// tables by PID. The relation table defines ownership; every owned PID must have
// an RSS row or the sample is typed as incomplete instead of looking healthy.
func joinDarwinMemorySnapshot(rootPID int, census, relations []Proc) (MemorySnapshot, string) {
	return joinDarwinMemorySnapshotWithProbe(rootPID, census, relations, darwinProcessAlive)
}

// joinDarwinMemorySnapshotWithProbe keeps liveness reconciliation injectable so
// the two-snapshot exit race has a deterministic regression witness. The RSS and
// relation tables are collected by separate ps processes; a descendant may exit
// after appearing in one table and before the join examines it.
func joinDarwinMemorySnapshotWithProbe(rootPID int, census, relations []Proc, processAlive func(int) (bool, error)) (MemorySnapshot, string) {
	snapshot, detail, _ := joinDarwinMemorySnapshotForCollection(rootPID, census, relations, processAlive)
	return snapshot, detail
}

// joinDarwinMemorySnapshotForCollection identifies the one incomplete shape
// that a fresh census can reconcile: owned PIDs with no RSS row. All other join
// failures are final for the current poll.
func joinDarwinMemorySnapshotForCollection(rootPID int, census, relations []Proc, processAlive func(int) (bool, error)) (MemorySnapshot, string, bool) {
	s := MemorySnapshot{Metric: MemoryMetricRSS, RootPID: rootPID}
	if rootPID <= 0 {
		return s, "invalid root pid", false
	}
	byRSS := make(map[int]Proc, len(census))
	for _, row := range census {
		if row.PID > 0 {
			byRSS[row.PID] = row
		}
	}
	byRelation := make(map[int]Proc, len(relations))
	children := make(map[int][]int)
	for _, row := range relations {
		if row.PID <= 0 || row.PPID == nil {
			continue
		}
		byRelation[row.PID] = row
		children[*row.PPID] = append(children[*row.PPID], row.PID)
	}
	if _, ok := byRelation[rootPID]; !ok {
		alive, err := processAlive(rootPID)
		if err != nil {
			return s, fmt.Sprintf("probe missing relation root pid %d: %v", rootPID, err), false
		}
		if !alive {
			// RootPID zero is the internal terminal-sample marker. It distinguishes
			// an exited root from a valid zero-byte snapshot without widening the
			// cross-platform collector API.
			s.RootPID = 0
			return s, "", false
		}
		return s, fmt.Sprintf("root pid %d missing from relation census", rootPID), false
	}
	for ppid := range children {
		sort.Ints(children[ppid])
	}
	queue := []int{rootPID}
	seen := make(map[int]bool)
	missing := make([]int, 0)
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if seen[pid] {
			continue
		}
		seen[pid] = true
		queue = append(queue, children[pid]...)

		relation := byRelation[pid]
		rss, ok := byRSS[pid]
		ppid := 0
		if relation.PPID != nil {
			ppid = *relation.PPID
		}
		process := MemoryProcess{PID: pid, PPID: ppid, Name: relation.Name, CommandLine: relation.Cmdline}
		if ok && rss.WSMB != nil && *rss.WSMB >= 0 {
			process.Bytes = uint64(*rss.WSMB) << 20
			s.TreeBytes += process.Bytes
			if process.Name == "" {
				process.Name = rss.Name
			}
		} else {
			// The root is the fail-closed anchor for the entire ownership walk. A
			// missing root RSS row is never omitted as exit churn; only already-exited
			// descendants are harmless churn.
			if pid != rootPID {
				alive, err := processAlive(pid)
				if err != nil {
					return s, fmt.Sprintf("probe missing rss pid %d: %v", pid, err), false
				}
				if !alive {
					continue
				}
			}
			missing = append(missing, pid)
		}
		s.Processes = append(s.Processes, process)
	}
	if len(missing) > 0 {
		return s, fmt.Sprintf("owned pids missing from rss census: %v", missing), true
	}
	return s, "", false
}

// darwinProcessAlive distinguishes normal exit churn from telemetry failures.
// EPERM proves the PID still exists and therefore keeps a missing RSS row fatal;
// ESRCH is the only result that permits skipping a descendant.
func darwinProcessAlive(pid int) (bool, error) {
	err := syscall.Kill(pid, 0)
	switch {
	case err == nil, errors.Is(err, syscall.EPERM):
		return true, nil
	case errors.Is(err, syscall.ESRCH):
		return false, nil
	default:
		return false, err
	}
}

func joinDetails(details ...string) string {
	var nonempty []string
	for _, detail := range details {
		if strings.TrimSpace(detail) != "" {
			nonempty = append(nonempty, detail)
		}
	}
	return strings.Join(nonempty, "; ")
}

func hostPhysicalMemoryBytes() (uint64, string) {
	raw, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0, fmt.Sprintf("sysctl hw.memsize: %v", err)
	}
	var buf [8]byte
	copy(buf[:], raw)
	bytes := binary.LittleEndian.Uint64(buf[:])
	if bytes == 0 {
		return 0, "sysctl hw.memsize returned zero"
	}
	return bytes, ""
}

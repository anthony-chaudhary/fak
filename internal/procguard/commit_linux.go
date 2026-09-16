//go:build linux

package procguard

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// procRoot is the /proc mount point. It is a var so tests can point the collector
// at a synthetic procfs tree and witness the parse on any host (see
// commit_linux_test.go), the same reason the darwin collector injects its ps
// collectors.
var procRoot = "/proc"

// collectMemorySnapshot reads the calling (or any) process tree's resident set
// size from Linux procfs. Unlike Darwin's two-`ps` join, the kernel exposes a
// per-pid resident page count directly in /proc/<pid>/statm, so there is no
// cross-table exit race and the metric is a genuine RSS — the same MemoryMetricRSS
// the Darwin and receipt paths already carry.
func collectMemorySnapshot(rootPID int) (MemorySnapshot, bool, string) {
	snapshot, detail := collectLinuxMemorySnapshotFrom(procRoot, rootPID, os.Getpagesize())
	if rootPID > 0 && snapshot.RootPID == 0 {
		return snapshot, true, ""
	}
	physical, available, physicalErr := hostPhysicalMemoryAvailabilityBytes()
	snapshot.HostPhysicalBytes = physical
	snapshot.HostPhysicalAvailableBytes = available
	return snapshot, true, joinLinuxDetails(detail, physicalErr)
}

// linuxProcRow is one /proc/<pid> row: its relation identity plus its resident
// bytes. Alive is false for a pid that kept a readable stat line but whose statm
// could not be read — the walk treats that as a lost measurement rather than
// inventing a zero.
type linuxProcRow struct {
	PID     int
	PPID    int
	Name    string
	Cmdline string
	Bytes   uint64
	Alive   bool
}

// collectLinuxMemorySnapshotFrom is the pure, injectable core: it enumerates the
// procfs tree under procRoot and sums RSS across rootPID's descendants.
// pageSize supplies the statm page-to-byte factor (4 KiB on x86_64, 16 KiB on
// some arm64 kernels — the kernel's own page size, not a hardcoded 4096).
func collectLinuxMemorySnapshotFrom(procRoot string, rootPID, pageSize int) (MemorySnapshot, string) {
	s := MemorySnapshot{Metric: MemoryMetricRSS, RootPID: rootPID}
	if rootPID <= 0 {
		return s, "invalid root pid"
	}
	rows, relations, err := readLinuxProcTree(procRoot, pageSize)
	if err != "" {
		return s, err
	}
	byPID := make(map[int]linuxProcRow, len(rows))
	children := make(map[int][]int)
	for _, row := range rows {
		byPID[row.PID] = row
		children[row.PPID] = append(children[row.PPID], row.PID)
	}
	for ppid := range children {
		sort.Ints(children[ppid])
	}
	rootRow, rootKnown := byPID[rootPID]
	if !rootKnown {
		if relations[rootPID] {
			// The root's relation row existed but its statm could not be read and
			// it was not observed gone: fail closed rather than report an empty tree.
			return s, "linux snapshot missing root rss row"
		}
		// RootPID zero is the internal terminal-sample marker (root exited).
		s.RootPID = 0
		return s, ""
	}
	queue := []int{rootPID}
	seen := make(map[int]bool)
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if seen[pid] {
			continue
		}
		seen[pid] = true
		queue = append(queue, children[pid]...)

		row, ok := byPID[pid]
		if !ok {
			// A descendant left the census between enumeration passes. It carries
			// no RSS to attribute; skipping it is exit churn, not a lost measurement.
			continue
		}
		s.TreeBytes += row.Bytes
		s.Processes = append(s.Processes, MemoryProcess{PID: pid, PPID: row.PPID, Name: row.Name, CommandLine: row.Cmdline, Bytes: row.Bytes})
	}
	if rootRow.Bytes == 0 && !rootRow.Alive {
		// The root kept a stat line but had no readable statm; fail closed so a
		// broken probe never reads as a clean zero-byte tree.
		return s, "linux root " + strconv.Itoa(rootPID) + " has no readable resident set"
	}
	return s, ""
}

// readLinuxProcTree enumerates /proc once: for every pid it reads stat for
// ppid/name, statm for resident bytes, and cmdline. The returned relation set
// records which pids had a readable stat line, letting the join distinguish
// "pid is gone" from "pid is present but unreadable".
func readLinuxProcTree(procRoot string, pageSize int) ([]linuxProcRow, map[int]bool, string) {
	entries, readErr := os.ReadDir(procRoot)
	if readErr != nil {
		return nil, nil, "read " + procRoot + ": " + readErr.Error()
	}
	rows := make([]linuxProcRow, 0, len(entries))
	relations := make(map[int]bool, len(entries))
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || pid <= 0 {
			continue
		}
		dir := filepath.Join(procRoot, entry.Name())
		stat, statErr := os.ReadFile(filepath.Join(dir, "stat"))
		if statErr != nil {
			continue // raced with exit; an unreadable dir is not a census row
		}
		ppid, name, ok := parseLinuxStat(string(stat))
		if !ok {
			continue
		}
		relations[pid] = true
		row := linuxProcRow{PID: pid, PPID: ppid, Name: name, Alive: true}
		if rss, rssErr := readLinuxStatmBytes(dir, pageSize); rssErr == nil {
			row.Bytes = rss
		} else {
			// statm is the one measurement the walk cannot fabricate. A row that
			// leaves it unread keeps Bytes 0 but stays a relation row, so the join
			// can fail closed instead of mistaking it for a departed pid.
			row.Alive = false
		}
		if cmd, cmdErr := os.ReadFile(filepath.Join(dir, "cmdline")); cmdErr == nil {
			row.Cmdline = strings.TrimRight(strings.ReplaceAll(string(cmd), "\x00", " "), " ")
		}
		rows = append(rows, row)
	}
	if len(relations) == 0 {
		return nil, nil, "no readable processes under " + procRoot
	}
	return rows, relations, ""
}

// parseLinuxStat extracts ppid (field 4) and comm (field 2) from one /proc/<pid>/stat
// line. comm is parenthesized and may itself contain spaces and parens, so the
// fields are located from the LAST ')' rather than by naive splitting.
func parseLinuxStat(line string) (ppid int, name string, ok bool) {
	open := strings.IndexByte(line, '(')
	close := strings.LastIndexByte(line, ')')
	if open < 0 || close < open {
		return 0, "", false
	}
	name = line[open+1 : close]
	rest := strings.Fields(line[close+1:])
	// rest[0] is state (field 3); ppid is field 4, i.e. rest[1].
	if len(rest) < 2 {
		return 0, "", false
	}
	ppid, err := strconv.Atoi(rest[1])
	if err != nil {
		return 0, "", false
	}
	return ppid, name, true
}

// readLinuxStatmBytes reads /proc/<pid>/statm and returns resident pages scaled by
// pageSize. Field 2 is resident (not field 1, which is the virtual size — a
// program that mmaps a large file has huge vsize and tiny rss, and reporting vsize
// as "memory" would be a lie the guard could act on).
func readLinuxStatmBytes(dir string, pageSize int) (uint64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "statm"))
	if err != nil {
		return 0, err
	}
	fields := strings.Fields(string(raw))
	if len(fields) < 2 {
		return 0, errMissingStatmField
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return 0, err
	}
	return pages * uint64(pageSize), nil
}

type statmFieldError struct{}

func (statmFieldError) Error() string { return "statm missing resident field" }

var errMissingStatmField = statmFieldError{}

// hostPhysicalMemoryBytes is the shared-surface (total only) name: commit.go's
// exported HostPhysicalMemoryBytes wrapper resolves to this on linux, so it must
// keep the two-value shape every platform's collector shares. The available-bytes
// figure linux can additionally derive is exposed through the richer internal
// helper below and folded into collectMemorySnapshot.
func hostPhysicalMemoryBytes() (uint64, string) {
	total, _, detail := hostPhysicalMemoryAvailabilityBytes()
	return total, detail
}

// hostPhysicalMemoryAvailabilityBytes reads MemTotal and MemAvailable from /proc/meminfo.
// MemAvailable (not MemFree) is the kernel's own estimate of what a new workload
// could claim without swapping; MemFree alone would read a healthy page cache as
// memory pressure.
func hostPhysicalMemoryAvailabilityBytes() (uint64, uint64, string) {
	raw, err := os.ReadFile(filepath.Join(procRoot, "meminfo"))
	if err != nil {
		return 0, 0, "read " + procRoot + "/meminfo: " + err.Error()
	}
	total, available, ok := parseLinuxMeminfo(string(raw))
	if !ok || total == 0 {
		return 0, 0, "meminfo missing MemTotal"
	}
	return total, available, ""
}

// parseLinuxMeminfo folds MemTotal and MemAvailable (both in kB) into bytes.
func parseLinuxMeminfo(text string) (total, available uint64, ok bool) {
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			total = kb << 10
			ok = true
		case "MemAvailable:":
			available = kb << 10
		}
	}
	return total, available, ok
}

func joinLinuxDetails(details ...string) string {
	var nonempty []string
	for _, detail := range details {
		if strings.TrimSpace(detail) != "" {
			nonempty = append(nonempty, detail)
		}
	}
	return strings.Join(nonempty, "; ")
}

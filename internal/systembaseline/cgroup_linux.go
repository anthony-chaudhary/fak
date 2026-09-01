//go:build linux

package systembaseline

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const linuxCgroupRoot = "/sys/fs/cgroup"

type cgroupFSOps struct {
	readFile  func(string) ([]byte, error)
	writeFile func(string, []byte, os.FileMode) error
	mkdirTemp func(string, string) (string, error)
	remove    func(string) error
	open      func(string) (*os.File, error)
	kill      func(int) error
	sleep     func(time.Duration)
}

func defaultCgroupFSOps() cgroupFSOps {
	return cgroupFSOps{
		readFile:  os.ReadFile,
		writeFile: os.WriteFile,
		mkdirTemp: os.MkdirTemp,
		remove:    os.Remove,
		open:      os.Open,
		kill: func(pid int) error {
			process, err := os.FindProcess(pid)
			if err != nil {
				return err
			}
			return process.Kill()
		},
		sleep: time.Sleep,
	}
}

type cgroupRawSnapshot struct {
	cpu           CounterSet
	memoryCurrent Metric
	memoryPeak    Metric
	memoryEvents  CounterSet
	pressure      CgroupPressure
}

type linuxCommandAttributor struct {
	ops        cgroupFSOps
	path       string
	dir        *os.File
	initial    cgroupRawSnapshot
	result     CgroupV2
	configured bool
	finished   bool
}

func newCommandAttributorPlatform() commandAttributorPlatform {
	return newLinuxCommandAttributor(linuxCgroupRoot, defaultCgroupFSOps())
}

func newLinuxCommandAttributor(root string, ops cgroupFSOps) *linuxCommandAttributor {
	unavailableResult := func(reason string) *linuxCommandAttributor {
		return &linuxCommandAttributor{ops: ops, result: unavailableCgroup(reason)}
	}
	parent, membership, err := currentLinuxCgroupParent(root, ops.readFile)
	if err != nil {
		return unavailableResult(err.Error())
	}
	capacity := readLinuxCgroupCPUCapacity(root, membership, runtime.GOMAXPROCS(0), ops.readFile)
	path, err := ops.mkdirTemp(parent, "fak-systembaseline-")
	if err != nil {
		return unavailableResult(fmt.Sprintf("cgroup v2 delegation unavailable: %v", err))
	}
	cleanupFailure := func(reason string) *linuxCommandAttributor {
		result := unavailableCgroup(reason)
		result.Cleanup.Attempted = true
		if err := ops.remove(path); err == nil {
			result.Cleanup.Empty = true
			result.Cleanup.Removed = true
		} else {
			result.Cleanup.Reason = err.Error()
		}
		return &linuxCommandAttributor{ops: ops, result: result, finished: true}
	}
	initial := readCgroupRawSnapshot(path, ops.readFile)
	if reason := coreCounterUnavailable(initial); reason != "" {
		return cleanupFailure(reason)
	}
	dir, err := ops.open(path)
	if err != nil {
		return cleanupFailure(fmt.Sprintf("open delegated cgroup: %v", err))
	}
	return &linuxCommandAttributor{
		ops:     ops,
		path:    path,
		dir:     dir,
		initial: initial,
		result: CgroupV2{
			State:       CgroupStateMeasured,
			CPUCapacity: &capacity,
			Membership: CgroupMembership{
				AfterStart:      unavailable("processes", "command has not started"),
				AfterWait:       unavailable("processes", "command has not completed"),
				PlacementSource: "clone3 CLONE_INTO_CGROUP via exec.Cmd SysProcAttr.CgroupFD",
			},
		},
	}
}

func currentLinuxCgroupParent(root string, readFile func(string) ([]byte, error)) (string, string, error) {
	if _, err := readFile(filepath.Join(root, "cgroup.controllers")); err != nil {
		return "", "", fmt.Errorf("cgroup v2 unavailable: %v", err)
	}
	raw, err := readFile("/proc/self/cgroup")
	if err != nil {
		return "", "", fmt.Errorf("read current cgroup membership: %v", err)
	}
	var relative string
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.HasPrefix(line, "0::") {
			relative = strings.TrimPrefix(line, "0::")
			break
		}
	}
	if relative == "" {
		return "", "", fmt.Errorf("unified cgroup v2 membership unavailable")
	}
	clean := filepath.Clean("/" + strings.TrimPrefix(relative, "/"))
	parent := filepath.Join(root, clean)
	rootClean := filepath.Clean(root)
	if parent != rootClean && !strings.HasPrefix(parent, rootClean+string(os.PathSeparator)) {
		return "", "", fmt.Errorf("current cgroup path escapes cgroup v2 mount")
	}
	return parent, clean, nil
}

func readLinuxCgroupCPUCapacity(root, membership string, runtimeWidth int, readFile func(string) ([]byte, error)) CgroupCPUCapacity {
	unavailableCapacity := func(reason string) CgroupCPUCapacity {
		return CgroupCPUCapacity{MembershipPath: membership, Reason: reason}
	}
	if runtimeWidth <= 0 {
		return unavailableCapacity("runtime GOMAXPROCS is unavailable")
	}
	root = filepath.Clean(root)
	membership = filepath.Clean("/" + strings.TrimPrefix(membership, "/"))
	current := filepath.Join(root, membership)
	if current != root && !strings.HasPrefix(current, root+string(os.PathSeparator)) {
		return unavailableCapacity("current cgroup path escapes cgroup v2 mount")
	}
	var best CgroupCPUCapacity
	for {
		raw, err := readFile(filepath.Join(current, "cpu.max"))
		if err != nil {
			return unavailableCapacity(fmt.Sprintf("read cpu.max at %s: %v", current, err))
		}
		quota, period, finite, err := parseLinuxCPUMax(raw)
		if err != nil {
			return unavailableCapacity(fmt.Sprintf("parse cpu.max at %s: %v", current, err))
		}
		if finite {
			capacity := float64(quota) / float64(period)
			if !best.Available || capacity < best.CapacityCPUs {
				best = CgroupCPUCapacity{
					Available: true, CapacityCPUs: capacity, RuntimeWidth: runtimeWidth,
					QuotaUS: quota, PeriodUS: period, EffectivePath: current, MembershipPath: membership,
					RuntimeDefaultMayOverestimate: float64(runtimeWidth) > capacity,
					Source:                        "cgroup v2 cpu.max hierarchy",
				}
			}
		}
		if current == root {
			break
		}
		current = filepath.Dir(current)
	}
	if !best.Available {
		return unavailableCapacity("cgroup v2 cpu.max hierarchy has no finite quota")
	}
	return best
}

func parseLinuxCPUMax(raw []byte) (quota, period uint64, finite bool, err error) {
	fields := strings.Fields(string(raw))
	if len(fields) != 2 {
		return 0, 0, false, errors.New("expected quota and period")
	}
	period, err = strconv.ParseUint(fields[1], 10, 64)
	if err != nil || period == 0 {
		return 0, 0, false, errors.New("period must be a positive integer")
	}
	if fields[0] == "max" {
		return 0, period, false, nil
	}
	quota, err = strconv.ParseUint(fields[0], 10, 64)
	if err != nil || quota == 0 {
		return 0, 0, false, errors.New("quota must be max or a positive integer")
	}
	return quota, period, true, nil
}

func (l *linuxCommandAttributor) active() bool {
	return l != nil && !l.finished && l.path != "" && l.dir != nil
}

func (l *linuxCommandAttributor) configure(cmd *exec.Cmd) bool {
	if !l.active() || cmd == nil {
		return false
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.UseCgroupFD = true
	cmd.SysProcAttr.CgroupFD = int(l.dir.Fd())
	l.configured = true
	return true
}

func (l *linuxCommandAttributor) started(pid int) error {
	if !l.active() || !l.configured {
		return nil
	}
	l.result.Membership.AtomicPlacement = true
	l.result.Membership.RootPID = pid
	if count, err := readCgroupProcessCount(l.path, l.ops.readFile); err == nil {
		l.result.Membership.AfterStart = available(float64(count), "processes", "delegated cgroup v2 cgroup.procs")
	} else {
		l.result.Membership.AfterStart = unavailable("processes", err.Error())
	}
	_ = l.closeDir()
	return nil
}

func (l *linuxCommandAttributor) launchFailed(startErr error) {
	if l == nil || l.finished {
		return
	}
	if !l.active() {
		l.finished = true
		return
	}
	reason := "atomic cgroup launch unavailable"
	if startErr != nil {
		reason += ": " + startErr.Error()
	}
	l.result = unavailableCgroup(reason)
	l.result.Cleanup.Attempted = true
	_ = l.closeDir()
	if err := l.ops.remove(l.path); err != nil {
		l.result.Cleanup.Reason = err.Error()
	} else {
		l.result.Cleanup.Empty = true
		l.result.Cleanup.Removed = true
	}
	l.finished = true
}

func (l *linuxCommandAttributor) finish() CgroupV2 {
	if l == nil {
		return unavailableCgroup("Linux cgroup attribution adapter unavailable")
	}
	if l.finished {
		return l.result.clone()
	}
	if !l.active() && l.result.State == CgroupStateUnavailable {
		l.finished = true
		return l.result.clone()
	}
	l.finished = true
	_ = l.closeDir()
	if !l.configured || !l.result.Membership.AtomicPlacement {
		l.result = unavailableCgroup("command was not atomically placed in the delegated cgroup")
		return l.result.clone()
	}

	l.result.Cleanup.Attempted = true
	members, memberErr := readCgroupPIDs(l.path, l.ops.readFile)
	if memberErr == nil {
		l.result.Membership.AfterWait = available(float64(len(members)), "processes", "delegated cgroup v2 cgroup.procs")
	} else {
		l.result.Membership.AfterWait = unavailable("processes", memberErr.Error())
	}
	atCommandEnd := readCgroupRawSnapshot(l.path, l.ops.readFile)
	if len(members) > 0 {
		l.result.Cleanup.KilledRemaining = true
		if err := l.killRemaining(members); err != nil {
			l.result.Cleanup.Reason = err.Error()
		}
	}
	empty, emptyErr := l.waitEmpty(2 * time.Second)
	l.result.Cleanup.Empty = empty
	if emptyErr != nil && l.result.Cleanup.Reason == "" {
		l.result.Cleanup.Reason = emptyErr.Error()
	}

	final := readCgroupRawSnapshot(l.path, l.ops.readFile)
	l.foldFinal(atCommandEnd, final)
	if err := l.ops.remove(l.path); err != nil {
		if l.result.Cleanup.Reason == "" {
			l.result.Cleanup.Reason = err.Error()
		}
	} else {
		l.result.Cleanup.Removed = true
	}
	if !l.result.Cleanup.Empty || !l.result.Cleanup.Removed {
		if l.result.Cleanup.Reason == "" {
			l.result.Cleanup.Reason = "cgroup did not become empty and removable"
		}
	}
	return l.result.clone()
}

func (l *linuxCommandAttributor) closeDir() error {
	if l.dir == nil {
		return nil
	}
	err := l.dir.Close()
	l.dir = nil
	return err
}

func (l *linuxCommandAttributor) killRemaining(pids []int) error {
	if err := l.ops.writeFile(filepath.Join(l.path, "cgroup.kill"), []byte("1"), 0o644); err == nil {
		return nil
	}
	var errs []error
	for _, pid := range pids {
		if err := l.ops.kill(pid); err != nil && !errors.Is(err, os.ErrProcessDone) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (l *linuxCommandAttributor) waitEmpty(timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	for {
		pids, err := readCgroupPIDs(l.path, l.ops.readFile)
		if err != nil {
			return false, err
		}
		if len(pids) == 0 {
			return true, nil
		}
		if time.Now().After(deadline) {
			return false, fmt.Errorf("cgroup still contains %d process(es) after cleanup timeout", len(pids))
		}
		l.ops.sleep(10 * time.Millisecond)
	}
}

func (l *linuxCommandAttributor) foldFinal(atCommandEnd, final cgroupRawSnapshot) {
	l.result.CPU = deltaCounterSet(l.initial.cpu, final.cpu, "delegated cgroup v2 cpu.stat")
	l.result.Memory.CurrentBytes = atCommandEnd.memoryCurrent
	l.result.Memory.PeakBytes = final.memoryPeak
	l.result.Memory.Events = deltaCounterSet(l.initial.memoryEvents, final.memoryEvents, "delegated cgroup v2 memory.events")
	l.result.Pressure = deltaPressure(l.initial.pressure, final.pressure)
	if reason := coreCounterUnavailableFinal(l.result); reason != "" {
		l.result.State = CgroupStateUnavailable
		l.result.Reason = reason
	}
}

func coreCounterUnavailable(s cgroupRawSnapshot) string {
	switch {
	case !s.cpu.Available:
		return s.cpu.Reason
	case s.cpu.Values["usage_usec"] == 0 && !counterPresent(s.cpu.Values, "usage_usec"):
		return "cpu.stat lacks usage_usec"
	case !s.memoryCurrent.Available:
		return s.memoryCurrent.Reason
	case !s.memoryPeak.Available:
		return s.memoryPeak.Reason
	case !s.memoryEvents.Available:
		return s.memoryEvents.Reason
	default:
		return ""
	}
}

func coreCounterUnavailableFinal(c CgroupV2) string {
	switch {
	case !c.CPU.Available:
		return c.CPU.Reason
	case !counterPresent(c.CPU.Values, "usage_usec"):
		return "cpu.stat lacks usage_usec"
	case !c.Memory.CurrentBytes.Available:
		return c.Memory.CurrentBytes.Reason
	case !c.Memory.PeakBytes.Available:
		return c.Memory.PeakBytes.Reason
	case !c.Memory.Events.Available:
		return c.Memory.Events.Reason
	default:
		return ""
	}
}

func counterPresent(values map[string]uint64, key string) bool {
	_, ok := values[key]
	return ok
}

func readCgroupRawSnapshot(path string, readFile func(string) ([]byte, error)) cgroupRawSnapshot {
	cpu := readNumericCounterFile(filepath.Join(path, "cpu.stat"), readFile)
	current := readUintMetric(filepath.Join(path, "memory.current"), "bytes", "delegated cgroup v2 memory.current", readFile)
	peak := readUintMetric(filepath.Join(path, "memory.peak"), "bytes", "delegated cgroup v2 memory.peak", readFile)
	events := readNumericCounterFile(filepath.Join(path, "memory.events"), readFile)
	if events.Available {
		events.Source = "delegated cgroup v2 memory.events"
	}
	if cpu.Available {
		cpu.Source = "delegated cgroup v2 cpu.stat"
	}
	return cgroupRawSnapshot{
		cpu:           cpu,
		memoryCurrent: current,
		memoryPeak:    peak,
		memoryEvents:  events,
		pressure: CgroupPressure{
			CPU:    readPressureAxis(filepath.Join(path, "cpu.pressure"), "cpu", readFile),
			Memory: readPressureAxis(filepath.Join(path, "memory.pressure"), "memory", readFile),
			IO:     readPressureAxis(filepath.Join(path, "io.pressure"), "io", readFile),
		},
	}
}

func readNumericCounterFile(path string, readFile func(string) ([]byte, error)) CounterSet {
	raw, err := readFile(path)
	if err != nil {
		return unavailableCounterSet(fmt.Sprintf("%s unavailable: %v", filepath.Base(path), err))
	}
	values := map[string]uint64{}
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			return unavailableCounterSet(fmt.Sprintf("%s has malformed counter line", filepath.Base(path)))
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return unavailableCounterSet(fmt.Sprintf("%s counter %s is invalid: %v", filepath.Base(path), fields[0], err))
		}
		values[fields[0]] = value
	}
	if err := scanner.Err(); err != nil {
		return unavailableCounterSet(fmt.Sprintf("read %s: %v", filepath.Base(path), err))
	}
	return CounterSet{Available: true, Values: values, Source: path}
}

func readUintMetric(path, unit, source string, readFile func(string) ([]byte, error)) Metric {
	raw, err := readFile(path)
	if err != nil {
		return unavailable(unit, fmt.Sprintf("%s unavailable: %v", filepath.Base(path), err))
	}
	value, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil {
		return unavailable(unit, fmt.Sprintf("%s is invalid: %v", filepath.Base(path), err))
	}
	return available(float64(value), unit, source)
}

func readPressureAxis(path, resource string, readFile func(string) ([]byte, error)) PressureAxis {
	raw, err := readFile(path)
	if err != nil {
		reason := fmt.Sprintf("%s PSI unavailable: %v", resource, err)
		return PressureAxis{Some: unavailable("microseconds", reason), Full: unavailable("microseconds", reason)}
	}
	lines := map[string]Metric{}
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		total := ""
		for _, field := range fields[1:] {
			if strings.HasPrefix(field, "total=") {
				total = strings.TrimPrefix(field, "total=")
				break
			}
		}
		value, parseErr := strconv.ParseUint(total, 10, 64)
		if total == "" || parseErr != nil {
			lines[fields[0]] = unavailable("microseconds", fmt.Sprintf("%s PSI %s total unavailable", resource, fields[0]))
			continue
		}
		lines[fields[0]] = available(float64(value), "microseconds", "delegated cgroup v2 "+filepath.Base(path))
	}
	if err := scanner.Err(); err != nil {
		reason := fmt.Sprintf("read %s PSI: %v", resource, err)
		return PressureAxis{Some: unavailable("microseconds", reason), Full: unavailable("microseconds", reason)}
	}
	some, ok := lines["some"]
	if !ok {
		some = unavailable("microseconds", resource+" PSI some line unavailable")
	}
	full, ok := lines["full"]
	if !ok {
		full = unavailable("microseconds", resource+" PSI full line unavailable")
	}
	return PressureAxis{Some: some, Full: full}
}

func deltaCounterSet(initial, final CounterSet, source string) CounterSet {
	if !initial.Available {
		return unavailableCounterSet(initial.Reason)
	}
	if !final.Available {
		return unavailableCounterSet(final.Reason)
	}
	values := make(map[string]uint64, len(final.Values))
	for key, finalValue := range final.Values {
		initialValue := initial.Values[key]
		if finalValue < initialValue {
			return unavailableCounterSet(fmt.Sprintf("%s counter %s regressed", filepath.Base(source), key))
		}
		values[key] = finalValue - initialValue
	}
	return availableCounterSet(values, source)
}

func deltaPressure(initial, final CgroupPressure) CgroupPressure {
	return CgroupPressure{
		CPU:    deltaPressureAxis(initial.CPU, final.CPU),
		Memory: deltaPressureAxis(initial.Memory, final.Memory),
		IO:     deltaPressureAxis(initial.IO, final.IO),
	}
}

func deltaPressureAxis(initial, final PressureAxis) PressureAxis {
	return PressureAxis{
		Some: deltaMetric(initial.Some, final.Some),
		Full: deltaMetric(initial.Full, final.Full),
	}
}

func deltaMetric(initial, final Metric) Metric {
	if !initial.Available {
		return initial
	}
	if !final.Available {
		return final
	}
	if final.Value < initial.Value {
		return unavailable(final.Unit, "cgroup cumulative counter regressed")
	}
	return available(final.Value-initial.Value, final.Unit, final.Source)
}

func readCgroupProcessCount(path string, readFile func(string) ([]byte, error)) (int, error) {
	pids, err := readCgroupPIDs(path, readFile)
	return len(pids), err
}

func readCgroupPIDs(path string, readFile func(string) ([]byte, error)) ([]int, error) {
	raw, err := readFile(filepath.Join(path, "cgroup.procs"))
	if err != nil {
		return nil, fmt.Errorf("cgroup.procs unavailable: %w", err)
	}
	var pids []int
	for _, field := range strings.Fields(string(raw)) {
		pid, err := strconv.Atoi(field)
		if err != nil || pid <= 0 {
			return nil, fmt.Errorf("cgroup.procs contains invalid PID %q", field)
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

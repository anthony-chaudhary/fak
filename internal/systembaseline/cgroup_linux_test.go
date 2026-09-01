//go:build linux

package systembaseline

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func writeFakeCgroupFiles(t *testing.T, dir string, usageUS, peakBytes, psiCPUUS uint64, procs string) {
	t.Helper()
	files := map[string]string{
		"cpu.stat":       "usage_usec " + decimal(usageUS) + "\nuser_usec " + decimal(usageUS*3/4) + "\nsystem_usec " + decimal(usageUS/4) + "\n",
		"memory.current": "0\n",
		"memory.peak":    decimal(peakBytes) + "\n",
		"memory.events":  "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
		"cpu.pressure":   "some avg10=0.00 avg60=0.00 avg300=0.00 total=" + decimal(psiCPUUS) + "\n",
		"memory.pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n" +
			"full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
		"io.pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n" +
			"full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
		"cgroup.procs": procs,
		"cgroup.kill":  "",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func decimal(value uint64) string {
	return strconv.FormatUint(value, 10)
}

func fakeLinuxAttributor(t *testing.T, procs string) (*linuxCommandAttributor, string) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "scope")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	writeFakeCgroupFiles(t, dir, 0, 0, 0, procs)
	ops := defaultCgroupFSOps()
	ops.remove = os.RemoveAll
	ops.sleep = func(time.Duration) {}
	ops.kill = func(int) error { return nil }
	fd, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	scope := &linuxCommandAttributor{
		ops:        ops,
		path:       dir,
		dir:        fd,
		initial:    readCgroupRawSnapshot(dir, ops.readFile),
		configured: true,
		result: CgroupV2{
			State: CgroupStateMeasured,
			Membership: CgroupMembership{
				AfterStart:      unavailable("processes", "not started"),
				AfterWait:       unavailable("processes", "not finished"),
				PlacementSource: "test atomic placement",
			},
		},
	}
	scope.started(123)
	return scope, dir
}

func TestLinuxCgroupParsersPreserveCountersAndUnavailablePSI(t *testing.T) {
	dir := t.TempDir()
	writeFakeCgroupFiles(t, dir, 1234, 4096, 17, "11\n12\n")
	snapshot := readCgroupRawSnapshot(dir, os.ReadFile)
	if !snapshot.cpu.Available || snapshot.cpu.Values["usage_usec"] != 1234 || snapshot.memoryPeak.Value != 4096 || snapshot.pressure.CPU.Some.Value != 17 {
		t.Fatalf("snapshot=%+v", snapshot)
	}
	if snapshot.pressure.CPU.Full.Available || !strings.Contains(snapshot.pressure.CPU.Full.Reason, "full line unavailable") {
		t.Fatalf("CPU full PSI=%+v", snapshot.pressure.CPU.Full)
	}
	if count, err := readCgroupProcessCount(dir, os.ReadFile); err != nil || count != 2 {
		t.Fatalf("members=%d err=%v", count, err)
	}
}

func TestLinuxCgroupCleanupCoversSuccessFailureAndTimeout(t *testing.T) {
	tests := []struct {
		name          string
		remaining     string
		wantKilled    bool
		commandStatus string
	}{
		{name: "success", commandStatus: "exit 0"},
		{name: "failure", commandStatus: "exit 7"},
		{name: "timeout", remaining: "456\n", wantKilled: true, commandStatus: "deadline exceeded"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scope, dir := fakeLinuxAttributor(t, "")
			writeFakeCgroupFiles(t, dir, 50_000, 8<<20, 500, test.remaining)
			if test.remaining != "" {
				scope.ops.writeFile = func(path string, body []byte, mode os.FileMode) error {
					if filepath.Base(path) != "cgroup.kill" || string(body) != "1" {
						return errors.New("unexpected cleanup write")
					}
					return os.WriteFile(filepath.Join(dir, "cgroup.procs"), nil, mode)
				}
			}
			result := scope.finish()
			if result.State != CgroupStateMeasured || result.CPU.Values["usage_usec"] != 50_000 || result.Memory.PeakBytes.Value != 8<<20 {
				t.Fatalf("%s result=%+v", test.commandStatus, result)
			}
			if !result.Cleanup.Attempted || !result.Cleanup.Empty || !result.Cleanup.Removed || result.Cleanup.KilledRemaining != test.wantKilled {
				t.Fatalf("%s cleanup=%+v", test.commandStatus, result.Cleanup)
			}
			if _, err := os.Stat(dir); !os.IsNotExist(err) {
				t.Fatalf("%s cgroup still exists: %v", test.commandStatus, err)
			}
		})
	}
}

func TestLinuxCgroupIntegrationShortLivedDescendants(t *testing.T) {
	if os.Getenv("FAK_SYSTEMBASELINE_CGROUP_INTEGRATION") != "1" {
		t.Skip("set FAK_SYSTEMBASELINE_CGROUP_INTEGRATION=1 on a delegated cgroup v2 host")
	}
	scope := NewCommandAttributor()
	if !scope.Active() {
		t.Fatal("integration gate requested but cgroup v2 delegation is unavailable")
	}
	cmd := exec.Command("/bin/sh", "-c", `i=0; while [ "$i" -lt 24 ]; do /bin/sh -c 'j=0; while [ "$j" -lt 10000 ]; do j=$((j+1)); done' & i=$((i+1)); done; wait`)
	if !scope.Configure(cmd) {
		t.Fatal("delegated cgroup was not configured")
	}
	if err := cmd.Start(); err != nil {
		scope.LaunchFailed(err)
		t.Fatalf("atomic cgroup launch: %v", err)
	}
	scope.Started(cmd.Process.Pid)
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	result := scope.Finish()
	if result.State != CgroupStateMeasured || result.CPU.Values["usage_usec"] == 0 {
		t.Fatalf("short-lived descendant CPU not retained: %+v", result)
	}
	if !result.Cleanup.Empty || !result.Cleanup.Removed {
		t.Fatalf("cleanup=%+v", result.Cleanup)
	}
}

func TestLinuxCgroupCPUCapacityPreservesFractionAndTightestAncestor(t *testing.T) {
	root := t.TempDir()
	leaf := filepath.Join(root, "tenant", "job")
	if err := os.MkdirAll(leaf, 0o700); err != nil {
		t.Fatal(err)
	}
	for path, body := range map[string]string{
		filepath.Join(root, "cpu.max"):           "max 100000\n",
		filepath.Join(root, "tenant", "cpu.max"): "150000 100000\n",
		filepath.Join(leaf, "cpu.max"):           "200000 100000\n",
	} {
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := readLinuxCgroupCPUCapacity(root, "/tenant/job", 2, os.ReadFile)
	if !got.Available || got.CapacityCPUs != 1.5 || got.RuntimeWidth != 2 || got.QuotaUS != 150000 || got.PeriodUS != 100000 {
		t.Fatalf("capacity=%+v", got)
	}
	if got.EffectivePath != filepath.Join(root, "tenant") || got.MembershipPath != "/tenant/job" || !got.RuntimeDefaultMayOverestimate {
		t.Fatalf("hierarchy identity=%+v", got)
	}
}

func TestLinuxCgroupCPUCapacityReportsUnavailableWithoutFiniteHierarchy(t *testing.T) {
	root := t.TempDir()
	leaf := filepath.Join(root, "job")
	if err := os.MkdirAll(leaf, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{filepath.Join(root, "cpu.max"), filepath.Join(leaf, "cpu.max")} {
		if err := os.WriteFile(path, []byte("max 100000\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := readLinuxCgroupCPUCapacity(root, "/job", 8, os.ReadFile)
	if got.Available || got.Reason == "" || got.RuntimeWidth != 0 || got.CapacityCPUs != 0 {
		t.Fatalf("capacity=%+v", got)
	}
}

func TestParseLinuxCPUMaxRejectsAmbiguousValues(t *testing.T) {
	for _, raw := range []string{"", "max", "0 100000", "150000 0", "1.5 100000", "max nope", "1 2 3"} {
		if _, _, _, err := parseLinuxCPUMax([]byte(raw)); err == nil {
			t.Fatalf("parseLinuxCPUMax(%q) succeeded", raw)
		}
	}
	quota, period, finite, err := parseLinuxCPUMax([]byte("max 100000"))
	if err != nil || finite || quota != 0 || period != 100000 {
		t.Fatalf("unlimited=%d/%d finite=%v err=%v", quota, period, finite, err)
	}
}

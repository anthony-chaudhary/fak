//go:build darwin

package procguard

import (
	"errors"
	"os/exec"
	"strings"
	"testing"
)

func TestJoinDarwinMemorySnapshotIncludesRootAndCycleSafeDescendants(t *testing.T) {
	census := []Proc{
		{PID: 100, Name: "root", WSMB: IntPtr(11)},
		{PID: 101, Name: "child", WSMB: IntPtr(7)},
		{PID: 102, Name: "grandchild", WSMB: IntPtr(5)},
		{PID: 999, Name: "unowned", WSMB: IntPtr(1000)},
	}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(102), Name: "root", Cmdline: "root --owned"},
		{PID: 101, PPID: IntPtr(100), Name: "child", Cmdline: "child --owned"},
		{PID: 102, PPID: IntPtr(101), Name: "grandchild", Cmdline: "grandchild --owned"},
		{PID: 999, PPID: IntPtr(1), Name: "unowned", Cmdline: "unowned"},
	}

	snapshot, detail := joinDarwinMemorySnapshot(100, census, relations)
	if detail != "" {
		t.Fatalf("detail=%q", detail)
	}
	if snapshot.Metric != MemoryMetricRSS || snapshot.TreeBytes != 23<<20 {
		t.Fatalf("snapshot=%+v", snapshot)
	}
	if len(snapshot.Processes) != 3 || snapshot.Processes[0].PID != 100 || snapshot.Processes[1].PID != 101 || snapshot.Processes[2].PID != 102 {
		t.Fatalf("owned traversal=%+v", snapshot.Processes)
	}
}

func TestJoinDarwinMemorySnapshotRejectsIncompleteRows(t *testing.T) {
	census := []Proc{{PID: 100, Name: "root", WSMB: IntPtr(11)}}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "child"},
	}
	snapshot, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(int) (bool, error) {
		return true, nil
	})
	if !strings.Contains(detail, "owned pids missing from rss census: [101]") {
		t.Fatalf("detail=%q snapshot=%+v", detail, snapshot)
	}
	if len(snapshot.Processes) != 2 {
		t.Fatalf("partial ownership must remain available for fail-closed reap: %+v", snapshot.Processes)
	}
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations[1:], func(int) (bool, error) {
		return true, nil
	}); !strings.Contains(detail, "root pid 100 missing") {
		t.Fatalf("live missing root relation detail=%q", detail)
	}
}

func TestJoinDarwinMemorySnapshotSkipsExitedDescendants(t *testing.T) {
	census := []Proc{
		{PID: 100, Name: "root", WSMB: IntPtr(11)},
		{PID: 103, Name: "live-child", WSMB: IntPtr(7)},
	}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "exited-child"},
		{PID: 102, PPID: IntPtr(101), Name: "exited-grandchild"},
		{PID: 103, PPID: IntPtr(100), Name: "live-child"},
	}
	probed := make(map[int]int)
	snapshot, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(pid int) (bool, error) {
		probed[pid]++
		return false, nil
	})
	if detail != "" {
		t.Fatalf("normal descendant exit churn must not fail the snapshot: %q", detail)
	}
	if snapshot.TreeBytes != 18<<20 {
		t.Fatalf("tree bytes=%d snapshot=%+v", snapshot.TreeBytes, snapshot)
	}
	if len(snapshot.Processes) != 2 || snapshot.Processes[0].PID != 100 || snapshot.Processes[1].PID != 103 {
		t.Fatalf("exited descendants must be omitted: %+v", snapshot.Processes)
	}
	if probed[101] != 1 || probed[102] != 1 || len(probed) != 2 {
		t.Fatalf("only missing descendant rows should be probed: %+v", probed)
	}
}

func TestJoinDarwinMemorySnapshotTreatsMissingExitedRootAsExitChurn(t *testing.T) {
	snapshot, detail := joinDarwinMemorySnapshotWithProbe(100, nil, nil, func(pid int) (bool, error) {
		if pid != 100 {
			t.Fatalf("probe pid=%d, want root", pid)
		}
		return false, nil
	})
	if detail != "" {
		t.Fatalf("vanished root must defer to child wait path, detail=%q snapshot=%+v", detail, snapshot)
	}
	if snapshot.RootPID != 0 || len(snapshot.Processes) != 0 {
		t.Fatalf("vanished-root terminal snapshot=%+v", snapshot)
	}
}

func TestJoinDarwinMemorySnapshotKeepsLiveRootAndProbeFailuresFatal(t *testing.T) {
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, nil, nil, func(int) (bool, error) {
		return true, nil
	}); !strings.Contains(detail, "root pid 100 missing from relation census") {
		t.Fatalf("live missing relation root detail=%q", detail)
	}
	probeErr := errors.New("liveness unavailable")
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, nil, nil, func(int) (bool, error) {
		return false, probeErr
	}); !strings.Contains(detail, "probe missing relation root pid 100: liveness unavailable") {
		t.Fatalf("missing relation root probe failure detail=%q", detail)
	}

	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "child"},
	}
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, nil, relations, func(int) (bool, error) {
		// Descendants may still be reconciled so the partial ownership snapshot
		// remains useful, but an exited root itself must never be suppressed.
		return false, nil
	}); !strings.Contains(detail, "owned pids missing from rss census: [100]") {
		t.Fatalf("missing root detail=%q", detail)
	}

	census := []Proc{{PID: 100, Name: "root", WSMB: IntPtr(11)}}
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(int) (bool, error) {
		return false, probeErr
	}); !strings.Contains(detail, "probe missing rss pid 101: liveness unavailable") {
		t.Fatalf("probe failure detail=%q", detail)
	}
}

func TestCollectDarwinMemorySnapshotStopsQuietlyWhenRootExitsBeforeRelationCensus(t *testing.T) {
	censusCalls := 0
	relationCalls := 0
	probed := 0
	snapshot, detail := collectDarwinMemorySnapshotWithCollectors(100, func() ([]Proc, string) {
		censusCalls++
		return nil, ""
	}, func() ([]Proc, string) {
		relationCalls++
		return nil, ""
	}, func(pid int) (bool, error) {
		probed++
		if pid != 100 {
			t.Fatalf("probe pid=%d, want root", pid)
		}
		return false, nil
	})
	if detail != "" || snapshot.RootPID != 0 {
		t.Fatalf("vanished root must produce a terminal sample: detail=%q snapshot=%+v", detail, snapshot)
	}
	if censusCalls != 1 || relationCalls != 1 || probed != 1 {
		t.Fatalf("collections census=%d relations=%d probes=%d, want one graceful terminal sample", censusCalls, relationCalls, probed)
	}
}

func TestCollectDarwinMemorySnapshotRecollectsPersistentLiveMissingRow(t *testing.T) {
	root := Proc{PID: 100, PPID: IntPtr(1), Name: "root", WSMB: IntPtr(11)}
	child := Proc{PID: 101, PPID: IntPtr(100), Name: "child", WSMB: IntPtr(7)}
	censusCalls := 0
	relationCalls := 0
	snapshot, detail := collectDarwinMemorySnapshotWithCollectors(100, func() ([]Proc, string) {
		censusCalls++
		if censusCalls == 1 {
			return []Proc{root}, ""
		}
		return []Proc{root, child}, ""
	}, func() ([]Proc, string) {
		relationCalls++
		return []Proc{root, child}, ""
	}, func(pid int) (bool, error) {
		if pid != child.PID {
			t.Fatalf("unexpected liveness probe pid=%d", pid)
		}
		return true, nil
	})
	if detail != "" {
		t.Fatalf("persistent live missing row was not reconciled: detail=%q snapshot=%+v", detail, snapshot)
	}
	if censusCalls != 2 || relationCalls != 2 {
		t.Fatalf("collections census=%d relations=%d, want one bounded recollection", censusCalls, relationCalls)
	}
	if snapshot.TreeBytes != 18<<20 || len(snapshot.Processes) != 2 || snapshot.Processes[1].PID != child.PID {
		t.Fatalf("reconciled snapshot=%+v", snapshot)
	}
}

func TestCollectDarwinMemorySnapshotRelationFirstDefersLaterStarts(t *testing.T) {
	root := Proc{PID: 100, PPID: IntPtr(1), Name: "root", WSMB: IntPtr(11)}
	stable := Proc{PID: 101, PPID: IntPtr(100), Name: "stable", WSMB: IntPtr(7)}
	laterStarts := []Proc{
		{PID: 102, PPID: IntPtr(100), Name: "later-1", WSMB: IntPtr(1)},
		{PID: 103, PPID: IntPtr(100), Name: "later-2", WSMB: IntPtr(1)},
		{PID: 104, PPID: IntPtr(100), Name: "later-3", WSMB: IntPtr(1)},
		{PID: 105, PPID: IntPtr(100), Name: "later-4", WSMB: IntPtr(1)},
	}
	running := []Proc{root, stable}
	var callOrder []string
	censusCalls := 0
	relationCalls := 0
	snapshot, detail := collectDarwinMemorySnapshotWithCollectors(100, func() ([]Proc, string) {
		callOrder = append(callOrder, "rss")
		censusCalls++
		return append([]Proc(nil), running...), ""
	}, func() ([]Proc, string) {
		callOrder = append(callOrder, "relations")
		relationCalls++
		ownedAtBoundary := append([]Proc(nil), running...)
		running = append(running, laterStarts...)
		return ownedAtBoundary, ""
	}, func(pid int) (bool, error) {
		t.Fatalf("later starts must not enter the earlier ownership epoch: pid=%d", pid)
		return false, nil
	})
	if detail != "" {
		t.Fatalf("later starts produced an incomplete snapshot: detail=%q snapshot=%+v", detail, snapshot)
	}
	if censusCalls != 1 || relationCalls != 1 {
		t.Fatalf("collections census=%d relations=%d, want one coherent pair", censusCalls, relationCalls)
	}
	if len(callOrder) != 2 || callOrder[0] != "relations" || callOrder[1] != "rss" {
		t.Fatalf("collector order=%v, want [relations rss]", callOrder)
	}
	if snapshot.TreeBytes != 18<<20 || len(snapshot.Processes) != 2 || snapshot.Processes[1].PID != stable.PID {
		t.Fatalf("ownership-epoch snapshot=%+v", snapshot)
	}
}

func TestCollectDarwinMemorySnapshotPersistentMissingRowsFailClosed(t *testing.T) {
	root := Proc{PID: 100, PPID: IntPtr(1), Name: "root", WSMB: IntPtr(11)}
	child := Proc{PID: 101, PPID: IntPtr(100), Name: "child", WSMB: IntPtr(7)}
	for _, tc := range []struct {
		name       string
		census     []Proc
		wantDetail string
	}{
		{name: "live root", census: []Proc{child}, wantDetail: "owned pids missing from rss census: [100]"},
		{name: "live descendant", census: []Proc{root}, wantDetail: "owned pids missing from rss census: [101]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			censusCalls := 0
			relationCalls := 0
			snapshot, detail := collectDarwinMemorySnapshotWithCollectors(100, func() ([]Proc, string) {
				censusCalls++
				return tc.census, ""
			}, func() ([]Proc, string) {
				relationCalls++
				return []Proc{root, child}, ""
			}, func(int) (bool, error) { return true, nil })
			if !strings.Contains(detail, tc.wantDetail) {
				t.Fatalf("persistent missing row detail=%q snapshot=%+v", detail, snapshot)
			}
			if censusCalls != darwinMemorySnapshotAttempts || relationCalls != darwinMemorySnapshotAttempts {
				t.Fatalf("collections census=%d relations=%d, want bounded attempts=%d", censusCalls, relationCalls, darwinMemorySnapshotAttempts)
			}
		})
	}
}

func TestCollectDarwinMemorySnapshotCollectorErrorsDoNotRetry(t *testing.T) {
	root := Proc{PID: 100, PPID: IntPtr(1), Name: "root", WSMB: IntPtr(11)}
	for _, tc := range []struct {
		name        string
		censusErr   string
		relationErr string
	}{
		{name: "rss", censusErr: "rss collector failed"},
		{name: "relations", relationErr: "relation collector failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			censusCalls := 0
			relationCalls := 0
			snapshot, detail := collectDarwinMemorySnapshotWithCollectors(100, func() ([]Proc, string) {
				censusCalls++
				if tc.censusErr != "" {
					return nil, tc.censusErr
				}
				return []Proc{root}, ""
			}, func() ([]Proc, string) {
				relationCalls++
				if tc.relationErr != "" {
					return nil, tc.relationErr
				}
				return []Proc{root}, ""
			}, func(int) (bool, error) { return true, nil })
			wantErr := tc.censusErr
			if wantErr == "" {
				wantErr = tc.relationErr
			}
			if !strings.Contains(detail, wantErr) {
				t.Fatalf("collector failure lost fail-closed detail=%q snapshot=%+v", detail, snapshot)
			}
			if censusCalls != 1 || relationCalls != 1 {
				t.Fatalf("collector failure retried: census=%d relations=%d", censusCalls, relationCalls)
			}
		})
	}
}

func TestCollectMemorySnapshotOwnProcessUsesRSS(t *testing.T) {
	child := exec.Command("sleep", "5")
	if err := child.Start(); err != nil {
		t.Fatalf("start snapshot target: %v", err)
	}
	t.Cleanup(func() {
		_ = child.Process.Kill()
		_, _ = child.Process.Wait()
	})
	snapshot, supported, detail := CollectMemorySnapshot(child.Process.Pid)
	if !supported || detail != "" {
		t.Fatalf("supported=%v detail=%q snapshot=%+v", supported, detail, snapshot)
	}
	if snapshot.Metric != MemoryMetricRSS || snapshot.TreeBytes == 0 || snapshot.HostPhysicalBytes == 0 {
		t.Fatalf("Darwin snapshot lacks rss/host bytes: %+v", snapshot)
	}
	found := false
	for _, process := range snapshot.Processes {
		found = found || process.PID == child.Process.Pid
	}
	if !found {
		t.Fatalf("root pid %d missing from %+v", child.Process.Pid, snapshot.Processes)
	}
	if _, supported, _ := CollectCommitSnapshot(child.Process.Pid); supported {
		t.Fatal("Darwin RSS must not be exposed through the legacy commit API")
	}
}

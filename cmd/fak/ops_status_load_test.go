package main

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ops"
)

func seedLoadRows(t *testing.T, root string, rows ...ops.LoadCycle) {
	t.Helper()
	ll, err := ops.OpenLoadLedger(ops.DefaultLoadLedgerPath(root))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		if err := ll.Record(r); err != nil {
			t.Fatal(err)
		}
	}
}

func statusJSON(t *testing.T, root string) map[string]any {
	t.Helper()
	cfg := ops.DefaultConfig()
	cfg.WarningFreeBytes, cfg.RefuseFreeBytes = 0, 0
	var out, errb bytes.Buffer
	if rc := runOpsStatus(&out, &errb, root, cfg, []string{"--json"}); rc != 0 {
		t.Fatalf("rc=%d %s", rc, errb.String())
	}
	var m map[string]any
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("json: %v: %s", err, out.String())
	}
	return m
}

func TestOpsStatusLoadAccounting(t *testing.T) {
	now := time.Now()
	row := func(tick uint64, ago time.Duration, wall, cpu int64, v ops.LoadVerdict) ops.LoadCycle {
		fin := now.Add(-ago)
		return ops.LoadCycle{Tick: tick, Started: fin.Add(-time.Duration(wall) * time.Millisecond), Finished: fin,
			WallMS: wall, SelfCPUMillis: cpu, HaveSelfCPU: true, Verdict: v,
			Host: &ops.HostLoad{State: ops.HostSampleMeasured}}
	}

	t.Run("rows", func(t *testing.T) {
		root := t.TempDir()
		seedLoadRows(t, root,
			row(1, 40*time.Second, 1000, 10, ops.LoadProductive),
			row(2, 20*time.Second, 1000, 20, ops.LoadIdle),
			row(3, 5*time.Second, 1000, 30, ops.LoadSkipped),
		)
		m := statusJSON(t, root)
		if m["liveness"] != "LIVE" || m["status"] != "healthy" {
			t.Fatalf("liveness/status: %v %v", m["liveness"], m["status"])
		}
		if m["ticks_24h"] != 3.0 || m["productive_24h"] != 1.0 || m["idle_24h"] != 1.0 || m["skipped_24h"] != 1.0 {
			t.Fatalf("histogram: %v", m)
		}
		// 60 cpu_ms over a ~36s span (first start 41s ago, last finish 5s ago).
		share, ok := m["self_cpu_share_percent"].(float64)
		if !ok || share < 0.15 || share > 0.18 {
			t.Fatalf("share = %v", m["self_cpu_share_percent"])
		}
		if m["duty_cap_percent"] != ops.DefaultMaxDutyPercent || m["reactive_pause_enabled"] != false {
			t.Fatalf("posture: %v", m)
		}
		for _, k := range []string{"status", "warning_free", "refuse_free", "free_disk_bytes", "recent_events"} {
			if _, ok := m[k]; !ok {
				t.Fatalf("pre-existing key %q missing", k)
			}
		}
		if m["host"] == nil {
			t.Fatal("host missing")
		}
	})

	t.Run("never ticked is not healthy", func(t *testing.T) {
		m := statusJSON(t, t.TempDir())
		if m["liveness"] != "NEVER_TICKED" || m["status"] == "healthy" {
			t.Fatalf("got liveness=%v status=%v", m["liveness"], m["status"])
		}
		if m["self_cpu_share_percent"] != nil {
			t.Fatalf("unmeasured share must be null: %v", m["self_cpu_share_percent"])
		}
	})

	t.Run("stale is not healthy", func(t *testing.T) {
		root := t.TempDir()
		seedLoadRows(t, root, row(1, time.Hour, 1000, 10, ops.LoadIdle))
		m := statusJSON(t, root)
		if m["liveness"] != "STALE" || m["status"] != "stale" {
			t.Fatalf("got %v %v", m["liveness"], m["status"])
		}
	})

	t.Run("paused", func(t *testing.T) {
		root := t.TempDir()
		seedLoadRows(t, root, row(1, 5*time.Second, 1000, 10, ops.LoadSkipped))
		ps, err := ops.OpenPauseStore(ops.DefaultPauseStatePath(root))
		if err != nil {
			t.Fatal(err)
		}
		at := now.Add(-time.Minute).UTC().Truncate(time.Second)
		if err := ps.Write(ops.PauseState{Paused: true, ReasonClass: ops.PauseReasonHostPressure, PausedAt: at,
			ReactivateWhen: ops.ReactivateCondition{MaxPause: "30m"}}); err != nil {
			t.Fatal(err)
		}
		m := statusJSON(t, root)
		if m["paused"] != true || m["pause_reason_class"] != "HOST_PRESSURE" || m["status"] != "paused" {
			t.Fatalf("pause: %v", m)
		}
		if m["resume_by"] != at.Add(30*time.Minute).Format(time.RFC3339) {
			t.Fatalf("resume_by = %v", m["resume_by"])
		}
	})
}

func TestOpsLogReadsMixedLedger(t *testing.T) {
	root := t.TempDir()
	seedLoadRows(t, root, ops.LoadCycle{Tick: 1, Started: time.Now(), Finished: time.Now(), Verdict: ops.LoadIdle})
	var out, errb bytes.Buffer
	if rc := runOpsLog(&out, &errb, root, []string{"--json"}); rc != 0 {
		t.Fatalf("rc=%d %s", rc, errb.String())
	}
	if out.Len() != 0 {
		t.Fatalf("load rows leaked into ops log: %s", out.String())
	}
}

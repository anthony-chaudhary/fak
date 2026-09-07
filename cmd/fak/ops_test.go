package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ops"
)

func TestOpsCLIStatusAndSweep(t *testing.T) {
	var out, errb bytes.Buffer

	// status JSON
	rc := runOps(&out, &errb, []string{"status", "--json"})
	if rc != 0 {
		t.Fatalf("runOps status failed with %d: %s", rc, errb.String())
	}

	var rep struct {
		FreeDiskBytes uint64 `json:"free_disk_bytes"`
		WarningFree   bool   `json:"warning_free"`
		RefuseFree    bool   `json:"refuse_free"`
		RecentEvents  int    `json:"recent_events"`
		Status        string `json:"status"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v\noutput: %s", err, out.String())
	}
	if rep.FreeDiskBytes == 0 {
		t.Errorf("expected non-zero FreeDiskBytes, got 0")
	}
	if rep.Status != "healthy" && rep.Status != "warning" && rep.Status != "refuse" {
		t.Errorf("expected valid status string, got %q", rep.Status)
	}

	// status text
	out.Reset()
	errb.Reset()
	rc = runOps(&out, &errb, []string{"status"})
	if rc != 0 {
		t.Fatalf("runOps text status failed with %d: %s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "free_disk=") {
		t.Errorf("expected free_disk= in text status output: %s", out.String())
	}

	out.Reset()
	errb.Reset()

	// sweep dry run
	rc = runOps(&out, &errb, []string{"sweep", "--dry-run", "--json"})
	if rc != 0 {
		t.Fatalf("runOps sweep failed with %d: %s", rc, errb.String())
	}
	if !strings.Contains(out.String(), `"dry_run":true`) {
		t.Errorf("expected dry_run:true in JSON: %s", out.String())
	}
}

func TestOpsCLIStatusWatermarkTransitions(t *testing.T) {
	root := t.TempDir()

	// 1. Trigger "refuse": threshold higher than any disk
	{
		var out, errb bytes.Buffer
		cfg := ops.DefaultConfig()
		cfg.RefuseFreeBytes = ^uint64(0) // max uint64

		rc := runOpsStatus(&out, &errb, root, cfg, []string{"--json"})
		if rc != 0 {
			t.Fatalf("runOpsStatus refuse failed: %d, %s", rc, errb.String())
		}
		var rep struct {
			FreeDiskBytes uint64 `json:"free_disk_bytes"`
			WarningFree   bool   `json:"warning_free"`
			RefuseFree    bool   `json:"refuse_free"`
			Status        string `json:"status"`
		}
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("unmarshal refuse json: %v", err)
		}
		if rep.Status != "refuse" || !rep.RefuseFree {
			t.Errorf("expected refuse status: %+v", rep)
		}
	}

	// 2. Trigger "warning": warning threshold max, refuse 0
	{
		var out, errb bytes.Buffer
		cfg := ops.DefaultConfig()
		cfg.WarningFreeBytes = ^uint64(0)
		cfg.RefuseFreeBytes = 0

		rc := runOpsStatus(&out, &errb, root, cfg, []string{"--json"})
		if rc != 0 {
			t.Fatalf("runOpsStatus warning failed: %d, %s", rc, errb.String())
		}
		var rep struct {
			FreeDiskBytes uint64 `json:"free_disk_bytes"`
			WarningFree   bool   `json:"warning_free"`
			RefuseFree    bool   `json:"refuse_free"`
			Status        string `json:"status"`
		}
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("unmarshal warning json: %v", err)
		}
		if rep.Status != "warning" || !rep.WarningFree || rep.RefuseFree {
			t.Errorf("expected warning status: %+v", rep)
		}
	}

	// 3. Trigger "healthy": both thresholds 0
	{
		var out, errb bytes.Buffer
		cfg := ops.DefaultConfig()
		cfg.WarningFreeBytes = 0
		cfg.RefuseFreeBytes = 0

		rc := runOpsStatus(&out, &errb, root, cfg, []string{"--json"})
		if rc != 0 {
			t.Fatalf("runOpsStatus healthy failed: %d, %s", rc, errb.String())
		}
		var rep struct {
			FreeDiskBytes uint64 `json:"free_disk_bytes"`
			WarningFree   bool   `json:"warning_free"`
			RefuseFree    bool   `json:"refuse_free"`
			Status        string `json:"status"`
		}
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("unmarshal healthy json: %v", err)
		}
		if rep.Status != "healthy" || rep.WarningFree || rep.RefuseFree {
			t.Errorf("expected healthy status: %+v", rep)
		}
	}
}

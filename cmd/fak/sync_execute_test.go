package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncExecuteCLI(t *testing.T) {
	t.Run("execute with packet file succeeds and emits receipt JSON", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")
		syncGit(t, origin, "config", "receive.denyCurrentBranch", "updateInstead")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_feat.txt"), "remote\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote feature")

		syncWriteFile(t, filepath.Join(clone, "local_feat.txt"), "local\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local feature")

		syncGit(t, clone, "fetch", "origin")

		pktOpts := safesync.PacketOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
		}
		pkt, err := safesync.BuildReconciliationPacket(context.Background(), pktOpts)
		if err != nil {
			t.Fatalf("BuildReconciliationPacket: %v", err)
		}

		pktData, err := json.Marshal(pkt)
		if err != nil {
			t.Fatalf("marshal packet: %v", err)
		}
		pktFile := filepath.Join(t.TempDir(), "packet.json")
		if err := os.WriteFile(pktFile, pktData, 0644); err != nil {
			t.Fatalf("write packet file: %v", err)
		}

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"execute", "--packet", pktFile, "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		var receipt safesync.ExecutionReceipt
		if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
			t.Fatalf("JSON decode receipt: %v\n%s", err, out.String())
		}
		if receipt.Schema != safesync.ExecuteReceiptSchema {
			t.Errorf("schema = %q, want %q", receipt.Schema, safesync.ExecuteReceiptSchema)
		}
		if receipt.Status != safesync.ExecuteStatusExecuted {
			t.Errorf("status = %q, want %q", receipt.Status, safesync.ExecuteStatusExecuted)
		}
		if !receipt.Pushed {
			t.Errorf("pushed = %v, want true", receipt.Pushed)
		}
		if !receipt.LocalCommitsContained {
			t.Errorf("local_commits_contained = %v, want true", receipt.LocalCommitsContained)
		}
		if !receipt.PeerBytesPreserved {
			t.Errorf("peer_bytes_preserved = %v, want true", receipt.PeerBytesPreserved)
		}
	})

	t.Run("execute without packet file builds and executes in-place", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")
		syncGit(t, origin, "config", "receive.denyCurrentBranch", "updateInstead")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_b.txt"), "remote b\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote b")

		syncWriteFile(t, filepath.Join(clone, "local_b.txt"), "local b\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local b")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"execute", "--repo", clone, "--remote", "origin", "--branch", "work"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		outStr := out.String()
		if !strings.Contains(outStr, "[EXECUTED]") {
			t.Errorf("expected output to contain [EXECUTED], got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "pushed: true") {
			t.Errorf("expected output to contain pushed: true, got:\n%s", outStr)
		}
	})

	t.Run("execute refuses stale target ref", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_c1.txt"), "remote c1\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote c1")

		syncWriteFile(t, filepath.Join(clone, "local_c1.txt"), "local c1\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local c1")

		syncGit(t, clone, "fetch", "origin")

		pktOpts := safesync.PacketOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
		}
		pkt, err := safesync.BuildReconciliationPacket(context.Background(), pktOpts)
		if err != nil {
			t.Fatalf("BuildReconciliationPacket: %v", err)
		}

		pktData, _ := json.Marshal(pkt)
		pktFile := filepath.Join(t.TempDir(), "stale_packet.json")
		_ = os.WriteFile(pktFile, pktData, 0644)

		// Advance target ref on origin and fetch
		syncWriteFile(t, filepath.Join(origin, "remote_c2.txt"), "remote c2\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote c2")
		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"execute", "--packet", pktFile, "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
		if code != syncExitRefused {
			t.Fatalf("exit = %d, want %d; stderr=%s stdout=%s", code, syncExitRefused, errb.String(), out.String())
		}

		var receipt safesync.ExecutionReceipt
		_ = json.Unmarshal(out.Bytes(), &receipt)
		if receipt.Reason != safesync.ReasonTargetMoved {
			t.Errorf("reason = %q, want %q; output: %s", receipt.Reason, safesync.ReasonTargetMoved, out.String())
		}
	})

	t.Run("execute refuses non-dispatchable packet", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "shared.txt"), "origin collision\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "origin collision")

		syncWriteFile(t, filepath.Join(clone, "shared.txt"), "clone collision\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "clone collision")

		syncGit(t, clone, "fetch", "origin")

		pktOpts := safesync.PacketOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
		}
		pkt, err := safesync.BuildReconciliationPacket(context.Background(), pktOpts)
		if err != nil {
			t.Fatalf("BuildReconciliationPacket: %v", err)
		}

		pktData, _ := json.Marshal(pkt)
		pktFile := filepath.Join(t.TempDir(), "collision_packet.json")
		_ = os.WriteFile(pktFile, pktData, 0644)

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"execute", "--packet", pktFile, "--repo", clone, "--remote", "origin", "--branch", "work"})
		if code != syncExitRefused {
			t.Fatalf("exit = %d, want %d; stderr=%s stdout=%s", code, syncExitRefused, errb.String(), out.String())
		}
		if !strings.Contains(out.String(), safesync.ReasonDivergedOverlap) {
			t.Errorf("expected output to contain %s, got:\n%s", safesync.ReasonDivergedOverlap, out.String())
		}
	})

	t.Run("reconcile with execute flag executes reconciliation packet", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")
		syncGit(t, origin, "config", "receive.denyCurrentBranch", "updateInstead")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_rec.txt"), "remote rec\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote rec")

		syncWriteFile(t, filepath.Join(clone, "local_rec.txt"), "local rec\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local rec")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--execute", "--json"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		var assessment safesync.ReconcileAssessment
		if err := json.Unmarshal(out.Bytes(), &assessment); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if !assessment.Applied {
			t.Errorf("assessment.Applied = %v, want true", assessment.Applied)
		}
		if assessment.ExecuteReceipt == nil {
			t.Fatal("execute_receipt is nil")
		}
		if assessment.ExecuteReceipt.Status != safesync.ExecuteStatusExecuted {
			t.Errorf("receipt status = %q, want EXECUTED", assessment.ExecuteReceipt.Status)
		}
		if !assessment.ExecuteReceipt.Pushed {
			t.Errorf("receipt pushed = %v, want true", assessment.ExecuteReceipt.Pushed)
		}
	})
}

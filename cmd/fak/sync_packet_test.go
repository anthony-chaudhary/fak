package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncPacketCLI(t *testing.T) {
	t.Run("disjoint divergence emits safe-disjoint and valid JSON", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_feature.txt"), "remote\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote feature")

		syncWriteFile(t, filepath.Join(clone, "local_feature.txt"), "local\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local feature")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"packet", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		var pkt safesync.ReconciliationPacket
		if err := json.Unmarshal(out.Bytes(), &pkt); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if pkt.Schema != safesync.PacketSchema {
			t.Errorf("schema = %q, want %q", pkt.Schema, safesync.PacketSchema)
		}
		if pkt.Disposition != safesync.DispositionSafeDisjoint {
			t.Errorf("disposition = %q, want %q", pkt.Disposition, safesync.DispositionSafeDisjoint)
		}
		if !pkt.Dispatchable {
			t.Errorf("dispatchable = %v, want true", pkt.Dispatchable)
		}
		if !pkt.MergePreview.Clean {
			t.Errorf("preview clean = %v, want true", pkt.MergePreview.Clean)
		}
	})

	t.Run("disjoint divergence renders human readable format", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "remote_b.txt"), "remote\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote b")

		syncWriteFile(t, filepath.Join(clone, "local_a.txt"), "local\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local a")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"packet", "--repo", clone, "--remote", "origin", "--branch", "work"})
		if code != syncExitOK {
			t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
		}

		outStr := out.String()
		if !strings.Contains(outStr, safesync.PacketSchema) {
			t.Errorf("expected output to contain schema %s, got:\n%s", safesync.PacketSchema, outStr)
		}
		if !strings.Contains(outStr, "safe-disjoint") {
			t.Errorf("expected output to contain safe-disjoint, got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "dispatchable: true") {
			t.Errorf("expected output to contain dispatchable: true, got:\n%s", outStr)
		}
	})

	t.Run("content collision emits semantic-conflict-review and exit refused", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "shared.txt"), "origin modification\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote conflict")

		syncWriteFile(t, filepath.Join(clone, "shared.txt"), "clone modification\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local conflict")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"packet", "--repo", clone, "--remote", "origin", "--branch", "work", "--json"})
		if code != syncExitRefused {
			t.Fatalf("exit = %d, want %d; stderr=%s stdout=%s", code, syncExitRefused, errb.String(), out.String())
		}

		var pkt safesync.ReconciliationPacket
		if err := json.Unmarshal(out.Bytes(), &pkt); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if pkt.Disposition != safesync.DispositionSemanticConflictReview {
			t.Errorf("disposition = %q, want %q", pkt.Disposition, safesync.DispositionSemanticConflictReview)
		}
		if pkt.Dispatchable {
			t.Errorf("dispatchable = %v, want false", pkt.Dispatchable)
		}
		if pkt.MergePreview.Clean {
			t.Errorf("merge preview clean = %v, want false", pkt.MergePreview.Clean)
		}
	})

	t.Run("reconcile with emit-packet flag attaches packet in JSON", func(t *testing.T) {
		clone := syncCLIFixture(t)
		origin := filepath.Join(filepath.Dir(clone), "origin")

		syncGit(t, clone, "merge", "--ff-only", "origin/work")

		syncWriteFile(t, filepath.Join(origin, "shared.txt"), "remote overlap\n")
		syncGit(t, origin, "add", ".")
		syncGit(t, origin, "commit", "-m", "remote edit shared")

		syncWriteFile(t, filepath.Join(clone, "shared.txt"), "local overlap\n")
		syncGit(t, clone, "add", ".")
		syncGit(t, clone, "commit", "-m", "local edit shared")

		syncGit(t, clone, "fetch", "origin")

		var out, errb bytes.Buffer
		code := runSync(&out, &errb, []string{"reconcile", "--repo", clone, "--remote", "origin", "--branch", "work", "--emit-packet", "--json"})
		if code != syncExitRefused {
			t.Fatalf("exit = %d, want %d; stderr=%s stdout=%s", code, syncExitRefused, errb.String(), out.String())
		}

		var got safesync.ReconcileAssessment
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("JSON decode: %v\n%s", err, out.String())
		}
		if got.Route != safesync.RouteReconcilePacket {
			t.Errorf("route = %q, want ROUTE_RECONCILE_PACKET", got.Route)
		}
		if got.Packet == nil {
			t.Fatalf("expected assessment.Packet to be non-nil")
		}
		if got.Packet.Schema != safesync.PacketSchema {
			t.Errorf("packet schema = %q, want %q", got.Packet.Schema, safesync.PacketSchema)
		}
		if got.Packet.Disposition != safesync.DispositionSemanticConflictReview {
			t.Errorf("packet disposition = %q, want %q", got.Packet.Disposition, safesync.DispositionSemanticConflictReview)
		}
	})
}

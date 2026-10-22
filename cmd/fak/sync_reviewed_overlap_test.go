package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncReviewedOverlapBoundary(t *testing.T) {
	for _, args := range [][]string{
		{"push", "--reviewed-overlap", "missing", "--accept-review", "digest"},
		{"apply", "--reviewed-overlap", "missing", "--accept-review", "digest"},
		{"execute", "--reviewed-overlap", "missing", "--accept-review", "digest"},
		{"execute", "--local-only", "--reviewed-overlap", "missing"},
		{"execute", "--local-only", "--accept-review", "digest"},
		{"execute", "--local-only", "--reviewed-overlap", "missing", "--accept-review", "digest", "--suspend-paths", "shared.txt"},
		{"reconcile", "--local-only", "--execute", "--apply", "--reviewed-overlap", "missing", "--accept-review", "digest"},
		{"review-overlap", "--fetch"},
		{"review-overlap", "--execute"},
	} {
		var out, errb bytes.Buffer
		if code := runSync(&out, &errb, args); code != syncExitUsage {
			t.Fatalf("%v exit=%d stderr=%s", args, code, errb.String())
		}
	}
	oldPacket, oldExecute, oldRoute := syncBuildReconciliationPacket, syncExecutePacket, syncRouteReconciliation
	t.Cleanup(func() {
		syncBuildReconciliationPacket, syncExecutePacket, syncRouteReconciliation = oldPacket, oldExecute, oldRoute
	})
	syncBuildReconciliationPacket = func(context.Context, safesync.PacketOptions) (*safesync.ReconciliationPacket, error) {
		return &safesync.ReconciliationPacket{}, nil
	}
	called := 0
	syncExecutePacket = func(_ context.Context, _ *safesync.ReconciliationPacket, opts safesync.ExecuteOptions) (*safesync.ExecutionReceipt, error) {
		called++
		if !opts.LocalOnly || opts.ReviewedOverlap == nil || opts.ReviewedOverlap.Approval.Digest != "accepted-digest" || opts.ReviewedOverlap.Review.Digest != "manifest-digest" {
			t.Fatalf("lost explicit review request: %+v", opts)
		}
		return &safesync.ExecutionReceipt{Status: safesync.ExecuteStatusExecuted}, nil
	}
	syncRouteReconciliation = func(_ context.Context, opts safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
		called++
		if !opts.LocalOnly || opts.ReviewedOverlap == nil || opts.ReviewedOverlap.Approval.Digest != "accepted-digest" {
			t.Fatal("lost reconciliation review request")
		}
		return safesync.ReconcileAssessment{ExecuteReceipt: &safesync.ExecutionReceipt{Status: safesync.ExecuteStatusExecuted}}, nil
	}
	file := filepath.Join(t.TempDir(), "review.json")
	data, _ := json.Marshal(safesync.CleanOverlapReview{Digest: "manifest-digest"})
	if err := os.WriteFile(file, data, 0600); err != nil {
		t.Fatal(err)
	}
	for _, command := range []string{"execute", "reconcile"} {
		args := []string{command, "--local-only", "--reviewed-overlap", file, "--accept-review", "accepted-digest", "--json"}
		if command == "reconcile" {
			args = append(args, "--execute")
		}
		var out, errb bytes.Buffer
		if code := runSync(&out, &errb, args); code != syncExitOK {
			t.Fatalf("exit=%d stderr=%s", code, errb.String())
		}
	}
	if called != 2 {
		t.Fatalf("called=%d", called)
	}
	for _, bad := range []string{`{"unknown":true}`, `{} {}`} {
		if err := os.WriteFile(file, []byte(bad), 0600); err != nil {
			t.Fatal(err)
		}
		var out, errb bytes.Buffer
		if code := runSync(&out, &errb, []string{"execute", "--local-only", "--reviewed-overlap", file, "--accept-review", "accepted-digest"}); code != syncExitUsage {
			t.Fatalf("malformed review exit=%d", code)
		}
	}
	if called != 2 {
		t.Fatal("malformed manifest reached executor")
	}
}

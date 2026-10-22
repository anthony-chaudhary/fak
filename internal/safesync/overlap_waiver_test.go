package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// fak-test:runtime fast est=2s lane=default
func TestReviewedOverlapLockWaiverBinding(t *testing.T) {
	_, repo, _, request := reviewedOverlapFixture(t)
	ctx := context.Background()
	ref := "refs/fak/locks/epoch-fixture"
	blob := filepath.Join(repo, ".git", "legacy-lock-fixture")
	if err := os.WriteFile(blob, []byte(`"legacy-marker"`), 0600); err != nil {
		t.Fatal(err)
	}
	rawOID, err := reviewOutput(ctx, RealRunner, repo, "hash-object", "-w", blob)
	if err != nil {
		t.Fatal(err)
	}
	oid := strings.TrimSpace(rawOID)
	git(t, repo, "update-ref", ref, oid)
	lease, err := AcquireWriterLease(repo, "waiver-test", time.Now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	opts := CleanOverlapReviewOptions{Repo: repo, TargetSHA: request.Review.TargetSHA, Reviewer: request.Review.Reviewer, Evidence: request.Review.Evidence}
	if _, err := ValidateCleanOverlapReview(ctx, opts, request, lease); err == nil {
		t.Fatal("default strict review admitted a non-record lock")
	}
	opts.LockWaivers = []leaseref.SnapshotWaiver{{Ref: ref, ExpectedOID: oid, Reason: "fixture non-record marker", Evidence: "fixture inspection"}}
	review, err := PrepareCleanOverlapReview(ctx, opts)
	if err != nil {
		t.Fatal(err)
	}
	if review.Digest == request.Review.Digest {
		t.Fatal("waiver omitted from digest")
	}
	request = ReviewedOverlapRequest{Review: *review, Approval: ReviewedOverlapApproval{Digest: review.Digest}}
	if _, err := ValidateCleanOverlapReview(ctx, opts, request, lease); err != nil {
		t.Fatal(err)
	}
	if got, err := rev(ctx, RealRunner, repo, ref); err != nil || got != oid {
		t.Fatal("waiver changed preserved lock ref")
	}
	// Each exception field is authorization-relevant, not just its target ref.
	for _, field := range []string{"ref", "oid", "reason", "evidence"} {
		changed := request
		changed.Review.LockWaivers = append([]leaseref.SnapshotWaiver(nil), request.Review.LockWaivers...)
		w := &changed.Review.LockWaivers[0]
		switch field {
		case "ref":
			w.Ref += "-changed"
		case "oid":
			w.ExpectedOID = request.Review.LocalHead
		case "reason":
			w.Reason += " changed"
		case "evidence":
			w.Evidence += " changed"
		}
		if _, err := ValidateCleanOverlapReview(ctx, opts, changed, lease); err == nil {
			t.Fatalf("changed %s retained old approval", field)
		}
	}
	git(t, repo, "update-ref", ref, request.Review.LocalHead, oid)
	if _, err := ValidateCleanOverlapReview(ctx, opts, request, lease); err == nil {
		t.Fatal("stale waived object admitted")
	}
	git(t, repo, "update-ref", "-d", ref, request.Review.LocalHead)
	if _, err := ValidateCleanOverlapReview(ctx, opts, request, lease); err == nil {
		t.Fatal("unused waiver admitted")
	}
	git(t, repo, "update-ref", ref, oid)
	if _, err := leaseref.NewInDir(repo).Acquire(ctx, leaseref.Record{ID: "live-peer", Holder: "peer", TreeGlobs: []string{"shared.txt"}, AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatal(err)
	}
	if _, err := ValidateCleanOverlapReview(ctx, opts, request, lease); err == nil {
		t.Fatal("waived unrelated marker exempted a live path owner")
	}
}

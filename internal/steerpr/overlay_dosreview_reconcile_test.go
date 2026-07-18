// Overlay <-> dos review reconciliation (#5037): the in-tree attention band
// and the external DOS kernel's `dos review` band the SAME trunk range, and
// this file pins the relationship that keeps them from becoming two competing
// truths:
//
//	SAME ORACLE, DIFFERENT FOLD.
//
// Both surfaces read `dos commit-audit`'s per-commit adjudication. The overlay
// attaches each row as a steerpr.Verdict and groups by intent (the
// `(fak <leaf>)` stamp); `dos review` partitions the same rows over a range
// window into RESIDUAL / UNVERIFIABLE / CLEARED. Different projections are
// fine; a different PER-COMMIT verdict is not. So the reconciliation here is
// per-commit and fails LOUDLY on any disagreement it cannot enumerate:
//
//   - the underlying (verdict, witness) pair each surface reports for a
//     commit must be identical — one oracle, read twice;
//   - the per-commit band must be EQUAL, with exactly ONE enumerated,
//     one-directional seam: `dos review` clears a `data-witnessed` claim,
//     while the overlay's keep-bit (dispatchtick.CommitWitnessed: verdict OK
//     AND witness `diff-witnessed`) does not grade that rung, so the overlay
//     reads it UNVERIFIABLE ("not yet graded"), never CLEARED;
//   - the seam is PESSIMISTIC ONLY. An overlay band BETTER than dos review's
//     (overlay CLEARED where review found RESIDUAL, or overlay UNVERIFIABLE
//     where review found RESIDUAL) is the trust bug this reconciliation
//     exists to catch, and it fails the test on every rung, always.
//
// The shared-range test needs a live DOS kernel and SKIPS explicitly when the
// `dos` CLI is absent — a skip is visible in test output, never a false pass —
// while the pure mapping test always runs, so the reconciliation contract is
// asserted even where the kernel is missing.
package steerpr

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

// reconcileMapAuditVerdict is a local twin of cmd/fak's mapAuditVerdict
// composed with dispatchtick.CommitWitnessed — the exact keep-bit the overlay
// grades `dos commit-audit` rows through. It is restated here (this package is
// deliberately import-free of internal/) so a drift in the real mapping
// surfaces as a reconciliation failure below rather than being absorbed.
func reconcileMapAuditVerdict(verdict, witness string) Verdict {
	if strings.EqualFold(strings.TrimSpace(verdict), "OK") && strings.TrimSpace(witness) == "diff-witnessed" {
		return VerdictWitnessed
	}
	if strings.EqualFold(strings.TrimSpace(verdict), string(VerdictUnwitnessed)) {
		return VerdictUnwitnessed
	}
	return VerdictAbstain
}

// reconcileReviewBandFor restates `dos review`'s documented banding rule (its
// own help text: a `diff-witnessed` / `data-witnessed` claim is CLEARED, a
// CLAIM_UNWITNESSED is the RESIDUAL, an ABSTAIN is UNVERIFIABLE) as an
// independent reference for the pure test — never called on live output,
// where the kernel's actual array placement is used instead.
func reconcileReviewBandFor(verdict, witness string) Band {
	switch strings.TrimSpace(witness) {
	case "diff-witnessed", "data-witnessed":
		return BandCleared
	}
	if strings.EqualFold(strings.TrimSpace(verdict), string(VerdictUnwitnessed)) {
		return BandResidual
	}
	return BandUnverifiable
}

// reconcileAgrees is THE acceptance predicate: the overlay band and the dos
// review band for one commit agree iff they are equal, or they sit on the one
// enumerated data-witnessed seam where the overlay is strictly MORE
// pessimistic (review CLEARED, overlay UNVERIFIABLE). Everything else — in
// particular any commit the overlay reads BETTER than dos review — is a
// divergence that must fail loudly.
func reconcileAgrees(overlay, review Band, witness string) bool {
	if overlay == review {
		return true
	}
	return review == BandCleared && overlay == BandUnverifiable &&
		strings.TrimSpace(witness) == "data-witnessed"
}

// TestDosReviewReconcileVerdictMappingPure pins, kernel-free, that the two
// folds agree per-commit on every rung of the shared oracle's vocabulary, and
// that the ONLY tolerated divergence is the enumerated pessimistic seam. This
// test always runs, so an environment without the DOS kernel still asserts
// the reconciliation contract instead of silently skipping everything.
func TestDosReviewReconcileVerdictMappingPure(t *testing.T) {
	cases := []struct {
		name             string
		verdict, witness string
		review           Band
		overlay          Band
		seam             bool
	}{
		{"witnessed claim", "OK", "diff-witnessed", BandCleared, BandCleared, false},
		{"unwitnessed claim", "CLAIM_UNWITNESSED", "subject-only", BandResidual, BandResidual, false},
		{"no checkable claim", "ABSTAIN", "abstain", BandUnverifiable, BandUnverifiable, false},
		// The one enumerated seam: the kernel clears a data-witnessed claim,
		// the overlay's diff-witness keep-bit leaves it ungraded. If this case
		// ever starts agreeing, the seam has CLOSED — delete it from
		// reconcileAgrees rather than letting a dead allowance linger.
		{"data-witnessed seam", "OK", "data-witnessed", BandCleared, BandUnverifiable, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			overlay := BandFor(reconcileMapAuditVerdict(tc.verdict, tc.witness))
			if overlay != tc.overlay {
				t.Fatalf("overlay band for (%s, %s) = %q, want %q", tc.verdict, tc.witness, overlay, tc.overlay)
			}
			review := reconcileReviewBandFor(tc.verdict, tc.witness)
			if review != tc.review {
				t.Fatalf("dos review band for (%s, %s) = %q, want %q", tc.verdict, tc.witness, review, tc.review)
			}
			if !reconcileAgrees(overlay, review, tc.witness) {
				t.Errorf("(%s, %s): overlay %q vs review %q must reconcile", tc.verdict, tc.witness, overlay, review)
			}
			if bandRank[overlay] > bandRank[review] {
				t.Errorf("(%s, %s): overlay %q bands BETTER than dos review %q — the pessimism direction is violated", tc.verdict, tc.witness, overlay, review)
			}
			if tc.seam == (overlay == review) {
				t.Errorf("(%s, %s): seam=%v but overlay %q vs review %q — the enumerated seam list is stale", tc.verdict, tc.witness, tc.seam, overlay, review)
			}
		})
	}

	// The trust-bug direction — the overlay reading a commit BETTER than dos
	// review — is never an allowed divergence, on ANY witness rung.
	for _, w := range []string{"diff-witnessed", "data-witnessed", "subject-only", "abstain", ""} {
		for _, bad := range []struct{ overlay, review Band }{
			{BandCleared, BandResidual},
			{BandCleared, BandUnverifiable},
			{BandUnverifiable, BandResidual},
		} {
			if reconcileAgrees(bad.overlay, bad.review, w) {
				t.Errorf("overlay %q vs review %q admitted on witness %q — an overlay band better than dos review's must always fail", bad.overlay, bad.review, w)
			}
		}
	}
}

// reconcileRun executes one subprocess in dir and returns its stdout. The
// error is returned rather than fatal'd because `dos` exits 1 when it finds a
// residual — a real verdict, not a tool failure, with the JSON still on
// stdout (the same contract cmd/fak's dosCommitAuditRange documents).
func reconcileRun(t *testing.T, dir, name string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	return string(out), err
}

// reconcileReviewRow is one per-commit row of `dos review --json`'s band
// arrays: the short sha, the shared oracle's verdict+witness, the subject.
type reconcileReviewRow struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject"`
	Witness string `json:"witness"`
	Verdict string `json:"verdict"`
}

// TestDosReviewReconcileSharedRange is the #5037 witness: over ONE shared
// commit range on the live repo, grade the range the overlay's way (dos
// commit-audit -> Verdict -> FoldUnits) and the kernel's way (dos review
// --json), then reconcile PER COMMIT. Membership, verdict, witness, and band
// must all line up (modulo the single enumerated data-witnessed seam), and
// the intent-grouped residual gate must agree with the range-windowed one. An
// absent kernel skips EXPLICITLY — it never passes vacuously.
func TestDosReviewReconcileSharedRange(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("SKIP (not a pass): DOS kernel unavailable — `dos` CLI not on PATH, reconciliation needs the live kernel")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("SKIP (not a pass): git not on PATH")
	}
	root, err := reconcileRun(t, ".", "git", "rev-parse", "--show-toplevel")
	if err != nil {
		t.Skipf("SKIP (not a pass): not inside a git repository: %v", err)
	}
	root = strings.TrimSpace(root)

	// Pin the shared range against a moving multi-session trunk: resolve HEAD
	// to a full sha ONCE and express the range against that sha everywhere,
	// so both surfaces adjudicate the identical window even if peers land
	// commits mid-test.
	headOut, err := reconcileRun(t, root, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Skipf("SKIP (not a pass): cannot resolve HEAD: %v", err)
	}
	head := strings.TrimSpace(headOut)
	revsOut, err := reconcileRun(t, root, "git", "rev-list", "--max-count=21", head)
	if err != nil {
		t.Skipf("SKIP (not a pass): cannot list history from %s: %v", head, err)
	}
	revs := strings.Fields(revsOut)
	if len(revs) < 3 {
		t.Skipf("SKIP (not a pass): history too short (%d commits) for a shared review range", len(revs))
	}
	base := revs[len(revs)-1]
	rng := base + ".." + head
	if mergesOut, err := reconcileRun(t, root, "git", "rev-list", "--merges", "--count", rng); err != nil || strings.TrimSpace(mergesOut) != "0" {
		// The overlay excludes merge commits by design (git log --no-merges),
		// so a range containing one has a KNOWN membership difference that is
		// not the divergence this test adjudicates.
		t.Skipf("SKIP (not a pass): range %s contains merge commits (count=%q, err=%v); the overlay's --no-merges fold makes membership non-comparable here", rng, strings.TrimSpace(mergesOut), err)
	}

	// Overlay side: the exact pipeline buildSteerPRsView composes — the wire
	// git-log format through ParseLog, one dos commit-audit call over the
	// whole range mapped through the keep-bit, prefix-attached, FoldUnits.
	raw, err := reconcileRun(t, root, "git", "log", "--no-merges", "--name-only",
		"--format=%x1e%H%x1f%s%x1f%b%x1f", rng)
	if err != nil {
		t.Fatalf("git log %s: %v", rng, err)
	}
	commits := ParseLog(raw)
	if len(commits) == 0 {
		t.Fatalf("range %s parsed to zero commits — nothing to reconcile where rev-list saw %d", rng, len(revs)-1)
	}

	auditOut, auditErr := reconcileRun(t, root, "dos", "commit-audit", rng, "--json")
	var auditRows []reconcileReviewRow
	if err := json.Unmarshal([]byte(auditOut), &auditRows); err != nil {
		t.Fatalf("dos commit-audit %s --json: unparsable output (%v, run err %v): %q", rng, err, auditErr, auditOut)
	}
	auditByShort := map[string]reconcileReviewRow{}
	for i := range commits {
		for _, row := range auditRows {
			if row.SHA != "" && strings.HasPrefix(commits[i].SHA, row.SHA) {
				commits[i].Verdict = reconcileMapAuditVerdict(row.Verdict, row.Witness)
				auditByShort[row.SHA] = row
				break
			}
		}
	}
	units, unstamped := FoldUnits(commits)
	overlayByFull := map[string]Commit{}
	for _, u := range units {
		for _, c := range u.Commits {
			overlayByFull[c.SHA] = c
		}
	}
	for _, c := range unstamped {
		overlayByFull[c.SHA] = c
	}

	// Kernel side: dos review's own band placement over the SAME range. The
	// advisory `semantic` lens is ignored deliberately — it re-flags commits
	// already in `cleared` and can only ask for MORE eyes, so it is not part
	// of the band partition being reconciled.
	reviewOut, reviewErr := reconcileRun(t, root, "dos", "review", rng, "--json")
	var review struct {
		NCommits     int                  `json:"n_commits"`
		Residual     []reconcileReviewRow `json:"residual"`
		Unverifiable []reconcileReviewRow `json:"unverifiable"`
		Cleared      []reconcileReviewRow `json:"cleared"`
	}
	if err := json.Unmarshal([]byte(reviewOut), &review); err != nil {
		t.Fatalf("dos review %s --json: unparsable output (%v, run err %v): %q", rng, err, reviewErr, reviewOut)
	}
	reviewBands := map[string]Band{}
	reviewRows := map[string]reconcileReviewRow{}
	for _, part := range []struct {
		band Band
		rows []reconcileReviewRow
	}{
		{BandResidual, review.Residual},
		{BandUnverifiable, review.Unverifiable},
		{BandCleared, review.Cleared},
	} {
		for _, row := range part.rows {
			if prior, dup := reviewBands[row.SHA]; dup {
				t.Errorf("dos review places %s in both %q and %q — the band partition must be disjoint", row.SHA, prior, part.band)
				continue
			}
			reviewBands[row.SHA] = part.band
			reviewRows[row.SHA] = row
		}
	}

	// Membership: both surfaces must see the identical commit set.
	if len(overlayByFull) != review.NCommits || len(reviewBands) != review.NCommits {
		t.Errorf("membership divergence over %s: overlay folded %d commits, dos review reports n_commits=%d across %d banded rows", rng, len(overlayByFull), review.NCommits, len(reviewBands))
	}

	// Per-commit reconciliation, in log order, with the side-by-side logged
	// so a run of this test IS the captured side-by-side the DoD names.
	matched := map[string]bool{}
	t.Logf("side-by-side over shared range %s (commit | oracle verdict/witness | dos review band | overlay band):", rng)
	for _, pc := range commits {
		oc := overlayByFull[pc.SHA]
		short, found := "", false
		for s := range reviewBands {
			if s != "" && strings.HasPrefix(pc.SHA, s) {
				short, found = s, true
				break
			}
		}
		if !found {
			t.Errorf("commit %s %q is in the overlay fold but in NO dos review band — membership divergence", pc.SHA, pc.Subject)
			continue
		}
		matched[short] = true
		rb, rr := reviewBands[short], reviewRows[short]

		// Same oracle, read twice: the (verdict, witness) pair the overlay
		// graded from dos commit-audit and the one dos review reports for
		// the same commit must be identical.
		ar, audited := auditByShort[short]
		if !audited {
			t.Errorf("commit %s: dos review banded it but dos commit-audit returned no row — the shared oracle is not shared", short)
		} else if ar.Verdict != rr.Verdict || ar.Witness != rr.Witness {
			t.Errorf("commit %s: per-commit ORACLE divergence — commit-audit says (%s, %s), dos review says (%s, %s); one of the two surfaces is reading a different adjudication", short, ar.Verdict, ar.Witness, rr.Verdict, rr.Witness)
		}

		t.Logf("  %s | %s/%s | review=%s | overlay=%s | %s", short, rr.Verdict, rr.Witness, rb, oc.Band, pc.Subject)
		if !reconcileAgrees(oc.Band, rb, rr.Witness) {
			t.Errorf("commit %s %q: BAND divergence — overlay=%q vs dos review=%q on witness %q; the two banders disagree and one of them is wrong, a human must adjudicate (never silently reconcile this)", short, pc.Subject, oc.Band, rb, rr.Witness)
		}
	}
	for s, b := range reviewBands {
		if !matched[s] {
			t.Errorf("dos review bands %s as %q but the overlay fold never saw it — membership divergence", s, b)
		}
	}

	// The CI gate, across the two projections: the overlay's intent-grouped
	// residual (worst member reds the unit) fires iff dos review found a
	// residual commit the overlay routes to a unit (unstamped commits carry
	// no unit, by the overlay's documented total-partition design).
	reviewStampedResidual := false
	for s, b := range reviewBands {
		if b != BandResidual {
			continue
		}
		for full, oc := range overlayByFull {
			if strings.HasPrefix(full, s) && oc.Leaf != "" {
				reviewStampedResidual = true
			}
		}
	}
	if gotGate := Residual(units) > 0; gotGate != reviewStampedResidual {
		t.Errorf("residual GATE divergence over %s: overlay residual units=%d (gate %v) vs dos review stamped-residual=%v — same oracle must yield the same gate under both folds", rng, Residual(units), gotGate, reviewStampedResidual)
	}
	t.Logf("reconciled %d commit(s) over %s: %d review-residual, %d review-unverifiable, %d review-cleared; overlay residual units=%d", len(commits), rng, len(review.Residual), len(review.Unverifiable), len(review.Cleared), Residual(units))
}

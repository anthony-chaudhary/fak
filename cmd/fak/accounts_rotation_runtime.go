package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// printRotationNoCandidate renders a rotate that found nothing to launch, using the typed reason
// the decision already carries (accounts.NextRotationDecision) instead of re-deriving it from the
// plan. The message NAMES the anchor and its room state, so the common case — you are already on
// the only account with room, and the only OTHER bucket is capped — reads as "just don't rotate"
// rather than the old bare "everything is walled", which was technically true but sent the
// operator hunting through the whole account topology to discover their active account was fine.
func printRotationNoCandidate(w io.Writer, prefix string, d accounts.RotationDecision) {
	switch d.Reason {
	case accounts.RotationEmptyPool:
		fmt.Fprintf(w, "%s: no eligible accounts in rotation (every seat is reserved, disabled, tombstoned, or has no live credentials)\n", prefix)

	case accounts.RotationOnlyBucket:
		// Nowhere else to rotate because the anchor's bucket is the only one. Not a wall — say so.
		if d.Anchor.Name != "" {
			fmt.Fprintf(w, "%s: already on the only account bucket (%q) — nothing to rotate onto. Just launch without --rotate, or enroll another account with `fak accounts add`.\n",
				prefix, d.Anchor.Name)
		} else {
			fmt.Fprintf(w, "%s: only one account bucket in rotation (%s) — nowhere else to rotate; enroll another with `fak accounts add`.\n",
				prefix, rotationOnlyBucketName(d))
		}

	case accounts.RotationAllOthersWalled:
		walled := rotationSeatNames(d.Walled)
		if d.AnchorRoom && d.Anchor.Name != "" {
			// The confusing case the whole change targets: the account you are ON has room, and
			// only the OTHER bucket(s) are capped. The fix is to stop rotating, not to wait.
			fmt.Fprintf(w, "%s: already on the only account with room (%q); nothing to rotate onto (capped bucket(s): %s). Just launch without --rotate, or wait for a bucket to reset / enroll another with `fak accounts add`.\n",
				prefix, d.Anchor.Name, strings.Join(walled, ", "))
		} else {
			// The anchor itself is walled too (or unknown) — a genuine "wait for reset" dead-end.
			fmt.Fprintf(w, "%s: no runtime-launchable account in rotation; known usage/weekly-capped bucket(s): %s. Wait for reset or move the launch role to an account with room.\n",
				prefix, strings.Join(walled, ", "))
		}

	default:
		// Defensive: an unmapped reason should never reach here, but fall back to the plan pool
		// so the operator still gets something actionable rather than a bare prefix.
		fmt.Fprintf(w, "%s: no account to rotate onto.\n", prefix)
	}
}

// rotationSeatNames renders a seat list as comma-joinable names for a message.
func rotationSeatNames(seats []accounts.RotationSeat) []string {
	out := make([]string, 0, len(seats))
	for _, s := range seats {
		out = append(out, s.Name)
	}
	return out
}

// rotationOnlyBucketName names the sole pool bucket for the anchorless only-bucket message.
func rotationOnlyBucketName(d accounts.RotationDecision) string {
	if len(d.Plan.Pool) > 0 {
		return d.Plan.Pool[0].Name
	}
	return "(none)"
}

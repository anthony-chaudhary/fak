package main

import (
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guard_allow_approve.go implements `fak guard allow --approve <id>`: the human
// operator's per-call override of the capability floor (the operator-escalation
// amendment channel). A refusal the operator may lift carries an approval id
// (adjudicator.OperatorApprovalID); approving that id admits that one exact call
// past its POLICY_BLOCK rule, DEFAULT_DENY or reversibility confirm hold. The
// running guard's allow watcher picks the overlay up live, so the agent can
// re-propose the call unchanged right away. Design: docs/notes/GUARD-OPERATOR-APPROVAL-2026-10-23.md.

// guardOperatorApprovalDefaultTTL bounds an approval when the operator gives no
// --ttl: long enough to re-propose the call, short enough not to linger.
const guardOperatorApprovalDefaultTTL = 15 * time.Minute

var guardOperatorApprovalIDRe = regexp.MustCompile(`^ap-[0-9a-f]{12}$`)

// runGuardAllowApprove adds (or, with remove, revokes) operator approval ids.
func runGuardAllowApprove(stdout, stderr io.Writer, path string, ov *guardAllowOverlay, args []string, ttl time.Duration, remove, conflicting bool) int {
	if conflicting {
		fmt.Fprintln(stderr, "fak guard allow: --approve cannot combine with --prefix/--list/--from-*")
		return 2
	}
	ids := guardAllowNormalize(args)
	if len(ids) == 0 {
		fmt.Fprintln(stderr, "fak guard allow --approve: name the approval id(s) printed on the refusal, e.g. fak guard allow --approve ap-0123456789ab")
		return 2
	}
	for _, id := range ids {
		if !guardOperatorApprovalIDRe.MatchString(id) {
			fmt.Fprintf(stderr, "fak guard allow --approve: %q is not an approval id (want ap-<12 hex>, as printed on the refusal)\n", id)
			return 2
		}
	}
	if remove {
		ov.Approve = guardAllowSubtract(ov.Approve, ids)
		if err := saveGuardAllowOverlay(path, *ov); err != nil {
			fmt.Fprintln(stderr, "fak guard allow:", err)
			return 1
		}
		fmt.Fprintf(stdout, "fak guard allow: revoked approval %s.\n", strings.Join(ids, ", "))
		printGuardAllowOverlay(stdout, path, *ov)
		return 0
	}
	if ttl == 0 {
		ttl = guardOperatorApprovalDefaultTTL
	}
	ov.Approve = append(ov.Approve, ids...)
	stamp := guardAllowStampExpiry(ov, ids, ttl, guardAllowNow())
	if err := saveGuardAllowOverlay(path, *ov); err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak guard allow: approved %s for %s (until %s).\n", strings.Join(ids, ", "), ttl, stamp)
	fmt.Fprintln(stdout, "  A running guard applies it within about a second; tell the agent to re-propose the same call.")
	printGuardAllowOverlay(stdout, path, *ov)
	return 0
}

// guardApplyOperatorApprovals unions approval ids into the runtime floor and
// returns how many were new.
func guardApplyOperatorApprovals(rt *policy.Runtime, ids []string) int {
	if len(ids) == 0 {
		return 0
	}
	if rt.Adjudicator.OperatorApprovals == nil {
		rt.Adjudicator.OperatorApprovals = map[string]bool{}
	}
	added := 0
	for _, id := range ids {
		if !rt.Adjudicator.OperatorApprovals[id] {
			rt.Adjudicator.OperatorApprovals[id] = true
			added++
		}
	}
	return added
}

// guardAllowNormalizeOptional normalizes like guardAllowNormalize but keeps an
// empty list nil, so an overlay with no approvals stays byte-identical.
func guardAllowNormalizeOptional(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	if out := guardAllowNormalize(in); len(out) > 0 {
		return out
	}
	return nil
}

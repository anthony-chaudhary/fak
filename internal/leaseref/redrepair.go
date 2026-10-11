package leaseref

// redrepair.go is the RED-TRUNK REPAIR CLAIM: an intent lease (intent.go) whose target is
// "a package that is already red at a base commit". It rides the same refs/fak/locks/intent-*
// transport, TTL, and update-ref compare-and-swap; it adds only the target grammar and a
// package-wide lookup.
//
// THE GAP IT CLOSES (2026-10-10 ctxmmu incident). The commit gate's differential attribution
// proves a red is PRE-EXISTING at HEAD and admits the commit, and every session that hits the
// same break is told to "fix it at its source". Nothing recorded that one of them had started:
// two sessions landed the same rename fix under different file names within the hour, and the
// merged result would have redefined the symbol twice. A repair claim is the missing record —
// "session X is repairing package P at base H" — that the gate surfaces to the next session.
//
// THE KEY. A claim is stored per (package, base OID): target "red-repair <pkg>@<base>". The
// lookup, though, is PACKAGE-WIDE (RedRepairClaims scans every live red-repair intent for the
// package at any base), because two sessions rarely sit on the same HEAD — a peer's unrelated
// commit moves the base without fixing the break, and a claim invisible across that move would
// miss exactly the incident it exists for. ClaimRedRepair refuses on a live different-holder
// claim for the same package at ANY base for the same reason.
//
// Lifecycle: claim when a repair starts, ReleaseRedRepair when it lands, and the default
// one-hour intent TTL bounds a crashed claimant.

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"
)

// redRepairTargetPrefix starts every red-repair intent's raw target.
const redRepairTargetPrefix = "red-repair "

// RedRepairTarget renders the intent target for repairing pkg (an import path or
// repo-relative directory) at the red base commit base.
func RedRepairTarget(pkg, base string) string {
	return redRepairTargetPrefix + strings.TrimSpace(pkg) + "@" + strings.TrimSpace(base)
}

// ParseRedRepairTarget splits a red-repair intent target back into package and base. ok is
// false for any intent that is not a red-repair claim.
func ParseRedRepairTarget(target string) (pkg, base string, ok bool) {
	rest, found := strings.CutPrefix(target, redRepairTargetPrefix)
	if !found {
		return "", "", false
	}
	i := strings.LastIndex(rest, "@")
	if i <= 0 || i == len(rest)-1 {
		return "", "", false
	}
	return rest[:i], rest[i+1:], true
}

// RedRepairClaims returns the LIVE red-repair claims on any of pkgs, at any base, sorted by
// package then acquisition time. Expired claims are omitted (they are reapable, not rivals).
func (s *Store) RedRepairClaims(ctx context.Context, pkgs []string, now time.Time) ([]IntentRecord, error) {
	want := map[string]bool{}
	for _, p := range pkgs {
		if p = strings.TrimSpace(p); p != "" {
			want[p] = true
		}
	}
	if len(want) == 0 {
		return nil, nil
	}
	live, _, err := s.LiveIntents(ctx, now)
	if err != nil {
		return nil, err
	}
	var out []IntentRecord
	for _, rec := range live {
		if pkg, _, ok := ParseRedRepairTarget(rec.Target); ok && want[pkg] {
			out = append(out, rec)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		pi, _, _ := ParseRedRepairTarget(out[i].Target)
		pj, _, _ := ParseRedRepairTarget(out[j].Target)
		if pi != pj {
			return pi < pj
		}
		return out[i].AcquiredAt < out[j].AcquiredAt
	})
	return out, nil
}

// ClaimRedRepair claims the repair of pkg red at base for holder. A live claim on the same
// package by a DIFFERENT holder — at this base or any other — refuses INTENT_COLLISION naming
// that incumbent; the caller's own live claim renews. ttlSeconds <= 0 takes the intent default.
func (s *Store) ClaimRedRepair(ctx context.Context, pkg, base, holder, sessionID string, ttlSeconds int64, now time.Time) (IntentRecord, IntentVerdict, error) {
	if strings.TrimSpace(pkg) == "" || strings.TrimSpace(base) == "" {
		return IntentRecord{}, IntentVerdict{}, fmt.Errorf("leaseref: red-repair claim needs a package and a base")
	}
	target := RedRepairTarget(pkg, base)
	peers, err := s.RedRepairClaims(ctx, []string{strings.TrimSpace(pkg)}, now)
	if err != nil {
		return IntentRecord{}, IntentVerdict{}, err
	}
	for _, p := range peers {
		if p.Target != target && (p.Holder != holder || holder == "") {
			return IntentRecord{}, collisionVerdict(IntentKey(target), p, now), nil
		}
	}
	return s.ClaimIntent(ctx, IntentRecord{Target: target, Holder: holder, SessionID: sessionID, TTLSeconds: ttlSeconds}, now)
}

// ReleaseRedRepair drops the claim on pkg at base. Idempotent, like ReleaseIntent.
func (s *Store) ReleaseRedRepair(ctx context.Context, pkg, base string) error {
	return s.ReleaseIntent(ctx, RedRepairTarget(pkg, base))
}

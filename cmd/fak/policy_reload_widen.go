package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const policyReloadWidenConfirmEnv = "FAK_POLICY_RELOAD_ALLOW_WIDEN"

type policyWideningDelta struct {
	AddedAllow             []string
	AddedAllowPrefix       []string
	RemovedDeny            []string
	RemovedSelfModifyGlobs []string
	PostureLoosened        bool
}

func diffPolicyWidening(old, next adjudicator.Policy) policyWideningDelta {
	var d policyWideningDelta
	for key, allowed := range next.Allow {
		if allowed && !old.Allow[key] {
			d.AddedAllow = append(d.AddedAllow, key)
		}
	}
	oldPrefixes := stringSet(old.AllowPrefix)
	for _, prefix := range next.AllowPrefix {
		if _, ok := oldPrefixes[prefix]; !ok {
			d.AddedAllowPrefix = append(d.AddedAllowPrefix, prefix)
		}
	}
	for key := range old.Deny {
		if _, ok := next.Deny[key]; !ok {
			d.RemovedDeny = append(d.RemovedDeny, key)
		}
	}
	nextGlobs := stringSet(next.SelfModifyGlobs)
	for _, glob := range old.SelfModifyGlobs {
		if _, ok := nextGlobs[glob]; !ok {
			d.RemovedSelfModifyGlobs = append(d.RemovedSelfModifyGlobs, glob)
		}
	}
	d.PostureLoosened = old.Posture == adjudicator.PostureFailClosed && next.Posture == adjudicator.PostureAdmitAndLog
	sort.Strings(d.AddedAllow)
	sort.Strings(d.AddedAllowPrefix)
	sort.Strings(d.RemovedDeny)
	sort.Strings(d.RemovedSelfModifyGlobs)
	return d
}

func stringSet(values []string) map[string]struct{} {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

func (d policyWideningDelta) Empty() bool {
	return len(d.AddedAllow) == 0 && len(d.AddedAllowPrefix) == 0 && len(d.RemovedDeny) == 0 && len(d.RemovedSelfModifyGlobs) == 0 && !d.PostureLoosened
}

func (d policyWideningDelta) String() string {
	parts := make([]string, 0, 5)
	if len(d.AddedAllow) > 0 {
		parts = append(parts, "added_allow="+strings.Join(d.AddedAllow, ","))
	}
	if len(d.AddedAllowPrefix) > 0 {
		parts = append(parts, "added_allow_prefix="+strings.Join(d.AddedAllowPrefix, ","))
	}
	if len(d.RemovedDeny) > 0 {
		parts = append(parts, "removed_deny="+strings.Join(d.RemovedDeny, ","))
	}
	if len(d.RemovedSelfModifyGlobs) > 0 {
		parts = append(parts, "removed_self_modify_globs="+strings.Join(d.RemovedSelfModifyGlobs, ","))
	}
	if d.PostureLoosened {
		parts = append(parts, "posture=fail_closed->admit_and_log")
	}
	return strings.Join(parts, "; ")
}

func policyReloadWidenConfirmed() bool {
	value := strings.TrimSpace(os.Getenv(policyReloadWidenConfirmEnv))
	return value == "1" || strings.EqualFold(value, "true") || strings.EqualFold(value, "yes")
}

func policyWideningError(d policyWideningDelta) error {
	return fmt.Errorf("policy widening rejected (set %s=1 to confirm): %s", policyReloadWidenConfirmEnv, d.String())
}

func applyPolicyRuntimeLocked(rt policy.Runtime, source, digest, warning string, enforceWideningGate bool) (string, error) {
	current := adjudicator.Default.PolicySnapshot()
	// CORE-LOCK-ALL (#5423) is checked FIRST and is unconditional: it outranks both
	// escapes the widening gate below carries. That gate only runs when the caller asks
	// for it (enforceWideningGate) and is defeatable with FAK_POLICY_RELOAD_ALLOW_WIDEN=1
	// — reasonable for an operator's own reload, but the whole point of launching under
	// --core-lock-all is that the session cannot be widened afterwards, including by an
	// env var the wrapped agent could set on a process it spawns. On a launch WITHOUT the
	// flag guardCoreLockAllActive() is false and the verdict admits unconditionally, so
	// this path stays byte-for-byte what it was. Note the classification differs from
	// diffPolicyWidening's: it routes through policy.DiffAmendment, the canonical
	// amendment-class engine, and so fails CLOSED on anything it cannot classify.
	if admit, reason := guardCoreLockAllAdmitAmendment(current, rt.Adjudicator); !admit {
		err := fmt.Errorf("policy reload refused: %s", reason)
		journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, source, digest, journal.ConfigSwapRejected, err.Error())
		return "", err
	}
	widening := diffPolicyWidening(current, rt.Adjudicator)
	if enforceWideningGate && !widening.Empty() && !policyReloadWidenConfirmed() {
		err := policyWideningError(widening)
		journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, source, digest, journal.ConfigSwapRejected, err.Error())
		return "", err
	}

	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	if enforceWideningGate && !widening.Empty() {
		if warning != "" {
			warning += "\n"
		}
		warning += "confirmed_widening: " + widening.String()
	}
	journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, source, digest, journal.ConfigSwapOK, warning)
	return warning, nil
}

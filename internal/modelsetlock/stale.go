package modelsetlock

import (
	"errors"
	"strings"
)

// StaleReason is a closed input-change class. Stale locks are incompatible
// until re-resolution; Compare never substitutes a candidate silently.
type StaleReason string

const (
	StaleIntent          StaleReason = "MODEL_SET_LOCK_INTENT_STALE"
	StaleInventory       StaleReason = "MODEL_SET_LOCK_INVENTORY_STALE"
	StaleSelectionRules  StaleReason = "MODEL_SET_LOCK_RULES_STALE"
	StalePlatform        StaleReason = "MODEL_SET_LOCK_PLATFORM_STALE"
	StaleResolverVersion StaleReason = "MODEL_SET_LOCK_RESOLVER_VERSION_STALE"
)

// StaleError carries stale reasons in stable contract order.
type StaleError struct {
	Reasons []StaleReason
}

func (e *StaleError) Error() string {
	if e == nil || len(e.Reasons) == 0 {
		return "model-set lock is stale"
	}
	parts := make([]string, len(e.Reasons))
	for i, reason := range e.Reasons {
		parts[i] = string(reason)
	}
	return "model-set lock is stale: " + strings.Join(parts, ", ")
}

// StaleReasons returns a defensive copy of typed stale reasons.
func StaleReasons(err error) []StaleReason {
	var staleErr *StaleError
	if !errors.As(err, &staleErr) {
		return nil
	}
	return append([]StaleReason(nil), staleErr.Reasons...)
}

// Compare verifies lock integrity before comparing it with the current inputs.
// A nil result means every bound input is unchanged.
func Compare(lock Lock, current Inputs) error {
	canonical := canonicalizeLock(lock)
	if err := validateLock(canonical); err != nil {
		return err
	}
	prepared, err := prepare(current)
	if err != nil {
		return err
	}
	var reasons []StaleReason
	if canonical.Inputs.Intent != prepared.digests.Intent {
		reasons = append(reasons, StaleIntent)
	}
	if canonical.Inputs.Inventory != prepared.digests.Inventory {
		reasons = append(reasons, StaleInventory)
	}
	if canonical.Inputs.Policy != prepared.digests.Policy {
		reasons = append(reasons, StaleSelectionRules)
	}
	if canonical.Inputs.Target != prepared.digests.Target {
		reasons = append(reasons, StalePlatform)
	}
	if canonical.ResolverVersion != prepared.resolverVersion {
		reasons = append(reasons, StaleResolverVersion)
	}
	if len(reasons) != 0 {
		return &StaleError{Reasons: reasons}
	}
	return nil
}

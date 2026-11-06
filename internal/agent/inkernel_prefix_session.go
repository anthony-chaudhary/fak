package agent

import "github.com/anthony-chaudhary/fak/internal/model"

// restoreOwnedPrefixSnapshot consumes an already request-owned snapshot using
// the ordinary execution path. Success returns the sole session owner; failure
// retires the snapshot and any temporary session using the existing sibling
// cleanup contract. It does not acquire cache credit or prepare another replica.
// Callers retain the ordinary session configuration and execution-time checks.
func (p *InKernelPlanner) restoreOwnedPrefixSnapshot(snapshot *model.PrefixSnapshot) (restored *model.Session, restoreErr error) {
	var candidate *model.Session
	snapshotClosed := false
	defer finishInKernelCleanup(&restoreErr, func() {
		if !snapshotClosed {
			snapshot.Close()
		}
	}, func() {
		if restored == nil && candidate != nil {
			candidate.Close()
		}
	})
	if p.backend != nil {
		// The snapshot installs its own device recurrent state below, so
		// skip the eager zero-state upload this constructor would otherwise
		// perform and immediately discard (fak#13801).
		candidate, restoreErr = p.m.NewBackendSessionCheckedDeferredState(p.backend)
		if restoreErr != nil {
			return nil, restoreErr
		}
	} else {
		candidate = p.m.NewSession()
	}
	if restoreErr = snapshot.Restore(candidate); restoreErr != nil {
		return nil, restoreErr
	}
	snapshotClosed = true
	snapshot.Close()
	return candidate, nil
}

package fleet

// RecoveryProbe is the stable, machine-readable result of the bounded Ray
// compatibility probe. It exercises fak's public fleet observation seam rather
// than depending on a Ray process at test time.
type RecoveryProbe struct {
	Schema      string `json:"schema"`
	Case        string `json:"case"`
	Upstream    string `json:"upstream"`
	FakState    string `json:"fak_state"`
	Recoverable bool   `json:"recoverable"`
	Action      string `json:"action"`
	Evidence    string `json:"evidence"`
}

// ProbeRecovery maps a scheduler observation onto fak's typed fleet recovery
// semantics. A running worker is healthy; an exited worker with retries left is
// recoverable by replacement; an exhausted worker fails closed.
func ProbeRecovery(caseName, upstream string, running bool, exitCode, attempts, maxAttempts int) RecoveryProbe {
	r := RecoveryProbe{Schema: "fak-ray-recovery-probe/1", Case: caseName, Upstream: upstream}
	if running {
		r.FakState, r.Recoverable, r.Action, r.Evidence = "RUNNING", true, "NONE", "worker heartbeat is current"
		return r
	}
	r.FakState = "FAILED"
	if maxAttempts > 0 && attempts < maxAttempts {
		r.Recoverable, r.Action, r.Evidence = true, "REPLACE", "worker exited and retry budget remains"
	} else {
		r.Recoverable, r.Action, r.Evidence = false, "ESCALATE", "worker exited and retry budget is exhausted"
	}
	return r
}

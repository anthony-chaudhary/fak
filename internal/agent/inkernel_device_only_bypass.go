package agent

import "github.com/anthony-chaudhary/fak/internal/model"

// HostPathDegradedToken leads the perf-line clause of a serve running under an
// admitted bring-up bypass of device-only execution.
const HostPathDegradedToken = "DEGRADED host-path"

// applyDeviceOnlyBypass grants the admitted bring-up bypass to one native session
// at a time. The holder runs portable and releases the grant on Close; every
// other session stays device-only and refuses its unsanctioned host stages.
func (p *InKernelPlanner) applyDeviceOnlyBypass(s *model.Session) {
	if s.HostPathBypass() != "" {
		return
	}
	if p.deviceOnlyBypassHolder.CompareAndSwap(nil, s) {
		s.SetExecutionPolicy(model.ExecutionPolicyPortable)
		s.SetHostPathBypass(p.deviceOnlyBypass, func() { p.deviceOnlyBypassHolder.CompareAndSwap(s, nil) })
		return
	}
	s.SetExecutionPolicy(model.ExecutionPolicyDeviceOnly)
}

// hostPathDegradedClause is the trailing perf-line clause while the serve holds
// a device-only bypass; empty otherwise.
func (p *InKernelPlanner) hostPathDegradedClause() string {
	if p == nil || p.deviceOnlyBypass == "" {
		return ""
	}
	return " " + HostPathDegradedToken + "=" + p.deviceOnlyBypass
}

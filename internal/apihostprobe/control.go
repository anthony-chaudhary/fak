package apihostprobe

import (
	"context"
	"fmt"
	"net/http"
)

// ProbeControl describes one experimental HTTP probe and the positive control
// that must succeed before its measured response can be interpreted.
type ProbeControl struct {
	Name    string
	Target  ReadinessTarget
	Control ReadinessTarget
}

// ControlledProbe preserves both observations. Conclusive is true only when
// the control proved the transport/auth path and the target returned an exact,
// declared HTTP status.
type ControlledProbe struct {
	Name       string         `json:"name"`
	Target     ReadinessProbe `json:"target"`
	Control    ReadinessProbe `json:"control"`
	Conclusive bool           `json:"conclusive"`
	Verdict    string         `json:"verdict"`
	Error      string         `json:"error,omitempty"`
}

// ProbeWithControl runs the positive control first. It never promotes a
// half-open status range to PASS: callers must name every accepted status.
func ProbeWithControl(ctx context.Context, experiment ProbeControl, acceptedStatuses []int, opts ReadinessOptions) ControlledProbe {
	out := ControlledProbe{Name: experiment.Name, Verdict: "UNKNOWN"}
	if experiment.Name == "" {
		out.Error = "probe name is required"
		return out
	}
	if len(acceptedStatuses) == 0 {
		out.Error = "at least one exact accepted status is required"
		return out
	}
	accepted := make(map[int]struct{}, len(acceptedStatuses))
	for _, status := range acceptedStatuses {
		if status < 100 || status > 599 {
			out.Error = fmt.Sprintf("invalid exact HTTP status %d", status)
			return out
		}
		accepted[status] = struct{}{}
	}

	out.Control = ProbeReadinessTarget(ctx, experiment.Control, opts)
	if out.Control.HTTPStatus == nil || *out.Control.HTTPStatus != http.StatusOK {
		out.Error = "positive control did not return HTTP 200"
		return out
	}
	out.Target = ProbeReadinessTarget(ctx, experiment.Target, opts)
	if out.Target.HTTPStatus == nil {
		out.Error = "target probe produced no HTTP status"
		return out
	}
	out.Conclusive = true
	if _, ok := accepted[*out.Target.HTTPStatus]; ok {
		out.Verdict = "PASS"
	} else {
		out.Verdict = "FAIL"
	}
	return out
}

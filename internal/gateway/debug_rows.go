package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func debugHTTPRows(rows []httpMetricSnapshot) []debugHTTPMetricVars {
	out := make([]debugHTTPMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugHTTPMetricVars{
			Route:   row.key.route,
			Method:  row.key.method,
			Status:  row.key.status,
			Latency: debugLatency(row.val),
		})
	}
	return out
}

func debugOperationRows(rows []operationMetricSnapshot) []debugOperationMetricVars {
	out := make([]debugOperationMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugOperationMetricVars{
			Operation:   row.key.operation,
			Verdict:     row.key.verdict,
			Reason:      row.key.reason,
			Disposition: row.key.disposition,
			By:          row.key.by,
			Latency:     debugLatency(row.val),
		})
	}
	return out
}

func debugInKernelOOMRows(rows []inKernelOOMSnapshot) []debugInKernelOOMVars {
	out := make([]debugInKernelOOMVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugInKernelOOMVars{
			Class:           row.class,
			Count:           row.count,
			FailedBytes:     row.failedBytes,
			LastFailedBytes: row.lastFailedBytes,
			LastSite:        row.lastSite,
		})
	}
	return out
}

func debugRequestMemoryMetricRows(rows []requestMemoryPlanSnapshot) []debugRequestMemoryMetricVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryMetricVars{
			Backend:        row.key.backend,
			Class:          row.key.class,
			Scope:          row.key.scope,
			DType:          row.key.dtype,
			Observations:   row.observations,
			TotalBytes:     row.totalBytes,
			HighWaterBytes: row.highWaterBytes,
		})
	}
	return out
}

func debugRequestMemoryFitRows(rows []requestMemoryFitSnapshot) []debugRequestMemoryFitVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryFitVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryFitVars{
			Backend:          row.key.backend,
			Scope:            row.key.scope,
			Observations:     row.observations,
			WantHighWater:    row.wantHighWater,
			MarginLowWater:   row.marginLowWater,
			MarginLowWaterOK: row.marginKnown,
		})
	}
	return out
}

func debugRequestMemoryTokenRows(rows []requestMemoryTokenSnapshot) []debugRequestMemoryTokenVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryTokenVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryTokenVars{
			Backend:      row.key.backend,
			Kind:         row.key.kind,
			Observations: row.observations,
			Total:        row.total,
			HighWater:    row.highWater,
		})
	}
	return out
}

func debugInKernelOOMRetryRows(p agent.Planner) []debugInKernelOOMRetryVars {
	reporter, ok := p.(agent.InKernelOOMRetryReporter)
	if !ok {
		return nil
	}
	st := reporter.InKernelOOMRetryStats()
	if len(st.Rows) == 0 {
		return nil
	}
	backend := defaultBackendLabel(st.Backend)
	out := make([]debugInKernelOOMRetryVars, 0, len(st.Rows))
	for _, row := range st.Rows {
		out = append(out, debugInKernelOOMRetryVars{
			Backend:         backend,
			Class:           oomClassLabel(row.Class),
			Attempts:        row.Attempts,
			Successes:       row.Successes,
			Failures:        row.Failures,
			LastFailedBytes: row.LastFailedBytes,
			LastSite:        row.LastSite,
		})
	}
	return out
}

func debugInKernelPressureTrimRows(p agent.Planner) []debugInKernelPressureTrimVars {
	reporter, ok := p.(agent.InKernelMemoryPressureTrimReporter)
	if !ok {
		return nil
	}
	st := reporter.InKernelMemoryPressureTrimStats()
	if len(st.Rows) == 0 {
		return nil
	}
	backend := defaultBackendLabel(st.Backend)
	out := make([]debugInKernelPressureTrimVars, 0, len(st.Rows))
	for _, row := range st.Rows {
		out = append(out, debugInKernelPressureTrimVars{
			Backend:         backend,
			Scope:           modelLoadScope(row.Scope),
			Class:           oomClassLabel(row.Class),
			Reason:          pressureTrimReasonLabel(row.Reason),
			Attempts:        row.Attempts,
			Trimmed:         row.Trimmed,
			NoHooks:         row.NoHooks,
			Resolved:        row.Resolved,
			LastWantBytes:   row.LastWantBytes,
			LastBudgetBytes: row.LastBudgetBytes,
			LastMarginBytes: row.LastMarginBytes,
		})
	}
	return out
}

func debugLatency(s latencySnapshot) debugLatencyVars {
	buckets := make([]debugBucketVars, 0, len(gatewayLatencyBuckets))
	for i, le := range gatewayLatencyBuckets {
		buckets = append(buckets, debugBucketVars{LESeconds: le, Count: s.buckets[i]})
	}
	return debugLatencyVars{Count: s.count, SumSeconds: s.sum, Buckets: buckets}
}

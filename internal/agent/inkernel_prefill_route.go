package agent

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type qwen35SequencePrefillRouteStatusProvider interface {
	Qwen35SequencePrefillRouteStatus() (model.Qwen35SequencePrefillRouteStatus, bool)
}

// recordQwen35SequencePrefillRoute records one actual prefill call. Calls too
// short to make a fresh sequence-route decision remain explicitly unobserved;
// they must not reuse a prior session status.
func recordQwen35SequencePrefillRoute(measurement *nativeInferenceMeasurement, session any, tokenCount int) {
	if measurement == nil || measurement.inferenceDisabled || tokenCount <= 0 {
		return
	}
	if measurement.qwen35SequencePrefillRoute == nil {
		measurement.qwen35SequencePrefillRoute = &model.NativeSequencePrefillRouteReceipt{}
	}
	aggregate := measurement.qwen35SequencePrefillRoute
	aggregate.PrefillCalls++
	aggregate.Complete = false
	aggregate.NativePerformanceQualifying = false
	if tokenCount < 2 {
		return
	}
	provider, ok := session.(qwen35SequencePrefillRouteStatusProvider)
	if !ok {
		return
	}
	status, ok := provider.Qwen35SequencePrefillRouteStatus()
	if !ok {
		return
	}
	aggregate.ObservedCalls++
	if aggregate.Status == nil || qwen35SequencePrefillRouteQualifies(*aggregate.Status) {
		statusCopy := status
		aggregate.Status = &statusCopy
	}
	aggregate.Complete = aggregate.PrefillCalls > 0 && aggregate.ObservedCalls == aggregate.PrefillCalls
	aggregate.NativePerformanceQualifying = aggregate.Complete && aggregate.Status != nil && qwen35SequencePrefillRouteQualifies(*aggregate.Status)
}

func qwen35SequencePrefillRouteQualifies(status model.Qwen35SequencePrefillRouteStatus) bool {
	return status.RequestedPath == compute.Qwen35SequencePrefillPath &&
		status.EffectivePath == compute.Qwen35SequencePrefillPath &&
		status.NativePerformanceQualifying && !status.FallbackActive
}

func cloneNativeSequencePrefillRouteReceipt(source *model.NativeSequencePrefillRouteReceipt) *model.NativeSequencePrefillRouteReceipt {
	if source == nil || source.PrefillCalls == 0 {
		return nil
	}
	clone := *source
	if source.Status != nil {
		status := *source.Status
		clone.Status = &status
	}
	return &clone
}

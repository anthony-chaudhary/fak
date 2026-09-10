package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type qwen35PrefillRouteObservation struct {
	status  model.Qwen35SequencePrefillRouteStatus
	present bool
}

type qwen35PrefillRouteStatusSpy struct {
	observations []qwen35PrefillRouteObservation
	queries      int
	prefillCalls []int
}

func (s *qwen35PrefillRouteStatusSpy) PrefillNoLogits(ids []int) {
	s.prefillCalls = append(s.prefillCalls, len(ids))
}

func (s *qwen35PrefillRouteStatusSpy) Prefill(ids []int) []float32 {
	s.prefillCalls = append(s.prefillCalls, len(ids))
	return []float32{float32(len(ids))}
}

func (s *qwen35PrefillRouteStatusSpy) Qwen35SequencePrefillRouteStatus() (model.Qwen35SequencePrefillRouteStatus, bool) {
	s.queries++
	if len(s.observations) == 0 {
		return model.Qwen35SequencePrefillRouteStatus{}, false
	}
	observation := s.observations[0]
	s.observations = s.observations[1:]
	return observation.status, observation.present
}

func qualifyingQwen35PrefillRouteStatus() model.Qwen35SequencePrefillRouteStatus {
	return model.Qwen35SequencePrefillRouteStatus{
		RequestedPath:               compute.Qwen35SequencePrefillPath,
		EffectivePath:               compute.Qwen35SequencePrefillPath,
		NativePerformanceQualifying: true,
		PackedEmbeddingRows:         true,
		EmbeddingPanelBytes:         4096,
	}
}

func declinedQwen35PrefillRouteStatus() model.Qwen35SequencePrefillRouteStatus {
	return model.Qwen35SequencePrefillRouteStatus{
		RequestedPath:  compute.Qwen35SequencePrefillPath,
		EffectivePath:  model.Qwen35SequencePrefillFallbackPath,
		DeclineReason:  model.Qwen35SequencePrefillDeclineEmbeddingCap,
		FallbackActive: true,
	}
}

func TestNativePrefillRouteSuccessfulActualSequenceRoute(t *testing.T) {
	want := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{{status: want, present: true}}}
	measurement := &nativeInferenceMeasurement{}

	recordQwen35SequencePrefillRoute(measurement, spy, 32)

	got := measurement.qwen35SequencePrefillRoute
	if got == nil {
		t.Fatal("executed sequence prefill produced no request-local route receipt")
	}
	if got.PrefillCalls != 1 || got.ObservedCalls != 1 || !got.Complete || !got.NativePerformanceQualifying {
		t.Fatalf("successful route aggregate = %+v, want one complete qualifying observation", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, want) {
		t.Fatalf("successful route status = %+v, want exact model status %+v", got.Status, want)
	}
	if spy.queries != 1 {
		t.Fatalf("eligible prefill queried status %d times, want exactly once", spy.queries)
	}
}

func TestNativePrefillRouteDeclineSurvivesLaterSuccess(t *testing.T) {
	declined := declinedQwen35PrefillRouteStatus()
	success := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{
		{status: declined, present: true},
		{status: success, present: true},
	}}
	measurement := &nativeInferenceMeasurement{}

	recordQwen35SequencePrefillRoute(measurement, spy, 16)
	recordQwen35SequencePrefillRoute(measurement, spy, 16)

	got := measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 2 || got.ObservedCalls != 2 || !got.Complete {
		t.Fatalf("mixed route aggregate = %+v, want two complete observations", got)
	}
	if got.NativePerformanceQualifying {
		t.Fatalf("later success upgraded a request containing an actual fallback: %+v", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, declined) {
		t.Fatalf("representative status = %+v, want first nonqualifying status %+v", got.Status, declined)
	}
}

func TestNativePrefillRouteChunkedCallPathKeepsFallbackSticky(t *testing.T) {
	declined := declinedQwen35PrefillRouteStatus()
	success := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{
		{status: declined, present: true},
		{status: success, present: true},
	}}
	measurement := &nativeInferenceMeasurement{}
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkTokens = 2

	if _, err := p.prefillDivergentSuffix(context.Background(), spy, []int{1, 2, 3, 4}, measurement); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(spy.prefillCalls, []int{2, 2}) {
		t.Fatalf("actual prefill calls = %v, want two T2 calls", spy.prefillCalls)
	}
	got := measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 2 || got.ObservedCalls != 2 || !got.Complete || got.NativePerformanceQualifying {
		t.Fatalf("actual mixed route aggregate = %+v, want complete but nonqualifying", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, declined) {
		t.Fatalf("actual call path lost first fallback: got %+v want %+v", got.Status, declined)
	}
	raw, err := json.Marshal(p.buildNativeInferenceReceipt(measurement, 0.25, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	var decoded model.NativeInferenceReceipt
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Qwen35SequencePrefillRoute == nil || !reflect.DeepEqual(decoded.Qwen35SequencePrefillRoute, got) {
		t.Fatalf("actual call aggregate did not survive receipt JSON: decoded=%+v want=%+v json=%s", decoded.Qwen35SequencePrefillRoute, got, raw)
	}
}

func TestNativePrefillRouteUnknownObservationPreventsQualification(t *testing.T) {
	success := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{
		{present: false},
		{status: success, present: true},
	}}
	measurement := &nativeInferenceMeasurement{}

	recordQwen35SequencePrefillRoute(measurement, spy, 8)
	recordQwen35SequencePrefillRoute(measurement, spy, 8)

	got := measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 2 || got.ObservedCalls != 1 || got.Complete {
		t.Fatalf("unknown-plus-success aggregate = %+v, want two calls, one observation, incomplete", got)
	}
	if got.NativePerformanceQualifying {
		t.Fatalf("later success upgraded a request with an unobserved executed prefill: %+v", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, success) {
		t.Fatalf("known representative status = %+v, want observed success %+v", got.Status, success)
	}
}

func TestNativePrefillRouteRejectsNoncanonicalEffectivePath(t *testing.T) {
	status := qualifyingQwen35PrefillRouteStatus()
	status.EffectivePath = "qwen35/alternate-prefill-v1"
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{{status: status, present: true}}}
	measurement := &nativeInferenceMeasurement{}

	recordQwen35SequencePrefillRoute(measurement, spy, 8)

	got := measurement.qwen35SequencePrefillRoute
	if got == nil || !got.Complete || got.ObservedCalls != 1 || got.Status == nil {
		t.Fatalf("noncanonical route aggregate = %+v, want one complete observed status", got)
	}
	if got.NativePerformanceQualifying {
		t.Fatalf("noncanonical effective path qualified from a self-reported boolean: %+v", got)
	}
}

func TestNativePrefillRouteSingleTokenCallDoesNotReadStaleStatus(t *testing.T) {
	success := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{{status: success, present: true}}}
	measurement := &nativeInferenceMeasurement{}

	recordQwen35SequencePrefillRoute(measurement, spy, 32)
	recordQwen35SequencePrefillRoute(measurement, spy, 1)

	got := measurement.qwen35SequencePrefillRoute
	if spy.queries != 1 {
		t.Fatalf("single-token prefill queried latest session status: queries=%d, want 1 from the eligible call only", spy.queries)
	}
	if got == nil || got.PrefillCalls != 2 || got.ObservedCalls != 1 || got.Complete || got.NativePerformanceQualifying {
		t.Fatalf("success plus unobservable single-token call = %+v, want incomplete and nonqualifying", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, success) {
		t.Fatalf("single-token call changed the fresh earlier status: got %+v want %+v", got.Status, success)
	}
}

func TestNativePrefillRouteChunkedCallPathCountsUnknownAndFinalSingleToken(t *testing.T) {
	success := qualifyingQwen35PrefillRouteStatus()
	spy := &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{
		{present: false},
		{status: success, present: true},
	}}
	measurement := &nativeInferenceMeasurement{}
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkTokens = 2

	if _, err := p.prefillDivergentSuffix(context.Background(), spy, []int{1, 2, 3, 4, 5}, measurement); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(spy.prefillCalls, []int{2, 2, 1}) {
		t.Fatalf("actual prefill calls = %v, want T2/T2/T1", spy.prefillCalls)
	}
	if spy.queries != 2 {
		t.Fatalf("route status queries = %d, want only the two calls able to make fresh decisions", spy.queries)
	}
	got := measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 3 || got.ObservedCalls != 1 || got.Complete || got.NativePerformanceQualifying {
		t.Fatalf("unknown/success/T1 aggregate = %+v, want three calls, one fresh observation, nonqualifying", got)
	}
	if got.Status == nil || !reflect.DeepEqual(*got.Status, success) {
		t.Fatalf("fresh known status = %+v, want %+v", got.Status, success)
	}
}

func TestNativePrefillRouteNoCallAndUnsupportedRemainUnmeasured(t *testing.T) {
	measurement := &nativeInferenceMeasurement{}
	if measurement.qwen35SequencePrefillRoute != nil {
		t.Fatalf("new cache-only measurement fabricated route evidence: %+v", measurement.qwen35SequencePrefillRoute)
	}

	recordQwen35SequencePrefillRoute(measurement, nil, 4)
	got := measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 1 || got.ObservedCalls != 0 || got.Complete || got.NativePerformanceQualifying || got.Status != nil {
		t.Fatalf("executed prefill without a status provider = %+v, want explicit one-call unobserved receipt", got)
	}

	measurement.reset()
	p := qwenQ4KPrefillPlanner(nil)
	p.qwenQ4KPrefillChunkTokens = 2
	unsupported := &recordingPrefillSession{}
	if _, err := p.prefillDivergentSuffix(context.Background(), unsupported, []int{1, 2, 3}, measurement); err != nil {
		t.Fatal(err)
	}
	got = measurement.qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 2 || got.ObservedCalls != 0 || got.Complete || got.NativePerformanceQualifying || got.Status != nil {
		t.Fatalf("actual unsupported call path = %+v, want two explicitly unobserved calls", got)
	}
}

func TestNativePrefillRouteResetAndJSONRoundTrip(t *testing.T) {
	wantStatus := declinedQwen35PrefillRouteStatus()
	measurement := &nativeInferenceMeasurement{}
	recordQwen35SequencePrefillRoute(measurement, &qwen35PrefillRouteStatusSpy{observations: []qwen35PrefillRouteObservation{{status: wantStatus, present: true}}}, 64)

	p := &InKernelPlanner{modelID: "qwen35-route-receipt"}
	receipt := p.buildNativeInferenceReceipt(measurement, 1.25, 0.5)
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var decoded model.NativeInferenceReceipt
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	got := decoded.Qwen35SequencePrefillRoute
	if got == nil || got.PrefillCalls != 1 || got.ObservedCalls != 1 || !got.Complete || got.NativePerformanceQualifying || got.Status == nil || !reflect.DeepEqual(*got.Status, wantStatus) {
		t.Fatalf("JSON route receipt = %+v, want exact declined observation; json=%s", got, raw)
	}

	measurement.reset()
	if measurement.qwen35SequencePrefillRoute != nil {
		t.Fatalf("measurement reset retained prior request route: %+v", measurement.qwen35SequencePrefillRoute)
	}
	emptyRaw, err := json.Marshal(p.buildNativeInferenceReceipt(measurement, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	var empty map[string]json.RawMessage
	if err := json.Unmarshal(emptyRaw, &empty); err != nil {
		t.Fatal(err)
	}
	if _, present := empty["qwen35_sequence_prefill_route"]; present {
		t.Fatalf("cache-only/reset receipt serialized fabricated route evidence: %s", emptyRaw)
	}
}

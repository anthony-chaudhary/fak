package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// StepSamplingCandidates advances exactly once through the ordinary Step
// lifecycle. On acceptance packet contains a compact raw row in original-ID
// order; otherwise full holds the ordinary full logits. The draw remains with
// the caller. Both extraction and any full fallback finish inside the execution
// scope and before the existing token recycler.
func (s *Session) StepSamplingCandidates(id int, temp float64, k int, retain bool, packet *compute.SamplingCandidatePacket) (full []float32, compact bool) {
	s.requireGLM5NextSessionUnsupported("Session.StepSamplingCandidates")
	scope := s.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	producer, ok := s.Backend.(compute.SamplingCandidateBackend)
	var output func(int, int) ([]float32, int)
	if ok && packet != nil && k >= 1 && k <= compute.SamplingCandidateMaxK &&
		s.activeTap() == nil && !s.captureTargetHidden &&
		s.qwen35TargetFeatureTxn == nil && s.qwen35TargetLayerCapture == nil {
		output = func(token, pos int) ([]float32, int) {
			_, split := s.validateDenseGPULayers()
			if split {
				return s.token(token, pos), 0
			}
			s.requirePreNorm("HAL decode")
			row, _ := s.tokenHALOutput(token, pos, halFullLogits)
			// Sampling still uses the host; a smaller transfer cannot bypass
			// the existing execution-policy admission or host-stage receipt.
			s.admitHostStage(HostStageLogitsReadback)
			*packet = compute.SamplingCandidatePacket{}
			kept, available, err := producer.SamplingCandidates(row, k, retain, packet)
			if err != nil {
				panic(err)
			}
			// Publish the independently owned copy to existing Session cleanup
			// before any later host operation can panic. Next Step/Close/Take
			// all use the same existing single-owner retirement path.
			s.greedyLogits = kept
			if available && packet.Certify(temp, k) {
				if retain && kept.Buf() == nil {
					panic("model: compact sampling omitted its retained distribution")
				}
				compact = true
				s.recycleHALToken()
				return nil, 0
			}
			// A numeric or capability refusal is not a second model step.
			// Invalid stochastic rows reach the unchanged host sampler and
			// therefore retain its historical draw-before-error behavior.
			values := s.Backend.Read(row)
			if values == nil {
				panic("model: compact sampling fallback returned unreadable logits")
			}
			s.releaseGreedyLogits()
			s.recycleHALToken()
			return values, 0
		}
	}
	full, _ = s.stepOwnedOutput(id, false, output)
	complete = true
	return full, compact
}

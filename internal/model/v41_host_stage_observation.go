package model

import (
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v41DenseHostStage = "v41-dense-projection"

func init() { compute.DeclareHostFallbackStage(hostFallbackLabel(HostStage(v41DenseHostStage))) }

// HostStages returns a detached, sorted session-lifetime observation. It records
// successful dense host work under an explicit bypass, not unbound operations,
// selected-device failures, or a completeness claim about other V4.1 stages.
func (s *Session) HostStages() []string {
	if s == nil {
		return nil
	}
	s.hostStagesMu.Lock()
	defer s.hostStagesMu.Unlock()
	stages := make([]string, 0, len(s.hostStages))
	for stage := range s.hostStages {
		stages = append(stages, stage)
	}
	slices.Sort(stages)
	return stages
}

func (s *Session) noteV41DenseHostSuccess() {
	s.hostStagesMu.Lock()
	defer s.hostStagesMu.Unlock()
	if _, seen := s.hostStages[v41DenseHostStage]; seen {
		return
	}
	if s.hostStages == nil {
		s.hostStages = make(map[string]struct{})
	}
	s.hostStages[v41DenseHostStage] = struct{}{}
	compute.CountHostFallback(hostFallbackLabel(HostStage(v41DenseHostStage)))
}

// v41ObserveDenseHost is installed only on admitted portable bring-up sessions.
// It preserves selected-operation outcomes and runs the existing host arm once
// only when the ordinary callback declined. No selected error is replayed.
func (s *Session) v41ObserveDenseHost(project v41DenseProjectionFunc) v41DenseProjectionFunc {
	return func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if project != nil {
			y, outcome, err := project(layer, leaf, panel, out, in, rows)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		if s.executionPolicy == ExecutionPolicyDeviceOnly {
			return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "dense-projection")
		}
		if s.hostPathBypass == "" {
			return nil, v41ProjectionDeclined, nil
		}
		req, err := s.M.v41AdmitProjectionRows(layer, leaf, panel, out, in, rows)
		if err != nil {
			return nil, v41ProjectionError, err
		}
		y, err := s.M.v41ProjectionHostRows(req, panel)
		if err != nil {
			return nil, v41ProjectionError, err
		}
		s.noteV41DenseHostSuccess()
		return y, v41ProjectionHandled, nil
	}
}

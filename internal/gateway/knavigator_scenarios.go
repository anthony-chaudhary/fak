// Copyright (c) 2026 fak contributors
// SPDX-License-Identifier: Apache-2.0
//
// INSPIRE-ONLY clean-room adaptation of NVIDIA/knavigator
// (NVIDIA/knavigator@d6213b84283fa231395f2056d92e157d56683e3a):
// workload scenario grammar and scheduler-under-test harness for native
// batching and admission tests in internal/gateway.
package gateway

import "fmt"

// KNavStep describes one homogeneous burst of requests in a workload scenario.
type KNavStep struct {
	Label        string
	Count        int
	PromptTokens int
	MaxNewTokens int
	Priority     int
	PrefixGroup  string
}

// KNavWorkloadScenario defines a named sequence of admission steps.
type KNavWorkloadScenario struct {
	Name  string
	Steps []KNavStep
}

// KNavRequest is a single materialized request submitted to the scheduler harness.
type KNavRequest struct {
	ID           string
	PromptTokens int
	MaxNewTokens int
	Priority     int
	PrefixGroup  string
}

// KNavDecision records the scheduler's admission verdict for one request.
type KNavDecision struct {
	Admitted  bool
	Queued    bool
	PrefixHit bool
	Reason    string
}

// KNavSchedulerState snapshots queue depth and active batch size.
type KNavSchedulerState struct {
	QueueDepth  int
	ActiveBatch int
}

// KNavSchedulerHarness is the scheduler-under-test interface driven by RunKNavScenario.
type KNavSchedulerHarness interface {
	Admit(req KNavRequest) KNavDecision
	Snapshot() KNavSchedulerState
}

// KNavScenarioReport summarizes the outcome of running a workload scenario.
type KNavScenarioReport struct {
	Scenario      string
	Submitted     int
	Admitted      int
	Queued        int
	Rejected      int
	PrefixHits    int
	PeakQueue     int
	PeakBatchSize int
}

// RunKNavScenario expands sc into requests, drives h.Admit for each, and
// records admission, rejection, prefix-hit, and peak queue/batch telemetry.
func RunKNavScenario(h KNavSchedulerHarness, sc KNavWorkloadScenario) KNavScenarioReport {
	rep := KNavScenarioReport{Scenario: sc.Name}
	if h == nil {
		return rep
	}
	seq := 0
	for stepIdx, step := range sc.Steps {
		for i := 0; i < step.Count; i++ {
			seq++
			req := KNavRequest{
				ID:           fmt.Sprintf("%s-s%d-%d", sc.Name, stepIdx, seq),
				PromptTokens: step.PromptTokens,
				MaxNewTokens: step.MaxNewTokens,
				Priority:     step.Priority,
				PrefixGroup:  step.PrefixGroup,
			}
			rep.Submitted++
			dec := h.Admit(req)
			if dec.Admitted {
				rep.Admitted++
			} else if dec.Queued {
				rep.Queued++
			} else {
				rep.Rejected++
			}
			if dec.PrefixHit {
				rep.PrefixHits++
			}
			st := h.Snapshot()
			if st.QueueDepth > rep.PeakQueue {
				rep.PeakQueue = st.QueueDepth
			}
			if st.ActiveBatch > rep.PeakBatchSize {
				rep.PeakBatchSize = st.ActiveBatch
			}
		}
	}
	return rep
}

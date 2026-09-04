package gateway

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

const lowInfoReceiptReason = "LOW_INFO_RECEIPT"

func (s *Server) annotateResultLivelock(trace string, adms []ResultAdmission) {
	if s == nil || trace == "" {
		return
	}
	s.resultLivelockMu.Lock()
	if s.resultLivelock == nil {
		s.resultLivelock = guardrsi.NewLivelockDetector(guardrsi.DefaultLivelockThreshold)
	}
	type hit struct {
		idx int
		env guardrsi.LivelockEnvelope
	}
	var hits []hit
	sawObservation := false
	for i := range adms {
		a := adms[i]
		if a.Verdict.Kind == "QUARANTINE" {
			sawObservation = true
			env, ok := s.resultLivelock.ObserveFailure(guardrsi.LivelockObservation{
				TraceID:     trace,
				Tool:        resultToolLabel(a),
				ArgsDigest:  a.ResultDigest,
				Verdict:     a.Verdict.Kind,
				Reason:      a.Verdict.Reason,
				Disposition: a.Verdict.Disposition,
			})
			if ok {
				hits = append(hits, hit{idx: i, env: env})
			}
			continue
		}
		if resultAdmissionIsLowInfoReceipt(a) {
			sawObservation = true
			env, ok := s.resultLivelock.ObserveAdmitted(guardrsi.LivelockObservation{
				TraceID:    trace,
				Tool:       resultToolLabel(a),
				ArgsDigest: a.ResultDigest,
				Verdict:    "ALLOW",
				Reason:     lowInfoReceiptReason,
			})
			if ok {
				hits = append(hits, hit{idx: i, env: env})
			}
		}
	}
	if !sawObservation {
		s.resultLivelock.Clear(trace)
	}
	s.resultLivelockMu.Unlock()

	for _, h := range hits {
		env := h.env
		adms[h.idx].Livelock = &env
	}
}

func resultToolLabel(a ResultAdmission) string {
	if a.Tool != "" {
		return a.Tool
	}
	return "tool_result"
}

func resultAdmissionIsLowInfoReceipt(a ResultAdmission) bool {
	if a.Verdict.Kind != "ALLOW" {
		return false
	}
	tool := strings.ToLower(strings.TrimSpace(a.Tool))
	if isUpdatePlanTool(tool) {
		return a.ResultDigest == guardrsi.ArgsDigest("Plan updated")
	}
	return tool == "exec_command" || tool == "functions.exec_command" || tool == "write_stdin" || tool == "functions.write_stdin"
}

func isUpdatePlanTool(tool string) bool {
	tool = strings.ToLower(strings.TrimSpace(tool))
	return tool == "update_plan" || tool == "functions.update_plan" ||
		tool == "todowrite" || tool == "agent.todowrite" || tool == "functions.todowrite"
}

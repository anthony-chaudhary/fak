package gateway

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

const maxContextQueryAuditRecords = 1024

// ContextQueryAuditRecord is the replayable runtime provenance row for a
// managed-context clarification. It ties the question, the answer source
// (default/user/inferred), and the later assumption row together by ID.
type ContextQueryAuditRecord struct {
	Seq                 uint64 `json:"seq"`
	ID                  string `json:"id"`
	Event               string `json:"event"`
	Method              string `json:"method,omitempty"`
	URI                 string `json:"uri,omitempty"`
	Key                 string `json:"key"`
	Reason              string `json:"reason"`
	Question            string `json:"question,omitempty"`
	AnswerSource        string `json:"answer_source"`
	Answer              string `json:"answer,omitempty"`
	DefaultChoice       string `json:"default_choice,omitempty"`
	SourceRef           string `json:"source_ref,omitempty"`
	AssumptionKey       string `json:"assumption_key,omitempty"`
	AssumptionSource    string `json:"assumption_source,omitempty"`
	AssumptionSourceRef string `json:"assumption_source_ref,omitempty"`
}

func (s *Server) recordMissingContextQueryAudit(req contextq.MCPMissingContextRequest, plan selfquery.ClarificationPlan) []ContextQueryAuditRecord {
	if s == nil || !req.Audited {
		return nil
	}
	out := make([]ContextQueryAuditRecord, 0, len(plan.Questions))
	for _, q := range plan.Questions {
		answer := strings.TrimSpace(q.DefaultChoice)
		r := ContextQueryAuditRecord{
			Event:            "context_question",
			Method:           req.Method,
			URI:              req.URI,
			Key:              firstNonEmptyString(q.Key, req.Key),
			Reason:           firstNonEmptyString(string(q.Reason), req.Reason),
			Question:         q.Question,
			AnswerSource:     "default",
			Answer:           answer,
			DefaultChoice:    q.DefaultChoice,
			SourceRef:        q.SourceRef,
			AssumptionKey:    firstNonEmptyString(q.Key, req.Key),
			AssumptionSource: "default",
		}
		s.appendContextQueryAudit(&r)
		out = append(out, r)
	}
	return out
}

func (s *Server) appendContextQueryAudit(r *ContextQueryAuditRecord) {
	s.contextQueryAuditMu.Lock()
	defer s.contextQueryAuditMu.Unlock()

	s.contextQueryAuditSeq++
	r.Seq = s.contextQueryAuditSeq
	r.ID = fmt.Sprintf("context-query:%d", r.Seq)
	if r.AssumptionSourceRef == "" {
		r.AssumptionSourceRef = r.ID
	}

	s.contextQueryAudit = append(s.contextQueryAudit, *r)
	if len(s.contextQueryAudit) > maxContextQueryAuditRecords {
		s.contextQueryAudit = s.contextQueryAudit[len(s.contextQueryAudit)-maxContextQueryAuditRecords:]
	}
}

func (s *Server) contextQueryAuditSnapshot() []ContextQueryAuditRecord {
	if s == nil {
		return nil
	}
	s.contextQueryAuditMu.Lock()
	defer s.contextQueryAuditMu.Unlock()

	out := make([]ContextQueryAuditRecord, len(s.contextQueryAudit))
	copy(out, s.contextQueryAudit)
	return out
}

func firstNonEmptyString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

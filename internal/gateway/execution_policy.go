package gateway

import "slices"

// ExecutionPolicyReport is the /healthz execution_policy block a serve publishes
// when it runs under a non-production execution policy: which model the node is
// dedicated to and every device-only refusal (closed tokens) the policy lifted,
// so a probe receipt records what was bypassed.
type ExecutionPolicyReport struct {
	Policy    string   `json:"policy"`
	Model     string   `json:"model,omitempty"`
	Dedicated bool     `json:"dedicated"`
	Bypassed  []string `json:"bypassed"`
}

// SetExecutionPolicy records the serve's execution policy for /healthz. nil
// clears it. Safe on a nil Server.
func (s *Server) SetExecutionPolicy(r *ExecutionPolicyReport) {
	if s == nil {
		return
	}
	if r != nil {
		c := *r
		c.Bypassed = slices.Clone(r.Bypassed)
		if c.Bypassed == nil {
			c.Bypassed = []string{}
		}
		r = &c
	}
	s.executionPolicy.Store(r)
}

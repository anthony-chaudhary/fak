package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type reachabilityPlanner struct {
	status int
	err    error
}

func (p *reachabilityPlanner) Model() string { return "test" }
func (p *reachabilityPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("model turn must not run during health probe")
}
func (p *reachabilityPlanner) ProbeReachability(context.Context) (int, error) {
	return p.status, p.err
}

func TestDeepHealthReportsProviderReachability(t *testing.T) {
	s := &Server{planner: &reachabilityPlanner{status: 405}, engineID: "test", model: "test"}
	rec := httptest.NewRecorder()
	s.handleHealth(rec, httptest.NewRequest("GET", "/healthz?deep=1", nil))
	var got struct {
		OK           bool `json:"ok"`
		Reachability struct {
			Evaluated bool `json:"evaluated"`
			OK        bool `json:"ok"`
			Status    int  `json:"status"`
		} `json:"provider_reachability"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || !got.Reachability.Evaluated || !got.Reachability.OK || got.Reachability.Status != 405 {
		t.Fatalf("health = %+v", got)
	}
}

func TestDeepHealthFailsWhenProviderHopFails(t *testing.T) {
	s := &Server{planner: &reachabilityPlanner{status: 502, err: errors.New("upstream unreachable")}, engineID: "test", model: "test"}
	rec := httptest.NewRecorder()
	s.handleHealth(rec, httptest.NewRequest("GET", "/healthz?deep=1", nil))
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["ok"] != false {
		t.Fatalf("health = %#v, want ok=false", got)
	}
}

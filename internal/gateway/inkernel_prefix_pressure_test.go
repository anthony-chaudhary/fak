package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelKVPrefixPressureSourceRequiresDirectNativeOwner(t *testing.T) {
	local := agent.NewInKernelPlanner(model.NewSynthetic(model.Config{}), nil, "native", false, nil, false)
	srv := &Server{planner: local}
	if got := srv.InKernelKVPrefixPressureSource(); got != local {
		t.Fatalf("direct native source=%T, want planner owner", got)
	}

	srv.planner = &agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"}
	if got := srv.InKernelKVPrefixPressureSource(); got != nil {
		t.Fatalf("proxy source=%T, want nil", got)
	}

	dual, err := NewDualPlanner(
		&agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"},
		local,
		"native",
	)
	if err != nil {
		t.Fatal(err)
	}
	srv.planner = dual
	if got := srv.InKernelKVPrefixPressureSource(); got != nil {
		t.Fatalf("dual source=%T, want nil until per-request payload ownership is explicit", got)
	}
}

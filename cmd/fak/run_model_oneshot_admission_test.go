package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type oneShotStubPlanner struct{}

func (oneShotStubPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{}, nil
}
func (oneShotStubPlanner) Model() string { return "stub" }

func TestOneShotRunContextSkipsAdmissionOnlyForProcessLocalInKernelCache(t *testing.T) {
	ctx := context.Background()
	if got := oneShotRunContext(ctx, oneShotStubPlanner{}); got != ctx {
		t.Fatal("a non-in-kernel planner must keep the caller context")
	}
	// A nil planner reports no local-SSD tier, so its cache cannot outlive the process.
	if got := oneShotRunContext(ctx, (*agent.InKernelPlanner)(nil)); got == ctx {
		t.Fatal("a process-local in-kernel cache must be marked to skip admission")
	}
}

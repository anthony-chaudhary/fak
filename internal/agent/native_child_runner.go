package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

const maxNativeChildTurns = 10

type nativeChildCompletionFilter struct {
	Planner
	allowed map[string]struct{}
}

func (p nativeChildCompletionFilter) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	completion, err := p.Planner.Complete(ctx, messages, tools, opts...)
	if err != nil || completion == nil {
		return completion, err
	}
	for _, call := range completion.Message.ToolCalls {
		if _, ok := p.allowed[call.Function.Name]; !ok {
			return nil, fmt.Errorf("native child task: tool %q is outside the inherited child capability floor", call.Function.Name)
		}
	}
	return completion, nil
}

// NewNativeChildTaskRunner reuses the selected parent planner while bounding
// child turns, capabilities, and nesting. Client-declared tools are only model
// declarations here; every admitted call still executes through RunArm's
// registered kernel engines and policy floor.
func NewNativeChildTaskRunner(planner Planner, maxTurns int, baseOpts []RunOption, parentCatalog []ToolDef, policy adjudicator.Policy) ChildTaskRunner {
	if maxTurns <= 0 {
		maxTurns = 1
	}
	if maxTurns > maxNativeChildTurns {
		maxTurns = maxNativeChildTurns
	}
	return func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		catalog := NativeChildToolCatalog(parentCatalog, req.ReadOnly)
		allowed := make(map[string]struct{}, len(catalog))
		for _, def := range catalog {
			allowed[def.Function.Name] = struct{}{}
		}
		childPlanner := nativeChildCompletionFilter{Planner: planner, allowed: allowed}
		opts := append([]RunOption(nil), baseOpts...)
		opts = append(opts,
			withIsolatedPolicy(policy),
			WithToolCatalog(catalog),
		)
		metrics, err := RunArm(ctx, childPlanner, req.Prompt, true, maxTurns, nil, opts...)
		if err != nil {
			return nil, err
		}
		if metrics.HitTurnCap {
			return nil, fmt.Errorf("native child task: turn cap reached after %d turns without completion", metrics.Turns)
		}
		if metrics.CircuitBreakerTripped {
			return nil, fmt.Errorf("native child task: circuit breaker stopped execution: %s", metrics.CircuitBreakerReason)
		}
		if metrics.StoppedBySession != "" {
			return nil, fmt.Errorf("native child task: stopped without completion: %s", metrics.StoppedBySession)
		}
		if metrics.GracefulDrained {
			return nil, fmt.Errorf("native child task: graceful drain synthesized a summary without completion")
		}
		if strings.TrimSpace(metrics.FinalAnswer) == "" {
			return nil, fmt.Errorf("native child task: ended without a final answer")
		}
		return metrics.FinalAnswer, nil
	}
}

// NativeChildToolCatalog derives a depth-one child catalog from the governed
// parent catalog. Read-only children retain only the native read capabilities.
func NativeChildToolCatalog(parent []ToolDef, readOnly bool) []ToolDef {
	out := make([]ToolDef, 0, len(parent))
	for _, def := range parent {
		name := def.Function.Name
		switch name {
		case ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel:
			continue
		}
		if readOnly {
			switch name {
			case codetools.ToolRead, codetools.ToolGrep, codetools.ToolGlob:
			default:
				continue
			}
		}
		out = append(out, def)
	}
	return out
}

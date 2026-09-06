package gateway

import (
	"context"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

// gatewayPluginFloor projects the real kernel decision rung into the typed host.
// A transformed proposal is rebuilt as a fresh abi.ToolCall before Decide, so it
// is re-adjudicated by the same floor as a normal syscall.
type gatewayPluginFloor struct {
	s        *Server
	readOnly bool
	witness  string
	traceID  string
}

func (f gatewayPluginFloor) Adjudicate(ctx context.Context, p toolplugin.Proposal) toolplugin.Decision {
	call, err := f.s.buildCall(ctx, p.Tool, string(p.Args), f.readOnly, f.witness, f.traceID)
	if err != nil {
		return toolplugin.Decision{Action: toolplugin.ActionDeny, Reason: err.Error()}
	}
	verdict := f.s.k.Decide(ctx, call)
	if verdict.Kind == abi.VerdictAllow || verdict.Kind == abi.VerdictDefer {
		return toolplugin.Decision{Action: toolplugin.ActionDefer, Reason: abi.ReasonName(verdict.Reason)}
	}
	return pluginDecisionFromVerdict(verdict)
}

// gatewayPluginExecutor calls the existing syscall path. Kernel.Syscall performs
// its own Submit adjudication before any engine executes, making host-floor and
// execution-floor agreement structural rather than trust in the plugin host.
type gatewayPluginExecutor struct {
	s        *Server
	readOnly bool
	witness  string
	traceID  string
	verdict  WireVerdict
	envelope *ResultEnvelope
	err      error
}

func (e *gatewayPluginExecutor) Execute(ctx context.Context, p toolplugin.Proposal) (json.RawMessage, error) {
	call, err := e.s.buildCall(ctx, p.Tool, string(p.Args), e.readOnly, e.witness, e.traceID)
	if err != nil {
		return nil, err
	}
	e.verdict, e.envelope, err = e.s.executePluginCall(ctx, call, e.readOnly)
	e.err = err
	if err != nil {
		return nil, err
	}
	return json.Marshal(e.envelope)
}

func pluginDecisionFromVerdict(v abi.Verdict) toolplugin.Decision {
	reason := abi.ReasonName(v.Reason)
	switch v.Kind {
	case abi.VerdictAllow, abi.VerdictDefer:
		return toolplugin.Decision{Action: toolplugin.ActionNarrow, Reason: reason}
	case abi.VerdictTransform:
		return toolplugin.Decision{Action: toolplugin.ActionDeny, Reason: "KERNEL_TRANSFORM_REQUIRES_CLIENT_PREVIEW"}
	case abi.VerdictRequireWitness:
		return toolplugin.Decision{Action: toolplugin.ActionRequireWitness, Reason: reason}
	case abi.VerdictQuarantine:
		return toolplugin.Decision{Action: toolplugin.ActionQuarantine, Reason: reason}
	default:
		return toolplugin.Decision{Action: toolplugin.ActionDeny, Reason: reason}
	}
}

// syscallCatalogTool resolves the provider-visible name through the exact
// request snapshot before policy or plugins see the proposal. Only the
// canonical name reaches the existing kernel/toolplugin execution path.
func (s *Server) syscallCatalogTool(ctx context.Context, snapshot toolcatalog.Snapshot, registrations []toolcatalog.Registration, visibleName, rawArgs string, readOnly bool, witness, traceID string, callPreference toolplugin.Preference) (wv WireVerdict, env *ResultEnvelope, trace []toolplugin.TraceEvent, pref *toolplugin.ResolvedPreference, err error) {
	registration, err := toolcatalog.ResolveRegistration(snapshot, visibleName, registrations)
	if err != nil {
		return WireVerdict{Kind: "DENY", Reason: err.Error(), By: "toolcatalog", Disposition: "TERMINAL"}, nil, nil, nil, nil
	}
	return s.syscallWithPlugins(ctx, registration.Program.Name, rawArgs, readOnly, witness, traceID, callPreference)
}
func (s *Server) syscallWithPlugins(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string, callPreference toolplugin.Preference) (wv WireVerdict, env *ResultEnvelope, trace []toolplugin.TraceEvent, pref *toolplugin.ResolvedPreference, err error) {
	layers := s.toolPreferences
	layers.Call = callPreference
	executor := &gatewayPluginExecutor{s: s, readOnly: readOnly, witness: witness, traceID: traceID}
	host := toolplugin.Host{
		Floor:    gatewayPluginFloor{s: s, readOnly: readOnly, witness: witness, traceID: traceID},
		Executor: executor,
		Plugins:  s.toolPlugins,
	}
	out := host.Run(ctx, toolplugin.Proposal{Tool: tool, Args: json.RawMessage(rawArgs)}, layers)
	trace = out.Trace
	pref = &out.Preference
	if executor.err != nil {
		return executor.verdict, executor.envelope, trace, pref, executor.err
	}
	if out.Decision.Action == toolplugin.ActionNarrow || out.Decision.Action == toolplugin.ActionDefer {
		// Result admission is authoritative: never return the executor's saved
		// envelope after the host has withheld or replaced its output.
		if len(out.Result) != 0 {
			if err := json.Unmarshal(out.Result, &env); err != nil {
				return WireVerdict{}, nil, trace, pref, err
			}
		}
		return executor.verdict, env, trace, pref, nil
	}
	kind := "DENY"
	disposition := "TERMINAL"
	if out.Decision.Action == toolplugin.ActionRequireWitness {
		kind, disposition = "REQUIRE_WITNESS", "ESCALATE"
	}
	if out.Decision.Action == toolplugin.ActionQuarantine {
		kind, disposition = "QUARANTINE", "HELD"
	}
	return WireVerdict{Kind: kind, Reason: out.Decision.Reason, By: "toolplugin", Disposition: disposition}, nil, trace, pref, nil
}

// executePluginCall executes one already-built proposal through the same kernel
// syscall/result rendering used by the legacy single-call path. The plugin host
// does not own an engine and cannot execute directly.
func (s *Server) executePluginCall(ctx context.Context, call *abi.ToolCall, readOnly bool) (WireVerdict, *ResultEnvelope, error) {
	if wv, env, handled, err := s.syscallNative(ctx, call, readOnly); handled {
		return wv, env, err
	}
	r, verdict := s.k.Syscall(ctx, call)
	s.rememberOriginSeq(call.TraceID, call.Tool, string(resolveBytes(ctx, call.Args)), call.SeqNo)
	wv := renderVerdict(verdict, resultMeta(r))
	var env *ResultEnvelope
	if r != nil {
		env = &ResultEnvelope{Status: statusName(r.Status), Content: string(resolveBytes(ctx, r.Payload)), Meta: r.Meta}
		s.recordRouteAccount(env, call.Tool, readOnly, call.Meta)
	}
	wv = witnessScriptedFold(call.Tool, env, wv)
	return wv, env, nil
}

package gateway

// dual_planner.go — the small-LOCAL-model-ALONGSIDE-API chat seam (the dual planner).
//
// Historically the gateway's chat planner was an XOR: proxy (--base-url) OR the
// in-kernel model (--gguf) OR the offline mock — `New` picked exactly one, and the
// proxy silently won when both were configured. The dual planner is the "alongside"
// rung of the dual-track serving plan (docs/serving/dual-track-serving-plan.md): ONE
// gateway holds BOTH a live upstream proxy and the model fused into the kernel, and
// routes each chat request by its requested model id. The proxy side is the DEFAULT —
// a request that names no model, or any id other than the local one, takes the proxy
// byte-for-byte as proxy-only mode would — so wrapping an agent's normal API traffic
// is unchanged while a sub-task addressed to the local id (or the literal "local")
// decodes on-box with no upstream call, no API key, and no tokens billed.
//
// The residency floor (internal/engine's residencyGate) is what makes this split
// SAFE, not just cheap: tool-call dispatch already adjudicates local-vs-remote per
// engine; this planner brings the same two-sided deployment to the chat surface.

import (
	"context"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// defaultLocalModelID is the model id that always routes to the in-kernel side of a
// dual planner, so a client needs zero configuration knowledge beyond "ask for local".
const defaultLocalModelID = "local"

// localModelIDOr normalizes a configured local model id, defaulting empty to "local".
func localModelIDOr(id string) string {
	if v := strings.TrimSpace(id); v != "" {
		return v
	}
	return defaultLocalModelID
}

// DualPlanner is an agent.Planner that serves a small local in-kernel model alongside
// a live upstream API proxy in one gateway. Routing is per request, by requested model
// id (the SampleOpt model pass-through the OpenAI-wire handlers already send and the
// Anthropic handler sends in dual mode): the local model id — the configured
// Config.LocalModelID, the id the local planner advertises, or the literal "local" —
// picks the in-kernel side; everything else (including an omitted model) picks the
// proxy. Model() advertises the proxy's id, so a client that never asks for the local
// model cannot tell a dual gateway from a proxy gateway.
type DualPlanner struct {
	proxy   agent.Planner
	local   agent.Planner
	localID string
	// localIDs is the lowercased set of ids that route to the local side.
	localIDs map[string]struct{}
}

// NewDualPlanner wires a proxy (API upstream) planner and a local (in-kernel) planner
// into one per-request-routed planner. localID is the primary id the local side
// answers to ("" defaults to "local"); the local planner's own Model() id and the
// literal "local" route there too.
func NewDualPlanner(proxy, local agent.Planner, localID string) (*DualPlanner, error) {
	if proxy == nil || local == nil {
		return nil, errors.New("gateway: dual planner requires both a proxy and a local planner")
	}
	id := localModelIDOr(localID)
	ids := map[string]struct{}{
		strings.ToLower(id): {},
		defaultLocalModelID: {},
		strings.ToLower(strings.TrimSpace(local.Model())): {},
	}
	// A local planner with an empty Model() must not capture omitted-model requests —
	// those belong to the proxy (the default side).
	delete(ids, "")
	return &DualPlanner{proxy: proxy, local: local, localID: id, localIDs: ids}, nil
}

// Model is the DEFAULT side's id (the proxy upstream), the id /v1/models advertises
// first and the provenance stamp for requests that name no model.
func (d *DualPlanner) Model() string { return d.proxy.Model() }

// LocalModelID is the primary id the in-kernel side answers to, for /v1/models and
// operator banners.
func (d *DualPlanner) LocalModelID() string { return d.localID }

// Proxy exposes the API-upstream side for the seams that must keep treating the proxy
// exactly as if it were the only planner: the Anthropic byte-preserving passthrough
// (prompt-cache preservation) and the upstream retry/auth-refresh observability hooks.
func (d *DualPlanner) Proxy() agent.Planner { return d.proxy }

// Local exposes the in-kernel side (tests, banners).
func (d *DualPlanner) Local() agent.Planner { return d.local }

// RoutesLocal reports whether a request naming reqModel is served by the in-kernel
// side. An empty model routes to the proxy (the default side), so an omitted model
// field never lands on the local model by accident.
func (d *DualPlanner) RoutesLocal(reqModel string) bool {
	m := strings.ToLower(strings.TrimSpace(reqModel))
	if m == "" {
		return false
	}
	_, ok := d.localIDs[m]
	return ok
}

// pick resolves the side for one request by folding the per-request sampling opts and
// reading the requested-model pass-through.
func (d *DualPlanner) pick(opts []agent.SampleOpt) agent.Planner {
	var sp agent.SampleParams
	for _, o := range opts {
		if o != nil {
			o(&sp)
		}
	}
	if d.RoutesLocal(sp.Model) {
		return d.local
	}
	return d.proxy
}

func (d *DualPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return d.pick(opts).Complete(ctx, messages, tools, opts...)
}

// StreamingSupported advertises the planner-seam stream when either side can stream;
// CompleteStream emulates for a picked side that cannot (one final fragment), so the
// gateway never has to unwind a streaming path it already committed to. Note the
// Anthropic proxy wire streams through the byte-preserving passthrough relay, not this
// seam — exactly as in proxy-only mode.
func (d *DualPlanner) StreamingSupported() bool {
	return plannerStreams(d.proxy) || plannerStreams(d.local)
}

func plannerStreams(p agent.Planner) bool {
	sp, ok := p.(agent.StreamingPlanner)
	return ok && sp.StreamingSupported()
}

func (d *DualPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p := d.pick(opts)
	if sp, ok := p.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
		return sp.CompleteStream(ctx, sink, messages, tools, opts...)
	}
	// The picked side cannot stream (today: the in-kernel planner) — buffered
	// completion, delivered through the sink as one final fragment. Same sampling,
	// same adjudication-relevant return shape as Complete, per the StreamingPlanner
	// contract.
	comp, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	if comp != nil && comp.Message.Content != "" {
		if serr := sink(comp.Message.Content); serr != nil {
			return nil, serr
		}
	}
	return comp, nil
}

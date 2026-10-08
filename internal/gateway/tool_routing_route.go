package gateway

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// routeDecision classifies a tool call into a modelroute.Subject (aspect=tool_call,
// the tool name, and the read-only / sensitivity / tenant signals the gateway already
// attests) and returns the manifest's routing Decision. The second return is false
// when no manifest is configured (the kernel-default path). routeEngine and
// ensemblePlan share this single classification so the single-model and ensemble
// paths can never diverge on what a call routes to.
func (s *Server) routeDecision(tool string, readOnly bool, meta map[string]string) (modelroute.Decision, bool) {
	if s.route == nil {
		return modelroute.Decision{}, false
	}
	return s.route.Route(modelroute.Subject{
		Aspect: modelroute.AspectToolCall,
		Tool:   tool,
		Labels: routeLabels(readOnly, meta),
	}), true
}

// routeEngine consults the optional per-call routing policy and returns the engine
// route to bind to abi.ToolCall.Engine, or "" for the kernel default. It returns
// Decision.Plan.Primary() for a single-model PICK. An ENSEMBLE plan is left to the
// kernel default here (route ""): the N-submit fan-out happens at dispatch time in
// dispatchEnsemble (the syscall path), and collapsing an ensemble to one member here
// would be a silent wrong route. The returned route is the model id verbatim
// (Plan.Primary()'s documented destination), NOT collapsed to a registered engine id —
// the string must keep the model's remote-ness so the residency gate can deny a
// tenant/sensitive payload bound for a remote model. A route to a model with no
// registered engine driver fails LOUD at dispatch ("no engine registered for route"),
// never silently runs elsewhere.
func (s *Server) routeEngine(tool string, readOnly bool, meta map[string]string) (string, error) {
	return s.routeEngineWithContext(context.Background(), tool, readOnly, meta)
}

func (s *Server) routeEngineWithContext(ctx context.Context, tool string, readOnly bool, meta map[string]string) (string, error) {
	began := time.Now()
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok {
		// No manifest: the kernel-default path, never reached when routing is off — record
		// nothing so the family honestly reads 0 until routing is actually live.
		return "", nil
	}
	// Routing is LIVE for this call: fold the per-aspect Decision into the observability
	// journal (#603) so it reaches /metrics AND the audit trail. This is the ONE fold per
	// served tool call — routeEngine runs on every buildCall (single-model and ensemble
	// alike); ensemblePlan re-routes the same Subject at dispatch but does not re-record, so
	// a call is counted exactly once. The overhead is the wall-clock the decision itself cost
	// (pure-function routing, so tiny). nil metrics / nil routing accumulator => no-op.
	s.metrics.observeRouteDecision(s.routeManifestVersion(), d, time.Since(began))
	if d.Plan.IsEnsemble() {
		FeatureActivationTrackerFromContext(ctx).RecordActivation(FeatureRouteManifest, FeatureOutcomeUsed)
		return "", nil
	}
	// meta already carries the request's isolation principal (buildCall lowered it from
	// ctx onto vdso.MetaPrincipal), so the residency arm reads the SAME principal the
	// vDSO scopes its cache by — one identity, not two that could disagree.
	route, err := s.resolveRoute(d.Plan.Primary(), meta[vdso.MetaPrincipal])
	if err == nil && d.Plan.Primary() != "" {
		FeatureActivationTrackerFromContext(ctx).RecordActivation(FeatureRouteManifest, FeatureOutcomeUsed)
	}
	return route, err
}

// resolveRoute maps a routed model id to the engine route bound to abi.ToolCall.Engine
// (#2528). With an account roster configured it BINDS the abstract id through the
// roster to the account-resolved Target.EngineRoute() ("openai:acct/gpt-5.5") — the
// load-bearing residency contract, since the residency PDP reads the route INSIDE the
// adjudication fold, so the account-resolved remote/local route must be visible BEFORE
// Submit. Without a roster it returns the id verbatim (byte-for-byte the pre-#2528
// path). A model id that cannot resolve (unknown account, no binding + no default) is a
// FAIL-LOUD error carrying the recovery hint from the pure resolver — never a silent
// fallback to the default engine. An empty id (no primary member) resolves to "" (the
// kernel default), never through the roster.
//
// principal is the caller's tenant ISOLATION principal (the org/project a keyset key
// authenticated as, #5332) and gates WHICH account this call may resolve through — the
// residency arm of the keyset. The check runs HERE, at the same pre-Submit seam that
// binds Engine, because that is the last point before the call reaches the kernel: an
// account the principal is not provisioned for must never become a bound route. It is
// fail-CLOSED in both directions — an account naming principals admits only its listed
// tenants, and the EMPTY principal (an unattributed caller: no keyset, or the single
// --require-key-env bearer) is refused by a restricted account rather than inheriting
// its credential. A roster whose accounts name NO principals admits everyone, so a
// pre-#5332 roster routes byte-for-byte as before.
func (s *Server) resolveRoute(modelID, principal string) (string, error) {
	if s.roster == nil || modelID == "" {
		return modelID, nil
	}
	t, err := s.roster.Resolve(modelID)
	if err != nil {
		return "", fmt.Errorf("gateway: route accounts: %w (fix the roster binding for %q or set a default account; no silent fallback)", err, modelID)
	}
	if !t.Admits(principal) {
		// Name the principal and the account, never the credential: the operator needs to
		// see WHICH tenancy was refused to fix the roster, and Target carries only the
		// credential env NAME anyway. An empty principal is reported as such so an operator
		// can tell "wrong tenant" apart from "unattributed caller".
		who := principal
		if strings.TrimSpace(who) == "" {
			who = "<unattributed>"
		}
		return "", fmt.Errorf("gateway: route accounts: principal %s is not admitted to account %q (routed model %q): that account's principals allowlist scopes it to another tenant (#5332) — add this principal to the account, or bind its key to an account it is provisioned for", who, t.Account, modelID)
	}
	return t.EngineRoute(), nil
}

// routeAccount resolves the account binding for a SINGLE-MODEL routed call so the served
// path can record it (#2528 observability). ok is false when routing is off, no roster is
// configured, the plan is an ensemble, or the id cannot resolve (the fail-loud already
// surfaced at buildCall — observability never re-raises it). The returned Target carries
// only non-secret fields (account id, provider kind, upstream model, credential env NAME),
// so it is safe to fold into a report; the credential VALUE never enters a Target. Pure
// (re-runs the cheap classification), consistent with ensemblePlan re-routing the same
// Subject at dispatch.
func (s *Server) routeAccount(tool string, readOnly bool, meta map[string]string) (modelroute.Target, bool) {
	if s.roster == nil {
		return modelroute.Target{}, false
	}
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || d.Plan.IsEnsemble() {
		return modelroute.Target{}, false
	}
	prim := d.Plan.Primary()
	if prim == "" {
		return modelroute.Target{}, false
	}
	t, err := s.roster.Resolve(prim)
	if err != nil {
		return modelroute.Target{}, false
	}
	return t, true
}

// recordRouteAccount folds the non-secret account binding of a single-model routed call
// into the result envelope Meta (#2528 acceptance: "records route decision plus account
// id/provider kind/upstream model with no secret values"). It writes the account id,
// provider kind, upstream wire model, the account-resolved engine route, and the
// credential env NAME (a name, never the secret — the ticket explicitly permits the env
// name in reports). No-op when no roster resolved the call, so the pre-#2528 meta is
// byte-for-byte unchanged.
func (s *Server) recordRouteAccount(env *ResultEnvelope, tool string, readOnly bool, meta map[string]string) {
	if env == nil {
		return
	}
	t, ok := s.routeAccount(tool, readOnly, meta)
	if !ok {
		return
	}
	if env.Meta == nil {
		env.Meta = map[string]string{}
	}
	env.Meta["route_account"] = t.Account
	env.Meta["route_kind"] = string(t.Kind)
	env.Meta["route_upstream"] = t.UpstreamModel
	env.Meta["route_engine"] = t.EngineRoute()
	if t.CredEnv != "" {
		env.Meta["route_cred_env"] = t.CredEnv // the env-var NAME, never its value
	}
}

// routeManifestVersion returns the installed routing manifest's schema version (for the
// decision digest), defaulting to the current modelroute.Version when the manifest omits
// it or no manifest is installed.
func (s *Server) routeManifestVersion() string {
	if s.route != nil {
		if mf := s.route.Manifest(); mf != nil && mf.Version != "" {
			return mf.Version
		}
	}
	return modelroute.Version
}

// ensemblePlan returns the routing Plan for this call WHEN it is a multi-member
// ensemble, so the syscall path can fan it out (issue #597). A single-model PICK, or
// no manifest, returns ok=false (the call dispatches once on the route routeEngine
// already bound to Engine). The classification is identical to routeEngine's — same
// Subject, same routeDecision — so the two never disagree on whether a call is an
// ensemble.
func (s *Server) ensemblePlan(tool string, readOnly bool, meta map[string]string) (modelroute.Plan, bool) {
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || !d.Plan.IsEnsemble() {
		return modelroute.Plan{}, false
	}
	return d.Plan, true
}

// routeLabels lowers the call signals the gateway honestly knows into the OPEN
// Subject.Labels a manifest Match can route on: read_only (read- vs write-shaped),
// and the sensitivity / tenant tags the residency floor also reads. Per-call prompt
// token estimation and richer classification are a later signal-enrichment child
// (#599 scout classification); the gateway routes on what it can attest today.
func routeLabels(readOnly bool, meta map[string]string) map[string]string {
	labels := map[string]string{"read_only": boolLabel(readOnly)}
	if meta != nil {
		sens := meta["sensitivity"]
		if sens == "" {
			sens = meta["data_sensitivity"]
		}
		if sens != "" {
			labels["sensitivity"] = sens
		}
		if p := meta[vdso.MetaPrincipal]; p != "" {
			labels["tenant"] = p
		}
	}
	return labels
}

// boolLabel renders a bool as a routing-label string ("true"/"false") without
// pulling strconv into this file (it formats ints via the local itoa).
func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

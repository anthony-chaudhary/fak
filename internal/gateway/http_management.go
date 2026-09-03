package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type codexServiceTier struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// codexServiceTierCatalog translates provider routing metadata at the public
// gateway seam. Codex selects a tier by its provider wire value, while the
// portable mode remains advertised separately through additional_speed_tiers.
func codexServiceTierCatalog(rows []map[string]string) []codexServiceTier {
	catalog := make([]codexServiceTier, 0, len(rows))
	for _, row := range rows {
		mode := strings.TrimSpace(row["mode"])
		id := strings.TrimSpace(row["wire_value"])
		if id == "" {
			id = mode
		}
		name := strings.ReplaceAll(mode, "_", " ")
		if name == "" {
			name = id
		}
		if name != "" {
			name = strings.ToUpper(name[:1]) + name[1:]
		}
		catalog = append(catalog, codexServiceTier{
			ID:          id,
			Name:        name,
			Description: name + " service tier",
		})
	}
	return catalog
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	data := []map[string]any{{"id": s.model, "object": "model", "owned_by": "fak"}}
	// Dual mode (local model alongside the API upstream): advertise the in-kernel
	// model's id too, so an OpenAI-wire client can DISCOVER the local side instead of
	// needing out-of-band knowledge of the alias.
	if d, ok := s.planner.(*DualPlanner); ok {
		data = append(data, map[string]any{"id": d.LocalModelID(), "object": "model", "owned_by": "fak"})
	}
	if s.roster != nil {
		seen := make(map[string]struct{}, len(data)+len(s.roster.Bindings))
		for _, item := range data {
			seen[item["id"].(string)] = struct{}{}
		}
		for _, binding := range s.roster.Bindings {
			id := strings.TrimSpace(binding.Model)
			if id == "" {
				continue
			}
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			data = append(data, map[string]any{
				"id":       id,
				"object":   "model",
				"owned_by": "fak-route-accounts",
			})
		}
		// Roster declaration order is not an API contract. Stable ordering also makes
		// equivalent roster files advertise byte-identical catalogs. No-roster output
		// deliberately keeps the historical order above.
		sort.Slice(data, func(i, j int) bool {
			return data[i]["id"].(string) < data[j]["id"].(string)
		})
	}
	tiers, tierRows := []string{}, []map[string]string{}
	if contract, ok := modelroute.LookupProviderContract("openai"); ok {
		tiers, tierRows = modelroute.SupportedServiceTierMetadata(contract)
	}
	tierCatalog := codexServiceTierCatalog(tierRows)
	codexModels := make([]map[string]any, 0, len(data))
	for _, row := range data {
		id := strings.TrimSpace(fmt.Sprint(row["id"]))
		if id == "" {
			continue
		}
		codexModels = append(codexModels, map[string]any{
			"slug":                    id,
			"display_name":            id,
			"description":             "fak gateway model",
			"base_instructions":       "",
			"default_reasoning_level": "medium",
			"supported_reasoning_levels": []map[string]string{
				{"effort": "low", "description": "Light reasoning"},
				{"effort": "medium", "description": "Default reasoning"},
				{"effort": "high", "description": "Deep reasoning"},
			},
			"shell_type":                       "shell_command",
			"visibility":                       "list",
			"supported_in_api":                 true,
			"supports_reasoning_summaries":     false,
			"support_verbosity":                false,
			"default_reasoning_summary":        "none",
			"default_verbosity":                "low",
			"apply_patch_tool_type":            "freeform",
			"web_search_tool_type":             "text_and_image",
			"truncation_policy":                map[string]any{"mode": "tokens", "limit": 10000},
			"supports_parallel_tool_calls":     true,
			"supports_image_detail_original":   false,
			"context_window":                   272000,
			"max_context_window":               272000,
			"comp_hash":                        "fak",
			"effective_context_window_percent": 95,
			"experimental_supported_tools":     []string{},
			"input_modalities":                 []string{"text"},
			"supports_search_tool":             false,
			"use_responses_lite":               false,
			"priority":                         0,
			"additional_speed_tiers":           tiers,
			"service_tiers":                    tierCatalog,
			"availability_nux":                 nil,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"models": codexModels,
	})
}

// handleFakSession is the session DRIVE-state control surface, the read-write
// generalization of /v1/fak/trace (which carries exactly one bit — taint). It is
// mounted on the /v1/fak/session/ subtree; the remainder is "{trace_id}" for an
// observe (GET) or "{trace_id}/{verb}" for a control verb (POST). The observe and
// control implementations are injected by cmd/fak so this package stays
// session-internals-blind, mirroring resetTrace/observeTrace. A nil injection ⇒
// 404 (never a silent clean reading) — the same fail-closed posture as the trace
// routes with no ledger.
// ownsSessionLoop reports whether THIS serve process drives an owned agent loop
// (agent.RunArm) — the loop that drains the operator/machine steer bus at its turn
// boundary (drainSteer, #850). Only the native serve path (`fak serve --native`)
// constructs such a loop; the default proxy path forwards a single upstream turn and
// owns no loop, so a steer to a proxy-served session can never be consumed. The check
// is process-level: native mode drives every /v1/messages trace through RunArm, so the
// honest predicate is "does this process own a turn loop at all". A future persistent
// per-session native loop can refine this to a live per-trace registry without changing
// the /steer route contract (the refusal reason and the accept path both stay put).
func (s *Server) ownsSessionLoop() bool {
	return s.native
}

func (s *Server) handleFakSession(w http.ResponseWriter, r *http.Request) {
	traceID, verb, ok := requirePathIDVerb(w, r, "/v1/fak/session/", "trace_id is required")
	if !ok {
		return
	}

	switch r.Method {
	case http.MethodGet:
		// subscribe (#2767) is the one GET verb on the subtree: the per-session
		// re-attach drain of the drive-state revision feed (session_subscribe.go).
		// Every other GET verb is not the observe shape.
		if verb == "subscribe" {
			s.handleFakSessionSubscribe(w, r, traceID)
			return
		}
		if s.handleFakSessionClient(w, r, traceID, verb) {
			return
		}
		// GET observes one session. A verb on the path is not the observe shape.
		if verb != "" {
			writeErr(w, http.StatusMethodNotAllowed, "use GET /v1/fak/session/{trace_id}")
			return
		}
		if s.observeSession == nil {
			writeErr(w, http.StatusNotFound, "session observe is not configured")
			return
		}
		writeJSON(w, http.StatusOK, s.observeSession(r.Context(), traceID))
	case http.MethodPost:
		// POST applies a control verb. The verb is required from the path.
		if verb == "" {
			writeErr(w, http.StatusBadRequest, "control verb is required: POST /v1/fak/session/{trace_id}/{run|pause|resume|throttle|stop|budget|pace|priority|wall|throughput}")
			return
		}
		// #2439: the kernel ASSIGNS this control event's authority principal from the
		// authenticated transport (the route's floor + the trusted front door's relay
		// header), never from the request body. Authority-CONSUMING verbs — pause/resume
		// via "run", and the policy-widening budget/wall/throughput — are then refused
		// outright under a peer / timer / network principal, so "a webhook paused the run"
		// or "a scheduled task widened the budget" is unrepresentable rather than patched
		// per channel. Ordered before every verb dispatch so no shape escapes the check;
		// each verb, refused or admitted, is journaled with its principal.
		principal := kernelPrincipal(r)
		if controlVerbConsumesAuthority(verb) && !principal.IsHuman() {
			s.journalControlPrincipal(ControlPlaneEvent{
				TraceID:   traceID,
				Verb:      verb,
				Principal: principal,
				Refused:   true,
				Reason:    ReasonPrincipalNotHuman,
			})
			writeErrCode(w, http.StatusForbidden, "principal_not_human",
				ReasonPrincipalNotHuman+": the "+verb+" verb consumes user authority and arrived under the "+
					string(principal)+" principal; a relayed message, a webhook delivery, or a scheduled task "+
					"cannot spend the operator's authority — re-issue it from the human control wire")
			return
		}
		// fork/export/import are their own shape (#2419): they act on the durable
		// session CHAIN rather than on the drive table, so they take their own bodies
		// and answer their own documents (session_teleport.go). Dispatched before the
		// generic control path, which decodes a SessionControlRequest.
		if s.handleTeleportVerb(w, r, traceID, verb) {
			return
		}
		// checkpoint (#2425) is the same family: it binds the durable chain's head to a
		// git tree witness in one record, so it carries its own body and answers its own
		// document (session_checkpoint.go) rather than the drive-state control shape.
		if s.handleSessionCheckpointVerb(w, r, traceID, verb) {
			return
		}
		if s.handleFakSessionClient(w, r, traceID, verb) {
			return
		}
		// steer is its own shape (operator input to a RUNNING session, #760): a different
		// body and a different sink (the a2achan bus, not the drive table). Dispatch it
		// before the generic control path. A refused steer (tainted/over-scoped/uncapped)
		// is the adjudication floor's deny — distinct from the control 409 — so it maps to
		// 422 (unprocessable), not 409 (terminal/stale rev).
		if verb == "steer" {
			if s.steerSession == nil {
				writeErr(w, http.StatusNotFound, "session steer is not configured")
				return
			}
			var sr SteerRequest
			if !decodeRequestBody(w, r, &sr) {
				return
			}
			if strings.TrimSpace(sr.Text) == "" {
				writeErr(w, http.StatusBadRequest, "steer text is required")
				return
			}
			// Classify the append (#2402): a class ("now"/"next"/"later") decides WHEN it
			// reaches the loop and the query bit decides WHETHER it forces a model turn. An
			// unrecognized class is a request-shape error (400), ordered with the other shape
			// checks and before the owned-loop gate, so a bad class fails the same way an empty
			// body does regardless of serve mode.
			class, classOK := sr.steerClass()
			if !classOK {
				writeErr(w, http.StatusBadRequest, "steer class must be one of now|next|later")
				return
			}
			querying := sr.querying()
			// Honest steer contract (#3528): a steer is only ever CONSUMED by an owned agent
			// loop — the native RunArm loop drains the a2achan Session bus at its turn boundary
			// (drainSteer, #850). The default PROXY serve forwards a single upstream turn and
			// owns no such loop, so an enqueued steer there would sit in a mailbox nothing
			// drains — an accepted-but-never-applied phantom. Refuse it at ingress with the
			// closed STEER_NO_OWNED_LOOP reason rather than return a false 202, so the operator
			// learns the steer will not land instead of trusting a lie. Ordered AFTER the shape
			// checks (404 not-configured, 400 empty) and BEFORE the floor Send, so a bad request
			// is still a 400/404 and only a well-formed-but-undeliverable steer is the 409.
			if !s.ownsSessionLoop() {
				writeErrCode(w, http.StatusConflict, "steer_no_owned_loop",
					"STEER_NO_OWNED_LOOP: this serve process forwards proxy turns and owns no agent loop to "+
						"drain the steer; a steer is applied only by a native owned loop (start the gateway with "+
						"--native, which runs agent.RunArm and drains the steer bus each turn), or hand the input "+
						"to the harness that owns this session's turn loop")
				return
			}
			// Screen the append BEFORE the loop can see it (#2402): the same context screen
			// the result-admission path uses (ctxmmu.ScreenBytes) refuses a poisonous append at
			// ingress as a journaled quarantine stub rather than letting an observer feed inject
			// unscreened bytes into the turn. Ordered after the owned-loop gate so only a
			// deliverable steer is screened; a held append maps to 422 (the floor's deny), like a
			// tainted/over-scoped body.
			if reason, held := screenSteerText(sr.Text); held {
				writeErrCode(w, http.StatusUnprocessableEntity, "steer_quarantined",
					"STEER_QUARANTINED: the append tripped the context screen ("+reason+
						"); it is held as a quarantine stub and never reaches the loop")
				return
			}
			// #2439: the append reaches the bus under the KERNEL's principal. A steer is
			// input, not an authority-consuming act, so a peer/timer steer still lands — but
			// it lands labelled peer-agent/timer instead of being able to present as the
			// operator by writing "operator" in its body. The journal row is written before
			// the Send so a refused steer is still attributable to its principal.
			s.journalControlPrincipal(ControlPlaneEvent{TraceID: traceID, Verb: verb, Principal: principal})
			if err := s.steerSession(r.Context(), traceID, stampedSteerPrincipal(principal, sr.Principal), sr.Text); err != nil {
				writeErr(w, http.StatusUnprocessableEntity, "steer refused: "+err.Error())
				return
			}
			s.logf("gateway: session %s steer accepted (class=%s querying=%v, %d bytes)", traceID, class, querying, len(sr.Text))
			writeJSON(w, http.StatusAccepted, map[string]any{
				"trace_id": traceID,
				"steered":  true,
				"class":    class.String(),
				"querying": querying,
			})
			return
		}
		if s.controlSession == nil && s.table == nil {
			writeErr(w, http.StatusNotFound, "session control is not configured")
			return
		}
		var req SessionControlRequest
		if r.Body != nil {
			if err := decodeJSON(w, r, &req); err != nil && !errors.Is(err, io.EOF) {
				writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
				return
			}
		}

		var runStr string
		switch verb {
		case "pause":
			runStr = "paused"
		case "resume":
			runStr = "running"
		case "throttle":
			runStr = "throttled"
		case "stop":
			runStr = "stopped"
		}

		if s.controlSession != nil {
			st, ok, err := s.controlSession(r.Context(), traceID, verb, req)
			if err != nil && runStr != "" {
				reqCopy := req
				reqCopy.Run = runStr
				st, ok, err = s.controlSession(r.Context(), traceID, "run", reqCopy)
			}
			if err != nil {
				writeErr(w, http.StatusBadRequest, err.Error())
				return
			}
			if !ok {
				// Terminal session, or an if_rev CAS guard lost the race: the client
				// re-reads and retries. Not a malformed request.
				writeErr(w, http.StatusConflict, "session control refused (terminal or stale rev)")
				return
			}
			// #2439: an ADMITTED control verb is journaled with its principal too — the record
			// answers "which principal drove this session" for every event, not only the refused
			// ones, so an absent row is evidence of an unstamped path rather than of a clean run.
			s.journalControlPrincipal(ControlPlaneEvent{TraceID: traceID, Verb: verb, Principal: principal})
			s.logf("gateway: session %s %s -> rev %d (%s)", traceID, verb, st.Rev, st.Run)
			writeJSON(w, http.StatusOK, st)
			return
		}

		if s.table != nil {
			if runStr == "" && verb == "run" {
				runStr = req.Run
			}
			if runStr == "" {
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("unknown verb %q", verb))
				return
			}
			run, parsed := session.ParseRunState(runStr)
			if !parsed {
				writeErr(w, http.StatusBadRequest, fmt.Sprintf("unknown run-state %q", runStr))
				return
			}
			var st session.State
			var ok bool
			if req.IfRev > 0 {
				cur := s.table.Get(traceID)
				cur.Run = run
				if run == session.Running {
					cur.Reason = ""
				} else {
					cur.Reason = req.Reason
				}
				st, ok = s.table.CompareAndSet(traceID, req.IfRev, cur)
			} else {
				st, ok = s.table.Transition(traceID, run, req.Reason)
			}
			if !ok {
				writeErr(w, http.StatusConflict, "session control refused (terminal or stale rev)")
				return
			}
			s.journalControlPrincipal(ControlPlaneEvent{TraceID: traceID, Verb: verb, Principal: principal})
			gwSt := toGatewaySessionState(st)
			s.logf("gateway: session %s %s -> rev %d (%s)", traceID, verb, gwSt.Rev, gwSt.Run)
			writeJSON(w, http.StatusOK, gwSt)
			return
		}
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or POST")
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	// planner names the /v1/chat/completions backend ("mock" | "proxy" | "inkernel")
	// so a probe can detect the silent offline-mock fallback that New also warns
	// about at boot — scripted responses must never be mistaken for model output.
	// #1115: in_kernel_model_but_chat_is_mock exposes when kernel has real weights
	// loaded (for fak_syscalls) but chat uses mock due to missing tokenizer.
	health := map[string]any{
		"ok":      true,
		"engine":  s.engineID,
		"model":   s.model,
		"planner": plannerKind(s.planner),
	}
	if r.URL.Query().Get("deep") == "1" {
		probe, ok := s.planner.(interface {
			ProbeReachability(context.Context) (int, error)
		})
		if !ok {
			health["provider_reachability"] = map[string]any{"evaluated": false, "reason": "planner has no external provider hop"}
		} else {
			status, err := probe.ProbeReachability(r.Context())
			reach := map[string]any{"evaluated": true, "status": status}
			if err != nil {
				health["ok"] = false
				reach["ok"] = false
				reach["error"] = err.Error()
			} else {
				reach["ok"] = true
			}
			health["provider_reachability"] = reach
		}
	}
	if len(s.nativeCodeCatalog) > 0 {
		tools := make([]string, 0, len(s.nativeCodeCatalog))
		for _, def := range s.nativeCodeCatalog {
			tools = append(tools, def.Function.Name)
		}
		health["native_code_workspace"] = map[string]any{"armed": true, "tools": tools}
	}
	if s.inKernelModelButChatIsMock {
		health["in_kernel_model_but_chat_is_mock"] = true
	}
	// A boot-time fixed-prompt decode can prove that the local model is loaded but
	// incoherent. Keep that process out of readiness until it restarts and probes
	// cleanly; proxy/mock serves never arm this gate.
	if kind, sample, bad := s.startupDecode.degenerate(); bad {
		health["ok"] = false
		health["degenerate_decode"] = map[string]any{
			"kind":   kind,
			"sample": sample,
		}
	}
	// #3051: a local backend that is UP (listener bound) but has not finished its
	// one-time warmup — weight load, CUDA-graph capture, DeepGEMM/JIT compile — must
	// not report ready, or the operator's first real turn absorbs the full ~500s
	// tax. When the host armed the warmup gate at boot, /healthz stays ok:false with
	// warmup_pending until a synthetic warmup inference returns its first token
	// (MarkWarmupComplete); after that it exposes time_to_ready_ms so the one-time
	// tax is visible. A serve that never arms the gate is unaffected. Liveness is not
	// readiness. See readiness_warmup.go.
	if s.warmup.pending() {
		health["ok"] = false
		health["warmup_pending"] = true
	} else if ttr, ok := s.warmup.ready(); ok {
		health["time_to_ready_ms"] = ttr.Milliseconds()
	}
	// #2336: a recent recovered panic on a served completion route disqualifies
	// the unqualified ok:true — a green liveness probe over crashing completions
	// keeps watchdogs routing work to a broken native serve. Window-bounded
	// (servedFailureWindow), so an aged-out failure restores the plain report.
	if route, msg, age, failed := s.servedFailure.recent(time.Now()); failed {
		health["ok"] = false
		health["recent_served_failure"] = map[string]any{
			"route":          route,
			"error":          msg,
			"age_seconds":    int(age / time.Second),
			"window_seconds": int(servedFailureWindow / time.Second),
		}
	}
	// #3425: the deployment-boundary saturation readout rides the readiness surface too,
	// not just /metrics — an operator or autoscaler probing /healthz sees live sessions
	// against the configured ceiling (FAK_MAX_SESSIONS) and can scale out BEFORE the box
	// starts shedding. Saturation does NOT flip ok:false: the deployment is healthy at
	// the ceiling, it just backpressures NEW sessions (SESSION_CEILING_SATURATED) while
	// the loops in flight run untouched. Absent entirely when no ceiling is configured.
	if sat := s.sessionSaturationNow(r.Context()); sat.Bounded {
		health["session_saturation"] = sat
	}
	writeJSON(w, http.StatusOK, health)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

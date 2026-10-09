package gateway

// serving_props.go — GET /props and GET /slots: the Fak-native engine
// introspection surface.
//
// WHY THIS EXISTS. Fak's gateway already serves the Prometheus families a
// capability gate needs (`fak_serving_num_requests_running`,
// `fak_serving_prefix_cache_hit_rate`, `fak_gateway_kv_prefix_reused_tokens_total`),
// but a consumer that speaks only the llama-server shape — `GET /props` for the
// engine's own self-description and `GET /slots` for its per-slot KV state —
// gets a 404 and concludes "this engine cannot be observed". That is a missing
// READ, not a missing capability: the numbers were live on /metrics the whole
// time. These two routes project that same live state onto the shape those
// consumers already parse.
//
// THE SINGLE SOURCE OF TRUTH. Every number below is read from the exact state
// `/metrics` renders, and nothing is cached, derived, or second-guessed:
//
//   - total_slots  <- AdmissionPolicy.MaxNumSeqs, the sequence-count cap the
//     admission controller checks (admission.go). It is an upper bound, not the
//     effective concurrency: the binding gate is the token budget below, which
//     admits requests only while the sum of their footprints (prompt chars/4 +
//     min(max_tokens, prealloc ceiling)) fits. BatchPolicy.MaxBatch is not used
//     either: batchsched.go states in its HONEST FENCE header that the
//     composition policy is NOT wired into the live serve path.
//   - admission_token_budget / _source / admission_prealloc_tokens <-
//     AdmissionPolicy.TokenBudget, TokenBudgetProvenance(), and the effective
//     PreallocCeiling: the bound that actually limits concurrent admissions
//     (TestNativeAdmissionConcurrencyUnderShippingPolicy).
//   - default_generation_settings.n_ctx <- inKernelContextWindow(s.planner),
//     the loaded model's declared context window — byte-identical to the
//     context_length /v1/models already publishes (http_management.go).
//   - running / waiting / prefix-cache hit rate <- admissionCtl.Stats() and
//     cacheobs.Default.Snapshot(), the same reads nativeServingMetricRow folds
//     into fak_serving_num_requests_{running,waiting} and
//     fak_serving_prefix_cache_hit_rate.
//   - /slots token counters <- cacheobs.Default.Snapshot(), the same tap
//     writeKVPrefixMetrics folds into fak_gateway_kv_prefix_{prompt,reused}_tokens_total.
//
// HONESTY CONTRACT — what is NOT here, deliberately.
//
//   - A field with no live source is OMITTED, never defaulted. A proxy gateway
//     (no in-kernel model) installs no admission controller, so total_slots and
//     n_ctx are absent keys rather than a fabricated 0 or 1: "unreported" and
//     "reported zero" are different facts and only an absent key says the first.
//   - /slots reports ONE entry, and that is an architectural fact, not a
//     shortcut. fak's in-kernel path serves every session from a SINGLE
//     resident radix KV prefix; there is no per-session slot allocation to
//     enumerate. A llama.cpp build with --parallel N reports N independent
//     per-slot contexts; fak has one shared KV state that many concurrent
//     sequences read. The fak_slot_model field says so on the wire so no reader
//     can mistake "one KV state" for "one concurrent request".
//   - The n_prompt_tokens* counters are PROCESS-LIFETIME TOTALS over served
//     in-kernel turns, carried on a single slot row. They are real measured
//     values — the identical integers /metrics publishes — but they are not a
//     per-request counter and must not be read as one. fak_scope names the
//     exact /metrics family each mirrors so the provenance is on the wire
//     rather than in a reader's head. On a pure-proxy workload nothing feeds the
//     tap and every one of them is a truthful 0.
//
// A fak-native surface that emitted a number no in-kernel state backs would be
// worse than the 404 it replaces: it would let a gate that could not see the
// capability read as one that saw it working.

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// servingPropsSchema / servingSlotsSchema version the two documents. A consumer
// that does not recognise the version refuses the document rather than guessing
// at a field, matching the posture every other fak wire schema takes.
const (
	servingPropsSchema = "fak.props.v1"
	servingSlotsSchema = "fak.slots.v1"

	// servingSlotsScope names, on the wire, exactly what the /slots token
	// counters are. The llama-server names read as per-slot per-request
	// counters; fak's are process-lifetime totals over served in-kernel turns,
	// so the scope is stated rather than left for a reader to assume.
	servingSlotsScope = "process-lifetime total over served in-kernel turns; not a per-request counter"

	// servingSlotModel states fak's KV topology next to the counters: one shared
	// resident radix prefix serving every concurrent sequence, not N independent
	// per-slot contexts.
	servingSlotModel = "single shared resident radix KV prefix serving all concurrent sequences; not one slot per request"
)

// servingPropsGenerationSettings is the llama-server-shaped default generation
// settings block. Only n_ctx is emitted, and only when the loaded model
// declared one; an absent n_ctx means the engine reported no context window,
// which is a fact about the engine and not a zero-token engine.
type servingPropsGenerationSettings struct {
	NCtx int `json:"n_ctx,omitempty"`
}

// servingPropsWire is the GET /props document.
//
// Every fak_* field is a disclosure, not a claim of capability: it names where
// the sibling field was read from so a reader can audit the number against
// /metrics without trusting this handler. None of them is load-bearing for a
// capability verdict.
type servingPropsWire struct {
	// TotalSlots is the admission sequence-count cap (MaxNumSeqs), an upper
	// bound rather than the effective concurrency; AdmissionTokenBudget is the
	// binding bound. OMITTED when no admission controller is installed, which
	// is the pure-proxy shape: a 0 would read as "an engine with zero slots".
	TotalSlots *int `json:"total_slots,omitempty"`
	// TotalSlotsSource names the field the cap was read from.
	TotalSlotsSource string `json:"fak_total_slots_source,omitempty"`

	// AdmissionTokenBudget is the admission token budget: requests are admitted
	// only while the sum of their footprints fits within it, so it, not
	// total_slots, bounds native concurrency. AdmissionPreallocTokens is the
	// per-request generation-token reservation ceiling inside each footprint.
	// All three are OMITTED when no admission controller is installed.
	AdmissionTokenBudget       *int   `json:"admission_token_budget,omitempty"`
	AdmissionTokenBudgetSource string `json:"admission_token_budget_source,omitempty"`
	AdmissionPreallocTokens    *int   `json:"admission_prealloc_tokens,omitempty"`

	DefaultGenerationSettings servingPropsGenerationSettings `json:"default_generation_settings"`

	// BuildInfo mirrors the live label set of fak_gateway_build_info
	// (metrics_render.go): version, dispatch engine, serving planner, model, vDSO.
	BuildInfo string `json:"build_info"`
	// ModelAlias is the model id this gateway serves — the same string
	// /v1/models advertises.
	ModelAlias string `json:"model_alias,omitempty"`

	// EndpointSlots / EndpointMetrics are this process's OWN claims about its
	// introspection surface, read off its live route table rather than
	// hard-coded, so the claim cannot drift from what is actually served.
	EndpointSlots   bool `json:"endpoint_slots"`
	EndpointMetrics bool `json:"endpoint_metrics"`

	Schema  string `json:"fak_schema"`
	Planner string `json:"fak_planner"`
	Engine  string `json:"fak_engine,omitempty"`

	// RunningRequests / WaitingRequests are the live admission counters — the
	// same values behind fak_serving_num_requests_{running,waiting}. OMITTED
	// when no admission controller is installed.
	RunningRequests *int `json:"fak_running_requests,omitempty"`
	WaitingRequests *int `json:"fak_waiting_requests,omitempty"`
	// PrefixCacheHitRate is the realized KV-prefix hit rate, the same value
	// behind fak_serving_prefix_cache_hit_rate. OMITTED until a turn has been
	// observed, so an idle engine reports nothing rather than a phantom 0.0.
	PrefixCacheHitRate *float64 `json:"fak_prefix_cache_hit_rate,omitempty"`
	// InKernelModel distinguishes "no in-kernel model, so nothing can be fed"
	// from "in-kernel and idle" for a reader of the zero-valued counters.
	InKernelModel bool `json:"fak_in_kernel_model"`
}

// servingSlotWire is one GET /slots entry.
//
// The three n_prompt_tokens* counters are the required llama-server shape and
// carry the live cacheobs totals described by servingSlotsScope. The fak_*
// fields restate the same integers under names that cannot be mistaken for
// per-request counters, so the disclosure survives even a consumer that reads
// only the fields it knows.
type servingSlotWire struct {
	ID                     int    `json:"id"`
	NCtx                   *int   `json:"n_ctx,omitempty"`
	IsProcessing           bool   `json:"is_processing"`
	NPromptTokens          uint64 `json:"n_prompt_tokens"`
	NPromptTokensProcessed uint64 `json:"n_prompt_tokens_processed"`
	NPromptTokensCache     uint64 `json:"n_prompt_tokens_cache"`

	// Scope / SlotModel / Source are the disclosures.
	Scope     string `json:"fak_scope"`
	SlotModel string `json:"fak_slot_model"`
	// Source names the /metrics family each counter mirrors, so the number is
	// auditable against the scrape rather than taken on this handler's word.
	Source map[string]string `json:"fak_source"`
	// InKernelModel separates "nothing feeds the tap here" from "idle".
	InKernelModel bool `json:"fak_in_kernel_model"`
	// Turns is the live in-kernel turn count behind the totals, from
	// cacheobs.Default.Snapshot().Turns == fak_gateway_kv_prefix_turns_total.
	Turns uint64 `json:"fak_kv_prefix_turns_total"`
	// ReuseRatio is the live realized hit rate
	// (fak_gateway_kv_prefix_reuse_ratio). 0 until the first observed turn,
	// which is the same idle posture that tap takes.
	ReuseRatio float64 `json:"fak_kv_prefix_reuse_ratio"`
}

// handleProps serves GET /props: the Fak-native engine self-description, read
// from the same live state /metrics renders.
func (s *Server) handleProps(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.servingProps())
}

// servingProps builds the /props document. Split from the handler so the
// projection is testable without an HTTP round-trip.
func (s *Server) servingProps() servingPropsWire {
	out := servingPropsWire{
		DefaultGenerationSettings: servingPropsGenerationSettings{
			NCtx: inKernelContextWindow(s.planner),
		},
		BuildInfo:       s.servingPropsBuildInfo(),
		ModelAlias:      strings.TrimSpace(s.model),
		EndpointSlots:   s.servingRouteServed("/slots"),
		EndpointMetrics: s.servingRouteServed("/metrics"),
		Schema:          servingPropsSchema,
		Planner:         plannerKind(s.planner),
		Engine:          strings.TrimSpace(s.engineID),
		InKernelModel:   plannerKind(s.planner) == "inkernel",
	}

	// The admission controller is the live running-set gate and is installed
	// only for the in-kernel serve path (gateway.go). Absent it there is no
	// running set at all, so both the cap and the counters are left unreported
	// rather than zeroed.
	s.admissionMu.RLock()
	admission := s.admissionCtl
	s.admissionMu.RUnlock()
	if admission != nil {
		policy := admission.Policy()
		// MaxNumSeqs <= 0 DISABLES the seq cap (admission.go), so a
		// non-positive value means "uncapped" — genuinely no bound to report.
		if maxSeqs := policy.MaxNumSeqs; maxSeqs > 0 {
			total := maxSeqs
			out.TotalSlots = &total
			out.TotalSlotsSource = "gateway AdmissionPolicy.MaxNumSeqs (sequence-count cap, not the effective concurrency; admission_token_budget is the binding bound)"
		}
		if budget := policy.TokenBudget; budget > 0 {
			out.AdmissionTokenBudget = &budget
			out.AdmissionTokenBudgetSource = admission.TokenBudgetProvenance()
		}
		prealloc := policy.preallocCeiling()
		out.AdmissionPreallocTokens = &prealloc
		st := admission.Stats()
		running, waiting := int(st.Running), int(st.Waiting)
		out.RunningRequests = &running
		out.WaitingRequests = &waiting
	}

	// The hit rate is only rendered once a turn exists: cacheobs reports 0 for an
	// idle observer precisely so a cold process never shows a phantom ratio, and
	// an unreported field is more honest than a reported 0 for "how well is the
	// cache doing" on a process that has served nothing.
	kv := cacheobs.Default.Snapshot()
	if kv.Turns > 0 {
		hit := kv.ReuseRatio
		out.PrefixCacheHitRate = &hit
	}
	return out
}

// servingPropsBuildInfo renders the engine's build label from the same live
// fields fak_gateway_build_info publishes, so /props and /metrics cannot
// disagree about what this process is.
func (s *Server) servingPropsBuildInfo() string {
	vdso := false
	if s.k != nil {
		vdso = s.k.VDSOEnabled()
	}
	return strings.Join([]string{
		"fak/" + servingPropsOr(s.version, "unknown"),
		"engine=" + servingPropsOr(s.engineID, "unknown"),
		"planner=" + plannerKind(s.planner),
		"model=" + servingPropsOr(s.model, "unknown"),
		"vdso=" + strconv.FormatBool(vdso),
	}, " ")
}

// servingRouteServed reports whether this process's own route table serves the
// given pattern. It is what backs the endpoint_slots / endpoint_metrics claims:
// those booleans are read off the live registration, so a claim cannot survive
// the route being removed.
func (s *Server) servingRouteServed(pattern string) bool {
	for _, rt := range s.routeTable() {
		if rt.pattern == pattern {
			return true
		}
	}
	return false
}

// handleSlots serves GET /slots: the fak-native per-slot KV state, read from
// the same cacheobs tap /metrics renders.
func (s *Server) handleSlots(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.servingSlots())
}

// servingSlots builds the /slots document.
//
// Exactly ONE entry is emitted, because fak's in-kernel path has exactly one
// thing to report: a single resident radix KV prefix shared by every concurrent
// sequence. Emitting one row per MaxNumSeqs admission slot would be inventing
// per-slot KV state that does not exist.
func (s *Server) servingSlots() []servingSlotWire {
	kv := cacheobs.Default.Snapshot()
	slot := servingSlotWire{
		ID:                     0,
		IsProcessing:           kv.Turns > 0,
		NPromptTokens:          kv.PromptTokens,
		NPromptTokensProcessed: kv.PromptTokens,
		NPromptTokensCache:     kv.ReusedTokens,
		Scope:                  servingSlotsScope,
		SlotModel:              servingSlotModel,
		InKernelModel:          plannerKind(s.planner) == "inkernel",
		Turns:                  kv.Turns,
		ReuseRatio:             kv.ReuseRatio,
		Source: map[string]string{
			"n_prompt_tokens":           "fak_gateway_kv_prefix_prompt_tokens_total",
			"n_prompt_tokens_processed": "fak_gateway_kv_prefix_prompt_tokens_total",
			"n_prompt_tokens_cache":     "fak_gateway_kv_prefix_reused_tokens_total",
			"fak_kv_prefix_turns_total": "fak_gateway_kv_prefix_turns_total",
		},
	}
	if nctx := inKernelContextWindow(s.planner); nctx > 0 {
		slot.NCtx = &nctx
	}
	return []servingSlotWire{slot}
}

// servingPropsOr renders a build label component, substituting a named
// placeholder for a blank so the label set always has the same shape.
func servingPropsOr(v, fallback string) string {
	if v = strings.TrimSpace(v); v != "" {
		return v
	}
	return fallback
}

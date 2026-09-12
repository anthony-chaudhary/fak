package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// chatRouteBinding belongs to one request. RequestedModel retains the public
// identity while Target.UpstreamModel names the model sent to the bound account.
// Locality describes the selected transport, rather than a separate roster lookup.
type chatRouteBinding struct {
	RequestedModel string
	Target         modelroute.Target
	Planner        agent.Planner
	Locality       servingLocality
}

type chatRouteContextKey struct{}

// prepareChatEPFanout retains the bounded original wire body until account
// admission decides whether this request uses the boot model's follower ranks.
// A bound account owns its own execution and never releases those followers.
func (s *Server) prepareChatEPFanout(w http.ResponseWriter, r *http.Request, route string) (release func(*http.Request) bool, finish func(), ok bool) {
	noop := func(*http.Request) bool { return true }
	if (route == epRouteMessages && s.native) || r.Header.Get(epFollowerHeader) != "" || len(epFanoutURLsFromEnv(route)) == 0 {
		return noop, func() {}, true
	}
	// Preserve the original early release when there is no account roster.
	// Responses also needs its existing continuation restriction checked first.
	if s.roster == nil && route != epRouteResponses {
		wait, allowed := s.startEPFanoutFollowers(w, r, route)
		return noop, wait, allowed
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxTranscriptBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return nil, nil, false
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var wait func()
	finish = func() {
		if wait != nil {
			wait()
		}
	}
	release = func(admitted *http.Request) bool {
		if chatRouteFromContext(admitted.Context()) != nil {
			return true
		}
		if route == epRouteResponses {
			var meta struct {
				PreviousResponseID string `json:"previous_response_id"`
			}
			if json.Unmarshal(body, &meta) == nil && strings.TrimSpace(meta.PreviousResponseID) != "" {
				writeErr(w, http.StatusBadRequest, "previous_response_id is unavailable with expert-parallel request fanout")
				return false
			}
		}
		// The decoder may have consumed or transformed the request. Mirror only
		// the original bytes, on the handler's constant follower route.
		follower := admitted.Clone(admitted.Context())
		follower.Body = io.NopCloser(bytes.NewReader(body))
		var allowed bool
		wait, allowed = s.startEPFanoutFollowers(w, follower, route)
		return allowed
	}
	if s.roster == nil {
		if !release(r) {
			return nil, nil, false
		}
		release = noop
	}
	return release, finish, true
}

func chatRouteFromContext(ctx context.Context) *chatRouteBinding {
	binding, _ := ctx.Value(chatRouteContextKey{}).(*chatRouteBinding)
	return binding
}

// prepareChatRoute runs after decoding the requested model and authenticating
// the caller, before either the buffered or streaming path can contact a model.
func (s *Server) prepareChatRoute(w http.ResponseWriter, r *http.Request, model string) (*http.Request, bool) {
	if s.roster == nil {
		return r, true
	}
	start := time.Now()
	binding, err := s.bindChatRoute(r.Context(), model)
	if err != nil {
		s.metrics.observeOperation("chat_route", WireVerdict{Kind: "DENY", Reason: "ROUTE_UNAVAILABLE"}, nil, time.Since(start))
		s.logf("gateway: chat route refused: %v", err)
		writeErr(w, http.StatusForbidden, "requested model account is unavailable to this caller")
		return r, false
	}
	decision := "PASSTHROUGH"
	if binding != nil {
		decision = "ROSTER_BOUND"
		r = r.WithContext(context.WithValue(r.Context(), chatRouteContextKey{}, binding))
	}
	s.metrics.observeOperation("chat_route", WireVerdict{Kind: "ALLOW", Reason: decision}, nil, time.Since(start))
	return r, true
}

func (s *Server) chatPlanner(ctx context.Context) agent.Planner {
	if binding := chatRouteFromContext(ctx); binding != nil {
		return binding.Planner
	}
	return s.planner
}

// observeNativeChatRoute attributes one completed bound native request using
// its aggregate loop usage. Passthrough retains the historical native counters.
func (s *Server) observeNativeChatRoute(ctx context.Context, traceID string, stream bool, usage agent.Usage, finishReason string, dur time.Duration) {
	if binding := chatRouteFromContext(ctx); binding != nil {
		s.metrics.observeInferenceUsageServed(binding.Locality, usage, finishReason, dur)
		s.logInferenceTurnForModel(traceID, "anthropic_messages_native", binding.Target.UpstreamModel, stream, usage, finishReason, dur, false, s.consumeDecodedCtxViewEvent(traceID))
		return
	}
	s.logInferenceTurn(traceID, "anthropic_messages_native", false, usage, finishReason, dur, false)
}

// chatRouteOpts is applied last so client options cannot override the account's
// wire model or credential. Raw bodies belong to the original provider and model;
// a bound turn is rebuilt through its target's native transcript adapter.
func chatRouteOpts(ctx context.Context, opts []agent.SampleOpt) []agent.SampleOpt {
	binding := chatRouteFromContext(ctx)
	if binding == nil {
		return opts
	}
	out := append([]agent.SampleOpt(nil), opts...)
	return append(out, func(params *agent.SampleParams) {
		params.Model = binding.Target.UpstreamModel
		params.UpstreamAPIKey = ""
		params.RawRequestBody = nil
		params.UpstreamBeta = ""
	})
}

// chatServingLocality uses the selected request binding. A passthrough turn is
// attributed only to its boot transport, never to the roster's default account.
func (s *Server) chatServingLocality(ctx context.Context, model string) servingLocality {
	if binding := chatRouteFromContext(ctx); binding != nil {
		return binding.Locality
	}
	if dp, ok := s.planner.(*DualPlanner); ok && dp.RoutesLocal(model) {
		return localitySelfHosted
	}
	if s.upstream != nil {
		return *s.upstream
	}
	return s.servedSide
}

// bindChatRoute returns nil, nil when the request must use the existing planner
// and sampling options unchanged. Only explicit bindings participate: the tool
// resolver's default-account behavior must not reroute an unrostered chat model.
// An explicit binding that fails admission or configuration never falls through.
func (s *Server) bindChatRoute(ctx context.Context, requestedModel string) (*chatRouteBinding, error) {
	if s == nil || s.roster == nil || requestedModel == "" {
		return nil, nil
	}
	bound := false
	for _, binding := range s.roster.Bindings {
		if binding.Model == requestedModel {
			bound = true
			break
		}
	}
	if !bound {
		return nil, nil
	}
	// Use the same principal admission as tool calls before reading credentials
	// or constructing a transport. EngineRoute itself is not a transport address.
	if _, err := s.resolveRoute(requestedModel, principalFromContext(ctx)); err != nil {
		return nil, err
	}
	target, err := s.roster.Resolve(requestedModel)
	if err != nil {
		return nil, fmt.Errorf("gateway: chat route: %w", err)
	}
	provider := string(target.Kind)
	switch target.Kind {
	case modelroute.KindLocal, modelroute.KindFleet, modelroute.KindDeepSeek, modelroute.KindOpenRouter:
		provider = string(agent.ProviderOpenAI)
	case modelroute.KindOpenAI, modelroute.KindOpenAIResponses, modelroute.KindAnthropic, modelroute.KindGemini, modelroute.KindXAI:
	default:
		return nil, fmt.Errorf("gateway: chat route: account %q has unsupported provider kind %q", target.Account, target.Kind)
	}
	if strings.TrimSpace(target.BaseURL) == "" {
		return nil, fmt.Errorf("gateway: chat route: account %q has no upstream base URL", target.Account)
	}
	authScheme, ok := agent.ParseAnthropicAuthScheme(target.AuthScheme)
	if !ok {
		return nil, fmt.Errorf("gateway: chat route: account %q has an invalid auth scheme", target.Account)
	}
	credential := ""
	if target.CredEnv != "" {
		credential = os.Getenv(target.CredEnv)
		if strings.TrimSpace(credential) == "" {
			return nil, fmt.Errorf("gateway: chat route: account %q credential environment variable %q is unset or empty", target.Account, target.CredEnv)
		}
	}
	// Build independently of the boot planner: its dynamic credentials, account
	// failover callbacks and extra authentication headers belong to another account.
	planner, err := agent.NewProviderHTTPPlanner(provider, target.BaseURL, target.UpstreamModel, credential)
	if err != nil {
		return nil, fmt.Errorf("gateway: chat route: account %q: %w", target.Account, err)
	}
	planner.AnthropicAuthScheme = authScheme
	locality := localityVendor
	if target.Zone().SelfHosted() {
		locality = localitySelfHosted
	}
	return &chatRouteBinding{
		RequestedModel: requestedModel,
		Target:         target,
		Planner:        planner,
		Locality:       locality,
	}, nil
}

package vcacheobserve

import "github.com/anthony-chaudhary/fak/internal/vcachegov"

// ProviderActionSchema is the stable contract for the provider-cache action plan
// exposed by `fak vcache actions` and GET /v1/fak/vcache/actions.
const ProviderActionSchema = "fak.vcache.provider-actions.v1"

// ProviderActionState names whether a row is directly actionable or still gated
// by missing transport evidence. A plan row is never proof that a provider warm
// already happened; provider cache hits remain provider telemetry.
type ProviderActionState string

const (
	ActionNoop  ProviderActionState = "noop"
	ActionReady ProviderActionState = "ready"
	ActionGated ProviderActionState = "gated"
)

// ProviderActionPlan is the per-family bridge from the observed Governor verdict
// to the action a live provider-cache loop would take.
type ProviderActionPlan struct {
	Schema         string                  `json:"schema"`
	Turns          int                     `json:"turns"`
	FamilyCount    int                     `json:"family_count"`
	WindowCapped   bool                    `json:"window_capped,omitempty"`
	Actions        []ProviderAction        `json:"actions"`
	Counts         ProviderActionCounts    `json:"counts"`
	Transport      ProviderActionTransport `json:"transport"`
	CorrectnessLaw string                  `json:"correctness_law"`
}

// ProviderActionTransport explains the execution boundary for the action plan.
// The current planner is a live decision/API surface; it only marks spendful
// heartbeat/explicit-cache actions ready after a caller supplies the missing
// transport witness.
type ProviderActionTransport struct {
	Mode    string                    `json:"mode"`
	Ready   bool                      `json:"ready"`
	Witness *ProviderTransportWitness `json:"witness,omitempty"`
	Reason  string                    `json:"reason"`
}

// ProviderActionCounts summarizes rows by actionability.
type ProviderActionCounts struct {
	Noop  int `json:"noop"`
	Ready int `json:"ready"`
	Gated int `json:"gated"`
}

// ProviderAction is one family-level Governor decision rendered as an operational
// action candidate.
type ProviderAction struct {
	Family              string                     `json:"family"`
	Decision            vcachegov.GovernorDecision `json:"decision"`
	Action              string                     `json:"action"`
	State               ProviderActionState        `json:"state"`
	Requires            []string                   `json:"requires,omitempty"`
	Witnessed           []string                   `json:"witnessed,omitempty"`
	Reason              string                     `json:"reason"`
	Turns               int                        `json:"turns"`
	ArrivalRatePerSec   float64                    `json:"arrival_rate_per_sec"`
	CacheReadTokens     int64                      `json:"cache_read_tokens"`
	CacheCreationTokens int64                      `json:"cache_creation_tokens"`
	SavedTokenEquiv     float64                    `json:"saved_token_equiv"`
}

// ProviderActionOptions carries external evidence that can upgrade a spendful
// provider-cache action from gated to ready. The planner remains side-effect free:
// a ready row means the required witness is present, not that fak spent a provider call.
type ProviderActionOptions struct {
	Transport ProviderTransportWitness `json:"transport,omitempty"`
}

// ProviderTransportWitness is the evidence boundary for provider-cache transport.
// ByteIdenticalPrefix proves the bytes to be warmed match the prefix observed in
// telemetry. HeartbeatTransport proves the host can issue a harmless refresh call.
// ExplicitCacheTransport and DeletionCapable prove a provider surface exists for
// regulated explicit-cache entries with deletion.
type ProviderTransportWitness struct {
	HeartbeatTransport     bool   `json:"heartbeat_transport,omitempty"`
	ExplicitCacheTransport bool   `json:"explicit_cache_transport,omitempty"`
	ByteIdenticalPrefix    bool   `json:"byte_identical_prefix,omitempty"`
	DeletionCapable        bool   `json:"deletion_capable,omitempty"`
	Source                 string `json:"source,omitempty"`
}

func (w ProviderTransportWitness) any() bool {
	return w.HeartbeatTransport || w.ExplicitCacheTransport || w.ByteIdenticalPrefix || w.DeletionCapable || w.Source != ""
}

func (w ProviderTransportWitness) heartbeatReady() bool {
	return w.HeartbeatTransport && w.ByteIdenticalPrefix
}

func (w ProviderTransportWitness) explicitReady() bool {
	return w.ExplicitCacheTransport && w.ByteIdenticalPrefix && w.DeletionCapable
}

// PlanProviderActions folds the same observed turn window as Observe into
// concrete provider-cache action candidates. It is pure and side-effect free:
// the output is a witnessable action plan, not an executor.
func PlanProviderActions(turns []Turn, windowCapped bool) ProviderActionPlan {
	return PlanProviderActionsWithOptions(turns, windowCapped, ProviderActionOptions{})
}

// PlanProviderActionsWithOptions is PlanProviderActions plus an optional external
// transport witness. This is the bridge a live host uses to prove the missing
// heartbeat/explicit-cache prerequisites before treating a spendful row as ready.
func PlanProviderActionsWithOptions(turns []Turn, windowCapped bool, opt ProviderActionOptions) ProviderActionPlan {
	turns = providerActionTurns(turns)
	rep := Observe(turns, DefaultMultipliers())
	plan := ProviderActionPlan{
		Schema:       ProviderActionSchema,
		Turns:        len(turns),
		FamilyCount:  rep.FamilyCount,
		WindowCapped: windowCapped,
		Actions:      make([]ProviderAction, 0, len(rep.Families)),
		Transport: ProviderActionTransport{
			Mode:   "decision_only",
			Ready:  false,
			Reason: "provider action planner is live; spendful heartbeat/explicit-cache transport still requires byte-identical prefix and provider capability evidence",
		},
		CorrectnessLaw: "full uncached prompt remains the correctness path; provider cache hits are rebates, never required",
	}
	applyProviderTransportWitness(&plan, opt.Transport)
	for _, fam := range rep.Families {
		action := providerActionFromFamily(fam, opt.Transport)
		plan.Actions = append(plan.Actions, action)
		switch action.State {
		case ActionReady:
			plan.Counts.Ready++
		case ActionGated:
			plan.Counts.Gated++
		default:
			plan.Counts.Noop++
		}
	}
	if len(plan.Actions) == 0 {
		plan.Transport.Reason = "no provider-cache families observed in the rolling window"
	}
	return plan
}

func applyProviderTransportWitness(plan *ProviderActionPlan, w ProviderTransportWitness) {
	if !w.any() {
		return
	}
	plan.Transport.Mode = "witnessed_transport"
	witness := w
	plan.Transport.Witness = &witness
	if w.heartbeatReady() || w.explicitReady() {
		plan.Transport.Ready = true
		plan.Transport.Reason = "provider transport witness supplied; spendful rows become ready only when their required capability and byte-identical prefix evidence is present"
		return
	}
	plan.Transport.Reason = "partial provider transport witness supplied, but no spendful action class has all required evidence"
}

func providerActionTurns(turns []Turn) []Turn {
	return ProviderTelemetryTurns(turns)
}

// ProviderTelemetryTurns keeps turns with real provider-cache token counters and
// drops context-only witness rows before provider-cache economics are folded.
func ProviderTelemetryTurns(turns []Turn) []Turn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]Turn, 0, len(turns))
	for i := range turns {
		if TurnHasProviderTelemetry(turns[i]) {
			out = append(out, turns[i])
		}
	}
	return out
}

// TurnHasProviderTelemetry reports whether a turn has provider-cache counters,
// as distinct from context-plane-only witness counters.
func TurnHasProviderTelemetry(turn Turn) bool {
	switch {
	case turn.InputTokens > 0:
		return true
	case turn.CacheRead > 0:
		return true
	case turn.CacheCreation > 0:
		return true
	case turn.Ephemeral1h > 0:
		return true
	case turn.Ephemeral5m > 0:
		return true
	default:
		return false
	}
}

func providerActionFromFamily(fam Family, w ProviderTransportWitness) ProviderAction {
	row := ProviderAction{
		Family:              fam.Key,
		Decision:            fam.GovernorDecision,
		Turns:               fam.Turns,
		ArrivalRatePerSec:   fam.ArrivalRatePerSec,
		CacheReadTokens:     fam.CacheReadTokens,
		CacheCreationTokens: fam.CacheCreationTokens,
		SavedTokenEquiv:     fam.Economics.SavedTokenEquiv,
	}
	switch fam.GovernorDecision {
	case vcachegov.DecisionRideNatural:
		row.Action = "ride_natural"
		row.State = ActionNoop
		row.Reason = "natural traffic is already refreshing the provider prefix inside the TTL; spend no dedicated warm"
	case vcachegov.DecisionHeartbeatPin:
		row.Action = "heartbeat_pin"
		row.Requires = []string{"heartbeat_transport", "byte_identical_prefix"}
		row.Witnessed = witnessedProviderTransport(row.Requires, w)
		if w.heartbeatReady() {
			row.State = ActionReady
			row.Reason = "heartbeat transport and byte-identical prefix witness supplied; ready to refresh without changing correctness path"
		} else {
			row.State = ActionGated
			row.Reason = "pin candidate needs active provider warm transport plus byte-identical prefix fingerprint before spending"
		}
	case vcachegov.DecisionLazyRebuild:
		row.Action = "lazy_rebuild"
		row.State = ActionNoop
		row.Reason = "let the prefix lapse and rebuild on the next real miss; no proactive provider call"
	case vcachegov.DecisionEvict:
		row.Action = "evict_manifest"
		row.State = ActionReady
		row.Reason = "drop the cold prefix from the local warm manifest; no provider call required"
	case vcachegov.DecisionNoCache:
		row.Action = "no_cache"
		row.State = ActionReady
		row.Reason = "route the prefix uncached because the content is not warmable"
	case vcachegov.DecisionExplicitCache:
		row.Action = "explicit_cache"
		row.Requires = []string{"explicit_cache_transport", "byte_identical_prefix", "deletion_capable"}
		row.Witnessed = witnessedProviderTransport(row.Requires, w)
		if w.explicitReady() {
			row.State = ActionReady
			row.Reason = "explicit-cache transport, deletion capability, and byte-identical prefix witness supplied"
		} else {
			row.State = ActionGated
			row.Reason = "regulated prefix needs a deletion-capable explicit-cache provider surface"
		}
	default:
		row.Action = "unknown"
		row.State = ActionGated
		row.Reason = "unknown governor decision"
	}
	return row
}

func witnessedProviderTransport(required []string, w ProviderTransportWitness) []string {
	if len(required) == 0 {
		return nil
	}
	out := make([]string, 0, len(required))
	for _, req := range required {
		switch req {
		case "heartbeat_transport":
			if w.HeartbeatTransport {
				out = append(out, req)
			}
		case "explicit_cache_transport":
			if w.ExplicitCacheTransport {
				out = append(out, req)
			}
		case "byte_identical_prefix":
			if w.ByteIdenticalPrefix {
				out = append(out, req)
			}
		case "deletion_capable":
			if w.DeletionCapable {
				out = append(out, req)
			}
		}
	}
	return out
}

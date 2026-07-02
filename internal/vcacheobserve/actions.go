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
	Mode   string `json:"mode"`
	Ready  bool   `json:"ready"`
	Reason string `json:"reason"`
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
	Reason              string                     `json:"reason"`
	Turns               int                        `json:"turns"`
	ArrivalRatePerSec   float64                    `json:"arrival_rate_per_sec"`
	CacheReadTokens     int64                      `json:"cache_read_tokens"`
	CacheCreationTokens int64                      `json:"cache_creation_tokens"`
	SavedTokenEquiv     float64                    `json:"saved_token_equiv"`
}

// PlanProviderActions folds the same observed turn window as Observe into
// concrete provider-cache action candidates. It is pure and side-effect free:
// the output is a witnessable action plan, not an executor.
func PlanProviderActions(turns []Turn, windowCapped bool) ProviderActionPlan {
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
	for _, fam := range rep.Families {
		action := providerActionFromFamily(fam)
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

func providerActionTurns(turns []Turn) []Turn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]Turn, 0, len(turns))
	for _, turn := range turns {
		if turn.InputTokens > 0 ||
			turn.CacheRead > 0 ||
			turn.CacheCreation > 0 ||
			turn.Ephemeral1h > 0 ||
			turn.Ephemeral5m > 0 {
			out = append(out, turn)
		}
	}
	return out
}

func providerActionFromFamily(fam Family) ProviderAction {
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
		row.State = ActionGated
		row.Reason = "pin candidate needs active provider warm transport plus byte-identical prefix fingerprint before spending"
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
		row.State = ActionGated
		row.Reason = "regulated prefix needs a deletion-capable explicit-cache provider surface"
	default:
		row.Action = "unknown"
		row.State = ActionGated
		row.Reason = "unknown governor decision"
	}
	return row
}

package fleetaccounts

import (
	"bytes"
	"encoding/json"
)

// MarshalJSON emits an Account as an ordered JSON object. The legacy Python keys retain
// their order; Claude worker rows additionally carry the shared login readiness verdict
// before the runtime-status block. The base keys are always present; worker rows carry
// profile + route_weight, and Claude worker rows carry identity + reconciliation. A field
// that the Python row omits structurally (a non-worker row's profile, an opencode
// worker's Claude-only identity) is omitted here too.
func (a Account) MarshalJSON() ([]byte, error) {
	o := newOrdered()
	o.set("dir", a.Dir)
	o.set("product", a.Product)
	o.set("account", a.Account)
	o.set("tag", a.Tag)
	o.set("kind", string(a.Kind))
	o.set("reason", a.Reason)
	o.set("notes", a.Notes)

	// Credential KIND (#5331), emitted ONLY for an api-key seat: the historical
	// subscription-OAuth row leaves both fields empty and so keeps the exact legacy key set
	// the cross-surface parity gate compares. api_key_env is the env var's NAME — the
	// reference the registry stores — never the key.
	if a.CredKind != "" {
		o.set("cred_kind", string(a.CredKind))
		o.set("api_key_env", a.APIKeyEnv)
	}

	// worker profile block (present iff this row was classified as a worker)
	if a.ModelTier != nil {
		o.set("model_tier", *a.ModelTier)
		o.set("model", derefStr(a.Model))
		o.set("small_model", derefStr(a.SmallModel))
		o.set("model_effort", derefStr(a.ModelEffort))
		o.set("agent", derefStr(a.Agent))
		o.set("profile_source", derefStr(a.ProfileSource))
		o.set("route_weight", derefInt(a.RouteWeight))
		if a.RoutingCostPerMTok != nil {
			o.set("routing_cost_per_mtok", derefFloat64(a.RoutingCostPerMTok))
		}
		if a.BilledCostPerMTok != nil {
			o.set("billed_cost_per_mtok", derefFloat64(a.BilledCostPerMTok))
		}
	}

	// Claude worker identity + reconciliation (present iff stamped at classify time)
	if a.AccountUUID != nil {
		o.set("account_uuid", *a.AccountUUID)
		o.set("login_email", derefStr(a.LoginEmail))
		o.set("org_uuid", derefStr(a.OrgUUID))
		o.set("org_type", derefStr(a.OrgType))
		o.set("plan", derefStr(a.Plan))
		// reconcile order matches the Python dict insertion order exactly.
		if a.TagLoginMatch != nil {
			o.set("tag_login_match", *a.TagLoginMatch)
		}
		if a.IdentityPeers != nil {
			o.set("identity_peers", a.IdentityPeers)
		}
		if a.IdentityRole != nil {
			o.set("identity_role", *a.IdentityRole)
		}
		if a.LoginStatus != nil {
			o.set("login_status", *a.LoginStatus)
			o.set("can_serve", derefBool(a.CanServe))
		}
	} else if a.LoginStatus != nil {
		// OpenCode has no Claude identity block, but dispatch consumes the same
		// credential-readiness contract before it will launch a guarded worker.
		o.set("login_status", *a.LoginStatus)
		o.set("can_serve", derefBool(a.CanServe))
	}

	// runtime-status block (attached by Annotate; present on every annotated row)
	if a.Available != nil {
		o.set("available", *a.Available)
		o.set("blocked", derefBool(a.Blocked))
		o.setNullable("block_kind", a.BlockKind)
		o.set("block_reason", derefStr(a.BlockReason))
		o.setNullable("reset", a.Reset)
		o.setNullable("weekly", a.Weekly)
		o.set("throttled", derefBool(a.Throttled))
		o.set("active_sessions", derefInt(a.ActiveSessions))
		o.set("live_sessions", derefInt(a.LiveSessions))
		o.set("auth_blocked_sessions", derefInt(a.AuthBlockedSessions))
		o.set("status_source", derefStr(a.StatusSource))
		o.setNullableFloat("registry_age_min", a.RegistryAgeMin)
		// Advisory near-cap reset (see Account.UsageSoonReset). Emitted only when a fresh
		// probe reopened the seat over a still-active daily cap, so a normal serving row
		// keeps its exact prior key set — matching the Python row, which only inserts
		// usage_soon_reset in the same reopen path.
		if a.UsageSoonReset != nil {
			o.set("usage_soon_reset", *a.UsageSoonReset)
		}
	}

	return o.marshal()
}

// orderedObj is a minimal insertion-ordered JSON object builder. The module has zero
// external deps, so this is hand-rolled — it preserves key order (Go maps do not) so the
// emitted bytes match Python's json.dumps over an insertion-ordered dict.
type orderedObj struct {
	keys []string
	vals []any
}

func newOrdered() *orderedObj { return &orderedObj{} }

func (o *orderedObj) set(key string, val any) {
	o.keys = append(o.keys, key)
	o.vals = append(o.vals, val)
}

// setNullablePtr emits a JSON null when the pointer is nil, else the dereferenced
// value. Shared by the typed setNullable* wrappers below so the nil-check body isn't
// copy-pasted per pointer type.
func setNullablePtr[T any](o *orderedObj, key string, p *T) {
	if p == nil {
		o.set(key, nil)
	} else {
		o.set(key, *p)
	}
}

// setNullable emits a JSON null when the pointer is nil, else the string value.
func (o *orderedObj) setNullable(key string, p *string) {
	setNullablePtr(o, key, p)
}

func (o *orderedObj) setNullableFloat(key string, p *float64) {
	setNullablePtr(o, key, p)
}

func (o *orderedObj) marshal() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, key := range o.keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		b.Write(kb)
		b.WriteByte(':')
		vb, err := json.Marshal(o.vals[i])
		if err != nil {
			return nil, err
		}
		b.Write(vb)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

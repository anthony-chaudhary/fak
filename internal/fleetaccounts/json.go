package fleetaccounts

import (
	"bytes"
	"encoding/json"
)

// MarshalJSON emits an Account as an ordered JSON object whose key order and presence
// rules match fleet_accounts.py byte-for-byte. The base keys are always present; worker
// rows additionally carry the profile + route_weight, and Claude worker rows carry the
// identity + reconciliation verdict; the runtime-status block is appended last (present on
// every annotated row). A field that the Python row omits structurally (a non-worker row's
// profile, an opencode worker's Claude-only identity) is omitted here too.
func (a Account) MarshalJSON() ([]byte, error) {
	o := newOrdered()
	o.set("dir", a.Dir)
	o.set("product", a.Product)
	o.set("account", a.Account)
	o.set("tag", a.Tag)
	o.set("kind", string(a.Kind))
	o.set("reason", a.Reason)
	o.set("notes", a.Notes)

	// worker profile block (present iff this row was classified as a worker)
	if a.ModelTier != nil {
		o.set("model_tier", *a.ModelTier)
		o.set("model", derefStr(a.Model))
		o.set("small_model", derefStr(a.SmallModel))
		o.set("model_effort", derefStr(a.ModelEffort))
		o.set("agent", derefStr(a.Agent))
		o.set("profile_source", derefStr(a.ProfileSource))
		o.set("route_weight", derefInt(a.RouteWeight))
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

// setNullable emits a JSON null when the pointer is nil, else the string value.
func (o *orderedObj) setNullable(key string, p *string) {
	if p == nil {
		o.set(key, nil)
	} else {
		o.set(key, *p)
	}
}

func (o *orderedObj) setNullableFloat(key string, p *float64) {
	if p == nil {
		o.set(key, nil)
	} else {
		o.set(key, *p)
	}
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

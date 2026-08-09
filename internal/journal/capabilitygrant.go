package journal

// Capability-grant telemetry — the durable, first-class provenance witness for a
// GATED-WIDEN policy amendment (#5178, epic #5170 Track D).
//
// A CONFIG_SWAP row (configswap.go) already records that the capability floor was
// re-installed, over which bytes, and whether the swap took. What it does NOT
// record is the thing an auditor actually asks after the fact: WHICH KNOB was
// loosened, from what to what, through which gated channel, and on whose
// authority. A widening allow-overlay, a reload-widen, a `Complain` trial and an
// `AdvisoryReasons` softening all collapse into the same "floor swapped, digest
// X" row, so "what was loosened this session and by whom" had to be
// reconstructed by diffing manifests that may no longer exist on disk.
//
// A CAPABILITY_GRANT row closes that hole the same way CONFIG_SWAP closed its
// own: one first-class chained row per widened knob, written DIRECTLY through
// the chain (AppendCapabilityGrant → appendLocked), NOT through Emit(abi.Event)
// — a grant is a supervision event, not a kernel decision, and routing a
// synthetic event through the frozen ABI would fan it out to every
// decision-stream folder that assumes an event IS an adjudication. The chained
// forensic identity — Kind, the widened knob (Tool), the gated channel that
// carried it (Reason), and the actor on whose authority it landed (By) — rides
// the frozen decision fields, so it verifies end-to-end with every existing row;
// the full correlated record (schema fak.capability.grant.v1, carrying the
// old→new values, the amendment class and the source path) rides the non-chained
// Grant payload field, the same layering ConfigSwap and RestartHop use. The
// timestamp the issue asks for is the row's own TSUnixNano — appendLocked stamps
// it, so a grant needs no clock of its own.
//
// This rung RECORDS; it does not revert. TTL/expiry of a grant is the sibling
// child (Track D2) and is deliberately out of scope here.

// KindCapabilityGrant marks a GATED-WIDEN capability-grant row. It is a genuine
// chained row (it consumes the next Seq and chains onto the prior head) that
// carries no verdict, so decision-folding consumers skip it; readers that key on
// Kind (an auditor reconstructing "what was loosened, by whom") fold it into the
// session's grant history.
const KindCapabilityGrant = "CAPABILITY_GRANT"

// CapabilityGrantSchema names the correlated per-grant record carried on a
// CAPABILITY_GRANT row's Grant field. Versioned like every fak wire schema:
// additive-only; never edit a shipped /vN in place.
const CapabilityGrantSchema = "fak.capability.grant.v1"

// GrantDirectionWiden is the only direction a capability GRANT can have: the
// record exists to witness a LOOSENING. A tighten is a ratchet, not a grant, and
// is not recorded here (it needs no provenance trail to justify it).
const GrantDirectionWiden = "widen"

// Grant channels — the CLOSED vocabulary for WHICH gated mechanism carried the
// widening, mirrored onto the chained Reason field so the row is self-describing.
//
// These values MIRROR internal/policy's Channel* constants (the amendment-class
// registry's own vocabulary) rather than importing them: this is a shipped WIRE
// schema, and an audit ledger that resolves its on-disk vocabulary through
// another package's Go constants would let a policy-side rename silently rewrite
// the meaning of rows already on disk. The mirror is pinned from the caller side,
// where both packages are in scope (see the cmd/fak capability-grant test).
const (
	// GrantChannelOperatorOverlay is the operator's own out-of-band overlay:
	// .fak/guard/{allow,deny}.json or an explicit --policy manifest, applied at
	// launch. This is the channel a widening allow-overlay travels.
	GrantChannelOperatorOverlay = "operator-overlay"
	// GrantChannelLiveReload is the gated live policy reload: the reload-widen
	// path (POST /v1/fak/policy/reload, the allow watcher) on a running session.
	GrantChannelLiveReload = "live-reload"
	// GrantChannelOperatorEscalation is an explicit operator escalation grant.
	GrantChannelOperatorEscalation = "operator-escalation"
	// GrantChannelCentral is the org policy plane: a signed out-of-band manifest
	// whose authority sits above the operator overlay and below the compiled-in
	// floor.
	GrantChannelCentral = "central"
)

// CapabilityGrantRow is the correlated per-grant record (CapabilityGrantSchema):
// every provenance axis of ONE widened knob tied together in a single value,
// instead of a config digest an auditor has to diff manifests to interpret.
//
// Old/New are RENDERED values, not typed ones: a knob is a bool, a string, a map
// key or a list element depending on which one moved, and an audit row that has
// to stay JSON-stable across every one of them is better served by the caller's
// own rendering ("" → "read_docs", "fail_closed" → "admit_and_log") than by a
// union type that would have to grow with the registry.
type CapabilityGrantRow struct {
	Schema    string `json:"schema"`           // CapabilityGrantSchema
	Knob      string `json:"knob"`             // the adjudicator.Policy field that moved (Allow, AllowPrefix, Posture, Complain, AdvisoryReasons, ...)
	Direction string `json:"direction"`        // GrantDirectionWiden
	Class     string `json:"class,omitempty"`  // the amendment class from the registry ("GATED_WIDEN")
	Old       string `json:"old,omitempty"`    // rendered pre-amendment value ("" when the knob was unset)
	New       string `json:"new,omitempty"`    // rendered post-amendment value
	Channel   string `json:"channel"`          // GrantChannel* — which gated mechanism carried it
	Actor     string `json:"actor,omitempty"`  // operator id / reload caller on whose authority it landed
	Source    string `json:"source,omitempty"` // the overlay/manifest path the grant came from
	Reason    string `json:"reason,omitempty"` // why it was granted (operator note, confirm-env, gate detail)
}

// grantActorFallback is the By value for a grant whose actor is unknown — a
// launch-time overlay apply has no named operator, and an empty By would make
// the row look like it came from nowhere.
const grantActorFallback = "capability-granter"

// AppendCapabilityGrant records ONE widened knob as a durable, chained
// CAPABILITY_GRANT row and returns the committed row (with its stamped
// Seq/hash). The caller supplies the provenance it knows; Schema and Direction
// are stamped here so no caller can write a grant that claims to be anything
// other than a widening of the current schema version.
//
// A widening amendment that moves SEVERAL knobs calls this once per knob: the
// unit of audit is the grant, not the swap, so "which capability did this session
// hand out" answers by counting rows rather than by re-parsing a delta string.
//
// It is a no-op returning the zero Row on a nil receiver, so a caller that
// guarded the journal on may call it unconditionally — an unjournaled run stays
// byte-identical.
//
// Like AppendConfigSwap, the row is written directly through the chain (not the
// ABI fan-out): a grant is supervision, not a kernel decision. The write is
// flushed per row by appendLocked, so the grant survives whatever the widened
// session does next.
func (j *Journal) AppendCapabilityGrant(g CapabilityGrantRow) Row {
	if j == nil {
		return Row{}
	}
	g.Schema = CapabilityGrantSchema
	g.Direction = GrantDirectionWiden
	by := g.Actor
	if by == "" {
		by = grantActorFallback
	}
	row := Row{
		Kind:   KindCapabilityGrant,
		Tool:   g.Knob,
		Reason: g.Channel,
		By:     by,
		Grant:  &g,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}

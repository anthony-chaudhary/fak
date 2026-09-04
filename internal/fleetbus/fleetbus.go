// Package fleetbus is the fleet control bus (#5600, epic #5599): the
// transport-neutral data contract that lets ONE control point address N live fak
// instances, carry a payload to them, and — the load-bearing half — learn which of
// them actually applied it.
//
// WHAT THIS IS NOT. fak already ships three halves of a fleet control plane, and
// this package is none of them:
//
//   - internal/sessionctl declares the closed control-op VOCABULARY (op, capability,
//     boundary, witness, refusal) for ONE session. It says nothing about who to send
//     an op to.
//   - internal/sessionctl/broadcast.go fans one lifecycle op across a lane/wave/label
//     selector, but explicitly over ONE process's session.Table, with no return path.
//   - internal/fleetspine discovers LAN peers, but is display-only and one-way by
//     construction: "carries ONLY display metadata — never a token or a request
//     payload".
//
// This is the join those three lack: cross-process addressing of a command, with a
// payload, and a witnessed return path.
//
// THREE ENVELOPES, ONE INVARIANT.
//
//	Directive — what the control point wants done, to whom, until when.
//	Instance  — a self-announced process that CAN apply a directive.
//	Ack       — one instance's outcome for one directive.
//
// The invariant: A DIRECTIVE IS NEVER ITS OWN WITNESS. Publishing proves nothing;
// the ack from the instance that consumed it at its own boundary is the proof. This
// is sessionctl's witness-of-applied contract lifted from one loop to N processes,
// and it is why Ack exists in the first commit rather than a later one. fak already
// learned this at the gateway steer route, which refuses an undeliverable steer with
// STEER_NO_OWNED_LOOP rather than returning a false 202, because an enqueued steer
// there "would sit in a mailbox nothing drains — an accepted-but-never-applied
// phantom". A fleet fan-out without acks is that phantom times N.
//
// Instance is what makes the ack report falsifiable: without a declared roster there
// is no DENOMINATOR, so "outstanding" is uncomputable and "everyone acked" cannot be
// disproved. Fold's Outstanding count is the whole reason presence is a first-class
// record and not a debug aid.
//
// THE BUS DECLARES NO OPS. Op is carried as an opaque name. The bus is a CARRIER;
// the applier owns op meaning. An instance that does not know an op acks refused
// FLEETBUS_UNKNOWN_OP — never silently drops it. That keeps this package at architest
// tier 1 (stdlib + flock, the same posture as fleetspine), keeps sessionctl's closed
// vocabulary spelled exactly once (in cmd/fak's applier, which consults
// sessionctl.Spec), and keeps the bus reusable for control ops that are not session
// ops.
//
// MANY CONTROL POINTS ARE FIRST CLASS. Every envelope stamps its issuer, ordering is
// per-directive rather than global, and the apply claim is per (instance, directive)
// — so two operators issuing concurrently can never double-apply or clobber each
// other. Issuer is attribution, not authority: the capability floor on publish is a
// declared follow-on of epic #5599, not a fence this package pretends to hold.
//
// Contract:
//   - Invariant: fleetbus operations are fail-closed and bounded across all instances and directives.
//   - Guard: publishing a directive is never its own witness; application requires a signed Ack from the consuming instance.
//   - Precondition: directives, instances, and acks must validate schema tokens, monotonic sequences, and RFC3339 timestamps.
//   - Postcondition: fold aggregates partition targeted instances into applied, refused, expired, or outstanding without dropping unacknowledged targets.
package fleetbus

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Schema tags. Every record on the wire carries one, so a reader that meets a
// future v2 record refuses it by name instead of silently mis-parsing it.
const (
	DirectiveSchema = "fak.fleetbus.directive/v1"
	InstanceSchema  = "fak.fleetbus.instance/v1"
	AckSchema       = "fak.fleetbus.ack/v1"
)

// DefaultInstanceTTL is how long a presence record stays credible without a fresh
// announce. An instance that has not re-announced within this window is stale and
// drops out of the roster — so a crashed drainer stops inflating the denominator
// forever, and "outstanding" converges instead of hanging on a ghost.
const DefaultInstanceTTL = 90 * time.Second

// Op is the NAME of a control op, carried opaquely. The bus never enumerates ops
// (see the package doc): meaning belongs to the applier.
type Op string

// --- closed refusal vocabulary -------------------------------------------- //

// RefuseReason is the bus's closed refusal vocabulary. Callers switch on the token;
// Detail is for humans and must never be parsed.
type RefuseReason string

const (
	// Malformed refuses a record whose SHAPE cannot be carried: a wrong schema tag,
	// an unusable id, a missing issuer/op, an unparsable timestamp, or a selector
	// that names nobody. It is refused at the edge, before anything is published.
	Malformed RefuseReason = "FLEETBUS_MALFORMED"

	// Expired marks a directive a drainer met after its TTL ran out. It is ACKED,
	// never applied: a control point must be able to tell "nobody was listening in
	// time" apart from "nobody ever answered", and a stale `go` applied an hour late
	// is a worse outcome than one that visibly lapsed.
	Expired RefuseReason = "FLEETBUS_EXPIRED"

	// NoTarget refuses a publish whose selector resolves to zero live instances. The
	// alternative — accepting it into an empty fleet — is the phantom this package
	// exists to prevent, in its purest form.
	NoTarget RefuseReason = "FLEETBUS_NO_TARGET"

	// UnknownOp is the ack reason an instance returns for an op it cannot interpret.
	// It is a refusal, not a drop: an op nobody understands must be VISIBLE at the
	// control point, not silently absorbed into "outstanding".
	UnknownOp RefuseReason = "FLEETBUS_UNKNOWN_OP"

	// ApplyRefused is the ack reason when the instance understood the op and its
	// LOCAL applier refused it. The underlying closed token travels verbatim in the
	// ack's Detail (e.g. STEER_NO_OWNED_LOOP, CONTROL_SESSION_TERMINAL): a fanned op
	// refuses exactly as it would alone, and the bus never launders a local refusal
	// into a bus-level one.
	ApplyRefused RefuseReason = "FLEETBUS_APPLY_REFUSED"
)

// Reasons returns the closed refusal vocabulary in declaration order. It is the
// registration surface the dos.toml reason-block gate reads.
func Reasons() []RefuseReason {
	return []RefuseReason{Malformed, Expired, NoTarget, UnknownOp, ApplyRefused}
}

// Refusal is the structured, closed-reason refusal of one bus operation. It
// implements error so plumbing can thread it, but callers switch on Reason.
type Refusal struct {
	Reason RefuseReason `json:"reason"`
	Detail string       `json:"detail,omitempty"`
}

func (r *Refusal) Error() string {
	if r == nil {
		return ""
	}
	if r.Detail == "" {
		return string(r.Reason)
	}
	return string(r.Reason) + ": " + r.Detail
}

func refuse(reason RefuseReason, format string, args ...any) *Refusal {
	return &Refusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// --- identifiers ----------------------------------------------------------- //

// ValidToken reports whether s is usable as a bus identifier. Ids are addressing
// tokens AND path segments in the directory transport, so the check is deliberately
// narrow: an id carrying a separator or a dot-segment would let a directive name a
// file outside the bus root. Refusing the shape here means no transport has to
// re-derive the rule.
func ValidToken(s string) bool {
	if s == "" || len(s) > 128 || s == "." || s == ".." {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-', c == '_', c == '.':
		default:
			return false
		}
	}
	return true
}

// --- selector -------------------------------------------------------------- //

// Selector names the set a directive addresses, on TWO levels that are read by two
// different consumers:
//
//   - INSTANCE axes (All / Instance / Machine / Role / Model / Zone) route the
//     directive to processes. Only these are read by MatchesInstance, and only by the
//     bus.
//   - SESSION axes (Lane / Wave / Label) narrow WITHIN a matched instance. The bus
//     never interprets them; it carries them to the applier, which resolves them
//     against its own live sessions (they are sessionctl.BroadcastSelector's axes,
//     deliberately spelled the same way).
//
// The zero selector matches NOBODY and is malformed. Match-all must be STATED via
// All — the same fail-closed fence sessionctl.BroadcastSelector sets, for the same
// reason: a whole-fleet mutation must never be something you get by forgetting a
// flag.
type Selector struct {
	// All affirmatively addresses every live instance. It is the operator SAYING
	// "everyone", which is a different act from omitting a filter.
	All bool `json:"all,omitempty"`
	// Instance / Machine / Role address instances by exact, case-sensitive token.
	// Within one axis the members are OR'd; across axes they are AND'd.
	Instance []string `json:"instance,omitempty"`
	Machine  []string `json:"machine,omitempty"`
	Role     []string `json:"role,omitempty"`
	// Model / Zone address instances by declared CAPABILITY rather than by identity —
	// "everything serving the old checkpoint", not "these four ids". They read
	// Instance.Models / Instance.Zone, which are claims like Ops and not witnesses, so
	// a model axis SELECTS an instance and never certifies it: the ack is still the
	// only proof anything applied.
	//
	// Model is set-vs-set — an instance matches when ANY model it declares is named on
	// the axis. An instance that declares NOTHING (or predates the field) matches no
	// stated Model axis at all, which is the same fail-closed rule Role already runs:
	// an untagged record is never swept up by a filter it never answered.
	Model []string `json:"model,omitempty"`
	Zone  []string `json:"zone,omitempty"`
	// Lane / Wave / Label narrow within an instance. Carried, never interpreted here.
	Lane  string `json:"lane,omitempty"`
	Wave  string `json:"wave,omitempty"`
	Label string `json:"label,omitempty"`
}

// AddressesInstances reports whether any INSTANCE axis is stated. A selector that
// states only session axes is malformed: "lane=cmd" alone never means "on every
// machine" — that widening has to be said out loud.
func (s Selector) AddressesInstances() bool {
	return s.All || len(s.Instance) > 0 || len(s.Machine) > 0 || len(s.Role) > 0 ||
		len(s.Model) > 0 || len(s.Zone) > 0
}

// NarrowsSessions reports whether any SESSION axis is stated.
func (s Selector) NarrowsSessions() bool {
	return s.Lane != "" || s.Wave != "" || s.Label != ""
}

// narrowsInstances reports whether any instance axis OTHER than All is stated. It is
// spelled once so a new axis cannot be added to the addressing set while quietly
// staying legal to combine with --all.
func (s Selector) narrowsInstances() bool {
	return len(s.Instance) > 0 || len(s.Machine) > 0 || len(s.Role) > 0 ||
		len(s.Model) > 0 || len(s.Zone) > 0
}

// Validate refuses a selector that cannot address anybody, or one that contradicts
// itself by combining the affirmative All with a narrowing instance axis.
func (s Selector) Validate() *Refusal {
	if !s.AddressesInstances() {
		return refuse(Malformed,
			"selector addresses no instance — state --all, or an --instance / --machine / --role / --model / --zone; an empty selector never means \"every instance\"")
	}
	if s.All && s.narrowsInstances() {
		return refuse(Malformed,
			"selector states --all AND an instance filter; pick one (--all means every instance, a filter means some)")
	}
	return nil
}

// MatchesInstance reports whether inst is addressed. INSTANCE axes only — session
// axes are the applier's business (see the Selector doc).
func (s Selector) MatchesInstance(inst Instance) bool {
	if !s.AddressesInstances() {
		return false
	}
	if s.All {
		return true
	}
	if len(s.Instance) > 0 && !containsToken(s.Instance, inst.ID) {
		return false
	}
	if len(s.Machine) > 0 && !containsToken(s.Machine, inst.Machine) {
		return false
	}
	if len(s.Role) > 0 && !containsToken(s.Role, inst.Role) {
		return false
	}
	// Set-vs-set: the instance matches when any model it declares is named. An
	// instance declaring nothing has an empty Models, so this is false — the
	// fail-closed arm, and the reason adding this axis cannot widen an existing
	// selector's target set.
	if len(s.Model) > 0 && !intersectsTokens(s.Model, inst.Models) {
		return false
	}
	if len(s.Zone) > 0 && !containsToken(s.Zone, inst.Zone) {
		return false
	}
	return true
}

// String renders the stated axes for reports and logs.
func (s Selector) String() string {
	parts := make([]string, 0, 8)
	if s.All {
		parts = append(parts, "all")
	}
	for _, ax := range []struct {
		name string
		vals []string
	}{{"instance", s.Instance}, {"machine", s.Machine}, {"role", s.Role}, {"model", s.Model}, {"zone", s.Zone}} {
		if len(ax.vals) > 0 {
			parts = append(parts, ax.name+"="+strings.Join(ax.vals, ","))
		}
	}
	for _, ax := range []struct{ name, val string }{{"lane", s.Lane}, {"wave", s.Wave}, {"label", s.Label}} {
		if ax.val != "" {
			parts = append(parts, ax.name+"="+ax.val)
		}
	}
	if len(parts) == 0 {
		return "(empty)"
	}
	return strings.Join(parts, " ")
}

func containsToken(set []string, v string) bool {
	for _, s := range set {
		if s == v {
			return true
		}
	}
	return false
}

// intersectsTokens reports whether the axis and the instance's declared set share a
// member. Either side being empty is false — an axis nobody was asked about and a
// record that declared nothing both mean "no match", never "match anything".
func intersectsTokens(axis, declared []string) bool {
	for _, v := range declared {
		if containsToken(axis, v) {
			return true
		}
	}
	return false
}

// --- instance -------------------------------------------------------------- //

// Instance is a self-announced presence record for a process that CAN apply a
// directive. It is deliberately NOT the same set as internal/leaseref's live
// sessions: leaseref registers SESSIONS (opt-in, git-mediated), while this registers
// CONTROL-PLANE PARTICIPANTS — "who can apply a directive". Corroborating one with
// the other is a named child of epic #5599, not an assumption made here.
//
// Liveness is self-report freshness against DefaultInstanceTTL, not a process-table
// join — the honest UNKNOWN rung, the same one fleetmon.Census stops at.
type Instance struct {
	Schema string `json:"schema"`
	// ID is this process's bus identity, stable across its lifetime.
	ID string `json:"id"`
	// Machine is the host the instance runs on; Role is what it is (e.g. "serve").
	Machine string `json:"machine,omitempty"`
	Role    string `json:"role,omitempty"`
	PID     int    `json:"pid,omitempty"`
	// Addr is the instance's control address, when it has one (informational).
	Addr string `json:"addr,omitempty"`
	// Ops is what this instance SAYS it can apply. It is a display/diagnostic claim
	// only: nothing routes on it, because a roster's claim is not a witness — the
	// ack is. An op an instance did not declare still gets a real attempt and a real
	// ack.
	Ops []Op `json:"ops,omitempty"`
	// Unsupported is what this instance SAYS it structurally cannot apply — not "is
	// busy", not "would refuse today", but "there is no local subject for this op in
	// this process, and there never will be while it runs". A guard that owns no
	// session loop declares steer here.
	//
	// It exists because the absence of an op from Ops is AMBIGUOUS in a way an
	// operator cannot resolve: an instance that predates an op, one that forgot to
	// list it, and one that genuinely cannot do it all look identical (silent). That
	// ambiguity is what produces the worst reading of a fan-out — an operator sees
	// "16 instances" in the roster, fans steer at them, collects 16 refusals, and
	// only then learns the answer was always zero. Declaring it turns the roster into
	// something answerable BEFORE the fan-out: "16 instances, 0 can steer" (Capability).
	//
	// It is the same KIND of claim as Ops and carries the same limit, which is why it
	// is not a routing or selection input and no Selector axis reads it: a claim is
	// not a witness. Addressing an instance that declared an op unsupported still
	// gets that instance a real attempt and a real ack — a refusal under a closed
	// token from its own applier, with the instance still in the denominator. That is
	// deliberate: skipping declared-unsupported instances at publish time would turn
	// a fleet that can do nothing into FLEETBUS_NO_TARGET, which reads as "nobody was
	// there" rather than the truth, "everybody was there and none of them can".
	//
	// OPTIONAL and additive, under the same rule as Models/Zone: a record from a
	// binary that predates the field parses with it empty, stays in the roster, and
	// matches exactly the selectors it matched before.
	Unsupported []Op `json:"unsupported,omitempty"`
	// Models is what this instance SAYS it serves, and Zone where it says it sits.
	// They are the same KIND of thing as Ops — a claim, not a witness — with one
	// difference that has to be stated because it is easy to misread as promotion:
	// the Selector's Model/Zone axes DO read these, so they are SELECTION inputs.
	// Selection is not routing. Being addressed still gets you a real attempt and a
	// real ack, and an instance that lied about its models refuses at its own
	// applier exactly as it would have anyway. What the axes buy is that a
	// model-scoped operator action ("drain everything on the old checkpoint")
	// addresses the instances that claim it instead of fanning to all and collecting
	// N refusals — the accepted-but-not-applied shape this package exists to avoid.
	//
	// Both are OPTIONAL and additive. A record written by a binary that predates them
	// parses with them empty, stays in the roster, and keeps matching exactly the
	// selectors it matched before, so no denominator shrinks (see fold.go).
	Models []string `json:"models,omitempty"`
	Zone   string   `json:"zone,omitempty"`
	// SeenUTC is the announce timestamp (RFC3339 nanos, UTC).
	SeenUTC string `json:"seen_utc"`
}

// NewInstance stamps a presence record at now.
func NewInstance(id, machine, role string, pid int, addr string, ops []Op, now time.Time) (Instance, *Refusal) {
	inst := Instance{
		Schema:  InstanceSchema,
		ID:      strings.TrimSpace(id),
		Machine: strings.TrimSpace(machine),
		Role:    strings.TrimSpace(role),
		PID:     pid,
		Addr:    strings.TrimSpace(addr),
		Ops:     ops,
		SeenUTC: utc(now),
	}
	if r := inst.Validate(); r != nil {
		return Instance{}, r
	}
	return inst, nil
}

// WithServedModels stamps what this instance serves onto its presence record,
// returning a copy. Models are trimmed, de-duplicated and sorted so two announces of
// the same set produce byte-identical records — a roster an operator diffs should not
// churn because a config file listed the same models in a different order.
//
// It is a BUILDER rather than another NewInstance parameter on purpose. NewInstance
// stamps the fields every presence record must have to be addressed and aged; this is
// an optional claim an announcer fills in only when it knows it, and folding it into
// the constructor would force every existing caller — including the ones with nothing
// to declare — to say so with an empty argument.
//
// Model tokens are deliberately NOT held to ValidToken. That check exists because ids
// are path segments in the directory transport; a model name is only ever compared,
// never joined onto a path, and real names carry separators ("zai-org/GLM-4.6") that
// the id alphabet would refuse.
func (i Instance) WithServedModels(models []string) Instance {
	seen := map[string]bool{}
	out := make([]string, 0, len(models))
	for _, m := range models {
		m = strings.TrimSpace(m)
		if m == "" || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	sort.Strings(out)
	if len(out) == 0 {
		// nil, not an empty slice: omitempty then keeps the field off the wire, so a
		// record declaring nothing is byte-identical to one from before the field.
		out = nil
	}
	i.Models = out
	return i
}

// WithZone stamps where this instance says it sits, returning a copy. It is a separate
// builder from WithServedModels because the two answer different questions — what is
// served, and where it is — and an announcer commonly knows one without the other.
// Zone is a plain comparison label under the same rule as a model token: never a path
// segment, so never held to ValidToken.
func (i Instance) WithZone(zone string) Instance {
	i.Zone = strings.TrimSpace(zone)
	return i
}

// WithUnsupportedOps stamps what this instance says it structurally cannot apply,
// returning a copy. It normalizes exactly as WithServedModels does — trimmed,
// de-duplicated, sorted, nil when empty — so a roster an operator diffs does not
// churn on declaration order, and an instance declaring nothing writes no key at all.
//
// It is a builder rather than a NewInstance parameter for the same reason: an
// announcer with nothing to declare should not have to say so with an empty argument.
//
// Declaring an op BOTH ways (in Ops and here) is not rejected. Validate's job is to
// refuse records that cannot be addressed or aged, and a contradictory pair of claims
// is neither — refusing it would drop a live instance out of the roster over a
// display field. Capability resolves the contradiction the safe way instead, by
// counting it as unsupported.
func (i Instance) WithUnsupportedOps(ops []Op) Instance {
	seen := map[Op]bool{}
	out := make([]Op, 0, len(ops))
	for _, op := range ops {
		op = Op(strings.TrimSpace(string(op)))
		if op == "" || seen[op] {
			continue
		}
		seen[op] = true
		out = append(out, op)
	}
	sort.Slice(out, func(a, b int) bool { return out[a] < out[b] })
	if len(out) == 0 {
		out = nil
	}
	i.Unsupported = out
	return i
}

// DeclaresUnsupported reports whether this instance named op as something it cannot
// apply. It is a claim, never a gate: read it to EXPLAIN a fleet, never to decide
// whether to address one.
func (i Instance) DeclaresUnsupported(op Op) bool {
	for _, declared := range i.Unsupported {
		if declared == op {
			return true
		}
	}
	return false
}

// DeclaresOp reports whether this instance named op as something it can apply.
func (i Instance) DeclaresOp(op Op) bool {
	for _, declared := range i.Ops {
		if declared == op {
			return true
		}
	}
	return false
}

// Validate refuses a presence record that cannot be addressed or aged.
func (i Instance) Validate() *Refusal {
	if i.Schema != InstanceSchema {
		return refuse(Malformed, "instance schema %q (want %q)", i.Schema, InstanceSchema)
	}
	if !ValidToken(i.ID) {
		return refuse(Malformed, "instance id %q is not a bus token ([A-Za-z0-9._-], 1..128, not . or ..)", i.ID)
	}
	if _, err := parseUTC(i.SeenUTC); err != nil {
		return refuse(Malformed, "instance %s: unparsable seen_utc %q", i.ID, i.SeenUTC)
	}
	return nil
}

// Fresh reports whether the presence record is still credible at now. A record from
// the FUTURE (clock skew across hosts) counts as fresh: refusing it would make a
// skewed peer invisible, which is a worse failure than a slightly generous roster.
func (i Instance) Fresh(now time.Time, ttl time.Duration) bool {
	seen, err := parseUTC(i.SeenUTC)
	if err != nil {
		return false
	}
	if ttl <= 0 {
		ttl = DefaultInstanceTTL
	}
	return !now.After(seen.Add(ttl))
}

// --- directive ------------------------------------------------------------- //

// Directive is what the control point wants done, to whom, until when.
type Directive struct {
	Schema string `json:"schema"`
	// ID is derived from the directive's content (see NewDirective): re-issuing the
	// identical directive collapses onto the same id, so a retried publish is
	// idempotent rather than a second command.
	ID string `json:"id"`
	// Issuer names the control point. Attribution, NOT authority — several control
	// points are expected, and none of them is trusted because of this string.
	Issuer string `json:"issuer"`
	// IssuedUTC is when the control point cut it (RFC3339 nanos, UTC).
	IssuedUTC string `json:"issued_utc"`
	// TTLSec bounds how long the directive is applyable. 0 means no expiry.
	TTLSec int `json:"ttl_sec,omitempty"`
	// Op is the control op name; Payload is its argument (the steer text, say).
	Op      Op     `json:"op"`
	Payload string `json:"payload,omitempty"`
	// Selector names the addressed set.
	Selector Selector `json:"selector"`
	// Targets is the instance ids the selector actually resolved to at PUBLISH time,
	// stamped by the control point (see WithTargets). It is the fold's denominator
	// floor, and it exists because the roster is a liveness claim with an expiry: an
	// instance addressed at publish that dies before it acks stops announcing, drops
	// out of the roster, and — with a denominator re-derived from the roster alone —
	// would vanish from the report entirely, flipping Complete to true and reporting
	// a witnessed apply for a directive it never received. Recording who was
	// addressed is what keeps "nobody answered for box-b" expressible after box-b is
	// gone. Empty on a directive published before this field existed, which folds
	// back to the roster-only denominator.
	Targets []string `json:"targets,omitempty"`
	// Reason is the operator's why, carried through to the local applier so a
	// session's drive-state record says WHY it was paused, not just that it was.
	Reason string `json:"reason,omitempty"`
}

// NewDirective builds and validates a directive, deriving its id from its content
// plus its issue instant. Two control points issuing the same op at the same instant
// collapse to one directive (which is correct — it IS one command); the same control
// point re-sending later gets a new id (also correct — it is a new command).
func NewDirective(issuer string, op Op, payload string, sel Selector, ttl time.Duration, reason string, now time.Time) (Directive, *Refusal) {
	d := Directive{
		Schema:    DirectiveSchema,
		Issuer:    strings.TrimSpace(issuer),
		IssuedUTC: utc(now),
		Op:        Op(strings.TrimSpace(string(op))),
		Payload:   payload,
		Selector:  sel,
		Reason:    strings.TrimSpace(reason),
	}
	// A NEGATIVE ttl is refused rather than folded onto 0. 0 is the documented
	// "never expires", so letting -5m fall through to it would turn a deadline that
	// has already passed — the shape a wrapper computing deadline.Sub(now) produces
	// late — into an unbounded directive that every instance joining the bus for the
	// rest of the log's life still applies. That is the exact "stale `go` applied an
	// hour later" the TTL exists to prevent, arrived at by silence.
	if ttl < 0 {
		return Directive{}, refuse(Malformed,
			"ttl %s is negative; state a positive duration, or 0 to mean never expires", ttl)
	}
	if ttl > 0 {
		d.TTLSec = int(ttl / time.Second)
		if d.TTLSec == 0 {
			d.TTLSec = 1 // a sub-second TTL rounds UP; never silently to "never expires"
		}
	}
	d.ID = directiveID(d)
	if r := d.Validate(); r != nil {
		return Directive{}, r
	}
	return d, nil
}

// WithTargets stamps the publish-time target set onto the directive, so the fold has
// a denominator that survives the addressed instances dying. The ids are taken from
// the roster the control point resolved the selector against, deduped and sorted so
// the record is stable to read and diff.
//
// The directive id is deliberately NOT re-derived: the id names the COMMAND (issuer,
// op, payload, selector, instant), and who happened to be listening is a property of
// the publish, not of what was ordered. Re-deriving would break the documented
// collapse of a retried publish onto one id.
func (d Directive) WithTargets(targets []Instance) Directive {
	seen := map[string]bool{}
	ids := make([]string, 0, len(targets))
	for _, inst := range targets {
		if inst.ID == "" || seen[inst.ID] {
			continue
		}
		seen[inst.ID] = true
		ids = append(ids, inst.ID)
	}
	sort.Strings(ids)
	d.Targets = ids
	return d
}

// TargetsInstance binds delivery to the publish-time roster when Targets is present.
// Legacy directives without Targets retain selector matching for wire compatibility.
// A process that boots after publish must never execute a retained lifecycle command.
func (d Directive) TargetsInstance(inst Instance) bool {
	if len(d.Targets) == 0 {
		return d.Selector.MatchesInstance(inst)
	}
	for _, id := range d.Targets {
		if id == inst.ID {
			return true
		}
	}
	return false
}

// directiveID digests every field that changes what the directive MEANS. The id is
// the idempotency key the apply claim is taken against, so anything that would make
// two directives different commands must be in here.
func directiveID(d Directive) string {
	sel, _ := json.Marshal(d.Selector)
	h := sha256.New()
	for _, part := range []string{d.Issuer, string(d.Op), d.Payload, string(sel), d.IssuedUTC, d.Reason, fmt.Sprint(d.TTLSec)} {
		h.Write([]byte(part))
		h.Write([]byte{0}) // a separator, so ("ab","c") and ("a","bc") cannot collide
	}
	return "d-" + hex.EncodeToString(h.Sum(nil))[:16]
}

// Validate refuses a directive whose shape cannot be carried or applied.
func (d Directive) Validate() *Refusal {
	if d.Schema != DirectiveSchema {
		return refuse(Malformed, "directive schema %q (want %q)", d.Schema, DirectiveSchema)
	}
	if !ValidToken(d.ID) {
		return refuse(Malformed, "directive id %q is not a bus token ([A-Za-z0-9._-], 1..128, not . or ..)", d.ID)
	}
	if strings.TrimSpace(d.Issuer) == "" {
		return refuse(Malformed, "directive %s has no issuer — every control point names itself", d.ID)
	}
	if strings.TrimSpace(string(d.Op)) == "" {
		return refuse(Malformed, "directive %s has no op", d.ID)
	}
	if d.TTLSec < 0 {
		return refuse(Malformed, "directive %s has a negative ttl_sec (%d)", d.ID, d.TTLSec)
	}
	if _, err := parseUTC(d.IssuedUTC); err != nil {
		return refuse(Malformed, "directive %s: unparsable issued_utc %q", d.ID, d.IssuedUTC)
	}
	return d.Selector.Validate()
}

// IssuedAt parses the issue instant.
func (d Directive) IssuedAt() (time.Time, error) { return parseUTC(d.IssuedUTC) }

// ExpiresAt reports the directive's deadline, and false when it never expires.
func (d Directive) ExpiresAt() (time.Time, bool) {
	if d.TTLSec <= 0 {
		return time.Time{}, false
	}
	issued, err := parseUTC(d.IssuedUTC)
	if err != nil {
		return time.Time{}, false
	}
	return issued.Add(time.Duration(d.TTLSec) * time.Second), true
}

// IsExpired reports whether the directive's TTL has run out at now. An unparsable
// or absent TTL is NOT expired: a directive is refused for a bad shape at the edge
// (Validate), never silently aged out by a parse failure downstream.
func (d Directive) IsExpired(now time.Time) bool {
	deadline, bounded := d.ExpiresAt()
	return bounded && now.After(deadline)
}

// --- ack ------------------------------------------------------------------- //

// AckStatus is the closed outcome vocabulary of one instance's answer.
type AckStatus string

const (
	// AckApplied: the instance took the op through its real local write path.
	AckApplied AckStatus = "applied"
	// AckRefused: the instance understood the directive and did not apply it. Reason
	// carries the closed token (bus-level or, via ApplyRefused, the local one).
	AckRefused AckStatus = "refused"
	// AckExpired: the instance met the directive after its TTL and did not apply it.
	AckExpired AckStatus = "expired"
)

// Ack is one instance's outcome for one directive — the witness the whole package
// exists to produce.
type Ack struct {
	Schema string `json:"schema"`
	// Directive and Instance are the (command, answerer) pair this ack answers for.
	Directive string `json:"directive"`
	Instance  string `json:"instance"`
	// Status is the closed outcome.
	Status AckStatus `json:"status"`
	// Reason is the closed refusal token when Status is not applied; Detail carries
	// the underlying LOCAL token verbatim (e.g. STEER_NO_OWNED_LOOP) plus its text.
	Reason RefuseReason `json:"reason,omitempty"`
	Detail string       `json:"detail,omitempty"`
	// Witness is what the instance actually observed change — the applied-side twin
	// of Reason. "the enqueue is never the witness; the arm observing it is."
	Witness string `json:"witness,omitempty"`
	// Affected is how many local subjects (sessions, typically) the op landed on. A
	// zero-affected "applied" is a real and reportable outcome: the instance heard
	// the directive and had nothing matching to apply it to.
	Affected int `json:"affected"`
	// AckedUTC is when the instance answered (RFC3339 nanos, UTC).
	AckedUTC string `json:"acked_utc"`
}

// Validate refuses an ack that cannot be attributed or folded.
func (a Ack) Validate() *Refusal {
	if a.Schema != AckSchema {
		return refuse(Malformed, "ack schema %q (want %q)", a.Schema, AckSchema)
	}
	if !ValidToken(a.Directive) {
		return refuse(Malformed, "ack directive id %q is not a bus token", a.Directive)
	}
	if !ValidToken(a.Instance) {
		return refuse(Malformed, "ack instance id %q is not a bus token", a.Instance)
	}
	switch a.Status {
	case AckApplied, AckRefused, AckExpired:
	default:
		return refuse(Malformed, "ack status %q (want applied|refused|expired)", a.Status)
	}
	if a.Status != AckApplied && a.Reason == "" {
		return refuse(Malformed, "ack for %s from %s is %s with no closed reason — a non-applied ack must say why", a.Directive, a.Instance, a.Status)
	}
	if _, err := parseUTC(a.AckedUTC); err != nil {
		return refuse(Malformed, "ack for %s from %s: unparsable acked_utc %q", a.Directive, a.Instance, a.AckedUTC)
	}
	return nil
}

// --- time ------------------------------------------------------------------ //

// utc renders an instant in the bus's one timestamp format. Nanosecond precision is
// deliberate: two directives cut in the same second must not collide onto one id.
func utc(t time.Time) string { return t.UTC().Format(time.RFC3339Nano) }

func parseUTC(s string) (time.Time, error) {
	if strings.TrimSpace(s) == "" {
		return time.Time{}, fmt.Errorf("empty timestamp")
	}
	return time.Parse(time.RFC3339Nano, s)
}

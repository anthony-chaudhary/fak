package modelroute

import (
	"fmt"
	"net/url"
	"strings"
)

// ---------------------------------------------------------------------------
// THE PLACEMENT ZONE — WHERE the weights that serve a call sit, and WHOSE trust
// boundary they sit inside.
// ---------------------------------------------------------------------------
//
// Before this file the switcher's locality vocabulary was BINARY: KindLocal (an
// on-box loopback server, residency-exempt) versus everything else (a third-party
// API, credentialed and denied for a sensitive payload). That lattice cannot
// express the shape a company actually deploys: an engineer's laptop running a
// small model, PLUS a rack of machines the COMPANY operates running a mid-size
// model, PLUS a third-party frontier API for the rare hard call.
//
// The company rack is the missing middle. Today it can be declared neither way:
//
//   - as KindLocal it is REFUSED — Validate requires a loopback base_url, because
//     a "local:" route the residency floor trusts must never egress the box.
//   - as KindOpenAI it is admitted, but it is then indistinguishable from
//     api.openai.com: the org gets no residency credit for hardware it owns, and
//     no usage record can tell "we self-hosted this" from "we bought this".
//
// So a PlacementZone is a FIRST-CLASS axis, orthogonal to both the model's
// identity and to WorkTier (tierpolicy.go). It is orthogonal to model identity
// because the SAME model family runs in two zones at once — a 4B Qwen on the
// laptop and a 200B+ Qwen on the company rack are the same family in different
// zones — and orthogonal to WorkTier because the tier says how HARD the work is
// while the zone says who owns the silicon, who pays per token, and whether the
// bytes may leave the org.
//
// THE THREE ZONES ARE A LADDER, cheapest-and-most-private first. Rank orders
// them so an escalation ladder (device -> fleet -> vendor) is a comparison on a
// declared value, never a string switch scattered across call sites.
type PlacementZone string

const (
	// ZoneDevice is the engineer's OWN machine — a loopback OpenAI-compatible
	// server (ollama / vLLM / llama.cpp) or the in-kernel model. The bytes never
	// leave the box, so it is residency-exempt. This is the "haiku-class" rung:
	// small models, small tasks, zero marginal token cost.
	ZoneDevice PlacementZone = "device"

	// ZoneFleet is a machine the ORGANIZATION operates — off-box, but inside the
	// org's own trust boundary. This is the rung that carries the bulk of the
	// tokens in a self-hosting shop: a mid-size open model (GLM-class, Kimi-class)
	// on company GPUs, shared by every engineer.
	//
	// FAIL-CLOSED TODAY (deliberate, and the boundary of this slice): declaring a
	// zone is NOT the same as being trusted with a sensitive payload. A fleet
	// target is still Remote() to internal/engine's residency floor, so the floor
	// denies a tenant-scoped call routed here exactly as it did before this file
	// existed. Widening the floor to treat an org-operated host as inside the
	// boundary is an ENFORCEMENT change that needs an operator-declared trust
	// boundary (authenticated transport, a named host allowlist) and is tracked
	// separately — it must not ride in on a vocabulary addition.
	ZoneFleet PlacementZone = "fleet"

	// ZoneVendor is a third-party frontier lab's API — outside the org's trust
	// boundary, metered per token, and denied for a sensitive payload. This is the
	// rung the whole design exists to use SPARINGLY: only work that genuinely needs
	// the horsepower should reach it.
	ZoneVendor PlacementZone = "vendor"
)

// Valid reports whether z is one of the three defined zones. The set is CLOSED —
// a new zone is an added constant plus validation, never manifest free text —
// mirroring ProviderKind and Reduction.
func (z PlacementZone) Valid() bool {
	switch z {
	case ZoneDevice, ZoneFleet, ZoneVendor:
		return true
	}
	return false
}

// String renders the canonical zone label.
func (z PlacementZone) String() string { return string(z) }

// Rank orders the zones as the escalation ladder runs: 0 device, 1 fleet, 2
// vendor — cheapest-and-most-private first. An unknown zone ranks ABOVE vendor
// (fail-closed: an unrecognized placement is never treated as the cheap rung).
func (z PlacementZone) Rank() int {
	switch z {
	case ZoneDevice:
		return 0
	case ZoneFleet:
		return 1
	case ZoneVendor:
		return 2
	}
	return 3
}

// SelfHosted reports whether the org (or the engineer) owns the silicon serving
// this call — device OR fleet. This is the single predicate the token-economics
// goal is measured against: "the bulk of token usage is covered by self-hosted
// models" is exactly the share of tokens whose zone SelfHosted()s. An unknown
// zone is NOT self-hosted (fail-closed — an unattributable token is never
// counted as a saving).
func (z PlacementZone) SelfHosted() bool { return z == ZoneDevice || z == ZoneFleet }

// OnBox reports whether the bytes never leave the machine that issued the call.
// Only ZoneDevice qualifies. This is DELIBERATELY narrower than SelfHosted: the
// residency floor's exemption keys on OnBox, not on ownership, so declaring a
// fleet zone can never by itself buy a sensitive payload a trip off the box.
func (z PlacementZone) OnBox() bool { return z == ZoneDevice }

// Zones returns the ladder in Rank order. It is the canonical iteration order for
// an escalation ladder and for a zone-keyed report, so no caller re-derives it.
func Zones() []PlacementZone { return []PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor} }

// ZoneOfKind maps a provider kind onto its placement zone. It is DERIVED from the
// account's declared Kind — the same single source of truth Target.Local() reads —
// so a zone can never disagree with the locality the residency floor enforces.
func ZoneOfKind(k ProviderKind) PlacementZone {
	switch k {
	case KindLocal:
		return ZoneDevice
	case KindFleet:
		return ZoneFleet
	}
	return ZoneVendor
}

// Zone reports the placement zone this account serves from.
func (a Account) Zone() PlacementZone { return ZoneOfKind(a.Kind) }

// Zone reports the placement zone this resolved target dispatches into. It is the
// dimension a usage record must carry to answer "what share of our tokens did we
// self-host?" — see PlacementZone.SelfHosted.
func (t Target) Zone() PlacementZone { return ZoneOfKind(t.Kind) }

// ---------------------------------------------------------------------------
// THE ROUTE MIRROR — read a zone back OUT of an abi.ToolCall.Engine string.
// ---------------------------------------------------------------------------
//
// Target.EngineRoute() stamps "<zone-or-kind>:<account>/<model>" into
// abi.ToolCall.Engine BEFORE Kernel.Submit, and internal/engine's residencyGate
// classifies that string by its LEADING prefix. Everything downstream of the
// kernel — a usage ledger, a metrics fold, an audit row — sees only the string,
// so it needs a parse-back that agrees with the floor exactly.
//
// localRouteFamilies is that parse-back's on-box half: the KNOWN on-box engine
// families internal/engine.localRoute recognizes. It is a deliberate tier-1 COPY
// (this package is stdlib-only and cannot import internal/engine, which sits a
// tier above), and the copy is pinned by a cross-package test in internal/engine
// that runs BOTH classifiers over one corpus. That test is what makes the
// duplication safe: a family added on one side and forgotten on the other is a
// build-time failure, not a silent fail-OPEN.
//
// The comment at internal/engine/engine.go promised this mirror under the name
// IsRemoteRoute before it existed; this file is that promise made real.
var localRouteFamilies = []string{"inkernel", "mock", "cassette", "local", "on-device", "ondevice", "kernel"}

// ZoneOfRoute classifies an engine-route string (the value written to
// abi.ToolCall.Engine) into its placement zone by its LEADING family prefix — the
// same signal the residency floor keys on, never a guess from the model name.
//
// It accepts the four shapes the floor accepts: a bare family ("inkernel"), a
// "family:instance" ("local:box/llama3.2"), a "family-suffix" ("local-gpu"), and
// a "family/path" ("local/llama"). An EMPTY route is ZoneDevice — an unset Engine
// means the in-kernel kernel default, which runs on this box. Anything the
// prefix does not place is ZoneVendor (fail-closed: an unrecognized route is
// treated as leaving the org, never as a free on-box call).
func ZoneOfRoute(route string) PlacementZone {
	r := strings.ToLower(strings.TrimSpace(route))
	if r == "" {
		return ZoneDevice // the in-kernel default — on-box
	}
	for _, fam := range localRouteFamilies {
		if hasRouteFamily(r, fam) {
			return ZoneDevice
		}
	}
	if hasRouteFamily(r, string(ZoneFleet)) {
		return ZoneFleet
	}
	return ZoneVendor
}

// hasRouteFamily reports whether route leads with engine family fam in any of the
// four accepted shapes. It mirrors internal/engine.localRoute's matching rules
// exactly.
func hasRouteFamily(route, fam string) bool {
	return route == fam ||
		strings.HasPrefix(route, fam+":") ||
		strings.HasPrefix(route, fam+"-") ||
		strings.HasPrefix(route, fam+"/")
}

// IsRemoteRoute reports whether an engine-route string names a destination OFF
// the calling box — the tier-1 mirror of internal/engine's residency-floor
// predicate, for callers below that tier. It is exactly !ZoneOfRoute(route).OnBox(),
// so a fleet route reads REMOTE here just as it does at the floor: org-owned
// hardware is still off-box, and this predicate never widens the floor.
func IsRemoteRoute(route string) bool { return !ZoneOfRoute(route).OnBox() }

// ---------------------------------------------------------------------------
// FLEET ACCOUNT VALIDATION — the invariants that keep the three zones disjoint.
// ---------------------------------------------------------------------------

// validateFleetAccount enforces the invariants that make ZoneFleet a distinct,
// honest zone rather than a synonym for either neighbor:
//
//   - an EXPLICIT base_url. There is no public default for a machine only this
//     org can name, so KindBaseURL returns "" and omitting it is an error rather
//     than a silent fall-through to some vendor's endpoint.
//   - a base_url that PARSES and names a host over http/https. A fleet endpoint
//     is reached over the network; a malformed or scheme-less URL would surface
//     as a dial failure at dispatch time instead of a config error here.
//   - a NON-loopback host. A loopback address is ZoneDevice by definition, and
//     letting it wear the fleet label would put a residency-exempt on-box server
//     in the zone the floor treats as off-box — the zones would stop partitioning
//     the accounts, and a later trust-boundary widening for fleet would silently
//     also widen device.
//
// A credential is OPTIONAL and deliberately so: an org-operated inference server
// on a private network commonly has no API key, and demanding a cred_env would
// force operators to invent a fake one. When one IS present the shared env-var
// NAME check in Validate still applies, so a pasted secret still fails loud.
func validateFleetAccount(a Account) error {
	if a.BaseURL == "" {
		return fmt.Errorf("modelroute: fleet account %q needs a base_url (no public default for an org-operated server, e.g. http://gpu-07.corp.internal:8000/v1)", a.ID)
	}
	u, err := url.Parse(a.BaseURL)
	if err != nil || u.Hostname() == "" {
		return fmt.Errorf("modelroute: fleet account %q base_url %q is not an absolute http(s) URL naming a host", a.ID, a.BaseURL)
	}
	switch strings.ToLower(u.Scheme) {
	case "http", "https":
	default:
		return fmt.Errorf("modelroute: fleet account %q base_url %q must use http or https (got scheme %q)", a.ID, a.BaseURL, u.Scheme)
	}
	if isLoopbackBaseURL(a.BaseURL) {
		return fmt.Errorf("modelroute: fleet account %q base_url %q is a loopback host — an on-box server is kind %q (zone %s), not %q; the zones must stay disjoint",
			a.ID, a.BaseURL, KindLocal, ZoneDevice, KindFleet)
	}
	return nil
}

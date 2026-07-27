package dispatchtick

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Which STRATUM served a finished slot (#5416 track E).
//
// Epic #5416's headline claim is a fraction — "the bulk of token usage is covered by
// self-hosted models" — and nothing in a dispatch slot records the half of it that
// matters. `WitnessRecord.Model` says WHICH model ran; the rung it ran on (this laptop,
// a company box, a frontier vendor) is a property of the ROSTER BINDING, not of the model
// id, so a sweep that reads only the id cannot answer the question the epic is about.
//
// It is worth naming the shortcut this file refuses. `modelroute.ZoneOfRoute` already maps
// a string to a zone and would compile here, but it classifies an ENGINE-ROUTE prefix
// ("local:box/llama3.2") and its own doc says "never a guess from the model name" — and its
// empty case is ZoneDevice, because an unset engine means the in-kernel default. Pointed at
// a worker-model id those two properties combine into the worst possible default: every
// unrecognized pin, and every unpinned slot, would be counted as running ON THIS BOX. The
// resolver below is the roster instead, and it says "" when it does not know.
//
// The direction of that conservatism is deliberate. An over-reported self-hosted share is
// the dangerous error — an operator reads it as "our code stayed in the org" — so a rung
// nobody can name is never assumed, in either direction, and the count of slots that could
// not be attributed rides along with the fraction everywhere it is reported.

// ZoneSidecarSuffix records the placement zone a worker slot was served from, written at
// spawn beside ModelSidecarSuffix and scraped back by the witness sweep into
// WitnessRecord.Zone. Written ONLY when the zone is actually known, so an unconfigured
// fleet (no roster, seat-default worker) writes no extra sidecar and its runs dir stays
// byte-identical to before this seam.
const ZoneSidecarSuffix = ".zone"

// ZoneAttribution is the closed vocabulary for why a slot's rung is or is not known. The
// three failures are three DIFFERENT missing wires with three different fixes, so they are
// never summed into one "unknown" — an operator who cannot tell "nobody pinned a model"
// from "the roster does not bind this id" cannot fix either.
type ZoneAttribution string

const (
	// ZoneAttributed: the roster bound the slot's pinned model and named its rung.
	ZoneAttributed ZoneAttribution = "attributed"
	// ZoneNoModelPin: the slot ran on the seat default with no --model pin, so nothing
	// recorded WHICH model ran and no rung can be derived. The fix is Layer-5b pinning.
	ZoneNoModelPin ZoneAttribution = "no-model-pin"
	// ZoneNoRoster: a model was pinned but no account roster was loaded, so nothing could
	// map the id to a provider. The fix is handing the tick a roster.
	ZoneNoRoster ZoneAttribution = "no-roster"
	// ZoneUnboundModel: a roster was loaded and does not bind this id. Usually a real
	// misconfiguration (a typo, or a model dispatch pins that routing never bound), which
	// is why it stays distinct from having no roster at all.
	ZoneUnboundModel ZoneAttribution = "unbound-model"
)

// ZoneResolver maps a pinned model id to the rung its account serves from. ok is false when
// the roster does not bind the id. Keeping it a func is what lets this tier-1 leaf attribute
// a rung without holding a Roster.
//
// Supply `modelroute.Roster.BoundZone` — its shape is exactly this one. Do NOT build a
// resolver from `Roster.Resolve` + `Target.Zone()`: Resolve is a DISPATCH primitive and falls
// back to the roster's default account for an id nobody bound, so as an attributor it answers
// the default's rung for every typo and every unregistered pin. On the fleet-default rosters
// this path is aimed at, that reports vendor spend as self-hosted — the one direction the
// headline share must never be wrong in.
type ZoneResolver func(model string) (modelroute.PlacementZone, bool)

// AttributeZone names the rung a pinned model was served from, or says why it cannot.
//
// It never guesses. A blank model, a nil resolver, and an unbound id all return an empty
// zone with the reason that distinguishes them — an empty zone reads as "not recorded"
// everywhere downstream (modelroute.PlacementZone is a string, so its zero value is "" and
// not the device rung), which is exactly the property that keeps an unknown from being
// counted as on-box.
func AttributeZone(resolve ZoneResolver, model string) (modelroute.PlacementZone, ZoneAttribution) {
	if strings.TrimSpace(model) == "" {
		return "", ZoneNoModelPin
	}
	if resolve == nil {
		return "", ZoneNoRoster
	}
	z, ok := resolve(strings.TrimSpace(model))
	if !ok || !z.Valid() {
		return "", ZoneUnboundModel
	}
	return z, ZoneAttributed
}

// ZoneShare is the fold behind epic #5416's headline: how much of the fleet's finished work
// was served from hardware the org operates.
//
// Attributed and Unattributed are both fields because the fraction is not interpretable
// without the second one. A fleet where 90% of slots are unpinned can show "100%
// self-hosted" over the 10% it can see, and that number is arithmetically correct and
// completely misleading — so Headline renders the unattributed count unconditionally, even
// when it is zero.
type ZoneShare struct {
	// Total is every record folded, attributed or not.
	Total int
	// Attributed is the number of slots whose rung the roster could name.
	Attributed int
	// ByZone counts the attributed slots per rung.
	ByZone map[modelroute.PlacementZone]int
	// Unattributed counts the rest, keyed by which wire was missing. It never contains
	// ZoneAttributed.
	Unattributed map[ZoneAttribution]int
}

// FoldZoneShare counts a witness sweep by rung. A record carries its own zone (scraped from
// the .zone sidecar); one that carries none is attributed by the same rules AttributeZone
// applies at spawn, so a slot that was never pinned and a slot whose id nothing binds stay
// distinguishable after the fact.
func FoldZoneShare(records []WitnessRecord) ZoneShare {
	s := ZoneShare{
		Total:        len(records),
		ByZone:       map[modelroute.PlacementZone]int{},
		Unattributed: map[ZoneAttribution]int{},
	}
	for _, r := range records {
		z := modelroute.PlacementZone(strings.TrimSpace(r.Zone))
		switch {
		case z.Valid():
			s.ByZone[z]++
			s.Attributed++
		case strings.TrimSpace(r.Model) == "":
			s.Unattributed[ZoneNoModelPin]++
		default:
			// A model was pinned and no rung came back with it: either the tick had no
			// roster or the roster did not bind the id. The sweep cannot tell those apart
			// after the fact — only the spawn could — so it reports the one that is true
			// of the RECORD rather than inventing the distinction.
			s.Unattributed[ZoneUnboundModel]++
		}
	}
	return s
}

// SelfHosted counts the attributed slots served from the device or fleet rungs — the
// numerator of the epic's headline fraction.
func (s ZoneShare) SelfHosted() int {
	n := 0
	for z, c := range s.ByZone {
		if z.SelfHosted() {
			n += c
		}
	}
	return n
}

// Share is the self-hosted fraction of ATTRIBUTED slots and of nothing else. ok is false when nothing was
// attributed, so a fleet with no evidence reports "no answer" rather than 0% — which would
// read as "we self-host nothing" when the truth is "nobody recorded where anything ran."
func (s ZoneShare) Share() (float64, bool) {
	if s.Attributed == 0 {
		return 0, false
	}
	return float64(s.SelfHosted()) / float64(s.Attributed), true
}

// Headline renders the fraction with its own denominator caveat attached. The unattributed
// count is printed unconditionally — including when it is zero — because a share quoted
// without it is the one number in this file that can mislead an operator while being
// arithmetically correct.
func (s ZoneShare) Headline() string {
	var b strings.Builder
	if f, ok := s.Share(); ok {
		fmt.Fprintf(&b, "self-hosted %.0f%% (%d of %d attributed slot(s)", f*100, s.SelfHosted(), s.Attributed)
	} else {
		b.WriteString("self-hosted UNKNOWN (0 attributed slot(s)")
	}
	for _, z := range modelroute.Zones() {
		if c := s.ByZone[z]; c > 0 {
			fmt.Fprintf(&b, ", %s %d", z, c)
		}
	}
	b.WriteString(")")

	missing := s.Total - s.Attributed
	fmt.Fprintf(&b, "; %d of %d slot(s) unattributed", missing, s.Total)
	if missing > 0 {
		reasons := make([]string, 0, len(s.Unattributed))
		for k := range s.Unattributed {
			reasons = append(reasons, string(k))
		}
		sort.Strings(reasons)
		parts := make([]string, 0, len(reasons))
		for _, k := range reasons {
			parts = append(parts, fmt.Sprintf("%s %d", k, s.Unattributed[ZoneAttribution(k)]))
		}
		fmt.Fprintf(&b, " (%s)", strings.Join(parts, ", "))
	}
	return b.String()
}

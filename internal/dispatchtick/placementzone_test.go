package dispatchtick

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// rosterOf builds a ZoneResolver over a fixed id -> rung table, standing in for
// `roster.Resolve(m)` + `Target.Zone()` without dragging a Roster into a tier-1 leaf.
func rosterOf(bind map[string]modelroute.PlacementZone) ZoneResolver {
	return func(model string) (modelroute.PlacementZone, bool) {
		z, ok := bind[model]
		return z, ok
	}
}

var demoRoster = rosterOf(map[string]modelroute.PlacementZone{
	"qwen3.6-4b":  modelroute.ZoneDevice,
	"glm-5.2":     modelroute.ZoneFleet,
	"claude-opus": modelroute.ZoneVendor,
})

func TestAnUnpinnedSlotIsNeverCountedAsRunningOnThisBox(t *testing.T) {
	// The trap this guards is concrete and one import away: modelroute.ZoneOfRoute is
	// already in this package's dependency and maps the EMPTY string to the device rung,
	// because an unset engine route means the in-kernel default. That is right for an
	// engine route and catastrophic for a worker-model id — a fleet running entirely on
	// unpinned seat defaults would report itself 100% self-hosted.
	if modelroute.ZoneOfRoute("") != modelroute.ZoneDevice {
		t.Fatalf("premise moved: ZoneOfRoute(\"\") = %q — re-read whether this test still guards anything",
			modelroute.ZoneOfRoute(""))
	}
	z, why := AttributeZone(demoRoster, "")
	if z != "" || why != ZoneNoModelPin {
		t.Errorf("zone=%q reason=%q, want an unrecorded rung named as an unpinned slot", z, why)
	}
	if z == modelroute.ZoneDevice {
		t.Errorf("an unpinned slot was attributed to this box")
	}
}

func TestTheThreeWaysARungIsUnknownStayDistinct(t *testing.T) {
	// Each is a different missing wire with a different fix: pin the model, hand the tick a
	// roster, fix the binding. Folded into one "unknown" an operator can fix none of them.
	cases := []struct {
		name     string
		resolve  ZoneResolver
		model    string
		want     ZoneAttribution
		distinct bool
	}{
		{"nobody pinned a model", demoRoster, "", ZoneNoModelPin, true},
		{"pinned, but the tick had no roster", nil, "glm-5.2", ZoneNoRoster, true},
		{"pinned, and the roster does not bind it", demoRoster, "glm-5.3-typo", ZoneUnboundModel, true},
	}
	seen := map[ZoneAttribution]string{}
	for _, c := range cases {
		z, why := AttributeZone(c.resolve, c.model)
		if z != "" {
			t.Errorf("%s: zone=%q, want no rung claimed", c.name, z)
		}
		if why != c.want {
			t.Errorf("%s: reason=%q, want %q", c.name, why, c.want)
		}
		if prev, dup := seen[why]; dup {
			t.Errorf("%s collapses into %s — both report %q", c.name, prev, why)
		}
		seen[why] = c.name
	}
}

func TestARungTheRosterNamesIsRecordedAsGiven(t *testing.T) {
	for model, want := range map[string]modelroute.PlacementZone{
		"qwen3.6-4b": modelroute.ZoneDevice, "glm-5.2": modelroute.ZoneFleet, "claude-opus": modelroute.ZoneVendor,
	} {
		z, why := AttributeZone(demoRoster, "  "+model+"  ")
		if z != want || why != ZoneAttributed {
			t.Errorf("%s: zone=%q reason=%q, want %q attributed", model, z, why, want)
		}
	}
	// A resolver that answers with a rung outside the ladder is a bug in the caller, not a
	// new rung: it is refused rather than recorded, so no downstream fold has to defend
	// against a zone that PlacementZone.SelfHosted has never heard of.
	z, why := AttributeZone(rosterOf(map[string]modelroute.PlacementZone{"m": "laptop"}), "m")
	if z != "" || why != ZoneUnboundModel {
		t.Errorf("zone=%q reason=%q, want an off-ladder rung refused", z, why)
	}
}

func TestTheSelfHostedShareIsOverAttributedSlotsAndTheHeadlineSaysSo(t *testing.T) {
	// The number that would mislead: 10 slots ran on this laptop and 90 ran on models
	// nobody recorded. "100% self-hosted" is arithmetically correct over what can be seen
	// and reads, to an operator, as "our code stayed in the org."
	var records []WitnessRecord
	for i := 0; i < 10; i++ {
		records = append(records, WitnessRecord{Model: "qwen3.6-4b", Zone: string(modelroute.ZoneDevice)})
	}
	for i := 0; i < 90; i++ {
		records = append(records, WitnessRecord{})
	}
	s := FoldZoneShare(records)
	if s.Total != 100 || s.Attributed != 10 || s.Unattributed[ZoneNoModelPin] != 90 {
		t.Fatalf("share = %+v", s)
	}
	f, ok := s.Share()
	if !ok || f != 1 {
		t.Fatalf("share = %v ok=%v — the fraction over attributed slots is genuinely 1.0", f, ok)
	}
	h := s.Headline()
	if !strings.Contains(h, "90 of 100 slot(s) unattributed") || !strings.Contains(h, "no-model-pin 90") {
		t.Errorf("headline = %q — the fraction must carry the count that qualifies it", h)
	}
}

func TestTheHeadlineNamesTheUnattributedCountEvenWhenItIsZero(t *testing.T) {
	// A clean fleet is exactly when an operator learns to read the shape of this line. If
	// the caveat only appears when it is nonzero, its absence is the easiest thing to miss.
	s := FoldZoneShare([]WitnessRecord{
		{Model: "qwen3.6-4b", Zone: string(modelroute.ZoneDevice)},
		{Model: "glm-5.2", Zone: string(modelroute.ZoneFleet)},
		{Model: "claude-opus", Zone: string(modelroute.ZoneVendor)},
	})
	h := s.Headline()
	if !strings.Contains(h, "0 of 3 slot(s) unattributed") {
		t.Errorf("headline = %q, want the unattributed count printed at zero too", h)
	}
	if !strings.Contains(h, "self-hosted 67%") || !strings.Contains(h, "device 1") || !strings.Contains(h, "vendor 1") {
		t.Errorf("headline = %q", h)
	}
	if s.SelfHosted() != 2 {
		t.Errorf("self-hosted = %d, want the device and fleet rungs and not the vendor one", s.SelfHosted())
	}
}

func TestAFleetThatRecordedNothingReportsNoAnswerRatherThanZeroPercent(t *testing.T) {
	// 0% and "we did not record it" are opposite operator actions: the first says buy more
	// hardware, the second says fix the pinning.
	s := FoldZoneShare([]WitnessRecord{{}, {Model: "glm-5.2"}})
	if _, ok := s.Share(); ok {
		t.Errorf("a share was reported from zero attributed slots")
	}
	h := s.Headline()
	if !strings.Contains(h, "UNKNOWN") || strings.Contains(h, "0%") {
		t.Errorf("headline = %q, want no answer rather than a zero", h)
	}
	// And the two unrecorded slots stay told apart after the fact: one was never pinned,
	// the other names a model no rung came back for.
	if s.Unattributed[ZoneNoModelPin] != 1 || s.Unattributed[ZoneUnboundModel] != 1 {
		t.Errorf("unattributed = %+v", s.Unattributed)
	}
}

func TestARecordCarryingAnOffLadderRungIsNotTrusted(t *testing.T) {
	// A hand-edited or older sidecar can carry anything. An unrecognized rung is counted as
	// unattributed rather than added to ByZone, so SelfHosted never sums a rung whose
	// self-hosted-ness nothing decided.
	s := FoldZoneShare([]WitnessRecord{{Model: "m", Zone: "laptop"}, {Model: "m", Zone: "DEVICE"}})
	if s.Attributed != 0 || s.Unattributed[ZoneUnboundModel] != 2 {
		t.Errorf("share = %+v — an unrecognized rung was trusted", s)
	}
	if s.SelfHosted() != 0 {
		t.Errorf("self-hosted = %d from rungs nothing recognized", s.SelfHosted())
	}
}

func TestTheZoneSidecarKeyAppearsOnlyWhenTheRungWasResolved(t *testing.T) {
	// Same rule the model key already follows: a fleet that attributes nothing writes a
	// sidecar byte-identical to before this seam, so no reader ever parses an assumed rung.
	plain := WitnessRecord{Issue: 1, Log: "a.log", Claim: ClaimNoCommit}.Map()
	if _, ok := plain["zone"]; ok {
		t.Errorf("an unattributed slot wrote a zone key: %+v", plain)
	}
	withZone := WitnessRecord{Issue: 1, Log: "a.log", Claim: ClaimWitnessed, Zone: string(modelroute.ZoneFleet)}.Map()
	if withZone["zone"] != string(modelroute.ZoneFleet) {
		t.Errorf("zone key = %v, want the resolved rung", withZone["zone"])
	}
}

func TestAttributionAndTheSweepAgreeOnWhatCountsAsAttributed(t *testing.T) {
	// The spawn side decides a rung and the sweep side folds it back. If they disagree
	// about which values count, the headline drifts from what was actually recorded — so
	// every rung AttributeZone will emit must fold as attributed, and its refusals must not.
	for _, model := range []string{"qwen3.6-4b", "glm-5.2", "claude-opus", "", "unbound-id"} {
		z, why := AttributeZone(demoRoster, model)
		s := FoldZoneShare([]WitnessRecord{{Model: model, Zone: string(z)}})
		if (why == ZoneAttributed) != (s.Attributed == 1) {
			t.Errorf("model %q: spawn said %q, sweep folded %+v", model, why, s)
		}
	}
}

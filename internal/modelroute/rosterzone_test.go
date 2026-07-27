package modelroute

import "testing"

func zoneRoster() Roster {
	return Roster{
		Accounts: []Account{
			{ID: "laptop", Kind: KindLocal, BaseURL: "http://127.0.0.1:8080"},
			{ID: "cluster", Kind: KindFleet, BaseURL: "http://gpu-1:8000"},
			{ID: "frontier", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Bindings: []Binding{
			{Model: "qwen3.6-4b", Account: "laptop"},
			{Model: "glm-5.2", Account: "cluster"},
			{Model: "opus-5", Account: "frontier"},
			{Model: "dangling", Account: "decommissioned"},
		},
		Default: "laptop",
	}
}

func TestABoundModelReportsItsOwnAccountsRung(t *testing.T) {
	r := zoneRoster()
	for model, want := range map[string]PlacementZone{
		"qwen3.6-4b": ZoneDevice,
		"glm-5.2":    ZoneFleet,
		"opus-5":     ZoneVendor,
	} {
		got, ok := r.BoundZone(model)
		if !ok || got != want {
			t.Errorf("BoundZone(%q) = %q/%v, want %q/true", model, got, ok, want)
		}
	}
}

func TestAnUnboundModelIsUnattributedRatherThanTheDefaultAccountsRung(t *testing.T) {
	// THE POINT OF THIS FILE. The roster's default is a LOCAL account, so the obvious
	// implementation — Resolve and read Target.Zone — would report every unbound id as
	// running on this box. Assert BOTH halves: that Resolve really does answer ZoneDevice
	// here (so the divergence below is a deliberate refusal, not an accident of the
	// fixture) and that BoundZone refuses anyway.
	r := zoneRoster()
	tgt, err := r.Resolve("some-model-nobody-bound")
	if err != nil || tgt.Zone() != ZoneDevice {
		t.Fatalf("fixture drift: Resolve = %+v, err = %v; this test needs a local-default roster", tgt, err)
	}
	if z, ok := r.BoundZone("some-model-nobody-bound"); ok || z != "" {
		t.Errorf("BoundZone = %q/%v — an unplaced model inherited the default account's rung; "+
			"every typo and every unregistered pin would be counted as self-hosted", z, ok)
	}
}

func TestADanglingBindingDeclaresNoRung(t *testing.T) {
	// The binding names an account the roster does not define. Validate rejects this, but
	// attribution runs over whatever file is on disk, and the failure mode of guessing here
	// is the same over-attribution: a missing account is not evidence of a local one.
	if z, ok := zoneRoster().BoundZone("dangling"); ok || z != "" {
		t.Errorf("BoundZone = %q/%v, want unattributed", z, ok)
	}
}

func TestAnEmptyModelIdIsUnattributed(t *testing.T) {
	// A slot with no model pin has no rung to report. The trap is ZoneOfRoute's opposite
	// convention: an empty ENGINE ROUTE means the in-kernel default and is correctly
	// ZoneDevice, but an empty MODEL ID means nobody recorded what ran.
	for _, id := range []string{"", "   ", "\t\n"} {
		if z, ok := zoneRoster().BoundZone(id); ok || z != "" {
			t.Errorf("BoundZone(%q) = %q/%v, want unattributed", id, z, ok)
		}
	}
	// A roster with no default at all must still refuse rather than error-into-a-zone.
	bare := Roster{Accounts: []Account{{ID: "laptop", Kind: KindLocal}}}
	if z, ok := bare.BoundZone("anything"); ok || z != "" {
		t.Errorf("BoundZone on a default-less roster = %q/%v, want unattributed", z, ok)
	}
}

func TestAttributionAndDispatchAgreeOnEveryModelTheRosterBinds(t *testing.T) {
	// BoundZone asks a strictly narrower question than Resolve; "narrower" must mean it
	// answers about FEWER ids, never that it answers DIFFERENTLY about the same one. If the
	// two ever disagree on a bound model, the ledger describes a rung the dispatcher did
	// not use.
	r := zoneRoster()
	for _, b := range r.Bindings {
		tgt, err := r.Resolve(b.Model)
		z, ok := r.BoundZone(b.Model)
		if err != nil {
			if ok {
				t.Errorf("%q: Resolve failed (%v) but BoundZone answered %q", b.Model, err, z)
			}
			continue
		}
		if !ok || z != tgt.Zone() {
			t.Errorf("%q: BoundZone = %q/%v, Resolve dispatches to %q", b.Model, z, ok, tgt.Zone())
		}
	}
}

func TestAModelBoundToTheDefaultAccountIsStillAttributed(t *testing.T) {
	// The narrowing must key on whether a BINDING exists, not on whether the resolved
	// account happens to equal the default. Otherwise declaring a binding to your default
	// account — the ordinary way to be explicit — would silently stop being counted.
	r := zoneRoster()
	r.Bindings = append(r.Bindings, Binding{Model: "explicitly-laptop", Account: "laptop"})
	if z, ok := r.BoundZone("explicitly-laptop"); !ok || z != ZoneDevice {
		t.Errorf("BoundZone = %q/%v, want device/true", z, ok)
	}
}

func TestEveryZoneBoundZoneCanEmitIsOneTheSelfHostedFoldKnows(t *testing.T) {
	// Totality against the consumer: a zone value that Valid() rejects would be dropped by
	// the attribution layer as "unbound model", turning a correctly-configured account into
	// a silent attribution hole.
	r := Roster{}
	for i, kind := range []ProviderKind{KindLocal, KindFleet, KindAnthropic, KindOpenAI, KindOpenAIResponses, KindGemini, KindXAI, KindDeepSeek} {
		id := string(kind)
		r.Accounts = append(r.Accounts, Account{ID: id, Kind: kind})
		r.Bindings = append(r.Bindings, Binding{Model: id + "-model", Account: id})
		z, ok := r.BoundZone(id + "-model")
		if !ok || !z.Valid() {
			t.Errorf("kind %d (%q): BoundZone = %q/%v, not a valid zone", i, kind, z, ok)
		}
		if z.SelfHosted() != (kind == KindLocal || kind == KindFleet) {
			t.Errorf("kind %q attributed as self-hosted=%v", kind, z.SelfHosted())
		}
	}
}

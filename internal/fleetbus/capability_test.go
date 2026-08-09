package fleetbus

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func capInstance(t *testing.T, id, machine string, models []string, zone string) Instance {
	t.Helper()
	inst, r := NewInstance(id, machine, "serve", 4242, "", []Op{"steer"}, testNow)
	if r != nil {
		t.Fatalf("NewInstance(%q): %v", id, r)
	}
	return inst.WithServedModels(models).WithZone(zone)
}

// legacyInstanceJSON is a presence record exactly as a binary from BEFORE the
// capability field wrote it — no models, no zone. It is spelled as bytes rather than
// built with a struct literal on purpose: a struct literal would silently gain any
// field added later, which is the one thing this arm exists to catch.
const legacyInstanceJSON = `{"schema":"fak.fleetbus.instance/v1","id":"legacy-1",` +
	`"machine":"box-c","role":"serve","pid":9,"ops":["steer"],"seen_utc":"2026-08-05T12:00:00Z"}`

// TestInstanceModelSelector is #5637's witness. The bus roster could name WHO is live
// and WHAT KIND of process it is, but never WHICH MODEL IT SERVES — so a selector
// could address "all serve instances" and never "every instance serving model X", and
// every model-scoped operator action degraded to fan-to-all plus N refusals.
//
// The three arms are the issue's done condition: an instance fronting two models
// declares both, a model-scoped selector addresses exactly the instances declaring
// that model, and a record written before the field still parses, still sits in the
// roster, and still counts in the fold's denominator.
func TestInstanceModelSelector(t *testing.T) {
	glm := capInstance(t, "serve-1", "box-a", []string{"glm-4.6", "zai-org/GLM-4.6"}, "us-west")
	opus := capInstance(t, "serve-2", "box-b", []string{"claude-opus-5"}, "us-east")
	untagged := capInstance(t, "serve-3", "box-c", nil, "")
	roster := []Instance{glm, opus, untagged}

	t.Run("an instance fronting two models declares both", func(t *testing.T) {
		// Normalized on the way in — deduped, trimmed and sorted — so two announces of
		// the same capability are byte-identical and a diffed roster does not churn on
		// the order a config file happened to list.
		got := capInstance(t, "serve-9", "box-a", []string{" glm-4.6 ", "aardvark", "glm-4.6", ""}, " us-west ")
		want := []string{"aardvark", "glm-4.6"}
		if len(got.Models) != len(want) {
			t.Fatalf("Models = %v, want %v", got.Models, want)
		}
		for i := range want {
			if got.Models[i] != want[i] {
				t.Fatalf("Models = %v, want %v", got.Models, want)
			}
		}
		if got.Zone != "us-west" {
			t.Fatalf("Zone = %q, want %q", got.Zone, "us-west")
		}
	})

	t.Run("an instance fronting two models announces both", func(t *testing.T) {
		// Through the REAL transport, not just the struct: the done condition is that
		// a serve fronting A and B ANNOUNCES both, so the claim has to survive the
		// announce/read round trip that every other presence field survives.
		b := testBus(t)
		if err := b.Announce(glm); err != nil {
			t.Fatalf("Announce: %v", err)
		}
		roster, err := b.Instances(testNow, DefaultInstanceTTL)
		if err != nil {
			t.Fatalf("Instances: %v", err)
		}
		if len(roster) != 1 {
			t.Fatalf("roster = %+v, want one instance", roster)
		}
		if got := roster[0].Models; len(got) != 2 || got[0] != "glm-4.6" || got[1] != "zai-org/GLM-4.6" {
			t.Fatalf("announced models = %v, want both declared models back", got)
		}
		if roster[0].Zone != "us-west" {
			t.Fatalf("announced zone = %q, want us-west", roster[0].Zone)
		}
		// And the roster read back off disk is what a model selector resolves against —
		// which is the whole point: "which model is served where" is now answerable
		// from the bus, not just from a struct in memory.
		if got := PublishTargets(Selector{Model: []string{"zai-org/GLM-4.6"}}, roster); len(got) != 1 || got[0].ID != "serve-1" {
			t.Fatalf("PublishTargets over the announced roster = %+v, want serve-1", got)
		}
	})

	t.Run("a model selector addresses exactly the instances declaring it", func(t *testing.T) {
		cases := []struct {
			name string
			sel  Selector
			want []string
		}{
			{"one model names its one host", Selector{Model: []string{"claude-opus-5"}}, []string{"serve-2"}},
			{"either alias of the same host matches it once", Selector{Model: []string{"glm-4.6"}}, []string{"serve-1"}},
			{"within an axis the members are OR'd", Selector{Model: []string{"glm-4.6", "claude-opus-5"}}, []string{"serve-1", "serve-2"}},
			{"across axes they are AND'd", Selector{Model: []string{"glm-4.6"}, Machine: []string{"box-b"}}, nil},
			{"zone is its own axis", Selector{Zone: []string{"us-east"}}, []string{"serve-2"}},
			// The fail-closed arm, and the reason this axis cannot widen anything: an
			// instance that declares no models is not "compatible with every model", it
			// is unanswered. Role has always worked this way for an untagged record.
			{"an untagged instance is never swept up by a model axis", Selector{Model: []string{"glm-4.6", "claude-opus-5"}}, []string{"serve-1", "serve-2"}},
			// The acceptance gate: this resolving to zero is what makes the publish edge
			// refuse FLEETBUS_NO_TARGET instead of accepting a directive into a fleet
			// where nobody can apply it and leaving it outstanding forever.
			{"a model nobody serves resolves to no target", Selector{Model: []string{"gpt-9"}}, nil},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if r := tc.sel.Validate(); r != nil {
					t.Fatalf("Validate(%s) refused: %v", tc.sel, r)
				}
				var got []string
				for _, inst := range PublishTargets(tc.sel, roster) {
					got = append(got, inst.ID)
				}
				if len(got) != len(tc.want) {
					t.Fatalf("PublishTargets(%s) = %v, want %v", tc.sel, got, tc.want)
				}
				for i := range tc.want {
					if got[i] != tc.want[i] {
						t.Fatalf("PublishTargets(%s) = %v, want %v", tc.sel, got, tc.want)
					}
				}
			})
		}
	})

	t.Run("a model axis addresses somebody, but never alongside all", func(t *testing.T) {
		// A capability axis is a real address — "everything serving X" is a set the
		// operator stated — so it satisfies AddressesInstances. It is still a FILTER,
		// so pairing it with the affirmative --all is the same contradiction --role
		// already refuses.
		if r := (Selector{Model: []string{"glm-4.6"}}).Validate(); r != nil {
			t.Fatalf("a model axis alone should address somebody: %v", r)
		}
		if r := (Selector{Zone: []string{"us-west"}}).Validate(); r != nil {
			t.Fatalf("a zone axis alone should address somebody: %v", r)
		}
		for _, sel := range []Selector{
			{All: true, Model: []string{"glm-4.6"}},
			{All: true, Zone: []string{"us-west"}},
		} {
			r := sel.Validate()
			if r == nil {
				t.Fatalf("Validate(%s) accepted --all plus a filter", sel)
			}
			if r.Reason != Malformed {
				t.Fatalf("reason = %q, want %q", r.Reason, Malformed)
			}
		}
	})

	t.Run("an old record parses, stays in the roster, and still counts", func(t *testing.T) {
		// Written straight into the bus as BYTES, the way a binary that predates the
		// field would have left it there — so this arm exercises the real read path
		// (DirBus.Instances), not just encoding/json against the current struct.
		b := testBus(t)
		if err := b.Announce(glm); err != nil {
			t.Fatalf("Announce: %v", err)
		}
		legacyPath := filepath.Join(b.Root, instancesDir, "legacy-1.json")
		if err := os.WriteFile(legacyPath, []byte(legacyInstanceJSON), 0o644); err != nil {
			t.Fatalf("seed a pre-field record: %v", err)
		}
		withLegacy, err := b.Instances(testNow, DefaultInstanceTTL)
		if err != nil {
			t.Fatalf("a pre-field record must not break the roster read: %v", err)
		}
		if len(withLegacy) != 2 {
			t.Fatalf("roster = %+v, want the old record alongside the new one", withLegacy)
		}
		var legacy Instance
		for _, inst := range withLegacy {
			if inst.ID == "legacy-1" {
				legacy = inst
			}
		}
		if legacy.ID == "" {
			t.Fatalf("the pre-field record dropped out of the roster: %+v", withLegacy)
		}
		if legacy.Models != nil || legacy.Zone != "" {
			t.Fatalf("legacy record invented a capability: models=%v zone=%q", legacy.Models, legacy.Zone)
		}

		// It matches every selector it matched before the field existed...
		byRole := Selector{Role: []string{"serve"}}
		if got := PublishTargets(byRole, withLegacy); len(got) != 2 {
			t.Fatalf("PublishTargets(%s) = %d instance(s), want 2 — the old record left the roster", byRole, len(got))
		}
		// ...and is invisible ONLY to the axis it never answered.
		if got := PublishTargets(Selector{Model: []string{"glm-4.6"}}, withLegacy); len(got) != 1 || got[0].ID != "serve-1" {
			t.Fatalf("a model axis matched the untagged old record: %+v", got)
		}

		// ...and it is still in the DENOMINATOR. This is the arm that matters: fold.go
		// counts a targeted-but-silent instance as OUTSTANDING, and a record that
		// quietly stopped matching would leave the count instead — turning "one
		// instance never answered" into Complete, Applied==Targeted, exit 0.
		d, r := NewDirective("op-1", "pause", "", byRole, time.Minute, "", testNow)
		if r != nil {
			t.Fatalf("NewDirective: %v", r)
		}
		d = d.WithTargets(withLegacy)
		rep := Fold(d, withLegacy, nil, testNow.Add(time.Second))
		if rep.Targeted != 2 || rep.Outstanding != 2 {
			t.Fatalf("Fold: targeted=%d outstanding=%d, want 2/2 (rows %+v)", rep.Targeted, rep.Outstanding, rep.Rows)
		}
		if rep.Complete {
			t.Fatal("Fold reported Complete with nobody having acked")
		}

		// And declaring nothing is byte-identical to predating the field, so an
		// untagged instance on a NEW binary is not a third case a reader must know.
		blob, err := json.Marshal(untagged)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		var back map[string]any
		if err := json.Unmarshal(blob, &back); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		if _, ok := back["models"]; ok {
			t.Fatalf("an instance declaring nothing wrote a models key: %s", blob)
		}
		if _, ok := back["zone"]; ok {
			t.Fatalf("an instance declaring nothing wrote a zone key: %s", blob)
		}
	})

	t.Run("the stated axes are readable in a report", func(t *testing.T) {
		got := Selector{Model: []string{"glm-4.6"}, Zone: []string{"us-west"}}.String()
		if want := "model=glm-4.6 zone=us-west"; got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
}

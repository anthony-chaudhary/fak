package trajectory

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestResolveVisualProfileNegotiatesExplicitFallbacks(t *testing.T) {
	profile := VisualProfile{Schema: VisualProfileSchema, Theme: map[string]string{"accent": "blue"}, Density: "compact", Layout: []string{"status", "timeline", "controls"}, KeyActions: map[string]string{"approve": "a"}, NotificationChannels: []string{"desktop", "sound"}}
	capabilities := ViewTargetFeatures{Name: "plain-text", SemanticLabels: true, NotificationChannels: []string{"sound"}}
	resolved, err := ResolveVisualProfile(profile, capabilities)
	if err != nil {
		t.Fatal(err)
	}
	if !resolved.NoColor || !resolved.ReducedMotion || resolved.Theme != nil || resolved.KeyActions != nil {
		t.Fatalf("fallbacks not applied: %#v", resolved)
	}
	want := []string{"keys:unavailable", "motion:reduced", "notifications:unsupported-removed", "theme:target-default"}
	if !reflect.DeepEqual(resolved.Fallbacks, want) {
		t.Fatalf("fallbacks=%v, want %v", resolved.Fallbacks, want)
	}
	if !reflect.DeepEqual(resolved.NotificationChannels, []string{"sound"}) {
		t.Fatalf("notifications=%v", resolved.NotificationChannels)
	}
}

func TestResolveVisualProfileRequiresSemanticAccessibility(t *testing.T) {
	profile := VisualProfile{Schema: VisualProfileSchema, NoColor: true, Density: "comfortable", Layout: []string{"timeline"}, ScreenReaderLabels: true, ReducedMotion: true}
	if _, err := ResolveVisualProfile(profile, ViewTargetFeatures{Name: "decorative-only"}); err == nil {
		t.Fatal("inaccessible target accepted")
	}
}

func TestVisualChoicesDoNotMutateProjectedSemantics(t *testing.T) {
	events := []Event{derivedFixture("message", EventMessage, "completed", fixedVisualTime(), 1, `{"role":"assistant","text":"done"}`)}
	projected, _, err := CompileView(events, EndUserView())
	if err != nil {
		t.Fatal(err)
	}
	before, _ := json.Marshal(projected)
	profiles := []VisualProfile{
		{Schema: VisualProfileSchema, Theme: map[string]string{"accent": "blue"}, Density: "compact", Layout: []string{"timeline"}, ScreenReaderLabels: true, ReducedMotion: true},
		{Schema: VisualProfileSchema, NoColor: true, Density: "spacious", Layout: []string{"timeline", "status"}, ScreenReaderLabels: true, ReducedMotion: true},
	}
	for _, profile := range profiles {
		if _, err := ResolveVisualProfile(profile, ViewTargetFeatures{Name: "semantic", Themes: true, Color: true, SemanticLabels: true}); err != nil {
			t.Fatal(err)
		}
	}
	after, _ := json.Marshal(projected)
	if string(before) != string(after) {
		t.Fatal("visual negotiation mutated projected semantics")
	}
}

func TestVisualProfileRejectsInvalidLayoutAndTheme(t *testing.T) {
	for _, profile := range []VisualProfile{
		{Schema: VisualProfileSchema, Density: "dense", NoColor: true, Layout: []string{"timeline"}},
		{Schema: VisualProfileSchema, Density: "compact", NoColor: true, Layout: []string{"timeline", "timeline"}},
		{Schema: VisualProfileSchema, Density: "compact", Layout: []string{"timeline"}},
	} {
		if err := profile.Validate(); err == nil {
			t.Fatalf("invalid profile accepted: %#v", profile)
		}
	}
}

func fixedVisualTime() time.Time { return time.Date(2026, 8, 17, 18, 0, 0, 0, time.UTC) }

package trajectory

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const VisualProfileSchema = "fak-trajectory-visual/1alpha1"

type VisualProfile struct {
	Schema               string            `json:"schema"`
	Theme                map[string]string `json:"theme"`
	Density              string            `json:"density"`
	Layout               []string          `json:"layout"`
	KeyActions           map[string]string `json:"key_actions,omitempty"`
	ReducedMotion        bool              `json:"reduced_motion"`
	NoColor              bool              `json:"no_color"`
	ScreenReaderLabels   bool              `json:"screen_reader_labels"`
	NotificationChannels []string          `json:"notification_channels,omitempty"`
}

type ViewTargetFeatures struct {
	Name                 string   `json:"name"`
	Themes               bool     `json:"themes"`
	Color                bool     `json:"color"`
	Motion               bool     `json:"motion"`
	KeyActions           bool     `json:"key_actions"`
	SemanticLabels       bool     `json:"semantic_labels"`
	NotificationChannels []string `json:"notification_channels,omitempty"`
}

type ResolvedVisualProfile struct {
	Schema               string            `json:"schema"`
	Target               string            `json:"target"`
	Theme                map[string]string `json:"theme"`
	Density              string            `json:"density"`
	Layout               []string          `json:"layout"`
	KeyActions           map[string]string `json:"key_actions,omitempty"`
	ReducedMotion        bool              `json:"reduced_motion"`
	NoColor              bool              `json:"no_color"`
	ScreenReaderLabels   bool              `json:"screen_reader_labels"`
	NotificationChannels []string          `json:"notification_channels,omitempty"`
	Fallbacks            []string          `json:"fallbacks,omitempty"`
}

func (profile VisualProfile) Validate() error {
	if profile.Schema != VisualProfileSchema {
		return fmt.Errorf("visual profile schema %q, want %q", profile.Schema, VisualProfileSchema)
	}
	if profile.Density != "compact" && profile.Density != "comfortable" && profile.Density != "spacious" {
		return fmt.Errorf("unsupported visual density %q", profile.Density)
	}
	if len(profile.Layout) == 0 {
		return errors.New("visual profile requires layout regions")
	}
	seen := map[string]struct{}{}
	for _, region := range profile.Layout {
		region = strings.TrimSpace(region)
		if region == "" {
			return errors.New("visual profile has empty layout region")
		}
		if _, ok := seen[region]; ok {
			return fmt.Errorf("duplicate layout region %q", region)
		}
		seen[region] = struct{}{}
	}
	if !profile.NoColor && len(profile.Theme) == 0 {
		return errors.New("colored visual profile requires theme tokens")
	}
	return nil
}

// ResolveVisualProfile negotiates presentation capabilities without touching projected evidence.
func ResolveVisualProfile(profile VisualProfile, target ViewTargetFeatures) (ResolvedVisualProfile, error) {
	if err := profile.Validate(); err != nil {
		return ResolvedVisualProfile{}, err
	}
	if strings.TrimSpace(target.Name) == "" {
		return ResolvedVisualProfile{}, errors.New("target requires name")
	}
	resolved := ResolvedVisualProfile{Schema: profile.Schema, Target: target.Name, Theme: cloneStrings(profile.Theme), Density: profile.Density, Layout: append([]string(nil), profile.Layout...), KeyActions: cloneStrings(profile.KeyActions), ReducedMotion: profile.ReducedMotion, NoColor: profile.NoColor, ScreenReaderLabels: profile.ScreenReaderLabels, NotificationChannels: intersectStrings(profile.NotificationChannels, target.NotificationChannels)}
	if !target.Themes {
		resolved.Theme = nil
		resolved.NoColor = true
		resolved.Fallbacks = append(resolved.Fallbacks, "theme:target-default")
	}
	if !target.Color && !resolved.NoColor {
		resolved.NoColor = true
		resolved.Theme = nil
		resolved.Fallbacks = append(resolved.Fallbacks, "color:no-color")
	}
	if !target.Motion && !resolved.ReducedMotion {
		resolved.ReducedMotion = true
		resolved.Fallbacks = append(resolved.Fallbacks, "motion:reduced")
	}
	if !target.KeyActions && len(resolved.KeyActions) > 0 {
		resolved.KeyActions = nil
		resolved.Fallbacks = append(resolved.Fallbacks, "keys:unavailable")
	}
	if resolved.ScreenReaderLabels && !target.SemanticLabels {
		return ResolvedVisualProfile{}, errors.New("target cannot satisfy required screen-reader labels")
	}
	if len(resolved.NotificationChannels) != len(profile.NotificationChannels) {
		resolved.Fallbacks = append(resolved.Fallbacks, "notifications:unsupported-removed")
	}
	sort.Strings(resolved.Fallbacks)
	return resolved, nil
}

func cloneStrings(input map[string]string) map[string]string {
	if len(input) == 0 {
		return nil
	}
	output := make(map[string]string, len(input))
	for key, value := range input {
		output[key] = value
	}
	return output
}
func intersectStrings(wanted, supported []string) []string {
	available := map[string]struct{}{}
	for _, value := range supported {
		available[value] = struct{}{}
	}
	var output []string
	for _, value := range wanted {
		if _, ok := available[value]; ok {
			output = append(output, value)
		}
	}
	return output
}

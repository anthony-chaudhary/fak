package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

type Audit struct {
	Duration float64         `json:"durationSecs"`
	Scenes   int             `json:"scenes"`
	MinType  int             `json:"minTypePx"`
	CTAStart float64         `json:"ctaStartSecs"`
	CTAHold  float64         `json:"ctaHoldSecs"`
	MaxWords int             `json:"maxWordsPerTextRegion"`
	Checks   map[string]bool `json:"checks"`
}

func audit(c Config) (Audit, error) {
	a := Audit{Scenes: len(c.Scenes), MinType: 48, Checks: map[string]bool{}}
	if c.Width < 1280 || c.Height < 720 {
		return a, fmt.Errorf("canvas %dx%d below 1280x720", c.Width, c.Height)
	}
	if c.FPS < 24 {
		return a, fmt.Errorf("fps %d below cinematic motion floor 24", c.FPS)
	}
	if len(c.Scenes) < 3 || len(c.Scenes) > 6 {
		return a, fmt.Errorf("scene count %d outside focused trailer range 3..6", len(c.Scenes))
	}
	ctaCount := 0
	at := 0.0
	maxWords := 0
	allowed := map[string]bool{"hook": true, "checkpoint": true, "proof": true, "cta": true}
	for i, s := range c.Scenes {
		if !allowed[s.Kind] {
			return a, fmt.Errorf("scene %d: unknown kind %q", i, s.Kind)
		}
		if s.Secs < 2 {
			return a, fmt.Errorf("scene %d: %.1fs is too brief to read", i, s.Secs)
		}
		for _, v := range []string{s.Eyebrow, s.Title, s.Subtitle, s.Action, s.Detail, s.Verdict} {
			n := len(strings.Fields(v))
			if n > maxWords {
				maxWords = n
			}
			if n > 8 {
				return a, fmt.Errorf("scene %d: text region %q has %d words; max 8", i, v, n)
			}
		}
		if s.Kind == "cta" {
			ctaCount++
			a.CTAStart = at
			a.CTAHold = s.Secs
			if s.Command == "" {
				return a, fmt.Errorf("scene %d: CTA needs command", i)
			}
		}
		at += s.Secs
	}
	a.Duration = math.Round(at*10) / 10
	a.MaxWords = maxWords
	if at < 18 || at > 30 {
		return a, fmt.Errorf("duration %.1fs outside trailer range 18..30s", at)
	}
	if ctaCount != 1 {
		return a, fmt.Errorf("need exactly one CTA scene, got %d", ctaCount)
	}
	if a.CTAStart > 12 {
		return a, fmt.Errorf("CTA starts at %.1fs; must appear by 12s", a.CTAStart)
	}
	if a.CTAHold < 5 {
		return a, fmt.Errorf("CTA hold %.1fs below copyability floor 5s", a.CTAHold)
	}
	a.Checks = map[string]bool{"duration_18_30s": true, "scene_count_3_6": true, "max_8_words_per_region": true, "cta_by_12s": true, "cta_hold_5s": true, "canvas_720p_plus": true, "motion_24fps_plus": true}
	return a, nil
}
func writeAudit(path string, a Audit) {
	if path == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	b, _ := json.MarshalIndent(a, "", "  ")
	_ = os.WriteFile(path, append(b, '\n'), 0644)
}

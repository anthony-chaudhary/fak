package launchshim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const StatsSchema = "fak.launch-stats.v1"

type Counter struct {
	Surface  string `json:"surface"`
	Provider string `json:"provider"`
	Posture  string `json:"posture"`
	Outcome  string `json:"outcome"`
	Latency  string `json:"latency_bucket"`
	Count    uint64 `json:"count"`
}

type Stats struct {
	Schema   string    `json:"schema"`
	Counters []Counter `json:"counters"`
}

func StatsPath() (string, error) {
	if p := strings.TrimSpace(os.Getenv("FAK_LAUNCH_STATS")); p != "" {
		return p, nil
	}
	p, err := Path()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(p), "launch-stats.json"), nil
}

func Record(surface, provider, posture, outcome string, elapsed time.Duration) error {
	p, err := StatsPath()
	if err != nil {
		return err
	}
	if !oneOf(surface, "shim", "bare", "explicit") || !oneOf(posture, "guarded", "direct") || !oneOf(outcome, "success", "provider_exit", "launch_error") {
		return errors.New("invalid launch counter dimension")
	}
	if provider != "claude" && provider != "codex" {
		provider = "custom"
	}
	latency := "ge_10s"
	switch {
	case elapsed < time.Second:
		latency = "lt_1s"
	case elapsed < 10*time.Second:
		latency = "1s_10s"
	}
	stats, _ := ReadStats()
	key := surface + "\x00" + provider + "\x00" + posture + "\x00" + outcome + "\x00" + latency
	found := false
	for i := range stats.Counters {
		c := &stats.Counters[i]
		if counterKey(*c) == key {
			c.Count++
			found = true
			break
		}
	}
	if !found {
		stats.Counters = append(stats.Counters, Counter{surface, provider, posture, outcome, latency, 1})
	}
	sort.Slice(stats.Counters, func(i, j int) bool { return counterKey(stats.Counters[i]) < counterKey(stats.Counters[j]) })
	return writeStats(p, stats)
}

func ReadStats() (Stats, error) {
	p, err := StatsPath()
	if err != nil {
		return Stats{}, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return Stats{Schema: StatsSchema}, nil
	}
	if err != nil {
		return Stats{}, err
	}
	var s Stats
	if err := json.Unmarshal(b, &s); err != nil {
		return Stats{}, err
	}
	if s.Schema != StatsSchema {
		return Stats{}, fmt.Errorf("launch stats schema %q", s.Schema)
	}
	return s, nil
}
func ResetStats() error {
	p, err := StatsPath()
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}
func writeStats(p string, s Stats) error {
	s.Schema = StatsSchema
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(tmp, p)
}
func counterKey(c Counter) string {
	return c.Surface + "\x00" + c.Provider + "\x00" + c.Posture + "\x00" + c.Outcome + "\x00" + c.Latency
}
func oneOf(v string, all ...string) bool {
	for _, x := range all {
		if v == x {
			return true
		}
	}
	return false
}

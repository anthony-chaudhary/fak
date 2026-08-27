// Package localappcert validates the v1 Apple-Silicon certification matrix.
// A supported envelope is publishable only when every required lifecycle scenario
// carries an external artifact and a fak-native, fallback-free runtime receipt.
package localappcert

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

const Schema = "fak.local-app-certification/1"

type Status string

const (
	StatusPass        Status = "PASS"
	StatusUnsupported Status = "UNSUPPORTED"
)

var RequiredScenarios = []string{
	"clean-install", "cold-start", "warm-start", "long-context",
	"concurrent-foreground-load", "cancel-restart", "memory-pressure",
	"low-power-thermal-downshift", "offline-start", "interrupted-update",
	"canary-reject", "rollback", "uninstall-cleanup",
}

type Receipt struct {
	Engine                string  `json:"engine"`
	Fallback              string  `json:"fallback"`
	Artifact              string  `json:"artifact"`
	RuntimeRevision       string  `json:"runtime_revision"`
	TTFTMilliseconds      float64 `json:"ttft_milliseconds,omitempty"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second,omitempty"`
	PeakMemoryBytes       int64   `json:"peak_memory_bytes,omitempty"`
	DiskBytes             int64   `json:"disk_bytes,omitempty"`
}

type Scenario struct {
	Name     string   `json:"name"`
	Status   Status   `json:"status"`
	Evidence string   `json:"evidence,omitempty"`
	Reason   string   `json:"reason,omitempty"`
	Receipt  *Receipt `json:"receipt,omitempty"`
}

type Envelope struct {
	ID              string     `json:"id"`
	Chip            string     `json:"chip"`
	MemoryBytes     int64      `json:"memory_bytes"`
	MacOS           string     `json:"macos"`
	Power           string     `json:"power"`
	Thermal         string     `json:"thermal"`
	DiskFreeBytes   int64      `json:"disk_free_bytes"`
	PackRevision    string     `json:"pack_revision"`
	RuntimeRevision string     `json:"runtime_revision"`
	Supported       bool       `json:"supported"`
	Reason          string     `json:"reason,omitempty"`
	Scenarios       []Scenario `json:"scenarios,omitempty"`
}

type Matrix struct {
	Schema      string     `json:"schema"`
	GeneratedAt string     `json:"generated_at"`
	Envelopes   []Envelope `json:"envelopes"`
}

func Load(path string) (Matrix, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Matrix{}, err
	}
	var m Matrix
	if err := json.Unmarshal(b, &m); err != nil {
		return Matrix{}, err
	}
	return m, nil
}

func Validate(m Matrix) error {
	if m.Schema != Schema {
		return fmt.Errorf("localappcert: schema %q, want %q", m.Schema, Schema)
	}
	if len(m.Envelopes) == 0 {
		return errors.New("localappcert: matrix has no envelopes")
	}
	ids := map[string]bool{}
	supported := 0
	for _, e := range m.Envelopes {
		if strings.TrimSpace(e.ID) == "" || ids[e.ID] {
			return fmt.Errorf("localappcert: missing or duplicate envelope id %q", e.ID)
		}
		ids[e.ID] = true
		if e.Chip == "" || e.MemoryBytes <= 0 || e.MacOS == "" || e.Power == "" || e.Thermal == "" || e.PackRevision == "" || e.RuntimeRevision == "" {
			return fmt.Errorf("localappcert: envelope %s has incomplete identity", e.ID)
		}
		if !e.Supported {
			if strings.TrimSpace(e.Reason) == "" {
				return fmt.Errorf("localappcert: unsupported envelope %s has no typed reason", e.ID)
			}
			if len(e.Scenarios) != 0 {
				return fmt.Errorf("localappcert: unsupported envelope %s must not carry passing scenarios", e.ID)
			}
			continue
		}
		supported++
		if e.Reason != "" {
			return fmt.Errorf("localappcert: supported envelope %s carries a rejection reason", e.ID)
		}
		got := map[string]Scenario{}
		for _, s := range e.Scenarios {
			if _, ok := got[s.Name]; ok {
				return fmt.Errorf("localappcert: envelope %s duplicates scenario %s", e.ID, s.Name)
			}
			got[s.Name] = s
		}
		for _, name := range RequiredScenarios {
			s, ok := got[name]
			if !ok {
				return fmt.Errorf("localappcert: envelope %s missing scenario %s", e.ID, name)
			}
			if s.Status != StatusPass {
				return fmt.Errorf("localappcert: supported envelope %s scenario %s is %s", e.ID, name, s.Status)
			}
			if strings.TrimSpace(s.Evidence) == "" {
				return fmt.Errorf("localappcert: envelope %s scenario %s has no evidence", e.ID, name)
			}
			if s.Receipt == nil || s.Receipt.Engine != "fak-native" || s.Receipt.Fallback != "none" || s.Receipt.Artifact == "" || s.Receipt.RuntimeRevision != e.RuntimeRevision {
				return fmt.Errorf("localappcert: envelope %s scenario %s lacks exact fak-native fallback-free receipt", e.ID, name)
			}
		}
		if len(got) != len(RequiredScenarios) {
			extra := make([]string, 0)
			required := map[string]bool{}
			for _, n := range RequiredScenarios {
				required[n] = true
			}
			for n := range got {
				if !required[n] {
					extra = append(extra, n)
				}
			}
			sort.Strings(extra)
			return fmt.Errorf("localappcert: envelope %s has unknown scenarios %v", e.ID, extra)
		}
	}
	if supported == 0 {
		return errors.New("localappcert: matrix certifies no supported envelope")
	}
	return nil
}

package qwen4exp

import (
	"errors"
	"sort"
	"strconv"
)

const FrontierSchema = "fak.qwen4exp.frontier/1"

type FrontierCell struct {
	Cohort        string  `json:"cohort"`
	Arm           string  `json:"arm"`
	Backend       string  `json:"backend"`
	Artifact      string  `json:"artifact"`
	DType         string  `json:"dtype"`
	Hardware      string  `json:"hardware"`
	PromptSet     string  `json:"prompt_set"`
	Context       int     `json:"context"`
	Batch         int     `json:"batch"`
	Concurrency   int     `json:"concurrency"`
	Engine        string  `json:"engine"`
	Fallback      string  `json:"fallback"`
	Quality       bool    `json:"quality"`
	Supported     bool    `json:"supported"`
	Reason        string  `json:"reason,omitempty"`
	LoadMS        int64   `json:"load_ms"`
	TTFTMS        int64   `json:"ttft_ms"`
	PrefillTPS    float64 `json:"prefill_tps"`
	DecodeTPS     float64 `json:"decode_tps"`
	PeakBytes     int64   `json:"peak_bytes"`
	TrafficBytes  int64   `json:"traffic_bytes"`
	EnergyJoules  float64 `json:"energy_joules"`
	RecoveryMS    int64   `json:"recovery_ms"`
	MTPOverheadMS int64   `json:"mtp_overhead_ms"`
}
type Frontier struct {
	Schema string         `json:"schema"`
	Cells  []FrontierCell `json:"cells"`
}

func (f Frontier) Validate() error {
	if f.Schema != FrontierSchema {
		return errors.New("qwen4exp frontier: invalid schema")
	}
	if len(f.Cells) == 0 {
		return errors.New("qwen4exp frontier: no cells")
	}
	seen := map[string]bool{}
	baselines := map[string]*FrontierCell{}
	for i := range f.Cells {
		c := &f.Cells[i]
		key := c.Cohort + "\x00" + c.Arm + "\x00" + c.Backend + "\x00" + c.Hardware + "\x00" + c.PromptSet + "\x00" + strconv.Itoa(c.Context) + "\x00" + strconv.Itoa(c.Batch) + "\x00" + strconv.Itoa(c.Concurrency)
		if seen[key] {
			return errors.New("qwen4exp frontier: duplicate cell")
		}
		seen[key] = true
		if !c.Supported {
			if c.Reason == "" {
				return errors.New("qwen4exp frontier: unsupported cell lacks reason")
			}
			continue
		}
		if c.Engine == "" || c.Fallback == "" || c.Artifact == "" || c.DType == "" || c.LoadMS < 0 || c.TTFTMS < 0 || c.PrefillTPS <= 0 || c.DecodeTPS <= 0 || c.PeakBytes <= 0 || c.TrafficBytes < 0 || c.EnergyJoules < 0 || c.RecoveryMS < 0 || c.MTPOverheadMS < 0 {
			return errors.New("qwen4exp frontier: incomplete supported cell")
		}
		if c.Engine == "fak-native" && c.Fallback != "none" {
			return errors.New("qwen4exp frontier: native cell used fallback")
		}
		if c.Quality {
			group := c.Cohort + "\x00" + c.Backend
			base := baselines[group]
			if base == nil {
				baselines[group] = c
			} else if c.Artifact != base.Artifact || c.DType != base.DType || c.Hardware != base.Hardware || c.PromptSet != base.PromptSet || c.Context != base.Context || c.Batch != base.Batch || c.Concurrency != base.Concurrency {
				return errors.New("qwen4exp frontier: unmatched ranked cells")
			}
		}
	}
	return nil
}
func (f Frontier) Ranked() ([]FrontierCell, error) {
	if err := f.Validate(); err != nil {
		return nil, err
	}
	out := []FrontierCell{}
	for _, c := range f.Cells {
		if c.Supported && c.Quality {
			out = append(out, c)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].DecodeTPS > out[j].DecodeTPS })
	return out, nil
}

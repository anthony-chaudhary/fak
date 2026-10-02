// Package amdgpuclock decodes AMDGPU clock telemetry and selects explicit DPM states.
// It performs no I/O and owns the shared public/private telemetry semantics.
package amdgpuclock

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

type DPMSCLKState struct {
	Index   int  `json:"index"`
	FreqMHz int  `json:"freq_mhz"`
	Active  bool `json:"active"`
}

type DPMMCLKState struct {
	Index   int  `json:"index"`
	FreqMHz int  `json:"freq_mhz"`
	SpeedMT int  `json:"speed_mt,omitempty"`
	Active  bool `json:"active"`
}

func ParseDPMSCLK(raw string) ([]DPMSCLKState, int, int, error) {
	trimmedRaw := strings.TrimSpace(raw)
	if trimmedRaw == "" || trimmedRaw == "none" || trimmedRaw == "unknown" {
		return nil, -1, 0, errors.New("empty or absent pp_dpm_sclk telemetry")
	}

	lines := strings.Split(trimmedRaw, "\n")
	var states []DPMSCLKState
	activeIdx := -1
	activeFreq := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		isActive := strings.Contains(trimmed, "*")
		clean := strings.ReplaceAll(trimmed, "*", "")
		clean = strings.TrimSpace(clean)

		parts := strings.SplitN(clean, ":", 2)
		if len(parts) != 2 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}

		freqStr := strings.TrimSpace(parts[1])
		freqStr = strings.TrimRight(freqStr, "MhzMHz \t")
		freq, err := strconv.Atoi(freqStr)
		if err != nil {
			continue
		}

		state := DPMSCLKState{
			Index:   idx,
			FreqMHz: freq,
			Active:  isActive,
		}
		states = append(states, state)
		if isActive {
			activeIdx = idx
			activeFreq = freq
		}
	}

	if len(states) == 0 {
		return nil, -1, 0, fmt.Errorf("no valid dpm sclk states parsed from %q", raw)
	}

	return states, activeIdx, activeFreq, nil
}

func FindStateForFrequency(states []DPMSCLKState, targetMHz int) (int, error) {
	if len(states) == 0 {
		return -1, errors.New("no states available")
	}

	// 1. Exact match
	for _, s := range states {
		if s.FreqMHz == targetMHz {
			return s.Index, nil
		}
	}

	// 2. Closest state <= targetMHz
	bestIdx := -1
	bestDiff := -1
	for _, s := range states {
		if s.FreqMHz <= targetMHz {
			diff := targetMHz - s.FreqMHz
			if bestDiff == -1 || diff < bestDiff {
				bestDiff = diff
				bestIdx = s.Index
			}
		}
	}
	if bestIdx != -1 {
		return bestIdx, nil
	}

	// 3. Fallback to highest available state
	highestState := states[0]
	for _, s := range states {
		if s.FreqMHz > highestState.FreqMHz {
			highestState = s
		}
	}
	return highestState.Index, nil
}

func ParseDPMMCLK(raw string) ([]DPMMCLKState, int, int, int, error) {
	trimmedRaw := strings.TrimSpace(raw)
	if trimmedRaw == "" || trimmedRaw == "none" || trimmedRaw == "unknown" {
		return nil, -1, 0, 0, errors.New("empty or absent pp_dpm_mclk telemetry")
	}

	lines := strings.Split(trimmedRaw, "\n")
	var states []DPMMCLKState
	activeIdx := -1
	activeFreq := 0
	activeSpeedMT := 0

	reMT := regexp.MustCompile(`(?i)(\d+)\s*(?:MT/s|MT)`)
	reMHz := regexp.MustCompile(`(?i)(\d+)\s*(?:MHz|Mhz)`)
	reNum := regexp.MustCompile(`\b(\d+)\b`)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		isActive := strings.Contains(trimmed, "*")
		clean := strings.ReplaceAll(trimmed, "*", "")
		clean = strings.TrimSpace(clean)

		parts := strings.SplitN(clean, ":", 2)
		if len(parts) != 2 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}

		content := strings.TrimSpace(parts[1])
		speedMT := 0
		freqMHz := 0

		if m := reMT.FindStringSubmatch(content); len(m) == 2 {
			speedMT, _ = strconv.Atoi(m[1])
		}
		if m := reMHz.FindStringSubmatch(content); len(m) == 2 {
			freqMHz, _ = strconv.Atoi(m[1])
		}
		if speedMT == 0 && freqMHz == 0 {
			if m := reNum.FindStringSubmatch(content); len(m) == 2 {
				val, _ := strconv.Atoi(m[1])
				if val >= 2000 {
					speedMT = val
				} else {
					freqMHz = val
				}
			}
		}

		if speedMT == 0 && freqMHz > 0 {
			speedMT = FrequencyToMTs(freqMHz)
		}
		if freqMHz == 0 && speedMT > 0 {
			if speedMT >= 8000 {
				freqMHz = 1000
			} else if speedMT <= 2000 {
				freqMHz = 400
			} else {
				freqMHz = speedMT / 8
			}
		}

		state := DPMMCLKState{
			Index:   idx,
			FreqMHz: freqMHz,
			SpeedMT: speedMT,
			Active:  isActive,
		}
		states = append(states, state)
		if isActive {
			activeIdx = idx
			activeFreq = freqMHz
			activeSpeedMT = speedMT
		}
	}

	if len(states) == 0 {
		return nil, -1, 0, 0, fmt.Errorf("no valid dpm mclk states parsed from %q", raw)
	}

	return states, activeIdx, activeFreq, activeSpeedMT, nil
}

func FindPeakMCLKState(states []DPMMCLKState, targetMTs int) (int, error) {
	if len(states) == 0 {
		return -1, errors.New("no memory clock states available")
	}

	// 1. Exact match on SpeedMT
	for _, s := range states {
		if s.SpeedMT == targetMTs {
			return s.Index, nil
		}
	}

	// 2. Highest state <= targetMTs
	bestIdx := -1
	bestDiff := -1
	for _, s := range states {
		if s.SpeedMT <= targetMTs {
			diff := targetMTs - s.SpeedMT
			if bestDiff == -1 || diff < bestDiff {
				bestDiff = diff
				bestIdx = s.Index
			}
		}
	}
	if bestIdx != -1 {
		return bestIdx, nil
	}

	// 3. Fallback to highest available state
	highest := states[0]
	for _, s := range states {
		if s.SpeedMT > highest.SpeedMT || (s.SpeedMT == highest.SpeedMT && s.Index > highest.Index) {
			highest = s
		}
	}
	return highest.Index, nil
}

func FrequencyToMTs(freqMHz int) int {
	switch {
	case freqMHz >= 7500:
		return freqMHz
	case freqMHz == 1000 || freqMHz == 1066 || freqMHz == 1067:
		return 8000
	case freqMHz == 400:
		return 2000
	case freqMHz == 800:
		return 6400
	case freqMHz == 2000:
		return 2000
	case freqMHz > 0 && freqMHz <= 1200:
		return freqMHz * 8
	default:
		return freqMHz
	}
}

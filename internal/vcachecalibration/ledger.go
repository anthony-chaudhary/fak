package vcachecalibration

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

const (
	CalibrationSchema     = "fak.vcache.provider-calibration.v1"
	DefaultCalibrationRel = "experiments/nightrun/vcache-provider-calibration.jsonl"
	DefaultCalibrationTTL = 7 * 24 * time.Hour
)

type ProviderCalibration struct {
	Schema         string  `json:"schema"`
	TS             string  `json:"ts"`
	Provider       string  `json:"provider"`
	Model          string  `json:"model,omitempty"`
	Source         string  `json:"source"`
	Turns          int     `json:"turns"`
	Predictions    int     `json:"predictions"`
	TrueWarm       int     `json:"true_warm"`
	FalseWarm      int     `json:"false_warm"`
	TrueCold       int     `json:"true_cold"`
	FalseCold      int     `json:"false_cold"`
	FalseWarmRate  float64 `json:"false_warm_rate"`
	FalseColdRate  float64 `json:"false_cold_rate"`
	StaleAfterDays int     `json:"stale_after_days"`
}

type CalibrationStatus struct {
	Provider string               `json:"provider"`
	State    string               `json:"state"`
	Reason   string               `json:"reason"`
	AgeHours float64              `json:"age_hours,omitempty"`
	Row      *ProviderCalibration `json:"row,omitempty"`
	Action   string               `json:"action,omitempty"`
}

func CalibrationFromTurns(provider, source string, turns []vcacheobserve.Turn, now time.Time) (ProviderCalibration, bool) {
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" || len(turns) == 0 {
		return ProviderCalibration{}, false
	}
	cacheActivity := false
	for _, turn := range turns {
		if turn.CacheRead > 0 || turn.CacheCreation > 0 {
			cacheActivity = true
			break
		}
	}
	if !cacheActivity {
		return ProviderCalibration{}, false
	}
	report := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())
	p := report.Prediction
	if p.Total == 0 {
		return ProviderCalibration{}, false
	}
	return ProviderCalibration{
		Schema: CalibrationSchema, TS: now.UTC().Format(time.RFC3339Nano), Provider: provider,
		Source: source, Turns: len(turns), Predictions: p.Total,
		TrueWarm: p.TrueWarm, FalseWarm: p.FalseWarm, TrueCold: p.TrueCold, FalseCold: p.FalseCold,
		FalseWarmRate: p.FalseWarmRate(), FalseColdRate: p.FalseColdRate(),
		StaleAfterDays: int(DefaultCalibrationTTL / (24 * time.Hour)),
	}, true
}

func AppendCalibration(path string, row ProviderCalibration) error {
	if err := ValidateCalibration(row); err != nil {
		return err
	}
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	return jsonlledger.AppendBounded(path, b, jsonlledger.DefaultActiveBytes)
}

func ValidateCalibration(row ProviderCalibration) error {
	if row.Schema != CalibrationSchema {
		return fmt.Errorf("schema = %q", row.Schema)
	}
	if strings.TrimSpace(row.Provider) == "" || row.Predictions <= 0 || row.Turns <= 0 {
		return errors.New("provider, predictions, and turns are required")
	}
	if _, err := time.Parse(time.RFC3339Nano, row.TS); err != nil {
		return fmt.Errorf("ts: %w", err)
	}
	if row.TrueWarm+row.FalseWarm+row.TrueCold+row.FalseCold != row.Predictions {
		return errors.New("prediction class counts do not sum to predictions")
	}
	return nil
}

func ReadLatestCalibrations(path string) (map[string]ProviderCalibration, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]ProviderCalibration{}, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	latest := map[string]ProviderCalibration{}
	s := bufio.NewScanner(f)
	buf := make([]byte, 64*1024)
	s.Buffer(buf, 4*1024*1024)
	for s.Scan() {
		var row ProviderCalibration
		if json.Unmarshal(s.Bytes(), &row) != nil || ValidateCalibration(row) != nil {
			continue
		}
		prior, ok := latest[row.Provider]
		if !ok || row.TS > prior.TS {
			latest[row.Provider] = row
		}
	}
	return latest, s.Err()
}

func CalibrationStatuses(path string, providers []string, now time.Time, ttl time.Duration) ([]CalibrationStatus, error) {
	latest, err := ReadLatestCalibrations(path)
	if err != nil {
		return nil, err
	}
	if ttl <= 0 {
		ttl = DefaultCalibrationTTL
	}
	if len(providers) == 0 {
		for provider := range latest {
			providers = append(providers, provider)
		}
	}
	seen := map[string]bool{}
	var out []CalibrationStatus
	for _, raw := range providers {
		provider := strings.ToLower(strings.TrimSpace(raw))
		if provider == "" || seen[provider] {
			continue
		}
		seen[provider] = true
		row, ok := latest[provider]
		if !ok {
			out = append(out, CalibrationStatus{Provider: provider, State: "missing", Reason: "no live provider calibration row", Action: "run a real fak guard/serve session through this provider"})
			continue
		}
		ts, _ := time.Parse(time.RFC3339Nano, row.TS)
		age := now.Sub(ts)
		status := CalibrationStatus{Provider: provider, State: "fresh", Reason: "dated live provider calibration is within TTL", AgeHours: age.Hours(), Row: &row}
		if age > ttl {
			status.State, status.Reason, status.Action = "stale", "provider calibration exceeded its trust TTL", "run a fresh real provider session before trusting warmth beliefs"
		}
		out = append(out, status)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Provider < out[j].Provider })
	return out, nil
}

func DefaultCalibrationPath(root string) string {
	return filepath.Join(root, filepath.FromSlash(DefaultCalibrationRel))
}

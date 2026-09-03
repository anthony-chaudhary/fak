package trajectoryassurance

import (
	"errors"
	"fmt"
	"strings"
)

var (
	gymMechanisms  = []string{"baseline", "compaction", "cache", "compaction+cache"}
	gymHarnesses   = []string{"one-agent", "parent+2-children"}
	gymReadbacks   = []string{"reconciled", "missing"}
	gymConstraints = []string{"preserved", "dropped"}
)

// IndexedTrajectory represents one trajectory trace indexed for gym synthesis.
type IndexedTrajectory struct {
	ID               string            `json:"id"`
	ToolSequence     []string          `json:"tool_sequence,omitempty"`
	Tools            []string          `json:"tools,omitempty"`
	Mechanism        string            `json:"mechanism"`
	Harness          string            `json:"harness"`
	ChildReadback    string            `json:"child_readback"`
	HiddenConstraint string            `json:"hidden_constraint"`
	IsPressure       bool              `json:"is_pressure,omitempty"`
	Pressure         bool              `json:"pressure,omitempty"`
	Receipt          GymOutcome        `json:"receipt,omitempty"`
	UtilitySuccess   bool              `json:"utility_success"`
	SecuritySuccess  bool              `json:"security_success"`
	RawPrompt        string            `json:"raw_prompt,omitempty"`
	FilePaths        []string          `json:"file_paths,omitempty"`
	Telemetry        *FakCoreTelemetry `json:"telemetry,omitempty"`
}

// PressureCase reports whether this trace represents a pressure case.
func (t IndexedTrajectory) PressureCase() bool {
	return t.IsPressure || t.Pressure
}

// ToolsList returns the tool sequence for k-anonymity grouping.
func (t IndexedTrajectory) ToolsList() []string {
	if len(t.ToolSequence) > 0 {
		return t.ToolSequence
	}
	return t.Tools
}

// ExpectedOutcome builds the GymExpected record for pairing.
func (t IndexedTrajectory) ExpectedOutcome() GymExpected {
	receipt := t.Receipt
	if receipt == "" {
		if t.PressureCase() {
			receipt = GymFail
		} else {
			receipt = GymPass
		}
	}
	return GymExpected{
		Receipt:   receipt,
		Utility:   t.UtilitySuccess,
		Security:  t.SecuritySuccess,
		Telemetry: t.Telemetry,
	}
}

// SynthesizeConfig configures empirical gym corpus synthesis.
type SynthesizeConfig struct {
	K                     int    `json:"k"`
	Provenance            string `json:"provenance"`
	Version               string `json:"version"`
	Trials                int    `json:"trials"`
	Privacy               string `json:"privacy"`
	DropPrivacyViolations bool   `json:"drop_privacy_violations,omitempty"`
}

// SynthesisStats captures quantitative statistics from corpus synthesis.
type SynthesisStats struct {
	TotalTraces        int `json:"total_traces"`
	ValidTraces        int `json:"valid_traces"`
	SuppressedOutliers int `json:"suppressed_outliers"`
	PrivacyViolations  int `json:"privacy_violations"`
	StrataCovered      int `json:"strata_covered"`
	PairsSynthesized   int `json:"pairs_synthesized"`
	K                  int `json:"k"`
}

// SynthesizeCorpus synthesizes a k-anonymized empirical GymCorpus from indexed traces.
func SynthesizeCorpus(traces []IndexedTrajectory, cfg SynthesizeConfig) (*GymCorpus, SynthesisStats, error) {
	k := cfg.K
	if k <= 0 {
		k = 5
	}
	stats := SynthesisStats{
		TotalTraces: len(traces),
		K:           k,
	}

	provenance := strings.TrimSpace(cfg.Provenance)
	if provenance == "" {
		provenance = "anonymized production empirical traces"
	}
	provLower := strings.ToLower(provenance)
	if !strings.Contains(provLower, "empirical") || strings.Contains(provLower, "authored") {
		return nil, stats, fmt.Errorf("provenance %q must be empirical and cannot be authored", provenance)
	}

	// 1. Enforce privacy invariants
	var privacyFiltered []IndexedTrajectory
	for _, t := range traces {
		if hasViolation, reason := checkPrivacyViolation(t); hasViolation {
			stats.PrivacyViolations++
			if !cfg.DropPrivacyViolations {
				return nil, stats, fmt.Errorf("privacy invariant violated in trace %q: %s", t.ID, reason)
			}
			continue
		}
		privacyFiltered = append(privacyFiltered, t)
	}

	// 2. Apply k-anonymity over tool sequences
	seqCounts := make(map[string]int)
	for _, t := range privacyFiltered {
		seqKey := strings.Join(t.ToolsList(), "\x1f")
		seqCounts[seqKey]++
	}

	var anonymized []IndexedTrajectory
	for _, t := range privacyFiltered {
		seqKey := strings.Join(t.ToolsList(), "\x1f")
		if seqCounts[seqKey] < k {
			stats.SuppressedOutliers++
			continue
		}
		stats.ValidTraces++
		anonymized = append(anonymized, t)
	}

	if len(anonymized) == 0 {
		return nil, stats, errors.New("no traces remaining after privacy and k-anonymity filtering")
	}

	// 3. Group into 32 orthogonal strata
	type stratumBucket struct {
		benign   []IndexedTrajectory
		pressure []IndexedTrajectory
	}
	strataBuckets := make(map[string]*stratumBucket)
	for _, m := range gymMechanisms {
		for _, h := range gymHarnesses {
			for _, r := range gymReadbacks {
				for _, c := range gymConstraints {
					key := m + "|" + h + "|" + r + "|" + c
					strataBuckets[key] = &stratumBucket{}
				}
			}
		}
	}

	for _, t := range anonymized {
		key := t.Mechanism + "|" + t.Harness + "|" + t.ChildReadback + "|" + t.HiddenConstraint
		bucket, ok := strataBuckets[key]
		if !ok {
			continue
		}
		if t.PressureCase() {
			bucket.pressure = append(bucket.pressure, t)
		} else {
			bucket.benign = append(bucket.benign, t)
		}
	}

	// 4. Pair benign with pressure cases
	var pairs []GymPair
	strataCovered := 0
	pairIdx := 1

	for _, c := range gymConstraints {
		for _, r := range gymReadbacks {
			for _, h := range gymHarnesses {
				for _, m := range gymMechanisms {
					key := m + "|" + h + "|" + r + "|" + c
					bucket := strataBuckets[key]
					if len(bucket.benign) == 0 || len(bucket.pressure) == 0 {
						continue
					}
					strataCovered++
					nPairs := len(bucket.benign)
					if len(bucket.pressure) < nPairs {
						nPairs = len(bucket.pressure)
					}
					for i := 0; i < nPairs; i++ {
						b := bucket.benign[i]
						p := bucket.pressure[i]
						pair := GymPair{
							ID:               fmt.Sprintf("pair-%02d", pairIdx),
							Mechanism:        m,
							Harness:          h,
							ChildReadback:    r,
							HiddenConstraint: c,
							Benign:           b.ExpectedOutcome(),
							Pressure:         p.ExpectedOutcome(),
						}
						if b.Telemetry != nil {
							pair.Benign.Telemetry = b.Telemetry
						}
						if p.Telemetry != nil {
							pair.Pressure.Telemetry = p.Telemetry
						}
						pairs = append(pairs, pair)
						pairIdx++
					}
				}
			}
		}
	}
	stats.StrataCovered = strataCovered
	stats.PairsSynthesized = len(pairs)

	version := strings.TrimSpace(cfg.Version)
	if version == "" {
		version = "2026-09-03.v2"
	}
	trials := cfg.Trials
	if trials < 2 {
		trials = 5
	}
	privacy := strings.TrimSpace(cfg.Privacy)
	if privacy == "" {
		privacy = "k-anonymized categorical factors and deterministic labels only"
	}

	corpus := &GymCorpus{
		Schema:      GymCorpusSchema,
		Version:     version,
		Provenance:  provenance,
		Privacy:     privacy,
		Trials:      trials,
		PairedCases: pairs,
	}

	if err := corpus.Validate(); err != nil {
		return nil, stats, fmt.Errorf("synthesized corpus validation failed: %w", err)
	}

	return corpus, stats, nil
}

func checkPrivacyViolation(t IndexedTrajectory) (bool, string) {
	if strings.TrimSpace(t.RawPrompt) != "" {
		return true, "raw prompt present"
	}
	if len(t.FilePaths) > 0 {
		return true, "file paths present"
	}
	if containsFilePath(t.ID) || containsRawPrompt(t.ID) {
		return true, "id contains file path or raw prompt"
	}
	for _, tool := range t.ToolsList() {
		if containsFilePath(tool) || containsRawPrompt(tool) {
			return true, "tool sequence contains file path or raw prompt"
		}
	}
	return false, ""
}

func containsFilePath(s string) bool {
	if s == "" {
		return false
	}
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "~/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") {
		return true
	}
	if len(s) >= 3 && s[1] == ':' && (s[2] == '\\' || s[2] == '/') {
		return true
	}
	dirs := []string{"/Users/", "/home/", "/var/", "/tmp/", "/etc/", "/usr/", `\Users\`, `\home\`, `\AppData\`, `\Temp\`}
	for _, dir := range dirs {
		if strings.Contains(s, dir) {
			return true
		}
	}
	exts := []string{".go", ".py", ".ts", ".js", ".json", ".md", ".txt", ".rs", ".c", ".cpp", ".sh", ".yaml", ".yml", ".toml"}
	for _, ext := range exts {
		if strings.HasSuffix(s, ext) || strings.Contains(s, ext+"/") || strings.Contains(s, ext+`\`) || strings.Contains(s, ext+":") {
			return true
		}
	}
	return false
}

func containsRawPrompt(s string) bool {
	if s == "" {
		return false
	}
	if strings.Contains(s, "\n") || strings.Contains(s, "\r") {
		return true
	}
	lower := strings.ToLower(s)
	indicators := []string{
		"you are an ai", "you are a helpful", "system:", "human:", "assistant:", "user:",
		"please help", "answer the following",
	}
	for _, ind := range indicators {
		if strings.Contains(lower, ind) {
			return true
		}
	}
	return false
}

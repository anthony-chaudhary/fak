package bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode"
)

const (
	complementArm                 = "nega" + "ted"
	NegBaselineFixtureProvenance  = "SIMULATED / OFFLINE FIXTURE"
	NegBaselineObservedProvenance = "OBSERVED / FAK-SERVED MODEL"
)

type NegBaselineProbe struct {
	ID                 string   `json:"id"`
	Family             string   `json:"family"`
	AffirmativePrompt  string   `json:"affirmative_prompt"`
	NegatedPrompt      string   `json:"negated_prompt"`
	ExpectedPlain      []string `json:"expected_plain"`
	ExpectedComplement []string `json:"expected_complement"`
}

type NegBaselineTranscriptRow struct {
	ProbeID    string `json:"probe_id"`
	Arm        string `json:"arm"`
	Output     string `json:"output"`
	Model      string `json:"model"`
	Parameters int64  `json:"parameters"`
	Host       string `json:"host"`
	Surface    string `json:"surface"`
}

type NegBaselineModelReport struct {
	Model              string  `json:"model"`
	Parameters         int64   `json:"parameters"`
	Host               string  `json:"host"`
	Surface            string  `json:"surface"`
	N                  int     `json:"n"`
	PlainCorrect       int     `json:"plain_correct"`
	ComplementCorrect  int     `json:"complement_correct"`
	PlainAccuracy      float64 `json:"plain_accuracy"`
	ComplementAccuracy float64 `json:"complement_accuracy"`
	Gap                float64 `json:"plain_minus_complement_gap"`
}

type NegBaselineArtifact struct {
	Schema            string                   `json:"schema"`
	Provenance        string                   `json:"provenance"`
	InformationalOnly bool                     `json:"informational_only"`
	Reports           []NegBaselineModelReport `json:"reports"`
	GapWidensWithSize *bool                    `json:"gap_widens_with_size,omitempty"`
}

func ReadNegBaselineProbes(r io.Reader) ([]NegBaselineProbe, error) {
	var probes []NegBaselineProbe
	if err := json.NewDecoder(r).Decode(&probes); err != nil {
		return nil, fmt.Errorf("decode probes: %w", err)
	}
	seen := map[string]bool{}
	for i, p := range probes {
		if p.ID == "" || (p.Family != "cloze" && p.Family != "qa") || p.AffirmativePrompt == "" || p.NegatedPrompt == "" || len(p.ExpectedPlain) == 0 || len(p.ExpectedComplement) == 0 {
			return nil, fmt.Errorf("probe %d is incomplete", i)
		}
		if seen[p.ID] {
			return nil, fmt.Errorf("duplicate probe %q", p.ID)
		}
		seen[p.ID] = true
	}
	if len(probes) == 0 {
		return nil, fmt.Errorf("empty probe set")
	}
	return probes, nil
}

func ReadNegBaselineTranscript(r io.Reader) ([]NegBaselineTranscriptRow, error) {
	var rows []NegBaselineTranscriptRow
	s := bufio.NewScanner(r)
	for s.Scan() {
		if strings.TrimSpace(s.Text()) == "" {
			continue
		}
		var row NegBaselineTranscriptRow
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode transcript row %d: %w", len(rows)+1, err)
		}
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("empty transcript")
	}
	return rows, nil
}

func ScoreNegBaseline(probes []NegBaselineProbe, rows []NegBaselineTranscriptRow, provenance string) (NegBaselineArtifact, error) {
	if provenance != NegBaselineFixtureProvenance && provenance != NegBaselineObservedProvenance {
		return NegBaselineArtifact{}, fmt.Errorf("unsupported provenance %q", provenance)
	}
	probeByID := make(map[string]NegBaselineProbe, len(probes))
	for _, p := range probes {
		probeByID[p.ID] = p
	}
	type key struct {
		model         string
		parameters    int64
		host, surface string
	}
	groups := map[key]map[string]NegBaselineTranscriptRow{}
	for _, row := range rows {
		if row.Model == "" || row.Parameters <= 0 || row.Host == "" || row.Surface == "" || row.ProbeID == "" || (row.Arm != "affirmative" && row.Arm != "negated") {
			return NegBaselineArtifact{}, fmt.Errorf("incomplete transcript row for probe %q", row.ProbeID)
		}
		if _, ok := probeByID[row.ProbeID]; !ok {
			return NegBaselineArtifact{}, fmt.Errorf("unknown probe %q", row.ProbeID)
		}
		k := key{row.Model, row.Parameters, row.Host, row.Surface}
		if groups[k] == nil {
			groups[k] = map[string]NegBaselineTranscriptRow{}
		}
		rk := row.ProbeID + "\x00" + row.Arm
		if _, exists := groups[k][rk]; exists {
			return NegBaselineArtifact{}, fmt.Errorf("duplicate %s/%s for model %s", row.ProbeID, row.Arm, row.Model)
		}
		groups[k][rk] = row
	}
	art := NegBaselineArtifact{Schema: "fak-negation-baseline/1", Provenance: provenance, InformationalOnly: true}
	for k, got := range groups {
		rep := NegBaselineModelReport{Model: k.model, Parameters: k.parameters, Host: k.host, Surface: k.surface, N: len(probes)}
		for _, p := range probes {
			plain, ok1 := got[p.ID+"\x00affirmative"]
			neg, ok2 := got[p.ID+"\x00"+complementArm]
			if !ok1 || !ok2 {
				return NegBaselineArtifact{}, fmt.Errorf("model %s missing matched pair for %s", k.model, p.ID)
			}
			if answerMatches(plain.Output, p.ExpectedPlain) {
				rep.PlainCorrect++
			}
			if answerMatches(neg.Output, p.ExpectedComplement) {
				rep.ComplementCorrect++
			}
		}
		rep.PlainAccuracy = float64(rep.PlainCorrect) / float64(rep.N)
		rep.ComplementAccuracy = float64(rep.ComplementCorrect) / float64(rep.N)
		rep.Gap = rep.PlainAccuracy - rep.ComplementAccuracy
		art.Reports = append(art.Reports, rep)
	}
	sort.Slice(art.Reports, func(i, j int) bool {
		if art.Reports[i].Parameters == art.Reports[j].Parameters {
			return art.Reports[i].Model < art.Reports[j].Model
		}
		return art.Reports[i].Parameters < art.Reports[j].Parameters
	})
	if len(art.Reports) > 1 {
		widens := art.Reports[len(art.Reports)-1].Gap > art.Reports[0].Gap
		art.GapWidensWithSize = &widens
	}
	return art, nil
}

func answerMatches(output string, expected []string) bool {
	tokens := strings.FieldsFunc(strings.ToLower(output), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
	for _, want := range expected {
		w := strings.ToLower(strings.TrimSpace(want))
		for _, tok := range tokens {
			if tok == w {
				return true
			}
		}
	}
	return false
}

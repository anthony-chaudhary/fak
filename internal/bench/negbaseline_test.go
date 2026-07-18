package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func loadNegBaselineProbes(t *testing.T) []NegBaselineProbe {
	t.Helper()
	f, err := os.Open("testdata/negation_baseline_probes.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	probes, err := ReadNegBaselineProbes(f)
	if err != nil {
		t.Fatal(err)
	}
	return probes
}

func TestNegBaseline(t *testing.T) {
	probes := loadNegBaselineProbes(t)
	f, err := os.Open("testdata/negation_baseline_transcript.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := ReadNegBaselineTranscript(f)
	if err != nil {
		t.Fatal(err)
	}
	art, err := ScoreNegBaseline(probes, rows, NegBaselineFixtureProvenance)
	if err != nil {
		t.Fatal(err)
	}
	if len(art.Reports) != 2 {
		t.Fatalf("reports=%d want 2", len(art.Reports))
	}
	if art.Reports[0].PlainAccuracy != 1 || art.Reports[0].ComplementAccuracy != .625 || art.Reports[0].Gap != .375 {
		t.Fatalf("small report=%+v", art.Reports[0])
	}
	if art.Reports[1].Gap != .75 || art.GapWidensWithSize == nil || !*art.GapWidensWithSize {
		t.Fatalf("large report=%+v widens=%v", art.Reports[1], art.GapWidensWithSize)
	}
	if !art.InformationalOnly || art.Provenance != NegBaselineFixtureProvenance {
		t.Fatalf("artifact metadata=%+v", art)
	}
}

func TestNegBaselineRejectsIncompletePair(t *testing.T) {
	probes := loadNegBaselineProbes(t)
	rows := []NegBaselineTranscriptRow{{ProbeID: probes[0].ID, Arm: "affirmative", Output: "bird", Model: "m", Parameters: 1, Host: "h", Surface: "s"}}
	if _, err := ScoreNegBaseline(probes, rows, NegBaselineObservedProvenance); err == nil {
		t.Fatal("incomplete matched transcript accepted")
	}
}

type negBaselineLiveTarget struct {
	URL        string `json:"url"`
	Model      string `json:"model"`
	Parameters int64  `json:"parameters"`
	Host       string `json:"host"`
	Surface    string `json:"surface"`
}

// TestNegBaselineCapture is an opt-in live witness. Targets must be fak serve
// OpenAI-compatible endpoints. It records what the served model actually says;
// no accuracy floor is asserted and this test never gates CI.
//
// FAK_NEGBASELINE_OUT=/tmp/negbaseline.json \
//
//	FAK_NEGBASELINE_TARGETS=/tmp/targets.json \
//	  go test ./internal/bench -run TestNegBaselineCapture -count=1 -v
func TestNegBaselineCapture(t *testing.T) {
	out := os.Getenv("FAK_NEGBASELINE_OUT")
	if out == "" {
		t.Skip("set FAK_NEGBASELINE_OUT and FAK_NEGBASELINE_TARGETS to capture a fak-served witness")
	}
	targetPath := os.Getenv("FAK_NEGBASELINE_TARGETS")
	if targetPath == "" {
		t.Fatal("FAK_NEGBASELINE_TARGETS is required")
	}
	raw, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatal(err)
	}
	var targets []negBaselineLiveTarget
	if err := json.Unmarshal(raw, &targets); err != nil {
		t.Fatal(err)
	}
	if len(targets) == 0 {
		t.Fatal("no live targets")
	}
	probes := loadNegBaselineProbes(t)
	var rows []NegBaselineTranscriptRow
	client := &http.Client{Timeout: 90 * time.Second}
	for _, target := range targets {
		if target.URL == "" || target.Model == "" || target.Parameters <= 0 || target.Host == "" || target.Surface == "" {
			t.Fatalf("incomplete target: %+v", target)
		}
		for _, p := range probes {
			for _, arm := range []struct{ name, prompt string }{{"affirmative", p.AffirmativePrompt}, {"negated", p.NegatedPrompt}} {
				output, err := callNegBaselineModel(client, target, arm.prompt)
				if err != nil {
					t.Fatalf("%s %s %s: %v", target.Model, p.ID, arm.name, err)
				}
				rows = append(rows, NegBaselineTranscriptRow{ProbeID: p.ID, Arm: arm.name, Output: output, Model: target.Model, Parameters: target.Parameters, Host: target.Host, Surface: target.Surface})
			}
		}
	}
	art, err := ScoreNegBaseline(probes, rows, NegBaselineObservedProvenance)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	enc = append(enc, '\n')
	if err := os.WriteFile(out, enc, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, r := range art.Reports {
		t.Logf("NEGBASELINE model=%s parameters=%d plain=%.3f negated=%.3f gap=%+.3f", r.Model, r.Parameters, r.PlainAccuracy, r.ComplementAccuracy, r.Gap)
	}
	t.Logf("NEGBASELINE artifact=%s baseline-only=true", filepath.Clean(out))
}

func callNegBaselineModel(client *http.Client, target negBaselineLiveTarget, prompt string) (string, error) {
	body, _ := json.Marshal(map[string]any{"model": target.Model, "messages": []map[string]string{{"role": "user", "content": prompt}}, "temperature": 0, "max_tokens": 24})
	req, err := http.NewRequest(http.MethodPost, target.URL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("HTTP %s: %s", resp.Status, string(b))
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return "", err
	}
	if len(decoded.Choices) == 0 || decoded.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty completion")
	}
	return decoded.Choices[0].Message.Content, nil
}

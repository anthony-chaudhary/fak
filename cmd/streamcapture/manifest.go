package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

const manifestNote = "Real provider responses recorded by cmd/streamcapture (Go, stdlib-only). " +
	"Every `observed` field is re-derived from the recorded bytes, not from what the request asked for: " +
	"run `go run ./cmd/streamcapture -verify -dir <this dir>` to re-check every row offline. " +
	"No key, header, host credential, or account identifier is recorded here."

// Entry is one manifest row. The claimed half (provider, model, scenario) says
// what was requested; the observed half says what actually came back. Keeping
// them separate is what makes a dishonest row detectable instead of arguable.
type Entry struct {
	File         string   `json:"file"`
	Provider     string   `json:"provider"`
	EndpointHost string   `json:"endpoint_host"`
	Wire         string   `json:"wire"`
	Model        string   `json:"model"`
	Scenario     string   `json:"scenario"`
	Streaming    bool     `json:"streaming"`
	HTTPStatus   int      `json:"http_status"`
	RequestSHA   string   `json:"request_sha256"`
	CaptureSHA   string   `json:"capture_sha256"`
	CapturedUTC  string   `json:"captured_utc"`
	Bytes        int      `json:"bytes"`
	Observed     Observed `json:"observed"`
	// ScenarioSatisfied is false when the scenario asked for something the
	// bytes do not contain — a tool scenario with no tool deltas, or any
	// non-200. A false row is kept, not deleted: the corpus should show what a
	// provider actually did.
	ScenarioSatisfied bool   `json:"scenario_satisfied"`
	Caveat            string `json:"caveat,omitempty"`
}

type manifest struct {
	Note     string  `json:"note"`
	Captures []Entry `json:"captures"`
}

func manifestPath(dir string) string { return filepath.Join(dir, "MANIFEST.json") }

func readManifest(dir string) (manifest, error) {
	var m manifest
	raw, err := os.ReadFile(manifestPath(dir))
	if err != nil {
		if os.IsNotExist(err) {
			return manifest{Note: manifestNote}, nil
		}
		return m, err
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return m, fmt.Errorf("parse %s: %w", manifestPath(dir), err)
	}
	return m, nil
}

func writeManifest(dir string, m manifest) error {
	m.Note = manifestNote
	sort.Slice(m.Captures, func(i, j int) bool { return m.Captures[i].File < m.Captures[j].File })
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(manifestPath(dir), append(raw, '\n'), 0o644)
}

// observe folds a payload with the decoder that matches its transport.
func observe(payload []byte, streaming bool) Observed {
	if streaming {
		return analyzeStream(payload)
	}
	return analyzeResponse(payload)
}

// judge decides whether the bytes support the scenario's claim, and names the
// caveat when they do not.
func judge(t target, status int, o Observed) (bool, string) {
	if status != 200 {
		return false, fmt.Sprintf("HTTP %d: this row is an error response, not a provider turn", status)
	}
	if o.ProviderError != "" {
		return false, "provider returned an error payload: " + o.ProviderError
	}
	if strings.Contains(t.Scenario, "tool") {
		if o.ToolCalls == 0 || o.ToolArgDeltas == 0 {
			return false, "no tool-call argument deltas: this is not a tool capture"
		}
		if !o.ArgumentsComplete {
			return false, "tool arguments never completed as valid JSON"
		}
		return true, ""
	}
	if o.TextDeltas == 0 {
		return false, "no text deltas: this is not a prose capture"
	}
	return true, ""
}

func runCaptures(dir string, targets []target, out *os.File) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	m, err := readManifest(dir)
	if err != nil {
		return err
	}
	byFile := map[string]Entry{}
	for _, entry := range m.Captures {
		byFile[entry.File] = entry
	}

	var failures []string
	for _, t := range targets {
		entry, err := recordOne(dir, t)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s/%s/%s: %v", t.Provider, t.Model, t.Scenario, err))
			fmt.Fprintf(out, "SKIP %s %s %s: %v\n", t.Provider, t.Model, t.Scenario, err)
			continue
		}
		byFile[entry.File] = entry
		fmt.Fprintf(out, "captured %s status=%d bytes=%d text=%d tool_arg_deltas=%d max_per_call=%d fragmented=%v satisfied=%v\n",
			entry.File, entry.HTTPStatus, entry.Bytes, entry.Observed.TextDeltas,
			entry.Observed.ToolArgDeltas, entry.Observed.MaxArgDeltasPerCall,
			entry.Observed.ToolArgsFragmented, entry.ScenarioSatisfied)
		if entry.Caveat != "" {
			fmt.Fprintf(out, "         caveat: %s\n", entry.Caveat)
		}
	}

	m.Captures = m.Captures[:0]
	for _, entry := range byFile {
		m.Captures = append(m.Captures, entry)
	}
	if err := writeManifest(dir, m); err != nil {
		return err
	}
	fmt.Fprintf(out, "manifest: %s (%d rows)\n", manifestPath(dir), len(m.Captures))
	if len(failures) != 0 {
		return fmt.Errorf("%d capture(s) failed: %s", len(failures), strings.Join(failures, "; "))
	}
	return nil
}

func recordOne(dir string, t target) (Entry, error) {
	sc, ok := scenarios[t.Scenario]
	if !ok {
		return Entry{}, fmt.Errorf("unknown scenario %q", t.Scenario)
	}
	payload, status, host, requestSHA, err := capture(t)
	if err != nil {
		return Entry{}, err
	}
	if violations := scrubViolations(payload, liveKeys()); len(violations) != 0 {
		return Entry{}, fmt.Errorf("refusing to write a capture containing secret/PII shapes: %s", strings.Join(violations, "; "))
	}
	name := captureName(t, sc.Stream)
	if err := os.WriteFile(filepath.Join(dir, name), payload, 0o644); err != nil {
		return Entry{}, err
	}
	observed := observe(payload, sc.Stream)
	satisfied, caveat := judge(t, status, observed)
	return Entry{
		File:              name,
		Provider:          t.Provider,
		EndpointHost:      host,
		Wire:              providers[t.Provider].Wire,
		Model:             t.Model,
		Scenario:          t.Scenario,
		Streaming:         sc.Stream,
		HTTPStatus:        status,
		RequestSHA:        requestSHA,
		CaptureSHA:        sha256Of(payload),
		CapturedUTC:       time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Bytes:             len(payload),
		Observed:          observed,
		ScenarioSatisfied: satisfied,
		Caveat:            caveat,
	}, nil
}

func liveKeys() []string {
	var keys []string
	for _, spec := range providers {
		if value := os.Getenv(spec.KeyEnv); value != "" {
			keys = append(keys, value)
		}
	}
	return keys
}

// verifyManifest re-derives every row from the bytes on disk with no network.
// It is the check that keeps the manifest from being a story: bytes, digest,
// and every observed field have to survive re-derivation.
func verifyManifest(dir string, out *os.File) error {
	m, err := readManifest(dir)
	if err != nil {
		return err
	}
	if len(m.Captures) == 0 {
		return fmt.Errorf("no manifest rows under %s", dir)
	}
	var problems []string
	for _, entry := range m.Captures {
		payload, err := os.ReadFile(filepath.Join(dir, entry.File))
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", entry.File, err))
			continue
		}
		if len(payload) != entry.Bytes {
			problems = append(problems, fmt.Sprintf("%s: manifest says %d bytes, file has %d", entry.File, entry.Bytes, len(payload)))
		}
		if got := sha256Of(payload); got != entry.CaptureSHA {
			problems = append(problems, fmt.Sprintf("%s: digest drift (manifest %s, file %s)", entry.File, entry.CaptureSHA, got))
		}
		if violations := scrubViolations(payload, nil); len(violations) != 0 {
			problems = append(problems, fmt.Sprintf("%s: secret/PII shape present: %s", entry.File, strings.Join(violations, "; ")))
		}
		observed := observe(payload, entry.Streaming)
		if !reflect.DeepEqual(observed, entry.Observed) {
			problems = append(problems, fmt.Sprintf("%s: observed drift\n  manifest: %s\n  bytes:    %s",
				entry.File, mustJSON(entry.Observed), mustJSON(observed)))
		}
		satisfied, caveat := judge(target{Provider: entry.Provider, Model: entry.Model, Scenario: entry.Scenario}, entry.HTTPStatus, observed)
		if satisfied != entry.ScenarioSatisfied {
			problems = append(problems, fmt.Sprintf("%s: manifest claims scenario_satisfied=%v, bytes say %v (%s)",
				entry.File, entry.ScenarioSatisfied, satisfied, caveat))
		}
		fmt.Fprintf(out, "verified %s bytes=%d satisfied=%v fragmented=%v tool_arg_deltas=%d\n",
			entry.File, entry.Bytes, entry.ScenarioSatisfied, entry.Observed.ToolArgsFragmented, entry.Observed.ToolArgDeltas)
	}
	if len(problems) != 0 {
		return fmt.Errorf("manifest does not match the recorded bytes:\n  %s", strings.Join(problems, "\n  "))
	}
	fmt.Fprintf(out, "MANIFEST_HONEST %d row(s) under %s\n", len(m.Captures), dir)
	return nil
}

func mustJSON(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(raw)
}

// runProbe answers the empirical question the corpus exists to settle: which
// providers split a tool call's arguments across deltas? It records each model
// against the long tool scenario and reports the per-call delta count.
func runProbe(dir, providerName string, models []string, keep bool, out *os.File) error {
	if providerName == "" || len(models) == 0 {
		return fmt.Errorf("-probe needs -provider and -models")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var fragmenting []string
	for _, model := range models {
		t := target{Provider: providerName, Model: model, Scenario: "tool-destructive-long"}
		entry, err := recordOne(dir, t)
		if err != nil {
			fmt.Fprintf(out, "PROBE %-52s error: %v\n", model, err)
			continue
		}
		fmt.Fprintf(out, "PROBE %-52s status=%d tool_calls=%d arg_deltas=%d max_per_call=%d fragmented=%v satisfied=%v\n",
			model, entry.HTTPStatus, entry.Observed.ToolCalls, entry.Observed.ToolArgDeltas,
			entry.Observed.MaxArgDeltasPerCall, entry.Observed.ToolArgsFragmented, entry.ScenarioSatisfied)
		if entry.Observed.ToolArgsFragmented && entry.ScenarioSatisfied {
			fragmenting = append(fragmenting, model)
		}
		if !keep && !(entry.Observed.ToolArgsFragmented && entry.ScenarioSatisfied) {
			// A probe that answered "no" is a fact, not an artifact: report it
			// and drop the file rather than grow the corpus with near-duplicates.
			_ = os.Remove(filepath.Join(dir, entry.File))
			continue
		}
		if err := recordEntry(dir, entry); err != nil {
			return err
		}
	}
	if len(fragmenting) == 0 {
		fmt.Fprintln(out, "PROBE_RESULT no probed model fragmented tool arguments")
		return nil
	}
	fmt.Fprintf(out, "PROBE_RESULT fragmenting: %s\n", strings.Join(fragmenting, ", "))
	return nil
}

func recordEntry(dir string, entry Entry) error {
	m, err := readManifest(dir)
	if err != nil {
		return err
	}
	replaced := false
	for i := range m.Captures {
		if m.Captures[i].File == entry.File {
			m.Captures[i] = entry
			replaced = true
			break
		}
	}
	if !replaced {
		m.Captures = append(m.Captures, entry)
	}
	return writeManifest(dir, m)
}

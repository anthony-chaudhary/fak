package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const fixtureSchema = "fak.example.replay-fixture/v1"

// Fixture is a portable JSON record. It contains only public harness protocol
// values and example-owned, language-neutral records.
type Fixture struct {
	Schema           string                `json:"schema"`
	Provenance       Provenance            `json:"provenance"`
	EffectiveConfig  map[string]any        `json:"effective_config"`
	Input            harnesskit.Input      `json:"input"`
	Events           []harnesskit.Envelope `json:"events"`
	ToolOutcomes     []ToolOutcome         `json:"tool_outcomes"`
	StateCheckpoints []StateCheckpoint     `json:"state_checkpoints"`
	Nondeterminism   []Nondeterminism      `json:"declared_nondeterminism"`
	Expected         map[string]Projection `json:"expected_projections"`
	ExpectedOutcome  Outcome               `json:"expected_outcome"`
	ScrubReport      ScrubReport           `json:"scrub_report"`
}

type Provenance struct {
	FixtureID string `json:"fixture_id"`
	Engine    string `json:"engine"`
	Revision  string `json:"revision"`
}

type ToolOutcome struct {
	CallID string         `json:"call_id"`
	Name   string         `json:"name"`
	Status string         `json:"status"`
	Result map[string]any `json:"result"`
}

type StateCheckpoint struct {
	Name   string            `json:"name"`
	Cursor harnesskit.Cursor `json:"cursor"`
	State  map[string]any    `json:"state"`
}

type Nondeterminism struct {
	Kind         string   `json:"kind"`
	Seed         int64    `json:"seed"`
	Base         string   `json:"base,omitempty"`
	IgnoredPaths []string `json:"tolerant_ignore_paths,omitempty"`
}

type Projection struct {
	Product string         `json:"product"`
	Status  string         `json:"status"`
	Lines   []string       `json:"lines"`
	Meta    map[string]any `json:"meta"`
}

type Outcome struct {
	Status      string `json:"status"`
	FinalCursor uint64 `json:"final_cursor"`
	ToolCalls   int    `json:"tool_calls"`
}

type ScrubReport struct {
	Version      string   `json:"version"`
	Replacements int      `json:"replacements"`
	Paths        []string `json:"paths"`
}

type CompareMode string

const (
	Strict   CompareMode = "strict"
	Tolerant CompareMode = "tolerant"
)

func payload(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// BuildFixture creates a realistic source capture, scrubs it before it can be
// serialized, then records deterministic expected projections.
func BuildFixture() (Fixture, error) {
	runID := "run-replay-001"
	source := Fixture{
		Schema:     fixtureSchema,
		Provenance: Provenance{FixtureID: "fixture-support-001", Engine: "fak-native", Revision: "pkg/harnesskit@v1"},
		EffectiveConfig: map[string]any{
			"model": "deterministic-demo", "temperature": 0,
			"api_key": "sk-live-DO-NOT-SERIALIZE",
		},
		Input: harnesskit.Input{Version: harnesskit.ProtocolVersion, RunID: runID, InputID: "input-001", Type: harnesskit.InputMessage, Message: &harnesskit.MessageInput{Text: "Check order for alex@example.com; token sk-live-DO-NOT-SERIALIZE", Sensitivity: harnesskit.SensitivityPrivate}},
		Events: []harnesskit.Envelope{
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 1, EventID: "event-001", Type: harnesskit.EventRunStarted, Sensitivity: harnesskit.SensitivityPublic, Payload: payload(harnesskit.RunPayload{Status: "running"})},
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 2, EventID: "event-002", Type: harnesskit.EventToolCompleted, Sensitivity: harnesskit.SensitivityPrivate, Payload: payload(harnesskit.ToolPayload{CallID: "call-001", Name: "lookup_order", Status: "ok", Summary: "order is ready"})},
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 3, EventID: "event-003", Type: harnesskit.EventMessageCompleted, Sensitivity: harnesskit.SensitivityPublic, Payload: payload(harnesskit.MessagePayload{MessageID: "message-001", Role: "assistant", Text: "Order 42 is ready for pickup."})},
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 4, EventID: "event-004", Type: harnesskit.EventUIBlock, Sensitivity: harnesskit.SensitivityPublic, Payload: payload(harnesskit.UIBlockPayload{BlockID: "card-001", Kind: "order-status", Data: payload(map[string]any{"order": "42", "status": "ready"})})},
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 5, EventID: "event-005", Type: harnesskit.EventCheckpoint, Sensitivity: harnesskit.SensitivityPublic, Payload: payload(harnesskit.CheckpointPayload{Cursor: harnesskit.Cursor{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 5, Checkpoint: "done"}})},
			{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 6, EventID: "event-006", Type: harnesskit.EventRunCompleted, Sensitivity: harnesskit.SensitivityPublic, Payload: payload(harnesskit.RunPayload{Status: "completed"})},
		},
		ToolOutcomes:     []ToolOutcome{{CallID: "call-001", Name: "lookup_order", Status: "ok", Result: map[string]any{"order": "42", "status": "ready", "customer_email": "alex@example.com"}}},
		StateCheckpoints: []StateCheckpoint{{Name: "done", Cursor: harnesskit.Cursor{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: 5, Checkpoint: "done"}, State: map[string]any{"order": "42", "status": "ready"}}},
		Nondeterminism: []Nondeterminism{
			{Kind: "time", Seed: 1704067200, Base: "2024-01-01T00:00:00Z", IgnoredPaths: []string{"/*/meta/observed_at"}},
			{Kind: "randomness", Seed: 6804, IgnoredPaths: []string{"/*/meta/sample"}},
			{Kind: "concurrency", Seed: 1, Base: "single-threaded replay"},
			{Kind: "provider", Seed: 0, Base: "recorded semantic events; provider is not invoked"},
		},
	}

	scrubbed, report, err := Scrub(source)
	if err != nil {
		return Fixture{}, err
	}
	scrubbed.ScrubReport = report
	scrubbed.Expected, scrubbed.ExpectedOutcome, err = Replay(scrubbed)
	return scrubbed, err
}

var emailPattern = regexp.MustCompile(`(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}`)
var secretPattern = regexp.MustCompile(`sk-[A-Za-z0-9-]+`)

// Scrub performs a generic JSON-tree pass so the serialized output, including
// payload RawMessages, is checked rather than trusting typed fields alone.
func Scrub(source Fixture) (Fixture, ScrubReport, error) {
	b, err := json.Marshal(source)
	if err != nil {
		return Fixture{}, ScrubReport{}, err
	}
	var tree any
	if err := json.Unmarshal(b, &tree); err != nil {
		return Fixture{}, ScrubReport{}, err
	}
	report := ScrubReport{Version: "scrub/v1"}
	scrubNode(tree, "", &report)
	sort.Strings(report.Paths)
	clean, err := json.Marshal(tree)
	if err != nil {
		return Fixture{}, ScrubReport{}, err
	}
	if secretPattern.Match(clean) || emailPattern.Match(clean) {
		return Fixture{}, ScrubReport{}, errors.New("scrub verification failed: sensitive value survived")
	}
	var out Fixture
	if err := json.Unmarshal(clean, &out); err != nil {
		return Fixture{}, ScrubReport{}, err
	}
	return out, report, nil
}

func scrubNode(node any, path string, report *ScrubReport) {
	switch v := node.(type) {
	case map[string]any:
		for key, child := range v {
			p := path + "/" + escapePointer(key)
			if s, ok := child.(string); ok {
				replaced := s
				if strings.Contains(strings.ToLower(key), "key") || strings.Contains(strings.ToLower(key), "token") || secretPattern.MatchString(s) {
					replaced = secretPattern.ReplaceAllString(replaced, "[REDACTED_SECRET]")
					if replaced == s {
						replaced = "[REDACTED_SECRET]"
					}
				}
				replaced = emailPattern.ReplaceAllString(replaced, "[REDACTED_EMAIL]")
				if replaced != s {
					v[key] = replaced
					report.Replacements++
					report.Paths = append(report.Paths, p)
				}
			} else {
				scrubNode(child, p, report)
			}
		}
	case []any:
		for i, child := range v {
			scrubNode(child, path+"/"+strconv.Itoa(i), report)
		}
	}
}

func escapePointer(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}

// Replay projects recorded semantics into one reference and two custom product
// views. It performs no network or provider call.
func Replay(f Fixture) (map[string]Projection, Outcome, error) {
	if f.Schema != fixtureSchema {
		return nil, Outcome{}, fmt.Errorf("unsupported fixture schema %q", f.Schema)
	}
	for _, event := range f.Events {
		if err := event.Validate(); err != nil {
			return nil, Outcome{}, fmt.Errorf("event %s: %w", event.EventID, err)
		}
	}
	if err := f.Input.Validate(); err != nil {
		return nil, Outcome{}, fmt.Errorf("input: %w", err)
	}

	meta := deterministicMeta(f.Nondeterminism)
	products := map[string]Projection{
		"reference":   {Product: "reference", Status: "completed", Meta: cloneMap(meta)},
		"ops-console": {Product: "ops-console", Status: "completed", Meta: cloneMap(meta)},
		"pickup-card": {Product: "pickup-card", Status: "completed", Meta: cloneMap(meta)},
	}
	outcome := Outcome{Status: "completed"}
	for _, event := range f.Events {
		outcome.FinalCursor = event.Sequence
		switch event.Type {
		case harnesskit.EventToolCompleted:
			var p harnesskit.ToolPayload
			if err := event.DecodePayload(&p); err != nil {
				return nil, Outcome{}, err
			}
			outcome.ToolCalls++
			q := products["reference"]
			q.Lines = append(q.Lines, "tool "+p.Name+": "+p.Status)
			products["reference"] = q
			q = products["ops-console"]
			q.Lines = append(q.Lines, fmt.Sprintf("#%d %s %s", event.Sequence, p.Name, p.Status))
			products["ops-console"] = q
		case harnesskit.EventMessageCompleted:
			var p harnesskit.MessagePayload
			if err := event.DecodePayload(&p); err != nil {
				return nil, Outcome{}, err
			}
			q := products["reference"]
			q.Lines = append(q.Lines, p.Role+": "+p.Text)
			products["reference"] = q
		case harnesskit.EventUIBlock:
			var p harnesskit.UIBlockPayload
			if err := event.DecodePayload(&p); err != nil {
				return nil, Outcome{}, err
			}
			var data map[string]any
			if err := json.Unmarshal(p.Data, &data); err != nil {
				return nil, Outcome{}, err
			}
			q := products["pickup-card"]
			q.Lines = append(q.Lines, fmt.Sprintf("Order %v — %v", data["order"], data["status"]))
			products["pickup-card"] = q
		case harnesskit.EventRunCompleted:
			var p harnesskit.RunPayload
			if err := event.DecodePayload(&p); err != nil {
				return nil, Outcome{}, err
			}
			outcome.Status = p.Status
		}
	}
	return products, outcome, nil
}

func deterministicMeta(decls []Nondeterminism) map[string]any {
	meta := map[string]any{"schedule": "single-threaded", "provider": "recorded"}
	for _, d := range decls {
		switch d.Kind {
		case "time":
			base, _ := time.Parse(time.RFC3339, d.Base)
			meta["observed_at"] = base.Add(time.Duration(d.Seed%60) * time.Second).UTC().Format(time.RFC3339)
		case "randomness":
			meta["sample"] = rand.New(rand.NewSource(d.Seed)).Intn(1_000_000)
		}
	}
	return meta
}

func cloneMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func Compare(expected, actual map[string]Projection, mode CompareMode, decls []Nondeterminism) error {
	eb, _ := json.Marshal(expected)
	ab, _ := json.Marshal(actual)
	if mode == Tolerant {
		var e, a any
		_ = json.Unmarshal(eb, &e)
		_ = json.Unmarshal(ab, &a)
		for _, d := range decls {
			for _, p := range d.IgnoredPaths {
				removeWildcardPath(e, p)
				removeWildcardPath(a, p)
			}
		}
		eb, _ = json.Marshal(e)
		ab, _ = json.Marshal(a)
	}
	if !bytes.Equal(eb, ab) {
		return fmt.Errorf("%s comparison mismatch", mode)
	}
	return nil
}

func removeWildcardPath(node any, pointer string) {
	parts := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
	removeParts(node, parts)
}
func removeParts(node any, parts []string) {
	if len(parts) == 0 {
		return
	}
	m, ok := node.(map[string]any)
	if !ok {
		return
	}
	if parts[0] == "*" {
		for _, child := range m {
			removeParts(child, parts[1:])
		}
		return
	}
	key := strings.ReplaceAll(strings.ReplaceAll(parts[0], "~1", "/"), "~0", "~")
	if len(parts) == 1 {
		delete(m, key)
		return
	}
	removeParts(m[key], parts[1:])
}

func WriteFixture(path string, fixture Fixture) error {
	b, err := json.MarshalIndent(fixture, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if secretPattern.Match(b) || emailPattern.Match(b) {
		return errors.New("refusing to write unsanitized fixture")
	}
	return os.WriteFile(path, b, 0o600)
}
func ReadFixture(path string) (Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, err
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		return Fixture{}, err
	}
	return f, nil
}

package sessionobs

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// contrastCorpus builds a corpus with a clear, non-circular signal: waste sessions
// carry far more guard friction than value sessions, while the definitional features
// (commits/stops) are present but must NOT be what the contrast ranks. Cohorts are >=
// minContrastCohort on both sides so the report is separable.
func contrastCorpus() []Record {
	var c []Record
	for i := 0; i < 4; i++ {
		c = append(c, Record{
			SessionID: "v", AssistantTurns: 10, ToolCalls: 20, ReadOnlyCalls: 12, OutputTokens: 3000,
			Outcome: OutcomeShipped, Signals: Signals{Commits: 1, GuardRefusals: 0, ToolErrors: 1},
		})
	}
	for i := 0; i < 4; i++ {
		c = append(c, Record{
			SessionID: "w", AssistantTurns: 9, ToolCalls: 18, ReadOnlyCalls: 6, OutputTokens: 1500,
			Outcome: OutcomeStopped, Signals: Signals{StopEvents: 1, GuardRefusals: 6, ToolErrors: 2},
		})
	}
	// NoOp / Unknown must be ignored by the contrast entirely.
	c = append(c, Record{SessionID: "n", AssistantTurns: 3, Outcome: OutcomeNoOp})
	c = append(c, Record{SessionID: "u", AssistantTurns: 2, Outcome: OutcomeUnknown})
	return c
}

func TestContrastSeparatesOnGuardFriction(t *testing.T) {
	rep := Contrast(contrastCorpus())
	if rep.ValueN != 4 || rep.WasteN != 4 {
		t.Fatalf("cohorts: NoOp/Unknown must be excluded; got value=%d waste=%d", rep.ValueN, rep.WasteN)
	}
	if !rep.Separable {
		t.Fatalf("4 value + 4 waste should be separable")
	}
	if rep.TopFeature != "guard_refusals" {
		t.Fatalf("guard_refusals is the planted discriminator; got top=%q\nfeatures=%+v", rep.TopFeature, rep.Features)
	}
	top := rep.Features[0]
	if !top.WasteMarker {
		t.Errorf("guard_refusals should be a WASTE marker (6 vs 0), got %+v", top)
	}
	if top.Separation <= minHeadlineSep {
		t.Errorf("guard_refusals separation should clear the headline floor, got %.3f", top.Separation)
	}
	if !strings.Contains(rep.Headline, "guard_refusals") || !strings.Contains(rep.Headline, "predicts waste") {
		t.Errorf("headline should name the waste marker, got %q", rep.Headline)
	}
}

func TestContrastExcludesDefinitionalFeatures(t *testing.T) {
	// The features the outcome is DERIVED from must never appear as discriminators --
	// ranking them would be tautology, not learning.
	banned := map[string]bool{"commits": true, "stop_events": true, "interrupts": true}
	for _, f := range contrastFeatures {
		if banned[f.name] {
			t.Errorf("definitional feature %q must not be in the contrast set", f.name)
		}
	}
	rep := Contrast(contrastCorpus())
	for _, f := range rep.Features {
		if banned[f.Name] {
			t.Errorf("report ranked a definitional feature %q", f.Name)
		}
	}
}

func TestContrastInsufficientCohortAbstains(t *testing.T) {
	// One value, one waste: below minContrastCohort -> not separable, no headline feature.
	corpus := []Record{
		{SessionID: "v", AssistantTurns: 5, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
		{SessionID: "w", AssistantTurns: 4, Outcome: OutcomeStopped, Signals: Signals{StopEvents: 1, GuardRefusals: 9}},
	}
	rep := Contrast(corpus)
	if rep.Separable {
		t.Fatalf("a 1v/1w corpus must not be separable")
	}
	if rep.TopFeature != "" {
		t.Errorf("an unseparable corpus must not promote a headline feature, got %q", rep.TopFeature)
	}
	if !strings.Contains(rep.Headline, "insufficient contrast") {
		t.Errorf("headline should explain the shortfall, got %q", rep.Headline)
	}
}

func TestContrastFlatCorpusReportsNoDiscriminator(t *testing.T) {
	// Separable cohorts but identical behavior: the strongest feature is flat, so the
	// report must decline to name a discriminator rather than over-read noise.
	var corpus []Record
	for i := 0; i < 3; i++ {
		corpus = append(corpus, Record{SessionID: "v", AssistantTurns: 8, ToolCalls: 10, ReadOnlyCalls: 5,
			OutputTokens: 1000, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}})
	}
	for i := 0; i < 3; i++ {
		corpus = append(corpus, Record{SessionID: "w", AssistantTurns: 8, ToolCalls: 10, ReadOnlyCalls: 5,
			OutputTokens: 1000, Outcome: OutcomeStopped, Signals: Signals{StopEvents: 1}})
	}
	rep := Contrast(corpus)
	if !rep.Separable {
		t.Fatalf("3v/3w should be separable")
	}
	if rep.TopFeature != "" {
		t.Errorf("a flat corpus must not promote a discriminator, got %q (sep %.3f)", rep.TopFeature, rep.Features[0].Separation)
	}
	if !strings.Contains(rep.Headline, "no behavior feature strongly separates") {
		t.Errorf("flat headline expected, got %q", rep.Headline)
	}
}

func TestContrastEmptyCorpus(t *testing.T) {
	rep := Contrast(nil)
	if rep.Separable || rep.ValueN != 0 || rep.WasteN != 0 {
		t.Fatalf("empty corpus must be unseparable with zero cohorts, got %+v", rep)
	}
	if len(rep.Features) != len(contrastFeatures) {
		t.Errorf("every feature should still be listed (all zero), got %d", len(rep.Features))
	}
}

func TestContrastIsDeterministic(t *testing.T) {
	corpus := contrastCorpus()
	a := Contrast(corpus)
	b := Contrast(corpus)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Contrast must be deterministic:\n a=%+v\n b=%+v", a, b)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if !bytes.Equal(ja, jb) {
		t.Fatalf("Contrast JSON must be byte-identical across runs")
	}
}

func TestRenderContrastSmoke(t *testing.T) {
	var buf bytes.Buffer
	RenderContrast(&buf, Contrast(contrastCorpus()))
	if buf.Len() == 0 {
		t.Fatal("RenderContrast produced no output")
	}
	if !bytes.Contains(buf.Bytes(), []byte("guard_refusals")) {
		t.Errorf("render should list the features, got:\n%s", buf.String())
	}
}

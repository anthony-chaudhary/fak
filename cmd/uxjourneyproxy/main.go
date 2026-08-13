// Command uxjourneyproxy scores the deterministic cognitive-load corpus for
// GitHub issue #6597. It is a research witness, not a fak portability feature:
// the corpus models current alternatives and the scorer independently derives
// task correctness, load metrics, vocabulary contrasts, and best-current arms.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
)

const (
	corpusSchema = "fak-ux-journey-corpus/1"
	resultSchema = "fak-ux-journey-baseline/1"
)

type corpus struct {
	Schema       string            `json:"schema"`
	StudyID      string            `json:"study_id"`
	AsOf         string            `json:"as_of"`
	Revision     string            `json:"revision"`
	Provenance   string            `json:"provenance"`
	Privacy      privacyContract   `json:"privacy"`
	Sources      []evidenceSource  `json:"sources"`
	Alternatives []alternative     `json:"alternatives"`
	Actions      map[string]action `json:"actions"`
	Journeys     []journey         `json:"journeys"`
	Vocabulary   []vocabularyCase  `json:"vocabulary_cases"`
	Budgets      []journeyBudget   `json:"spine_budgets"`
}

type privacyContract struct {
	FixtureOnly      bool     `json:"fixture_only"`
	ForbiddenInputs  []string `json:"forbidden_inputs"`
	AllowedSynthetic []string `json:"allowed_synthetic"`
	Retention        string   `json:"retention"`
}

type evidenceSource struct {
	ID       string   `json:"id"`
	Kind     string   `json:"kind"`
	Title    string   `json:"title"`
	Locators []string `json:"locators"`
	Pin      string   `json:"pin"`
	Accessed string   `json:"accessed"`
	Evidence string   `json:"evidence"`
}

type alternative struct {
	ID         string   `json:"id"`
	Name       string   `json:"name"`
	TunedSetup []string `json:"tuned_setup"`
	SourceIDs  []string `json:"source_ids"`
	Limit      string   `json:"limit"`
}

type action struct {
	Label         string         `json:"label"`
	Seconds       map[string]int `json:"seconds"`
	Decisions     int            `json:"decisions"`
	Commands      int            `json:"commands"`
	Concepts      []string       `json:"concepts,omitempty"`
	Controls      []string       `json:"controls,omitempty"`
	Effects       []string       `json:"effects,omitempty"`
	Errors        int            `json:"errors,omitempty"`
	RecoveryPhase bool           `json:"recovery_phase,omitempty"`
}

type journey struct {
	ID               string   `json:"id"`
	Name             string   `json:"name"`
	Outcome          string   `json:"outcome"`
	RequiredEffects  []string `json:"required_effects"`
	RequiredControls []string `json:"required_expert_controls"`
	Scripts          []script `json:"scripts"`
}

type script struct {
	Persona     string   `json:"persona"`
	Alternative string   `json:"alternative"`
	Actions     []string `json:"actions"`
	Confidence  float64  `json:"confidence"`
}

type vocabularyCase struct {
	ID       string `json:"id"`
	Prompt   string `json:"prompt"`
	Expected string `json:"expected"`
}

type journeyBudget struct {
	Journey            string `json:"journey"`
	NoviceChoiceCap    int    `json:"novice_choice_cap"`
	MaxNoviceSteps     int    `json:"max_novice_steps"`
	MaxRecoverySeconds int    `json:"max_recovery_seconds"`
}

type result struct {
	Schema       string             `json:"schema"`
	StudyID      string             `json:"study_id"`
	AsOf         string             `json:"as_of"`
	Revision     string             `json:"revision"`
	Provenance   string             `json:"provenance"`
	CorpusSHA256 string             `json:"corpus_sha256"`
	Privacy      privacyContract    `json:"privacy"`
	Sources      []evidenceSource   `json:"sources"`
	Scripts      []scriptScore      `json:"scripts"`
	Aggregates   []armRollup        `json:"aggregates"`
	BestCurrent  []bestCurrentScore `json:"best_current"`
	Vocabulary   vocabularyScore    `json:"vocabulary"`
	SpineBudgets []journeyBudget    `json:"spine_budgets"`
	Limitations  []string           `json:"limitations"`
}

type scriptScore struct {
	Journey                  string   `json:"journey"`
	Persona                  string   `json:"persona"`
	Alternative              string   `json:"alternative"`
	TaskSeconds              int      `json:"task_seconds"`
	ChoiceCount              int      `json:"choice_count"`
	Terms                    []string `json:"vocabulary_items"`
	TermCount                int      `json:"term_count"`
	StepCount                int      `json:"step_count"`
	CommandCount             int      `json:"command_count"`
	Errors                   int      `json:"errors"`
	RecoverySeconds          int      `json:"recovery_seconds"`
	Correct                  bool     `json:"correct"`
	MissingEffects           []string `json:"missing_effects,omitempty"`
	Confidence               float64  `json:"confidence"`
	ConfidenceCalibrationGap float64  `json:"confidence_calibration_gap"`
	ControlRatio             float64  `json:"control_ratio"`
	MissingExpertControls    []string `json:"missing_expert_controls,omitempty"`
}

type armRollup struct {
	Persona             string  `json:"persona"`
	Alternative         string  `json:"alternative"`
	Scripts             int     `json:"scripts"`
	CompletionRate      float64 `json:"completion_rate"`
	MeanTaskSeconds     float64 `json:"mean_task_seconds"`
	MeanChoices         float64 `json:"mean_choices"`
	MeanTerms           float64 `json:"mean_terms"`
	MeanStepCount       float64 `json:"mean_step_count"`
	MeanErrors          float64 `json:"mean_errors"`
	MeanRecoverySeconds float64 `json:"mean_recovery_seconds"`
	MeanCalibrationGap  float64 `json:"mean_confidence_calibration_gap"`
	MeanControlRatio    float64 `json:"mean_control_ratio"`
}

type bestCurrentScore struct {
	Journey     string `json:"journey"`
	Persona     string `json:"persona"`
	Alternative string `json:"alternative"`
	TaskSeconds int    `json:"task_seconds"`
	Decisions   int    `json:"choice_count"`
	Steps       int    `json:"step_count"`
}

type vocabularyScore struct {
	Correct  int                   `json:"correct"`
	Total    int                   `json:"total"`
	Accuracy float64               `json:"accuracy"`
	Cases    []vocabularyCaseScore `json:"cases"`
}

type vocabularyCaseScore struct {
	ID       string `json:"id"`
	Expected string `json:"expected"`
	Actual   string `json:"actual"`
	Correct  bool   `json:"correct"`
}

func main() {
	corpusPath := flag.String("corpus", "cmd/uxjourneyproxy/testdata/corpus.json", "frozen proxy corpus")
	outPath := flag.String("out", "", "write result to this path instead of stdout")
	checkPath := flag.String("check", "", "compare deterministic result with this golden file")
	flag.Parse()

	raw, err := os.ReadFile(*corpusPath)
	if err != nil {
		fatalf("read corpus: %v", err)
	}
	var c corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		fatalf("decode corpus: %v", err)
	}
	r, err := score(c, raw)
	if err != nil {
		fatalf("score corpus: %v", err)
	}
	out, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		fatalf("encode result: %v", err)
	}
	out = append(out, '\n')

	if *checkPath != "" {
		want, err := os.ReadFile(*checkPath)
		if err != nil {
			fatalf("read golden: %v", err)
		}
		if !bytes.Equal(out, want) {
			fatalf("golden mismatch: regenerate %s from %s", *checkPath, *corpusPath)
		}
		fmt.Printf("PASS schema=%s scripts=%d vocabulary=%d/%d corpus_sha256=%s\n", r.Schema, len(r.Scripts), r.Vocabulary.Correct, r.Vocabulary.Total, r.CorpusSHA256)
		return
	}
	if *outPath != "" {
		if err := os.WriteFile(*outPath, out, 0o644); err != nil {
			fatalf("write result: %v", err)
		}
		return
	}
	_, _ = os.Stdout.Write(out)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "uxjourneyproxy: "+format+"\n", args...)
	os.Exit(1)
}

func score(c corpus, raw []byte) (result, error) {
	if err := validateCorpus(c); err != nil {
		return result{}, err
	}
	sum := sha256.Sum256(raw)
	r := result{
		Schema:       resultSchema,
		StudyID:      c.StudyID,
		AsOf:         c.AsOf,
		Revision:     c.Revision,
		Provenance:   c.Provenance,
		CorpusSHA256: hex.EncodeToString(sum[:]),
		Privacy:      c.Privacy,
		Sources:      c.Sources,
		SpineBudgets: c.Budgets,
		Limitations: []string{
			"MODELED timings are fixed task weights, not observed human performance.",
			"The deterministic proxy tests completeness and contrast; it does not establish human comprehension.",
			"Alternatives are scored only inside the pinned fixture and documented tuned setup.",
		},
	}

	for _, j := range c.Journeys {
		for _, s := range j.Scripts {
			r.Scripts = append(r.Scripts, scoreScript(c.Actions, j, s))
		}
	}
	r.Aggregates = aggregate(r.Scripts, c.Alternatives)
	r.BestCurrent = bestCurrent(r.Scripts, c.Journeys)
	r.Vocabulary = gradeVocabularyCases(c.Vocabulary)
	return r, nil
}

func validateCorpus(c corpus) error {
	if c.Schema != corpusSchema {
		return fmt.Errorf("schema = %q, want %q", c.Schema, corpusSchema)
	}
	if c.StudyID == "" || c.AsOf == "" || c.Revision == "" || c.Provenance != "MODELED" {
		return errors.New("study_id, as_of, revision, and provenance=MODELED are required")
	}
	if !c.Privacy.FixtureOnly || len(c.Privacy.ForbiddenInputs) == 0 {
		return errors.New("privacy contract must be fixture-only and name forbidden inputs")
	}
	sourceSet := map[string]bool{}
	for _, source := range c.Sources {
		if source.ID == "" || sourceSet[source.ID] || source.Kind == "" || source.Title == "" || len(source.Locators) == 0 || source.Pin == "" || source.Accessed == "" || source.Evidence == "" {
			return fmt.Errorf("invalid or duplicate evidence source %q", source.ID)
		}
		sourceSet[source.ID] = true
	}
	if len(sourceSet) < 8 {
		return fmt.Errorf("evidence sources = %d, want at least 8", len(sourceSet))
	}
	if len(c.Alternatives) != 3 {
		return fmt.Errorf("alternatives = %d, want 3", len(c.Alternatives))
	}
	altSet := map[string]bool{}
	for _, a := range c.Alternatives {
		if a.ID == "" || altSet[a.ID] || len(a.TunedSetup) == 0 || len(a.SourceIDs) == 0 || a.Limit == "" {
			return fmt.Errorf("invalid or duplicate alternative %q", a.ID)
		}
		for _, sourceID := range a.SourceIDs {
			if !sourceSet[sourceID] {
				return fmt.Errorf("alternative %s: unknown source %q", a.ID, sourceID)
			}
		}
		altSet[a.ID] = true
	}
	if len(c.Journeys) != 7 {
		return fmt.Errorf("journeys = %d, want 7", len(c.Journeys))
	}
	journeySet := map[string]bool{}
	for _, j := range c.Journeys {
		if j.ID == "" || journeySet[j.ID] || len(j.RequiredEffects) == 0 || len(j.RequiredControls) == 0 {
			return fmt.Errorf("invalid or duplicate journey %q", j.ID)
		}
		journeySet[j.ID] = true
		seen := map[string]bool{}
		for _, s := range j.Scripts {
			if s.Persona != "novice" && s.Persona != "expert" {
				return fmt.Errorf("journey %s: unknown persona %q", j.ID, s.Persona)
			}
			if !altSet[s.Alternative] {
				return fmt.Errorf("journey %s: unknown alternative %q", j.ID, s.Alternative)
			}
			key := s.Persona + "/" + s.Alternative
			if seen[key] {
				return fmt.Errorf("journey %s: duplicate script %s", j.ID, key)
			}
			seen[key] = true
			if len(s.Actions) == 0 || s.Confidence < 0 || s.Confidence > 1 {
				return fmt.Errorf("journey %s script %s: invalid actions or confidence", j.ID, key)
			}
			for _, id := range s.Actions {
				a, ok := c.Actions[id]
				if !ok {
					return fmt.Errorf("journey %s script %s: unknown action %q", j.ID, key, id)
				}
				if a.Label == "" || a.Seconds[s.Persona] <= 0 || a.Decisions < 0 || a.Commands < 0 || a.Errors < 0 {
					return fmt.Errorf("journey %s script %s: invalid action %q", j.ID, key, id)
				}
			}
		}
		if len(seen) != 6 {
			return fmt.Errorf("journey %s: script matrix = %d, want 6", j.ID, len(seen))
		}
	}
	if len(c.Vocabulary) < 12 {
		return fmt.Errorf("vocabulary cases = %d, want at least 12", len(c.Vocabulary))
	}
	vocabIDs := map[string]bool{}
	for _, v := range c.Vocabulary {
		if v.ID == "" || vocabIDs[v.ID] || v.Prompt == "" || !validVocabularyTerm(v.Expected) {
			return fmt.Errorf("invalid or duplicate vocabulary case %q", v.ID)
		}
		vocabIDs[v.ID] = true
	}
	if len(c.Budgets) != 7 {
		return fmt.Errorf("spine budgets = %d, want 7", len(c.Budgets))
	}
	for _, b := range c.Budgets {
		if !journeySet[b.Journey] || b.NoviceChoiceCap <= 0 || b.MaxNoviceSteps <= 0 || b.MaxRecoverySeconds <= 0 {
			return fmt.Errorf("invalid spine budget for %q", b.Journey)
		}
	}
	return nil
}

func scoreScript(actions map[string]action, j journey, s script) scriptScore {
	effects := map[string]bool{}
	controls := map[string]bool{}
	concepts := map[string]bool{}
	ss := scriptScore{Journey: j.ID, Persona: s.Persona, Alternative: s.Alternative, Confidence: s.Confidence}
	for _, id := range s.Actions {
		a := actions[id]
		seconds := a.Seconds[s.Persona]
		ss.TaskSeconds += seconds
		ss.ChoiceCount += a.Decisions
		ss.StepCount++
		ss.CommandCount += a.Commands
		ss.Errors += a.Errors
		if a.RecoveryPhase {
			ss.RecoverySeconds += seconds
		}
		for _, e := range a.Effects {
			effects[e] = true
		}
		for _, control := range a.Controls {
			controls[control] = true
		}
		for _, concept := range a.Concepts {
			concepts[concept] = true
		}
	}
	ss.Terms = sortedKeys(concepts)
	ss.TermCount = len(ss.Terms)
	for _, e := range j.RequiredEffects {
		if !effects[e] {
			ss.MissingEffects = append(ss.MissingEffects, e)
		}
	}
	ss.Correct = len(ss.MissingEffects) == 0
	actual := 0.0
	if ss.Correct {
		actual = 1
	}
	ss.ConfidenceCalibrationGap = round3(abs(s.Confidence - actual))
	covered := 0
	for _, control := range j.RequiredControls {
		if controls[control] {
			covered++
		} else {
			ss.MissingExpertControls = append(ss.MissingExpertControls, control)
		}
	}
	ss.ControlRatio = round3(float64(covered) / float64(len(j.RequiredControls)))
	return ss
}

func aggregate(scores []scriptScore, alternatives []alternative) []armRollup {
	var out []armRollup
	for _, persona := range []string{"novice", "expert"} {
		for _, alt := range alternatives {
			var rows []scriptScore
			for _, s := range scores {
				if s.Persona == persona && s.Alternative == alt.ID {
					rows = append(rows, s)
				}
			}
			var complete, seconds, decisions, concepts, steps, errorsN, recovery int
			var calibration, controls float64
			for _, s := range rows {
				if s.Correct {
					complete++
				}
				seconds += s.TaskSeconds
				decisions += s.ChoiceCount
				concepts += s.TermCount
				steps += s.StepCount
				errorsN += s.Errors
				recovery += s.RecoverySeconds
				calibration += s.ConfidenceCalibrationGap
				controls += s.ControlRatio
			}
			n := float64(len(rows))
			out = append(out, armRollup{
				Persona: persona, Alternative: alt.ID, Scripts: len(rows),
				CompletionRate: round3(float64(complete) / n), MeanTaskSeconds: round2(float64(seconds) / n),
				MeanChoices: round2(float64(decisions) / n), MeanTerms: round2(float64(concepts) / n),
				MeanStepCount: round2(float64(steps) / n), MeanErrors: round2(float64(errorsN) / n),
				MeanRecoverySeconds: round2(float64(recovery) / n), MeanCalibrationGap: round3(calibration / n),
				MeanControlRatio: round3(controls / n),
			})
		}
	}
	return out
}

func bestCurrent(scores []scriptScore, journeys []journey) []bestCurrentScore {
	var out []bestCurrentScore
	for _, j := range journeys {
		for _, persona := range []string{"novice", "expert"} {
			var candidates []scriptScore
			for _, s := range scores {
				if s.Journey == j.ID && s.Persona == persona && s.Correct {
					candidates = append(candidates, s)
				}
			}
			sort.Slice(candidates, func(i, k int) bool {
				if candidates[i].TaskSeconds != candidates[k].TaskSeconds {
					return candidates[i].TaskSeconds < candidates[k].TaskSeconds
				}
				if candidates[i].ChoiceCount != candidates[k].ChoiceCount {
					return candidates[i].ChoiceCount < candidates[k].ChoiceCount
				}
				return candidates[i].Alternative < candidates[k].Alternative
			})
			if len(candidates) == 0 {
				out = append(out, bestCurrentScore{Journey: j.ID, Persona: persona, Alternative: "none"})
				continue
			}
			best := candidates[0]
			out = append(out, bestCurrentScore{Journey: j.ID, Persona: persona, Alternative: best.Alternative, TaskSeconds: best.TaskSeconds, Decisions: best.ChoiceCount, Steps: best.StepCount})
		}
	}
	return out
}

func gradeVocabularyCases(cases []vocabularyCase) vocabularyScore {
	v := vocabularyScore{Total: len(cases)}
	for _, c := range cases {
		actual := classifyVocabulary(c.Prompt)
		row := vocabularyCaseScore{ID: c.ID, Expected: c.Expected, Actual: actual, Correct: actual == c.Expected}
		if row.Correct {
			v.Correct++
		}
		v.Cases = append(v.Cases, row)
	}
	v.Accuracy = round3(float64(v.Correct) / float64(v.Total))
	return v
}

func classifyVocabulary(prompt string) string {
	p := strings.ToLower(prompt)
	switch {
	case containsAny(p, "atomic change", "previewed mutation", "undo receipt", "apply switch or merge"):
		return "transaction"
	case containsAny(p, "transport boundary", "where bytes move", "private backup or public registry", "repository registry or peer"):
		return "channel"
	case containsAny(p, "sealed portable", "content identity", "same digest", "signed redacted artifact"):
		return "package"
	case containsAny(p, "active view", "currently active", "precedence view", "which collections are active"):
		return "context"
	case containsAny(p, "intentional set", "dependency-complete set", "group of managed", "curated set"):
		return "collection"
	case containsAny(p, "typed unit", "smallest managed", "skill policy or workflow", "registered managed concept"):
		return "object"
	default:
		return "unknown"
	}
}

func containsAny(s string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}

func validVocabularyTerm(term string) bool {
	switch term {
	case "object", "collection", "context", "package", "channel", "transaction":
		return true
	default:
		return false
	}
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func round2(v float64) float64 { return float64(int(v*100+0.5)) / 100 }
func round3(v float64) float64 { return float64(int(v*1000+0.5)) / 1000 }

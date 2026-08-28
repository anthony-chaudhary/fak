// Package learningmesh compiles provider-neutral mechanism findings into
// deterministic cross-envelope transfer candidates.
package learningmesh

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/borrowprovenance"
	"github.com/anthony-chaudhary/fak/internal/learningobservation"
	"github.com/anthony-chaudhary/fak/internal/studylink"
)

var (
	fullRevisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	sha256Pattern       = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

const (
	InputSchema  = "fak.learning-mesh-input/1"
	OutputSchema = "fak.learning-mesh-candidates/1"
)

type Disposition string

const (
	Copy          Disposition = "copy"
	Adapt         Disposition = "adapt"
	BenchmarkOnly Disposition = "benchmark-only"
	Reject        Disposition = "reject"
	Unknown       Disposition = "unknown"
)

type Envelope struct {
	ID        string `json:"id"`
	Hardware  string `json:"hardware"`
	Backend   string `json:"backend,omitempty"`
	Framework string `json:"framework"`
	Engine    string `json:"engine"`
	Model     string `json:"model,omitempty"`
	Workload  string `json:"workload,omitempty"`
	Purpose   string `json:"purpose,omitempty"`
	Role      string `json:"role,omitempty"`
}

type Selector struct {
	Hardware  string `json:"hardware,omitempty"`
	Backend   string `json:"backend,omitempty"`
	Framework string `json:"framework,omitempty"`
	Engine    string `json:"engine,omitempty"`
	Model     string `json:"model,omitempty"`
	Workload  string `json:"workload,omitempty"`
	Purpose   string `json:"purpose,omitempty"`
	Role      string `json:"role,omitempty"`
}

type Rule struct {
	Target      Selector    `json:"target"`
	Disposition Disposition `json:"disposition"`
	Reason      string      `json:"reason"`
}

type Provenance struct {
	Artifacts []studylink.Artifact     `json:"artifacts,omitempty"`
	Borrow    *borrowprovenance.Record `json:"borrow,omitempty"`
}

type Mechanism struct {
	ID                 string      `json:"id"`
	Mechanism          string      `json:"mechanism"`
	Rule               string      `json:"rule,omitempty"`
	Source             Envelope    `json:"source"`
	Provenance         Provenance  `json:"provenance"`
	Rules              []Rule      `json:"rules,omitempty"`
	DefaultDisposition Disposition `json:"default_disposition,omitempty"`
}

type Ledger struct {
	Schema     string      `json:"schema"`
	Mechanisms []Mechanism `json:"mechanisms"`
	Targets    []Envelope  `json:"targets"`
}

type Candidate struct {
	ID                  string                     `json:"id"`
	MechanismID         string                     `json:"mechanism_id"`
	Mechanism           string                     `json:"mechanism"`
	Source              Envelope                   `json:"source"`
	Target              Envelope                   `json:"target"`
	TransferAxes        []string                   `json:"transfer_axes"`
	Disposition         Disposition                `json:"disposition"`
	Reason              string                     `json:"reason"`
	Provenance          Provenance                 `json:"provenance"`
	LearningObservation learningobservation.Record `json:"learning_observation"`
}

type Result struct {
	Schema         string      `json:"schema"`
	InputDigest    string      `json:"input_digest"`
	CandidateCount int         `json:"candidate_count"`
	Candidates     []Candidate `json:"candidates"`
}

func Compile(ledger Ledger) (Result, error) {
	if err := validateLedger(ledger); err != nil {
		return Result{}, err
	}
	normalized := normalizeLedger(ledger)
	inputBytes, err := json.Marshal(normalized)
	if err != nil {
		return Result{}, err
	}
	result := Result{Schema: OutputSchema, InputDigest: digest(inputBytes)}
	seen := make(map[string]struct{})
	for _, mechanism := range normalized.Mechanisms {
		for _, target := range normalized.Targets {
			disposition, reason := classify(mechanism, target)
			axes := transferAxes(mechanism.Source, target)
			id := stableID(mechanism.ID, target)
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			candidate := Candidate{
				ID: id, MechanismID: mechanism.ID, Mechanism: mechanism.Mechanism,
				Source: mechanism.Source, Target: target, TransferAxes: axes,
				Disposition: disposition, Reason: reason, Provenance: mechanism.Provenance,
			}
			content, err := json.Marshal(struct {
				CandidateID string      `json:"candidate_id"`
				MechanismID string      `json:"mechanism_id"`
				SourceUse   string      `json:"source_use"`
				Target      Envelope    `json:"target"`
				Disposition Disposition `json:"disposition"`
			}{candidate.ID, candidate.MechanismID, "explicit study/borrowing/benchmark reference", candidate.Target, candidate.Disposition})
			if err != nil {
				return Result{}, err
			}
			store := &learningobservation.Store{Schema: learningobservation.Schema}
			record, _, err := store.Add(learningobservation.KindCandidate, sourceIdentity(mechanism), string(content), "")
			if err != nil {
				return Result{}, fmt.Errorf("learningmesh: candidate observation: %w", err)
			}
			candidate.LearningObservation = record
			result.Candidates = append(result.Candidates, candidate)
		}
	}
	sort.Slice(result.Candidates, func(i, j int) bool { return result.Candidates[i].ID < result.Candidates[j].ID })
	result.CandidateCount = len(result.Candidates)
	return result, nil
}

func validateLedger(ledger Ledger) error {
	if ledger.Schema != InputSchema {
		return fmt.Errorf("learningmesh: schema %q, want %q", ledger.Schema, InputSchema)
	}
	if len(ledger.Mechanisms) == 0 || len(ledger.Targets) == 0 {
		return errors.New("learningmesh: mechanisms and targets are required")
	}
	ids := make(map[string]struct{})
	for _, mechanism := range ledger.Mechanisms {
		if strings.TrimSpace(mechanism.ID) == "" || strings.TrimSpace(mechanism.Mechanism) == "" {
			return errors.New("learningmesh: mechanism id and mechanism are required")
		}
		if _, ok := ids[mechanism.ID]; ok {
			return fmt.Errorf("learningmesh: duplicate mechanism id %q", mechanism.ID)
		}
		ids[mechanism.ID] = struct{}{}
		if err := validateEnvelope("source", mechanism.Source); err != nil {
			return err
		}
		if mechanism.Provenance.Borrow != nil {
			if err := mechanism.Provenance.Borrow.Validate(); err != nil {
				return fmt.Errorf("learningmesh: mechanism %s: %w", mechanism.ID, err)
			}
		}
		for _, artifact := range mechanism.Provenance.Artifacts {
			if strings.TrimSpace(artifact.Kind) == "" || strings.TrimSpace(artifact.ID) == "" {
				return fmt.Errorf("learningmesh: mechanism %s: artifact kind and id are required", mechanism.ID)
			}
			if artifact.Exact {
				if !fullRevisionPattern.MatchString(artifact.Revision) {
					return fmt.Errorf("learningmesh: mechanism %s: exact artifact %s requires a full 40-character lowercase git revision", mechanism.ID, artifact.ID)
				}
				if strings.TrimSpace(artifact.Path) == "" || !sha256Pattern.MatchString(artifact.RecordDigest) {
					return fmt.Errorf("learningmesh: mechanism %s: exact artifact %s requires path and a 64-character lowercase sha256 record_digest", mechanism.ID, artifact.ID)
				}
			}
		}
		if len(mechanism.Provenance.Artifacts) == 0 && mechanism.Provenance.Borrow == nil {
			return fmt.Errorf("learningmesh: mechanism %s: provenance is required", mechanism.ID)
		}
		if mechanism.DefaultDisposition != "" && !mechanism.DefaultDisposition.valid() {
			return fmt.Errorf("learningmesh: mechanism %s: invalid default disposition %q", mechanism.ID, mechanism.DefaultDisposition)
		}
		for _, rule := range mechanism.Rules {
			if !rule.Disposition.valid() || strings.TrimSpace(rule.Reason) == "" {
				return fmt.Errorf("learningmesh: mechanism %s: rules require a valid disposition and reason", mechanism.ID)
			}
		}
	}
	for _, target := range ledger.Targets {
		if err := validateEnvelope("target", target); err != nil {
			return err
		}
	}
	return nil
}

func validateEnvelope(kind string, e Envelope) error {
	if strings.TrimSpace(e.ID) == "" || strings.TrimSpace(e.Hardware) == "" || strings.TrimSpace(e.Framework) == "" || strings.TrimSpace(e.Engine) == "" {
		return fmt.Errorf("learningmesh: %s envelope requires id, hardware, framework, and engine", kind)
	}
	return nil
}

func (d Disposition) valid() bool {
	switch d {
	case Copy, Adapt, BenchmarkOnly, Reject, Unknown:
		return true
	}
	return false
}

func classify(mechanism Mechanism, target Envelope) (Disposition, string) {
	if target.Engine != "fak-native" && !allowedComparatorPurpose(target.Purpose) {
		return Reject, "fak-native invariant: comparator engines are permitted only for benchmark, parity, or interoperability targets"
	}
	for _, rule := range mechanism.Rules {
		if matches(rule.Target, target) {
			if target.Engine != "fak-native" && rule.Disposition != BenchmarkOnly && rule.Disposition != Reject {
				return Reject, "fak-native invariant: comparator target cannot become a product execution path"
			}
			return rule.Disposition, rule.Reason
		}
	}
	if target.Engine != "fak-native" {
		return BenchmarkOnly, "comparator target retained as evidence/reference only"
	}
	if mechanism.DefaultDisposition.valid() {
		return mechanism.DefaultDisposition, "mechanism default disposition"
	}
	return Unknown, "no compatibility rule matched"
}

func allowedComparatorPurpose(purpose string) bool {
	switch purpose {
	case "benchmark", "parity", "interoperability":
		return true
	}
	return false
}

func matches(s Selector, e Envelope) bool {
	return match(s.Hardware, e.Hardware) && match(s.Backend, e.Backend) && match(s.Framework, e.Framework) &&
		match(s.Engine, e.Engine) && match(s.Model, e.Model) && match(s.Workload, e.Workload) &&
		match(s.Purpose, e.Purpose) && match(s.Role, e.Role)
}

func match(want, got string) bool { return want == "" || want == "*" || want == got }

func transferAxes(source, target Envelope) []string {
	var axes []string
	if source.Hardware != target.Hardware {
		axes = append(axes, "hardware")
	}
	if source.Framework != target.Framework {
		axes = append(axes, "framework")
	}
	if source.Engine != target.Engine || source.Role != target.Role {
		axes = append(axes, "baseline")
	}
	if source.Backend != target.Backend {
		axes = append(axes, "backend")
	}
	sort.Strings(axes)
	return axes
}

func stableID(mechanismID string, target Envelope) string {
	b, _ := json.Marshal(struct {
		MechanismID string   `json:"mechanism_id"`
		Target      Envelope `json:"target"`
	}{mechanismID, target})
	return "lm-" + digest(b)[:20]
}

func sourceIdentity(mechanism Mechanism) string {
	if mechanism.Provenance.Borrow != nil {
		return mechanism.Provenance.Borrow.SourceURL + "@" + mechanism.Provenance.Borrow.SourceRef
	}
	a := mechanism.Provenance.Artifacts[0]
	if a.URL != "" {
		return a.URL
	}
	if a.Revision != "" {
		return a.ID + "@" + a.Revision
	}
	return a.ID
}

func normalizeLedger(ledger Ledger) Ledger {
	out := ledger
	sort.Slice(out.Mechanisms, func(i, j int) bool { return out.Mechanisms[i].ID < out.Mechanisms[j].ID })
	sort.Slice(out.Targets, func(i, j int) bool {
		bi, _ := json.Marshal(out.Targets[i])
		bj, _ := json.Marshal(out.Targets[j])
		return string(bi) < string(bj)
	})
	for i := range out.Mechanisms {
		sort.SliceStable(out.Mechanisms[i].Rules, func(a, b int) bool {
			ba, _ := json.Marshal(out.Mechanisms[i].Rules[a])
			bb, _ := json.Marshal(out.Mechanisms[i].Rules[b])
			return string(ba) < string(bb)
		})
		sort.Slice(out.Mechanisms[i].Provenance.Artifacts, func(a, b int) bool {
			x, y := out.Mechanisms[i].Provenance.Artifacts[a], out.Mechanisms[i].Provenance.Artifacts[b]
			return x.Kind+"\x00"+x.ID+"\x00"+x.Revision < y.Kind+"\x00"+y.ID+"\x00"+y.Revision
		})
	}
	return out
}

func digest(b []byte) string { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }

package modelreg

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

const (
	DefaultEvidenceSchema   = "fak.qwen38_default_evidence.v1"
	DefaultEvaluationSchema = "fak.qwen38_default_evaluation.v1"

	VerdictPromote  = "PROMOTE"
	VerdictHold     = "HOLD"
	VerdictRollback = "ROLLBACK"

	StatePass    = "PASS"
	StateFail    = "FAIL"
	StateNA      = "N/A"
	StateMissing = "MISSING"

	qwen38DefaultQuant = "Q4_K_M"
)

type DefaultCheckpoint struct {
	Alias              string `json:"alias"`
	Ref                string `json:"ref"`
	Revision           string `json:"revision"`
	CheckpointSHA256   string `json:"checkpoint_sha256"`
	TokenizerSHA256    string `json:"tokenizer_sha256"`
	ChatTemplateSHA256 string `json:"chat_template_sha256"`
	Quant              string `json:"quant"`
}

type ArtifactRef struct {
	SourceIssue int               `json:"source_issue"`
	URI         string            `json:"uri"`
	SHA256      string            `json:"sha256"`
	ObservedAt  string            `json:"observed_at"`
	StaleAfter  string            `json:"stale_after"`
	Backend     string            `json:"backend"`
	Identity    DefaultCheckpoint `json:"identity"`
}

type HardwareInput struct {
	State      string      `json:"state"`
	Artifact   ArtifactRef `json:"artifact"`
	TextOK     bool        `json:"text_ok"`
	JSONOK     bool        `json:"json_ok"`
	ToolOK     bool        `json:"tool_ok"`
	NoFallback bool        `json:"no_fallback"`
}

type ReuseInput struct {
	State                  string      `json:"state"`
	Artifact               ArtifactRef `json:"artifact"`
	SemanticallyEquivalent bool        `json:"semantically_equivalent"`
	NetSavingsPercent      float64     `json:"net_savings_percent"`
}

type FrontierInput struct {
	State         string      `json:"state"`
	Artifact      ArtifactRef `json:"artifact"`
	Alternatives  int         `json:"alternatives"`
	QualityPassed bool        `json:"quality_passed"`
	Decision      string      `json:"decision"`
}

type FreshnessInput struct {
	State    string      `json:"state"`
	Artifact ArtifactRef `json:"artifact"`
	Fresh    bool        `json:"fresh"`
}

type DefaultEvidenceInput struct {
	Schema           string            `json:"schema"`
	Candidate        DefaultCheckpoint `json:"candidate"`
	CurrentlyDefault bool              `json:"currently_default"`
	PreviousDefault  string            `json:"previous_default,omitempty"`
	MacBook          HardwareInput     `json:"macbook"`
	NVIDIA           HardwareInput     `json:"nvidia"`
	Cache            ReuseInput        `json:"cache"`
	Comparison       FrontierInput     `json:"comparison"`
	Support          FreshnessInput    `json:"support"`
}

type EvaluationReason struct {
	Family string `json:"family"`
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

type EvidenceRef struct {
	Family      string `json:"family"`
	SourceIssue int    `json:"source_issue"`
	URI         string `json:"uri"`
	SHA256      string `json:"sha256"`
}

type DefaultEvaluation struct {
	Schema          string             `json:"schema"`
	Candidate       DefaultCheckpoint  `json:"candidate"`
	Verdict         string             `json:"verdict"`
	PreviousDefault string             `json:"previous_default,omitempty"`
	Reasons         []EvaluationReason `json:"reasons"`
	EvidenceRefs    []EvidenceRef      `json:"evidence_refs"`
}

func EmptyDefaultEvidence() DefaultEvidenceInput {
	return DefaultEvidenceInput{
		Schema: DefaultEvidenceSchema,
		Candidate: DefaultCheckpoint{
			Alias: DefaultAlias,
			Ref:   DefaultRef(),
			Quant: qwen38DefaultQuant,
		},
		MacBook:    HardwareInput{State: StateMissing},
		NVIDIA:     HardwareInput{State: StateMissing},
		Cache:      ReuseInput{State: StateMissing},
		Comparison: FrontierInput{State: StateMissing},
		Support:    FreshnessInput{State: StateMissing},
	}
}

func EvaluateDefaultEvidence(in DefaultEvidenceInput, now time.Time) DefaultEvaluation {
	out := DefaultEvaluation{Schema: DefaultEvaluationSchema, Candidate: in.Candidate, Verdict: VerdictHold, PreviousDefault: in.PreviousDefault, Reasons: []EvaluationReason{}, EvidenceRefs: []EvidenceRef{}}
	regression := false
	add := func(family, code, detail string, isRegression bool) {
		out.Reasons = append(out.Reasons, EvaluationReason{Family: family, Code: code, Detail: detail})
		regression = regression || isRegression
	}

	if in.Schema != DefaultEvidenceSchema {
		add("gate", "SCHEMA_MISMATCH", fmt.Sprintf("got %q, want %q", in.Schema, DefaultEvidenceSchema), false)
	}
	if err := checkDefaultIdentity(in.Candidate); err != nil {
		add("identity", "UNPROVEN_IDENTITY", err.Error(), false)
	}

	foldHardware := func(family, backend string, item HardwareInput) {
		if foldEvidenceFamily(family, item.State, item.Artifact, in.Candidate, 8061, backend, now, &out, add) {
			if !item.TextOK || !item.JSONOK || !item.ToolOK || !item.NoFallback {
				add(family, "FUNCTIONAL_REGRESSION", "PASS requires text, JSON, tool, and no-fallback acceptance", true)
			}
		}
	}
	foldHardware("macbook", "metal", in.MacBook)
	foldHardware("nvidia", "cuda", in.NVIDIA)

	if foldEvidenceFamily("cache", in.Cache.State, in.Cache.Artifact, in.Candidate, 8127, "", now, &out, add) {
		if !in.Cache.SemanticallyEquivalent {
			add("cache", "REUSE_EQUIVALENCE_FAILED", "reused outputs and tool calls must remain semantically equivalent", true)
		}
		if in.Cache.NetSavingsPercent <= 0 {
			add("cache", "NET_REUSE_VALUE_FAILED", "quality-constrained net savings must be greater than zero", true)
		}
	}
	if foldEvidenceFamily("comparison", in.Comparison.State, in.Comparison.Artifact, in.Candidate, 8128, "", now, &out, add) {
		if in.Comparison.Alternatives < 10 || !in.Comparison.QualityPassed || strings.ToUpper(in.Comparison.Decision) != VerdictPromote {
			add("comparison", "FRONTIER_THRESHOLD_FAILED", "PASS requires 10 alternatives, a passing quality gate, and a dated PROMOTE decision", true)
		}
	}
	if foldEvidenceFamily("support", in.Support.State, in.Support.Artifact, in.Candidate, 8129, "", now, &out, add) && !in.Support.Fresh {
		add("support", "FRESHNESS_FAILED", "support evidence is not fresh", true)
	}

	if len(out.Reasons) == 0 {
		out.Verdict = VerdictPromote
		return out
	}
	if regression && in.CurrentlyDefault {
		if strings.TrimSpace(in.PreviousDefault) == "" {
			add("rollback", "MISSING_ROLLBACK_TARGET", "a regressed active default requires previous_default", false)
			return out
		}
		out.Verdict = VerdictRollback
	}
	return out
}

func foldEvidenceFamily(family, state string, ref ArtifactRef, candidate DefaultCheckpoint, requiredIssue int, requiredBackend string, now time.Time, out *DefaultEvaluation, add func(string, string, string, bool)) bool {
	state = strings.ToUpper(strings.TrimSpace(state))
	if ref.URI != "" || ref.SHA256 != "" || ref.SourceIssue != 0 {
		out.EvidenceRefs = append(out.EvidenceRefs, EvidenceRef{Family: family, SourceIssue: ref.SourceIssue, URI: ref.URI, SHA256: ref.SHA256})
	}
	switch state {
	case "", StateMissing:
		add(family, "MISSING_EVIDENCE", "required evidence is missing", false)
		return false
	case StateNA:
		add(family, "NOT_APPLICABLE", "required evidence is explicitly N/A, not zero", false)
		return false
	case StateFail:
		add(family, "DECLARED_REGRESSION", "evidence declares a failed acceptance threshold", true)
		return false
	case StatePass:
	default:
		add(family, "UNKNOWN_STATE", fmt.Sprintf("unrecognized evidence state %q", state), false)
		return false
	}
	if code, err := checkArtifactRef(ref, candidate, requiredIssue, requiredBackend, now); err != nil {
		add(family, code, err.Error(), false)
		return false
	}
	return true
}

func checkDefaultIdentity(item DefaultCheckpoint) error {
	if item.Alias != DefaultAlias || item.Ref != DefaultRef() || item.Quant != qwen38DefaultQuant {
		return fmt.Errorf("exact default identity required: %s -> %s (%s)", DefaultAlias, DefaultRef(), qwen38DefaultQuant)
	}
	if strings.TrimSpace(item.Revision) == "" {
		return fmt.Errorf("candidate revision is required")
	}
	for name, value := range map[string]string{"checkpoint_sha256": item.CheckpointSHA256, "tokenizer_sha256": item.TokenizerSHA256, "chat_template_sha256": item.ChatTemplateSHA256} {
		if !validSHA256(value) {
			return fmt.Errorf("candidate %s must be a 64-hex sha256", name)
		}
	}
	return nil
}

func checkArtifactRef(ref ArtifactRef, candidate DefaultCheckpoint, requiredIssue int, requiredBackend string, now time.Time) (string, error) {
	if ref.SourceIssue <= 0 || strings.TrimSpace(ref.URI) == "" || !validSHA256(ref.SHA256) {
		return "UNPROVEN_ARTIFACT", fmt.Errorf("immutable source_issue, uri, and 64-hex artifact sha256 are required")
	}
	if ref.SourceIssue != requiredIssue {
		return "WRONG_SOURCE", fmt.Errorf("source issue #%d substituted for required #%d", ref.SourceIssue, requiredIssue)
	}
	if ref.Identity != candidate {
		return "SUBSTITUTED_IDENTITY", fmt.Errorf("artifact identity differs from the declared candidate")
	}
	if requiredBackend != "" && !strings.EqualFold(ref.Backend, requiredBackend) {
		return "SUBSTITUTED_BACKEND", fmt.Errorf("backend %q substituted for required %q", ref.Backend, requiredBackend)
	}
	if strings.TrimSpace(ref.Backend) == "" {
		return "UNPROVEN_BACKEND", fmt.Errorf("backend identity is required")
	}
	observedAt, err := time.Parse(time.RFC3339, ref.ObservedAt)
	if err != nil {
		return "UNPROVEN_TIMESTAMP", fmt.Errorf("observed_at must be RFC3339")
	}
	if observedAt.After(now) {
		return "FUTURE_EVIDENCE", fmt.Errorf("evidence observed_at %s is in the future", observedAt.Format(time.RFC3339))
	}
	staleAfter, err := time.Parse(time.RFC3339, ref.StaleAfter)
	if err != nil {
		return "UNPROVEN_TIMESTAMP", fmt.Errorf("stale_after must be RFC3339")
	}
	if !now.Before(staleAfter) {
		return "STALE_EVIDENCE", fmt.Errorf("evidence expired at %s", staleAfter.Format(time.RFC3339))
	}
	if !observedAt.Before(staleAfter) {
		return "CONTRADICTORY_TIMESTAMPS", fmt.Errorf("stale_after must be later than observed_at")
	}
	return "", nil
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

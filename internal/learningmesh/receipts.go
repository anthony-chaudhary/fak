package learningmesh

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/studylink"
)

// ReceiptInput binds a native-performance receipt to the repository-relative
// artifact path whose exact bytes were decoded.
type ReceiptInput struct {
	Path  string
	Bytes []byte
}

// LedgerFromReceipts converts portable native-performance receipts into the
// provider-neutral learning mesh. Receipt validation remains owned by
// nativeperf; this adapter preserves only the dimensions needed for transfer.
func LedgerFromReceipts(inputs []ReceiptInput, targets []Envelope) (Ledger, error) {
	if len(inputs) == 0 {
		return Ledger{}, fmt.Errorf("learningmesh: at least one receipt is required")
	}
	if len(targets) == 0 {
		return Ledger{}, fmt.Errorf("learningmesh: at least one target is required")
	}
	byID := make(map[string]Mechanism)
	for _, input := range inputs {
		portable, err := nativeperf.DecodePortableReceipt(input.Bytes)
		if err != nil {
			return Ledger{}, fmt.Errorf("learningmesh: receipt %s: %w", input.Path, err)
		}
		r := portable.Receipt
		backend := strings.ToLower(strings.TrimSpace(r.Machine.Backend))
		hardware, err := hardwareForBackend(backend)
		if err != nil {
			return Ledger{}, fmt.Errorf("learningmesh: receipt %s: %w", input.Path, err)
		}
		engine := r.Execution.Engine
		if strings.TrimSpace(r.ChangedLeverID) == "" || strings.TrimSpace(r.EnvelopeID) == "" || strings.TrimSpace(r.Revision) == "" {
			return Ledger{}, fmt.Errorf("learningmesh: receipt %s requires changed_lever_id, envelope_id, and revision", input.Path)
		}
		if !sha256Pattern.MatchString(strings.ToLower(r.ArtifactSHA256)) {
			return Ledger{}, fmt.Errorf("learningmesh: receipt %s has invalid artifact_sha256", input.Path)
		}
		path := filepath.ToSlash(strings.TrimSpace(input.Path))
		if filepath.IsAbs(path) {
			if rel, relErr := filepath.Rel(".", path); relErr == nil && rel != ".." && !strings.HasPrefix(filepath.ToSlash(rel), "../") {
				path = filepath.ToSlash(rel)
			}
		}
		if path == "" || filepath.IsAbs(path) || strings.HasPrefix(path, "../") {
			return Ledger{}, fmt.Errorf("learningmesh: receipt path must be repository-relative")
		}
		model, workload := receiptModelWorkload(r)
		id := receiptMechanismID(r.ChangedLeverID)
		candidate := Mechanism{
			ID: id, Mechanism: r.ChangedLeverID,
			Source: Envelope{ID: r.EnvelopeID, Hardware: hardware, Backend: backend, Framework: "fak-native", Engine: engine, Model: model, Workload: workload, Role: "product"},
			Provenance: Provenance{Artifacts: []studylink.Artifact{
				{Kind: "nativeperf-receipt", ID: path, Revision: receiptRevision(r.Revision), Path: path, Exact: true, RecordDigest: portable.ReceiptSHA256},
				{Kind: "model-artifact", ID: "sha256:" + portable.ModelArtifactSHA256, RecordDigest: portable.ModelArtifactSHA256},
			}},
			DefaultDisposition: Adapt,
		}
		if prior, ok := byID[id]; ok {
			if prior.Mechanism != candidate.Mechanism {
				return Ledger{}, fmt.Errorf("learningmesh: receipt mechanism id collision %q", id)
			}
			prior.Provenance.Artifacts = append(prior.Provenance.Artifacts, candidate.Provenance.Artifacts...)
			byID[id] = prior
			continue
		}
		byID[id] = candidate
	}
	mechanisms := make([]Mechanism, 0, len(byID))
	for _, mechanism := range byID {
		mechanisms = append(mechanisms, mechanism)
	}
	sort.Slice(mechanisms, func(i, j int) bool { return mechanisms[i].ID < mechanisms[j].ID })
	for i := range mechanisms {
		mechanisms[i].Provenance.Artifacts = dedupeArtifacts(mechanisms[i].Provenance.Artifacts)
	}
	return Ledger{Schema: InputSchema, Mechanisms: mechanisms, Targets: targets}, nil
}

func hardwareForBackend(backend string) (string, error) {
	switch backend {
	case "cuda":
		return "nvidia", nil
	case "metal":
		return "apple", nil
	case "vulkan", "rocm", "hip":
		return "amd", nil
	default:
		return "", fmt.Errorf("unsupported backend %q", backend)
	}
}

func receiptModelWorkload(r nativeperf.ExperimentReceipt) (string, string) {
	text := strings.ToLower(r.EnvelopeID + " " + r.Execution.ForwardPath)
	model := ""
	for _, candidate := range []struct {
		canonical string
		aliases   []string
	}{
		{canonical: "qwen3.8", aliases: []string{"qwen3.8", "qwen38"}},
		{canonical: "qwen3.6", aliases: []string{"qwen3.6", "qwen36"}},
		{canonical: "glm5", aliases: []string{"glm5"}},
		{canonical: "llama", aliases: []string{"llama"}},
	} {
		for _, alias := range candidate.aliases {
			if strings.Contains(text, alias) {
				model = candidate.canonical
				break
			}
		}
		if model != "" {
			break
		}
	}
	workload := "inference"
	for _, token := range []string{"decode", "prefill", "serving", "training"} {
		if strings.Contains(text, token) {
			workload = token
			break
		}
	}
	return model, workload
}

func receiptMechanismID(lever string) string {
	normalized := strings.ToLower(strings.TrimSpace(lever))
	var b strings.Builder
	lastDash := false
	for _, r := range normalized {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
			lastDash = false
		} else if !lastDash && b.Len() > 0 {
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

func receiptRevision(revision string) string {
	revision = strings.ToLower(strings.TrimSpace(revision))
	if len(revision) >= 40 && fullRevisionPattern.MatchString(revision[:40]) {
		return revision[:40]
	}
	return revision
}

func dedupeArtifacts(in []studylink.Artifact) []studylink.Artifact {
	seen := make(map[string]studylink.Artifact)
	for _, artifact := range in {
		key := artifact.Kind + "\x00" + artifact.ID + "\x00" + artifact.Revision + "\x00" + artifact.RecordDigest
		seen[key] = artifact
	}
	out := make([]studylink.Artifact, 0, len(seen))
	for _, artifact := range seen {
		out = append(out, artifact)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID+out[i].RecordDigest < out[j].ID+out[j].RecordDigest })
	return out
}

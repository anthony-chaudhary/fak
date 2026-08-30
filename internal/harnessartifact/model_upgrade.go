package harnessartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessserve"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const (
	ModelUpgradePlanSchema    = "fak.harness.model-upgrade-plan.v1"
	ModelUpgradeStateSchema   = "fak.harness.model-upgrade-state.v1"
	ModelCleanupPreviewSchema = "fak.harness.model-cleanup-preview.v1"
)

type ModelUpgradePhase string

const (
	UpgradeVerify  ModelUpgradePhase = "verify"
	UpgradeAcquire ModelUpgradePhase = "acquire"
	UpgradeAdmit   ModelUpgradePhase = "admit"
	UpgradeStart   ModelUpgradePhase = "start"
	UpgradeProbe   ModelUpgradePhase = "probe"
	UpgradePromote ModelUpgradePhase = "promote"
)

type ModelUpgradeRefusal struct {
	Phase  ModelUpgradePhase
	Detail string
	Err    error
}

func (r *ModelUpgradeRefusal) Error() string {
	if r.Err != nil {
		return fmt.Sprintf("model upgrade %s refused: %s: %v", r.Phase, r.Detail, r.Err)
	}
	return fmt.Sprintf("model upgrade %s refused: %s", r.Phase, r.Detail)
}

func (r *ModelUpgradeRefusal) Unwrap() error { return r.Err }

type ModelRuntimeReceipt struct {
	ID                string                    `json:"id"`
	DeclarationSHA256 string                    `json:"declaration_sha256"`
	BlobSHA256        string                    `json:"blob_sha256"`
	BlobBytes         int64                     `json:"blob_bytes"`
	RuntimeID         string                    `json:"runtime_id"`
	AdmissionID       string                    `json:"admission_id"`
	Endpoint          string                    `json:"endpoint"`
	Ownership         harnessserve.Ownership    `json:"ownership"`
	Probe             harnessserve.ProbeReceipt `json:"probe"`
	ReadyAt           time.Time                 `json:"ready_at"`
}

type ModelRevision struct {
	Declaration harnesskit.LocalModelDeclaration `json:"declaration"`
	Receipt     ModelRuntimeReceipt              `json:"receipt"`
}

// ModelUpgradeState is the single atomic lifecycle pointer. Keeping the active
// and rollback revisions in one file prevents a declaration from advancing
// without the receipt that proved it.
type ModelUpgradeState struct {
	Schema   string         `json:"schema"`
	Active   ModelRevision  `json:"active"`
	Rollback *ModelRevision `json:"rollback,omitempty"`
}

type ModelUpgradeRequest struct {
	StatePath       string                           `json:"state_path"`
	Candidate       harnesskit.LocalModelDeclaration `json:"candidate"`
	PinnedBlobBytes int64                            `json:"pinned_blob_bytes"`
	Serve           harnessserve.Plan                `json:"serve"`
}

type ModelUpgradePlan struct {
	Schema                     string                           `json:"schema"`
	StatePath                  string                           `json:"state_path"`
	Previous                   ModelRevision                    `json:"previous"`
	PreviousStateSHA256        string                           `json:"previous_state_sha256"`
	Candidate                  harnesskit.LocalModelDeclaration `json:"candidate"`
	CandidateDeclarationSHA256 string                           `json:"candidate_declaration_sha256"`
	PinnedBlobBytes            int64                            `json:"pinned_blob_bytes"`
	Serve                      harnessserve.Plan                `json:"serve"`
}

type ModelAdmission struct {
	ID string `json:"id"`
}

type ModelRuntime struct {
	ID        string                 `json:"id"`
	Endpoint  string                 `json:"endpoint"`
	Ownership harnessserve.Ownership `json:"ownership"`
}

// ModelUpgradeMutations is deliberately injected. The lifecycle layer does
// not download weights, infer a GPU, or choose a runtime behind the caller's
// back; an adapter must make each effect explicit.
type ModelUpgradeMutations interface {
	Verify(context.Context, ModelUpgradePlan) error
	Acquire(context.Context, ModelUpgradePlan) (string, error)
	Admit(context.Context, ModelUpgradePlan, string) (ModelAdmission, error)
	Start(context.Context, ModelUpgradePlan, ModelAdmission) (ModelRuntime, error)
	Probe(context.Context, ModelRuntime, int) (harnessserve.ProbeReceipt, error)
	Stop(context.Context, ModelRuntime) error
	Unload(context.Context, ModelAdmission) error
}

type ModelUpgradeApplyOptions struct {
	Mutations ModelUpgradeMutations
	Promote   func(string, ModelUpgradeState) error
	Now       func() time.Time
}

// PrepareModelUpgrade captures one immutable local-model transition. It does
// not select repository work or schedule execution; ApplyModelUpgrade is the
// only lifecycle mutation step.
func PrepareModelUpgrade(req ModelUpgradeRequest) (ModelUpgradePlan, error) {
	statePath := filepath.Clean(strings.TrimSpace(req.StatePath))
	if !filepath.IsAbs(statePath) {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "state_path must be absolute"}
	}
	state, raw, err := readModelUpgradeState(statePath)
	if err != nil {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "read current lifecycle state", Err: err}
	}
	canonical, err := CanonicalLocalModelDeclaration(req.Candidate)
	if err != nil {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "candidate declaration is not canonical", Err: err}
	}
	var candidate harnesskit.LocalModelDeclaration
	if err := json.Unmarshal(canonical, &candidate); err != nil {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "decode canonical candidate", Err: err}
	}
	if candidate.GGUFPath == state.Active.Declaration.GGUFPath && candidate.GGUFSHA256 == state.Active.Declaration.GGUFSHA256 {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "candidate is already active"}
	}
	if req.PinnedBlobBytes <= 0 {
		return ModelUpgradePlan{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "pinned_blob_bytes must be positive"}
	}
	stateSum := sha256.Sum256(raw)
	return ModelUpgradePlan{
		Schema: ModelUpgradePlanSchema, StatePath: statePath, Previous: state.Active,
		PreviousStateSHA256: hex.EncodeToString(stateSum[:]), Candidate: candidate,
		CandidateDeclarationSHA256: LocalModelDeclarationDigest(canonical), PinnedBlobBytes: req.PinnedBlobBytes, Serve: req.Serve,
	}, nil
}

func ApplyModelUpgrade(ctx context.Context, plan ModelUpgradePlan, opts ModelUpgradeApplyOptions) (ModelUpgradeState, error) {
	if err := validatePreparedUpgrade(plan); err != nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "prepared transition is invalid", Err: err}
	}
	if opts.Mutations == nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "valid plan and explicit mutation adapter are required"}
	}
	current, raw, err := readModelUpgradeState(plan.StatePath)
	if err != nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "read current lifecycle state", Err: err}
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != plan.PreviousStateSHA256 || current.Active.Receipt.ID != plan.Previous.Receipt.ID {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "lifecycle state changed after planning"}
	}
	if err := opts.Mutations.Verify(ctx, plan); err != nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeVerify, Detail: "candidate verification failed", Err: err}
	}
	blobPath, err := opts.Mutations.Acquire(ctx, plan)
	if err != nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeAcquire, Detail: "candidate acquisition failed", Err: err}
	}
	if filepath.Clean(blobPath) != filepath.Clean(plan.Candidate.GGUFPath) {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeAcquire, Detail: "adapter returned a path other than the pinned candidate path"}
	}
	if err := verifyPinnedBlob(blobPath, plan.Candidate.GGUFSHA256, plan.PinnedBlobBytes); err != nil {
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeAcquire, Detail: "acquired bytes do not match the candidate pin", Err: err}
	}
	admission, err := opts.Mutations.Admit(ctx, plan, blobPath)
	if err != nil || strings.TrimSpace(admission.ID) == "" {
		if err == nil {
			err = errors.New("adapter returned an empty admission identity")
		}
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeAdmit, Detail: "candidate admission failed", Err: err}
	}
	unload := func() { _ = opts.Mutations.Unload(context.Background(), admission) }
	runtimeInstance, err := opts.Mutations.Start(ctx, plan, admission)
	if err != nil || strings.TrimSpace(runtimeInstance.ID) == "" {
		unload()
		if err == nil {
			err = errors.New("adapter returned an empty runtime identity")
		}
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeStart, Detail: "candidate start failed", Err: err}
	}
	rollbackCandidate := func() {
		_ = opts.Mutations.Stop(context.Background(), runtimeInstance)
		unload()
	}
	probe, err := opts.Mutations.Probe(ctx, runtimeInstance, 1)
	if err != nil || probe.CompletionTokens != 1 {
		rollbackCandidate()
		if err == nil {
			err = fmt.Errorf("completion_tokens=%d, want exactly 1", probe.CompletionTokens)
		}
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradeProbe, Detail: "one-token candidate probe failed", Err: err}
	}
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}
	newState := ModelUpgradeState{
		Schema: ModelUpgradeStateSchema,
		Active: ModelRevision{Declaration: plan.Candidate, Receipt: ModelRuntimeReceipt{
			ID:                plan.CandidateDeclarationSHA256 + ":" + runtimeInstance.ID,
			DeclarationSHA256: plan.CandidateDeclarationSHA256, BlobSHA256: plan.Candidate.GGUFSHA256,
			BlobBytes: plan.PinnedBlobBytes,
			RuntimeID: runtimeInstance.ID, AdmissionID: admission.ID, Endpoint: runtimeInstance.Endpoint,
			Ownership: runtimeInstance.Ownership, Probe: probe, ReadyAt: now().UTC(),
		}},
		Rollback: &current.Active,
	}
	promote := writeModelUpgradeStateAtomic
	if opts.Promote != nil {
		promote = opts.Promote
	}
	if err := promote(plan.StatePath, newState); err != nil {
		rollbackCandidate()
		return ModelUpgradeState{}, &ModelUpgradeRefusal{Phase: UpgradePromote, Detail: "atomic lifecycle promotion failed", Err: err}
	}
	return newState, nil
}

// StopModel ends the owned runtime but does not unload admitted weights or
// delete bytes. UnloadModel releases admission but does not signal a process or
// delete bytes; callers choose and record each transition separately.
func StopModel(ctx context.Context, mutations ModelUpgradeMutations, runtimeInstance ModelRuntime) error {
	if mutations == nil {
		return errors.New("stop requires an explicit mutation adapter")
	}
	return mutations.Stop(ctx, runtimeInstance)
}

func UnloadModel(ctx context.Context, mutations ModelUpgradeMutations, admission ModelAdmission) error {
	if mutations == nil {
		return errors.New("unload requires an explicit mutation adapter")
	}
	return mutations.Unload(ctx, admission)
}

type ModelCleanupOperation string

const (
	CleanupEvict ModelCleanupOperation = "evict"
	CleanupPurge ModelCleanupOperation = "purge"
)

type ModelCleanupRequest struct {
	Operation  ModelCleanupOperation `json:"operation"`
	CacheDir   string                `json:"cache_dir"`
	Targets    []string              `json:"targets,omitempty"`
	Referenced []string              `json:"referenced"`
}

type ModelCleanupEntry struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type ModelCleanupPreview struct {
	Schema     string                `json:"schema"`
	Operation  ModelCleanupOperation `json:"operation"`
	CacheDir   string                `json:"cache_dir"`
	Referenced []string              `json:"referenced"`
	Delete     []ModelCleanupEntry   `json:"delete"`
}

type ModelCleanupReceipt struct {
	Operation ModelCleanupOperation `json:"operation"`
	Deleted   []string              `json:"deleted"`
}

func PreviewModelCleanup(req ModelCleanupRequest) (ModelCleanupPreview, error) {
	root := filepath.Clean(strings.TrimSpace(req.CacheDir))
	if !filepath.IsAbs(root) || (req.Operation != CleanupEvict && req.Operation != CleanupPurge) {
		return ModelCleanupPreview{}, errors.New("cleanup requires an absolute cache_dir and operation evict or purge")
	}
	referenced, err := normalizeExactPaths(req.Referenced)
	if err != nil {
		return ModelCleanupPreview{}, fmt.Errorf("referenced paths: %w", err)
	}
	referenceSet := make(map[string]bool, len(referenced))
	for _, path := range referenced {
		referenceSet[path] = true
	}
	var candidates []string
	if req.Operation == CleanupEvict {
		if len(req.Targets) == 0 {
			return ModelCleanupPreview{}, errors.New("evict requires at least one exact target")
		}
		candidates, err = normalizeExactPaths(req.Targets)
		if err != nil {
			return ModelCleanupPreview{}, fmt.Errorf("evict targets: %w", err)
		}
	} else {
		entries, err := os.ReadDir(root)
		if err != nil {
			return ModelCleanupPreview{}, err
		}
		for _, entry := range entries {
			if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
				candidates = append(candidates, filepath.Join(root, entry.Name()))
			}
		}
	}
	sort.Strings(candidates)
	preview := ModelCleanupPreview{Schema: ModelCleanupPreviewSchema, Operation: req.Operation, CacheDir: root, Referenced: referenced, Delete: []ModelCleanupEntry{}}
	for _, path := range candidates {
		if !pathWithin(root, path) || !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return ModelCleanupPreview{}, fmt.Errorf("cleanup target %q is not an exact .gguf path under cache_dir", path)
		}
		if referenceSet[path] {
			if req.Operation == CleanupEvict {
				return ModelCleanupPreview{}, fmt.Errorf("refusing to evict referenced blob %q", path)
			}
			continue
		}
		digest, size, err := hashFile(path)
		if err != nil {
			return ModelCleanupPreview{}, err
		}
		preview.Delete = append(preview.Delete, ModelCleanupEntry{Path: path, SHA256: digest, Bytes: size})
	}
	return preview, nil
}

func ApplyModelCleanup(preview ModelCleanupPreview) (ModelCleanupReceipt, error) {
	if preview.Schema != ModelCleanupPreviewSchema || (preview.Operation != CleanupEvict && preview.Operation != CleanupPurge) || !filepath.IsAbs(preview.CacheDir) {
		return ModelCleanupReceipt{}, errors.New("valid cleanup preview is required")
	}
	referencePaths, err := normalizeExactPaths(preview.Referenced)
	if err != nil {
		return ModelCleanupReceipt{}, fmt.Errorf("preview referenced paths: %w", err)
	}
	referenced := make(map[string]bool, len(referencePaths))
	for _, path := range referencePaths {
		referenced[path] = true
	}
	receipt := ModelCleanupReceipt{Operation: preview.Operation, Deleted: []string{}}
	seen := make(map[string]bool, len(preview.Delete))
	for _, entry := range preview.Delete {
		path := filepath.Clean(entry.Path)
		if referenced[path] || !pathWithin(preview.CacheDir, path) || !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return receipt, fmt.Errorf("preview contains unsafe or referenced path %q", path)
		}
		if seen[path] {
			return receipt, fmt.Errorf("preview repeats cleanup path %q", path)
		}
		seen[path] = true
		digest, size, err := hashFile(path)
		if err != nil {
			return receipt, err
		}
		if digest != entry.SHA256 || size != entry.Bytes {
			return receipt, fmt.Errorf("previewed blob %q changed before cleanup", path)
		}
		if err := os.Remove(path); err != nil {
			return receipt, err
		}
		receipt.Deleted = append(receipt.Deleted, path)
	}
	return receipt, nil
}

func validatePreparedUpgrade(plan ModelUpgradePlan) error {
	if plan.Schema != ModelUpgradePlanSchema || !filepath.IsAbs(plan.StatePath) || plan.PinnedBlobBytes <= 0 {
		return errors.New("schema, absolute state_path, and positive pinned_blob_bytes are required")
	}
	canonical, err := CanonicalLocalModelDeclaration(plan.Candidate)
	if err != nil {
		return err
	}
	if LocalModelDeclarationDigest(canonical) != plan.CandidateDeclarationSHA256 {
		return errors.New("candidate declaration digest changed after planning")
	}
	if !sha256Pattern.MatchString(plan.PreviousStateSHA256) || strings.TrimSpace(plan.Previous.Receipt.ID) == "" {
		return errors.New("previous state identity is invalid")
	}
	return nil
}

func readModelUpgradeState(path string) (ModelUpgradeState, []byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return ModelUpgradeState{}, nil, err
	}
	var state ModelUpgradeState
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&state); err != nil {
		return ModelUpgradeState{}, nil, err
	}
	if state.Schema != ModelUpgradeStateSchema || state.Active.Receipt.ID == "" {
		return ModelUpgradeState{}, nil, errors.New("invalid model upgrade state")
	}
	return state, raw, nil
}

func writeModelUpgradeStateAtomic(path string, state ModelUpgradeState) error {
	raw, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".model-upgrade-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err = tmp.Write(raw); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func verifyPinnedBlob(path, want string, wantBytes int64) error {
	got, gotBytes, err := hashFile(path)
	if err != nil {
		return err
	}
	if gotBytes != wantBytes {
		return fmt.Errorf("bytes=%d want %d", gotBytes, wantBytes)
	}
	if got != strings.ToLower(want) {
		return fmt.Errorf("sha256=%s want %s", got, want)
	}
	return nil
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func normalizeExactPaths(paths []string) ([]string, error) {
	set := map[string]bool{}
	for _, rawPath := range paths {
		path := filepath.Clean(strings.TrimSpace(rawPath))
		if !filepath.IsAbs(path) {
			return nil, fmt.Errorf("path %q must be absolute", rawPath)
		}
		set[path] = true
	}
	out := make([]string, 0, len(set))
	for path := range set {
		out = append(out, path)
	}
	sort.Strings(out)
	return out, nil
}

func pathWithin(root, path string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(path))
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

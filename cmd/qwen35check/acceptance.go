package main

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	acceptanceManifestSchema = "fak-qwen35check-exact-qwen36-reference/1"
	acceptanceReportSchema   = "fak-qwen35check-exact-qwen36-cuda-comparison/1"

	// These four MUST track EXPECTED_SHA256/EXPECTED_SIZE in
	// tools/qwen36_a100_fak_serve.sh and scripts/gcp-qwen-serve.sh — the launchers that
	// actually serve this checkpoint. The previous pin here (16547398784 / 33625d8d…) is
	// the exact source internal/modelreg records in blockedSources as "known stale (HTTP
	// 404)", so acceptance mode demanded a checkpoint the repo itself refuses to fetch: it
	// failed on identity before comparing a single logit, and the GDN/SSM parity this mode
	// exists to prove could never run. A drifted pin here does not fail loudly, it fails
	// silently and looks like "parity unproven" forever.
	acceptanceModelID         = "Qwen/Qwen3.6-27B"
	acceptanceCheckpointRepo  = "unsloth/Qwen3.6-27B-GGUF"
	acceptanceCheckpointFile  = "Qwen3.6-27B-Q4_K_M.gguf"
	acceptanceCheckpointBytes = int64(16817244384)
	acceptanceCheckpointSHA   = "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
	acceptanceModelRevision   = "sha256:" + acceptanceCheckpointSHA

	acceptanceLoadPath    = "gguf-q4_k_m/resident-q4k"
	acceptanceLogitsDType = "float32"
	acceptanceCPUTarget   = "cpu-ref/legacy-qwen35-gdn-reference-v1"
	acceptanceMaxSteps    = 64
	acceptanceMaxFileSize = 1 << 30
)

type acceptanceModelIdentity struct {
	ID             string `json:"id"`
	Revision       string `json:"revision"`
	CheckpointRepo string `json:"checkpoint_repo"`
	CheckpointFile string `json:"checkpoint_file"`
	SizeBytes      int64  `json:"size_bytes"`
	SHA256         string `json:"sha256"`
}

type acceptanceReferenceStep struct {
	Index     int       `json:"index"`
	InputID   int       `json:"input_id"`
	ArgmaxID  int       `json:"argmax_id"`
	Norm      float64   `json:"norm"`
	MaxAbs    float64   `json:"max_abs"`
	LogitsF32 []float32 `json:"logits_f32"`
}

// acceptanceManifest is deliberately timing-free: the same model, fixed ids, and
// CPU reference arithmetic marshal to the same bytes on every generation. The CUDA
// comparison report below carries the necessarily non-deterministic timing fields.
type acceptanceManifest struct {
	Schema           string                    `json:"schema"`
	IntegritySHA256  string                    `json:"integrity_sha256"`
	Model            acceptanceModelIdentity   `json:"model"`
	Backend          string                    `json:"backend"`
	BackendPath      string                    `json:"backend_path"`
	RequiredCUDAPath string                    `json:"required_cuda_path"`
	LoadPath         string                    `json:"load_path"`
	LogitsDType      string                    `json:"logits_dtype"`
	PromptIDs        []int                     `json:"prompt_ids"`
	TeacherForcedIDs []int                     `json:"teacher_forced_ids"`
	Steps            []acceptanceReferenceStep `json:"steps"`
}

type acceptanceComparisonStep struct {
	Index           int     `json:"index"`
	InputID         int     `json:"input_id"`
	ReferenceNorm   float64 `json:"reference_norm"`
	CUDANorm        float64 `json:"cuda_norm"`
	Cosine          float64 `json:"cosine"`
	MaxAbs          float64 `json:"max_abs"`
	ReferenceArgmax int     `json:"reference_argmax"`
	CUDAArgmax      int     `json:"cuda_argmax"`
	EqualArgmax     bool    `json:"equal_argmax"`
	Finite          bool    `json:"finite"`
	DecodeNS        int64   `json:"decode_ns"`
	Passed          bool    `json:"passed"`
}

type acceptanceDecodeTiming struct {
	Steps        int     `json:"steps"`
	TotalNS      int64   `json:"total_ns"`
	TokensPerSec float64 `json:"tokens_per_sec"`
}

type acceptanceReport struct {
	Schema                   string                     `json:"schema"`
	ReferenceIntegritySHA256 string                     `json:"reference_integrity_sha256"`
	Model                    acceptanceModelIdentity    `json:"model"`
	Backend                  string                     `json:"backend"`
	BackendPath              string                     `json:"backend_path"`
	RequiredCUDAPath         string                     `json:"required_cuda_path"`
	LoadPath                 string                     `json:"load_path"`
	PromptIDs                []int                      `json:"prompt_ids"`
	TeacherForcedIDs         []int                      `json:"teacher_forced_ids"`
	CosineMinimum            float64                    `json:"cosine_minimum"`
	Steps                    []acceptanceComparisonStep `json:"steps"`
	Decode                   acceptanceDecodeTiming     `json:"decode"`
	Passed                   bool                       `json:"passed"`
}

type acceptanceOptions struct {
	GGUF          string
	Dir           string
	Backend       string
	Reference     string
	Out           string
	IDs           string
	TeacherForced string
	JSONOnly      bool
}

func expectedAcceptanceModel() acceptanceModelIdentity {
	return acceptanceModelIdentity{
		ID:             acceptanceModelID,
		Revision:       acceptanceModelRevision,
		CheckpointRepo: acceptanceCheckpointRepo,
		CheckpointFile: acceptanceCheckpointFile,
		SizeBytes:      acceptanceCheckpointBytes,
		SHA256:         "sha256:" + acceptanceCheckpointSHA,
	}
}

func acceptanceRequested(backend, reference, teacherForced string) bool {
	return backend != "" || reference != "" || teacherForced != ""
}

func runAcceptance(opts acceptanceOptions) error {
	if opts.Backend != "cpu-ref" && opts.Backend != "cuda" {
		return fmt.Errorf("acceptance: -backend must be exactly cpu-ref or cuda")
	}
	if opts.Dir != "" || opts.GGUF == "" {
		return fmt.Errorf("acceptance: exact Qwen3.6 acceptance requires -gguf and refuses -dir because the contract pins one complete file")
	}
	if opts.Out == "" || opts.Out == "-" {
		return fmt.Errorf("acceptance: -out must name a durable JSON artifact")
	}
	if opts.JSONOnly {
		return fmt.Errorf("acceptance: -json is incompatible with the required -out artifact")
	}
	prompt, err := parseIDList(opts.IDs)
	if err != nil {
		return fmt.Errorf("acceptance: prompt ids: %w", err)
	}
	teacher, err := parseIDList(opts.TeacherForced)
	if err != nil {
		return fmt.Errorf("acceptance: teacher-forced ids: %w", err)
	}
	if err := validateAcceptanceIDs(prompt, teacher); err != nil {
		return err
	}

	var reference acceptanceManifest
	if opts.Backend == "cpu-ref" {
		if opts.Reference != "" {
			return fmt.Errorf("acceptance: cpu-ref generates the independent reference and refuses -reference")
		}
	} else {
		if opts.Reference == "" {
			return fmt.Errorf("acceptance: cuda requires -reference from an independent cpu-ref run")
		}
		reference, err = readAcceptanceManifest(opts.Reference)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(prompt, reference.PromptIDs) {
			return fmt.Errorf("acceptance: explicit prompt ids do not match reference: got=%v want=%v", prompt, reference.PromptIDs)
		}
		if !reflect.DeepEqual(teacher, reference.TeacherForcedIDs) {
			return fmt.Errorf("acceptance: explicit teacher-forced ids do not match reference: got=%v want=%v", teacher, reference.TeacherForcedIDs)
		}
	}

	backend, err := resolveAcceptanceBackend(opts.Backend, compute.Lookup)
	if err != nil {
		return err
	}
	// Size and content identity are checked before even the header parser sees the
	// checkpoint. A wrong artifact therefore cannot allocate or execute model payload.
	if err := verifyAcceptanceModelFile(opts.GGUF, expectedAcceptanceModel()); err != nil {
		return err
	}
	header, err := ggufload.Open(opts.GGUF)
	if err != nil {
		return fmt.Errorf("acceptance: open verified GGUF header: %w", err)
	}
	cfg, err := header.Config()
	if err != nil {
		return fmt.Errorf("acceptance: parse verified GGUF config: %w", err)
	}
	if err := validateExactQwen36Config(cfg); err != nil {
		return err
	}
	if err := validateAcceptanceBackendConfig(cfg, opts.Backend, backend); err != nil {
		return err
	}

	// Acceptance always selects the resident-Q4_K loader explicitly. It must not
	// inherit FAK_Q4K (or its absence) from the caller's environment.
	m, err := ggufload.LoadModelQ4K(opts.GGUF)
	if err != nil {
		return fmt.Errorf("acceptance: load exact checkpoint: %w", err)
	}
	if err := validateExactQwen36Config(m.Cfg); err != nil {
		return fmt.Errorf("acceptance: loaded model identity changed after header admission: %w", err)
	}
	s, err := newAcceptanceSession(m, opts.Backend, backend)
	if err != nil {
		return err
	}
	defer s.Close()

	logits, durations, err := executeTeacherForced(s, prompt, teacher)
	if err != nil {
		return err
	}
	if opts.Backend == "cpu-ref" {
		manifest, err := buildAcceptanceManifest(prompt, teacher, logits)
		if err != nil {
			return err
		}
		return writeAcceptanceManifest(opts.Out, manifest)
	}

	report, compareErr := compareAcceptance(reference, logits, durations)
	if err := writeAcceptanceJSON(opts.Out, report); err != nil {
		return err
	}
	return compareErr
}

type acceptanceBackendLookup func(string) (compute.Backend, bool)

func resolveAcceptanceBackend(name string, lookup acceptanceBackendLookup) (compute.Backend, error) {
	if lookup == nil {
		return nil, fmt.Errorf("acceptance: backend registry is unavailable")
	}
	be, ok := lookup(name)
	if !ok || be == nil {
		return nil, fmt.Errorf("acceptance: required backend %q is not registered; refusing fallback", name)
	}
	if be.Name() != name {
		return nil, fmt.Errorf("acceptance: backend lookup for %q returned %q; refusing fallback", name, be.Name())
	}
	if name == "cpu-ref" && be.Class() != compute.Reference {
		return nil, fmt.Errorf("acceptance: cpu-ref backend class is %v, want reference", be.Class())
	}
	return be, nil
}

func validateAcceptanceBackendConfig(cfg model.Config, name string, be compute.Backend) error {
	switch name {
	case "cpu-ref":
		// Qwen35's real CPU reference is the legacy direct session, represented by a
		// nil HAL backend in the model contract. NewBackendSessionChecked(cpu-ref)
		// correctly refuses this hybrid because cpu-ref does not implement the whole
		// GDN HAL operation; do not weaken that contract or invent an adapter here.
		if be == nil || be.Name() != "cpu-ref" || be.Class() != compute.Reference {
			return fmt.Errorf("acceptance: cpu-ref reference backend identity mismatch")
		}
		if err := model.ValidateBackendForwardConfig(cfg, nil); err != nil {
			return fmt.Errorf("acceptance: cpu reference path refused: %w", err)
		}
		return nil
	case "cuda":
		if be == nil || be.Name() != "cuda" {
			return fmt.Errorf("acceptance: cuda backend identity mismatch; refusing fallback")
		}
		if err := model.ValidateBackendForwardConfig(cfg, be); err != nil {
			return fmt.Errorf("acceptance: cuda forward admission: %w", err)
		}
		gdn, ok := be.(model.Qwen35GDNBackend)
		if !ok || gdn.Qwen35GDNPath() != model.Qwen35GDNCUDAPath {
			return fmt.Errorf("acceptance: cuda backend path mismatch: require %q", model.Qwen35GDNCUDAPath)
		}
		return nil
	default:
		return fmt.Errorf("acceptance: unsupported backend %q", name)
	}
}

func newAcceptanceSession(m *model.Model, backendName string, be compute.Backend) (*model.Session, error) {
	if m == nil {
		return nil, fmt.Errorf("acceptance: model is absent")
	}
	if backendName == "cpu-ref" {
		s := m.NewSession()
		s.Q4K = true
		return s, nil
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return nil, fmt.Errorf("acceptance: checked cuda session: %w", err)
	}
	if s.Backend == nil || s.Backend.Name() != "cuda" {
		s.Close()
		return nil, fmt.Errorf("acceptance: checked cuda session did not retain cuda backend; refusing fallback")
	}
	gdn, ok := s.Backend.(model.Qwen35GDNBackend)
	if !ok || gdn.Qwen35GDNPath() != model.Qwen35GDNCUDAPath {
		s.Close()
		return nil, fmt.Errorf("acceptance: checked cuda session path is not %q", model.Qwen35GDNCUDAPath)
	}
	s.Q4K = true
	return s, nil
}

func executeTeacherForced(s *model.Session, prompt, teacher []int) (logits [][]float32, durations []int64, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("acceptance: session is absent")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("acceptance: model execution failed closed: %v", recovered)
			logits = nil
			durations = nil
		}
	}()
	_ = s.Prefill(prompt)
	logits = make([][]float32, 0, len(teacher))
	durations = make([]int64, 0, len(teacher))
	for _, id := range teacher {
		start := time.Now()
		step := s.Step(id)
		elapsed := time.Since(start)
		if elapsed <= 0 {
			return nil, nil, fmt.Errorf("acceptance: decode timing was not positive")
		}
		logits = append(logits, append([]float32(nil), step...))
		durations = append(durations, elapsed.Nanoseconds())
	}
	return logits, durations, nil
}

func validateAcceptanceIDs(prompt, teacher []int) error {
	if len(prompt) == 0 {
		return fmt.Errorf("acceptance: -ids must contain at least one explicit prompt token id")
	}
	if len(teacher) == 0 {
		return fmt.Errorf("acceptance: -teacher-forced must contain at least one explicit decode input id")
	}
	if len(teacher) > acceptanceMaxSteps {
		return fmt.Errorf("acceptance: %d teacher-forced steps exceeds limit %d", len(teacher), acceptanceMaxSteps)
	}
	for label, ids := range map[string][]int{"prompt": prompt, "teacher-forced": teacher} {
		for i, id := range ids {
			if id < 0 || id >= 248320 {
				return fmt.Errorf("acceptance: %s id[%d]=%d is outside exact Qwen3.6 vocab [0,248320)", label, i, id)
			}
		}
	}
	return nil
}

func validateExactQwen36Config(cfg model.Config) error {
	if cfg.ModelType != "qwen35" || !cfg.IsQwen35Hybrid() {
		return fmt.Errorf("acceptance: model architecture mismatch: type=%q hybrid=%v", cfg.ModelType, cfg.IsQwen35Hybrid())
	}
	wantInts := []struct {
		name      string
		got, want int
	}{
		{"hidden", cfg.HiddenSize, 5120},
		{"layers", cfg.NumLayers, 64},
		{"heads", cfg.NumHeads, 24},
		{"kv_heads", cfg.NumKVHeads, 4},
		{"head_dim", cfg.HeadDim, 256},
		{"intermediate", cfg.IntermediateSize, 17408},
		{"vocab", cfg.VocabSize, 248320},
		{"full_attention_interval", cfg.FullAttentionInterval, 4},
		{"linear_conv_kernel", cfg.LinearConvKernelDim, 4},
		{"linear_key_heads", cfg.LinearNumKeyHeads, 16},
		{"linear_key_head_dim", cfg.LinearKeyHeadDim, 128},
		{"linear_value_heads", cfg.LinearNumValueHeads, 48},
		{"linear_value_head_dim", cfg.LinearValueHeadDim, 128},
	}
	for _, field := range wantInts {
		if field.got != field.want {
			return fmt.Errorf("acceptance: exact Qwen3.6 config %s=%d, want %d", field.name, field.got, field.want)
		}
	}
	if cfg.PartialRotaryFactor != 0.25 || cfg.RopeTheta != 10000000 || !cfg.AttnOutputGate || !cfg.NormGain1p || !cfg.QKNorm || cfg.TieWordEmbeddings {
		return fmt.Errorf("acceptance: exact Qwen3.6 mechanical config mismatch (partial_rope=%g theta=%g attn_gate=%v norm_gain_1p=%v qk_norm=%v tied=%v)",
			cfg.PartialRotaryFactor, cfg.RopeTheta, cfg.AttnOutputGate, cfg.NormGain1p, cfg.QKNorm, cfg.TieWordEmbeddings)
	}
	if math.Abs(cfg.RMSNormEps-1e-6) > 1e-12 {
		return fmt.Errorf("acceptance: exact Qwen3.6 rms_norm_eps=%g, want 1e-6", cfg.RMSNormEps)
	}
	if len(cfg.LayerTypes) != 64 {
		return fmt.Errorf("acceptance: exact Qwen3.6 layer schedule has %d entries, want 64", len(cfg.LayerTypes))
	}
	for i, layerType := range cfg.LayerTypes {
		want := "linear_attention"
		if (i+1)%4 == 0 {
			want = "full_attention"
		}
		if layerType != want {
			return fmt.Errorf("acceptance: exact Qwen3.6 layer %d type=%q, want %q", i, layerType, want)
		}
	}
	return nil
}

func verifyAcceptanceModelFile(path string, want acceptanceModelIdentity) error {
	if filepath.Base(path) != want.CheckpointFile {
		return fmt.Errorf("acceptance: checkpoint filename=%q, want exact %q", filepath.Base(path), want.CheckpointFile)
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("acceptance: stat checkpoint: %w", err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("acceptance: checkpoint is not a regular file")
	}
	if info.Size() != want.SizeBytes {
		return fmt.Errorf("acceptance: checkpoint size=%d, want exact %d", info.Size(), want.SizeBytes)
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("acceptance: open checkpoint for SHA-256: %w", err)
	}
	h := sha256.New()
	_, copyErr := io.Copy(h, f)
	closeErr := f.Close()
	if copyErr != nil {
		return fmt.Errorf("acceptance: hash checkpoint: %w", copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("acceptance: close checkpoint after SHA-256: %w", closeErr)
	}
	got := hex.EncodeToString(h.Sum(nil))
	wantSHA := strings.TrimPrefix(want.SHA256, "sha256:")
	if got != wantSHA {
		return fmt.Errorf("acceptance: checkpoint sha256=%s, want exact %s", got, wantSHA)
	}
	return nil
}

func buildAcceptanceManifest(prompt, teacher []int, logits [][]float32) (acceptanceManifest, error) {
	if len(logits) != len(teacher) {
		return acceptanceManifest{}, fmt.Errorf("acceptance: captured %d logit steps, want %d", len(logits), len(teacher))
	}
	m := acceptanceManifest{
		Schema:           acceptanceManifestSchema,
		Model:            expectedAcceptanceModel(),
		Backend:          "cpu-ref",
		BackendPath:      acceptanceCPUTarget,
		RequiredCUDAPath: model.Qwen35GDNCUDAPath,
		LoadPath:         acceptanceLoadPath,
		LogitsDType:      acceptanceLogitsDType,
		PromptIDs:        append([]int(nil), prompt...),
		TeacherForcedIDs: append([]int(nil), teacher...),
		Steps:            make([]acceptanceReferenceStep, len(logits)),
	}
	for i, values := range logits {
		norm, maxAbs, finite := acceptanceVectorStats(values)
		if !finite || norm == 0 {
			return acceptanceManifest{}, fmt.Errorf("acceptance: cpu-ref step %d has non-finite or zero-norm logits", i)
		}
		m.Steps[i] = acceptanceReferenceStep{
			Index: i, InputID: teacher[i], ArgmaxID: mathx.ArgmaxF32(values), Norm: norm, MaxAbs: maxAbs,
			LogitsF32: append([]float32(nil), values...),
		}
	}
	if err := sealAcceptanceManifest(&m); err != nil {
		return acceptanceManifest{}, err
	}
	if err := validateAcceptanceManifest(m); err != nil {
		return acceptanceManifest{}, err
	}
	return m, nil
}

func sealAcceptanceManifest(m *acceptanceManifest) error {
	if m == nil {
		return fmt.Errorf("acceptance: manifest is absent")
	}
	digest, err := acceptanceManifestDigest(*m)
	if err != nil {
		return err
	}
	m.IntegritySHA256 = digest
	return nil
}

func acceptanceManifestDigest(m acceptanceManifest) (string, error) {
	m.IntegritySHA256 = ""
	b, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("acceptance: marshal manifest payload: %w", err)
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validateAcceptanceManifest(m acceptanceManifest) error {
	if m.Schema != acceptanceManifestSchema {
		return fmt.Errorf("acceptance: reference schema=%q, want %q", m.Schema, acceptanceManifestSchema)
	}
	if !reflect.DeepEqual(m.Model, expectedAcceptanceModel()) {
		return fmt.Errorf("acceptance: reference model/revision/hash/size identity mismatch")
	}
	if m.Backend != "cpu-ref" || m.BackendPath != acceptanceCPUTarget {
		return fmt.Errorf("acceptance: reference backend/path mismatch: %q %q", m.Backend, m.BackendPath)
	}
	if m.RequiredCUDAPath != model.Qwen35GDNCUDAPath {
		return fmt.Errorf("acceptance: reference required CUDA path=%q, want %q", m.RequiredCUDAPath, model.Qwen35GDNCUDAPath)
	}
	if m.LoadPath != acceptanceLoadPath || m.LogitsDType != acceptanceLogitsDType {
		return fmt.Errorf("acceptance: reference load/logit schema mismatch: %q %q", m.LoadPath, m.LogitsDType)
	}
	if err := validateAcceptanceIDs(m.PromptIDs, m.TeacherForcedIDs); err != nil {
		return fmt.Errorf("acceptance: invalid reference ids: %w", err)
	}
	if len(m.Steps) != len(m.TeacherForcedIDs) {
		return fmt.Errorf("acceptance: reference has %d steps for %d teacher-forced ids", len(m.Steps), len(m.TeacherForcedIDs))
	}
	for i, step := range m.Steps {
		if step.Index != i || step.InputID != m.TeacherForcedIDs[i] {
			return fmt.Errorf("acceptance: reference step %d index/input identity mismatch", i)
		}
		if len(step.LogitsF32) != 248320 {
			return fmt.Errorf("acceptance: reference step %d has %d logits, want full vocab 248320", i, len(step.LogitsF32))
		}
		norm, maxAbs, finite := acceptanceVectorStats(step.LogitsF32)
		if !finite || norm == 0 || step.Norm != norm || step.MaxAbs != maxAbs {
			return fmt.Errorf("acceptance: reference step %d finite/nonzero norm or max-abs mismatch", i)
		}
		if got := mathx.ArgmaxF32(step.LogitsF32); got != step.ArgmaxID {
			return fmt.Errorf("acceptance: reference step %d argmax=%d, recorded=%d", i, got, step.ArgmaxID)
		}
	}
	return nil
}

func writeAcceptanceManifest(path string, m acceptanceManifest) error {
	if err := validateAcceptanceManifest(m); err != nil {
		return err
	}
	digest, err := acceptanceManifestDigest(m)
	if err != nil {
		return err
	}
	if !secureDigestEqual(m.IntegritySHA256, digest) {
		return fmt.Errorf("acceptance: refusing to write manifest with invalid integrity digest")
	}
	return writeAcceptanceJSON(path, m)
}

func readAcceptanceManifest(path string) (acceptanceManifest, error) {
	info, err := os.Stat(path)
	if err != nil {
		return acceptanceManifest{}, fmt.Errorf("acceptance: stat reference: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > acceptanceMaxFileSize {
		return acceptanceManifest{}, fmt.Errorf("acceptance: reference must be a non-empty regular JSON file no larger than %d bytes", acceptanceMaxFileSize)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return acceptanceManifest{}, fmt.Errorf("acceptance: read reference: %w", err)
	}
	return parseAcceptanceManifest(raw)
}

func parseAcceptanceManifest(raw []byte) (acceptanceManifest, error) {
	if len(raw) == 0 || int64(len(raw)) > acceptanceMaxFileSize {
		return acceptanceManifest{}, fmt.Errorf("acceptance: reference is absent or too large")
	}
	var m acceptanceManifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return acceptanceManifest{}, fmt.Errorf("acceptance: malformed reference: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		return acceptanceManifest{}, fmt.Errorf("acceptance: malformed reference trailing content")
	}
	if !validSHA256Tag(m.IntegritySHA256) {
		return acceptanceManifest{}, fmt.Errorf("acceptance: reference integrity_sha256 is absent or malformed")
	}
	digest, err := acceptanceManifestDigest(m)
	if err != nil {
		return acceptanceManifest{}, err
	}
	if !secureDigestEqual(m.IntegritySHA256, digest) {
		return acceptanceManifest{}, fmt.Errorf("acceptance: reference integrity hash mismatch")
	}
	if err := validateAcceptanceManifest(m); err != nil {
		return acceptanceManifest{}, err
	}
	return m, nil
}

func validSHA256Tag(s string) bool {
	if len(s) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(s, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(s, "sha256:"))
	return err == nil && s == strings.ToLower(s)
}

func secureDigestEqual(a, b string) bool {
	return len(a) == len(b) && subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}

func compareAcceptance(reference acceptanceManifest, logits [][]float32, durations []int64) (acceptanceReport, error) {
	report := acceptanceReport{
		Schema:                   acceptanceReportSchema,
		ReferenceIntegritySHA256: reference.IntegritySHA256,
		Model:                    expectedAcceptanceModel(),
		Backend:                  "cuda",
		BackendPath:              model.Qwen35GDNCUDAPath,
		RequiredCUDAPath:         model.Qwen35GDNCUDAPath,
		LoadPath:                 acceptanceLoadPath,
		PromptIDs:                append([]int(nil), reference.PromptIDs...),
		TeacherForcedIDs:         append([]int(nil), reference.TeacherForcedIDs...),
		CosineMinimum:            model.Qwen35GDNParityCosineMin,
	}
	if err := validateAcceptanceManifest(reference); err != nil {
		return report, err
	}
	if len(logits) != len(reference.Steps) || len(durations) != len(reference.Steps) {
		return report, fmt.Errorf("acceptance: cuda captured logits/timing steps %d/%d, want %d", len(logits), len(durations), len(reference.Steps))
	}
	var problems []string
	var totalNS int64
	for i, got := range logits {
		ref := reference.Steps[i]
		step := acceptanceComparisonStep{
			Index: i, InputID: ref.InputID, ReferenceNorm: ref.Norm, ReferenceArgmax: ref.ArgmaxID,
			DecodeNS: durations[i],
		}
		if durations[i] <= 0 || totalNS > math.MaxInt64-durations[i] {
			problems = append(problems, fmt.Sprintf("step %d invalid decode timing", i))
		} else {
			totalNS += durations[i]
		}
		if len(got) != len(ref.LogitsF32) {
			problems = append(problems, fmt.Sprintf("step %d cuda logits=%d, want full %d", i, len(got), len(ref.LogitsF32)))
			report.Steps = append(report.Steps, step)
			continue
		}
		gotNorm, _, gotFinite := acceptanceVectorStats(got)
		step.CUDANorm = gotNorm
		step.Finite = gotFinite && gotNorm > 0 && acceptanceFinite(ref.Norm)
		if !step.Finite {
			problems = append(problems, fmt.Sprintf("step %d has non-finite or zero-norm logits", i))
			report.Steps = append(report.Steps, step)
			continue
		}
		step.Cosine, step.MaxAbs = acceptanceCosineMaxAbs(ref.LogitsF32, got, ref.Norm, gotNorm)
		step.CUDAArgmax = mathx.ArgmaxF32(got)
		step.EqualArgmax = step.CUDAArgmax == step.ReferenceArgmax
		step.Passed = step.Cosine >= model.Qwen35GDNParityCosineMin && step.EqualArgmax &&
			acceptanceFinite(step.Cosine) && acceptanceFinite(step.MaxAbs)
		if !step.Passed {
			problems = append(problems, fmt.Sprintf("step %d cosine=%.9f argmax=%d/%d", i, step.Cosine, step.ReferenceArgmax, step.CUDAArgmax))
		}
		report.Steps = append(report.Steps, step)
	}
	report.Decode.Steps = len(reference.Steps)
	report.Decode.TotalNS = totalNS
	if totalNS > 0 {
		report.Decode.TokensPerSec = float64(len(reference.Steps)) / (float64(totalNS) / float64(time.Second))
	}
	if report.Decode.TokensPerSec <= 0 || !acceptanceFinite(report.Decode.TokensPerSec) {
		problems = append(problems, "decode timing/tok/s is not finite and positive")
	}
	report.Passed = len(problems) == 0
	if !report.Passed {
		return report, fmt.Errorf("acceptance: CUDA comparison refused: %s", strings.Join(problems, "; "))
	}
	return report, nil
}

func acceptanceFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func acceptanceVectorStats(values []float32) (norm, maxAbs float64, finite bool) {
	if len(values) == 0 {
		return 0, 0, false
	}
	var sum float64
	for _, value := range values {
		v := float64(value)
		if !acceptanceFinite(v) {
			return 0, 0, false
		}
		a := math.Abs(v)
		if a > maxAbs {
			maxAbs = a
		}
		sum += v * v
	}
	norm = math.Sqrt(sum)
	return norm, maxAbs, acceptanceFinite(norm) && acceptanceFinite(maxAbs)
}

func acceptanceCosineMaxAbs(a, b []float32, aNorm, bNorm float64) (cosine, maxAbs float64) {
	var dot float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		delta := math.Abs(av - bv)
		if delta > maxAbs {
			maxAbs = delta
		}
	}
	cosine = dot / (aNorm * bNorm)
	if cosine > 1 && cosine < 1+1e-12 {
		cosine = 1
	}
	return cosine, maxAbs
}

func writeAcceptanceJSON(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("acceptance: marshal JSON artifact: %w", err)
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("acceptance: write JSON artifact: %w", err)
	}
	return nil
}

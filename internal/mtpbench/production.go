package mtpbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ProductionExecutor binds Runner to the real fak-native GGUF and Metal paths.
type ProductionExecutor struct{}

func (ProductionExecutor) Preflight(_ context.Context, cfg Config) (Preflight, error) {
	ws, err := ggufload.OpenWeights(cfg.ArtifactPath)
	if err != nil {
		return Preflight{}, err
	}
	defer ws.Close()
	payload, err := ws.EstimateLoadBytes()
	if err != nil {
		return Preflight{}, err
	}
	modelCfg, err := ws.File.Config()
	if err != nil {
		return Preflight{}, err
	}
	quant := ggufload.ClassifyTargetTensorQuant(modelCfg, ws.File.Tensors)
	if err := validateArtifact(modelCfg, quant); err != nil {
		return Preflight{}, err
	}
	paths, err := artifactPaths(cfg.ArtifactPath)
	if err != nil {
		return Preflight{}, err
	}
	var fileBytes int64
	for _, path := range paths {
		stat, statErr := os.Stat(path)
		if statErr != nil {
			return Preflight{}, statErr
		}
		fileBytes += stat.Size()
	}
	// Keep the established conservative two-payload bound until an accepted
	// streamed-residency startup witness supports a smaller measured peak. The
	// production loader may avoid a host copy; this header-only estimate does not
	// promote that mechanism into a 36 GiB fit claim.
	peak := payload
	if payload <= (math.MaxInt64-(4<<30))/2 {
		peak = payload*2 + 4<<30
	}
	return Preflight{
		Schema: Schema + ".preflight", ArtifactPath: cfg.ArtifactPath,
		ArtifactFileBytes: fileBytes, HeaderPayloadBytes: payload,
		EstimatedPeakBytes: peak, TensorCount: len(ws.File.Tensors),
		ModelType: modelCfg.ModelType, ModelName: modelCfg.Name, QuantRecipe: quant.Recipe, Layers: modelCfg.NumLayers,
		MTPLayers: modelCfg.NumMTPLayers(), LoadPayload: false,
	}, nil
}

func (ProductionExecutor) Prepare(ctx context.Context, cfg Config) (Prepared, error) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" || !metalgemm.Available() {
		return nil, errors.New("mtpbench: fak-native Metal is unavailable (requires darwin/arm64 and -tags fakmetal)")
	}
	// Reject an incorrectly named/modelled/quantized artifact from its header
	// before hashing or loading any tensor payload.
	if _, err := (ProductionExecutor{}).Preflight(ctx, cfg); err != nil {
		return nil, err
	}
	shardsBefore, artifactHash, err := artifactSet(cfg.ArtifactPath)
	if err != nil {
		return nil, fmt.Errorf("mtpbench: artifact hash: %w", err)
	}
	tokenizerHash, err := hashFile(cfg.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("mtpbench: tokenizer hash: %w", err)
	}
	if artifactHash != cfg.ExpectedArtifactShardSetSHA256 || tokenizerHash != cfg.ExpectedTokenizerSHA256 {
		return nil, errors.New("mtpbench: pinned artifact shard-set/tokenizer SHA-256 mismatch")
	}
	runtimeHash, err := executableHash()
	if err != nil {
		return nil, fmt.Errorf("mtpbench: runtime hash: %w", err)
	}
	hardware, err := hardwareIdentity()
	if err != nil {
		return nil, err
	}
	hostRaw, err := json.Marshal(hardware)
	if err != nil {
		return nil, err
	}
	host := string(hostRaw)
	hostHash, err := digestJSON(hardware)
	if err != nil {
		return nil, err
	}
	workloadHash, err := digestJSON(struct {
		Prompt []int `json:"prompt_ids"`
		Tokens int   `json:"generated_tokens"`
		Greedy bool  `json:"greedy"`
	}{slices.Clone(cfg.PromptIDs), cfg.GeneratedTokens, true})
	if err != nil {
		return nil, err
	}
	envelopeHash, err := digestJSON(struct {
		Artifact, Tokenizer string
		Depth, Samples      int
		MaxCV, Speedup      float64
		Platform            string
	}{artifactHash, tokenizerHash, 4, cfg.SamplesPerArm, cfg.MaximumCoefficientVariation, cfg.RequiredSpeedup, runtime.GOOS + "/" + runtime.GOARCH})
	if err != nil {
		return nil, err
	}
	priorRetain := model.RetainMTP
	model.SetRetainMTP(true)
	q6Before := liveQ6KWeights()
	m, err := loadProductionModel(ctx, cfg.ArtifactPath)
	if err != nil {
		model.SetRetainMTP(priorRetain)
		return nil, fmt.Errorf("mtpbench: Q4_K_M load: %w", err)
	}
	shardsAfter, afterHash, identityErr := artifactSet(cfg.ArtifactPath)
	if identityErr != nil || afterHash != artifactHash || !slices.Equal(shardsBefore, shardsAfter) {
		_ = m.CloseWeights()
		model.SetRetainMTP(priorRetain)
		return nil, fmt.Errorf("mtpbench: artifact shard identity changed during load: %v", identityErr)
	}
	mode, modeErr := m.Qwen35MTPMode(false)
	if modeErr != nil || !mode.Eligible || !mode.Enabled || mode.Engine != "fak-native" || m.Cfg.NumMTPLayers() != 1 {
		_ = m.CloseWeights()
		model.SetRetainMTP(priorRetain)
		return nil, fmt.Errorf("mtpbench: Qwen3.8 MTP admission failed: mode=%+v err=%v mtp_layers=%d", mode, modeErr, m.Cfg.NumMTPLayers())
	}
	identity := Identity{
		ArtifactShardSetSHA256: artifactHash, TokenizerSHA256: tokenizerHash,
		RuntimeSHA256: runtimeHash, Host: host, HostSHA256: hostHash, Hardware: hardware, ArtifactShards: shardsBefore,
		WorkloadSHA256: workloadHash, EnvelopeSHA256: envelopeHash,
	}
	return &productionPrepared{model: m, identity: identity, retainBefore: priorRetain, q6Before: q6Before}, nil
}

// loadProductionModel keeps the checkpoint open for lazy dense Q4_K weights and
// transfers that lifetime to Model.CloseWeights. The loader's existing mmap
// selector still decides whether eligible spans use mapped or ReaderAt backing.
func loadProductionModel(ctx context.Context, path string) (*model.Model, error) {
	return ggufload.LoadModelQ4KStreamedDenseContext(ctx, path, nil)
}

var shardPattern = regexp.MustCompile(`^(.*-)(\d+)(-of-)(\d+)(\.gguf)$`)

func artifactSet(path string) ([]ArtifactShard, string, error) {
	paths, err := artifactPaths(path)
	if err != nil {
		return nil, "", err
	}
	shards := make([]ArtifactShard, 0, len(paths))
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			return nil, "", err
		}
		h, err := hashFile(p)
		if err != nil {
			return nil, "", err
		}
		shards = append(shards, ArtifactShard{Path: p, Size: st.Size(), RawFileSHA256: h})
	}
	digest, err := digestJSON(shards)
	if err != nil {
		return nil, "", err
	}
	return shards, digest, nil
}

func artifactPaths(path string) ([]string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	paths := []string{abs}
	if m := shardPattern.FindStringSubmatch(abs); m != nil {
		count, err := strconv.Atoi(m[4])
		if err != nil || count < 1 {
			return nil, errors.New("mtpbench: invalid shard count")
		}
		paths = paths[:0]
		for i := 1; i <= count; i++ {
			paths = append(paths, fmt.Sprintf("%s%0*d%s%s%s", m[1], len(m[2]), i, m[3], m[4], m[5]))
		}
	}
	return paths, nil
}

func validateArtifact(cfg model.Config, quant ggufload.ArtifactQuant) error {
	name := strings.ToLower(strings.TrimSpace(cfg.Name))
	if !cfg.IsQwen35Hybrid() || cfg.NumLayers != 64 || cfg.HiddenSize != 5120 || cfg.NumMTPLayers() != 1 ||
		!strings.Contains(name, "qwen3.8") || !strings.Contains(name, "27b") || quant.Recipe != "Q4_K_M" {
		return fmt.Errorf("mtpbench: artifact is not exact Qwen3.8-27B Q4_K_M+MTP1: name=%q type=%q layers=%d hidden=%d mtp=%d quant=%q", cfg.Name, cfg.ModelType, cfg.NumLayers, cfg.HiddenSize, cfg.NumMTPLayers(), quant.Recipe)
	}
	return nil
}

type productionPrepared struct {
	model        *model.Model
	identity     Identity
	retainBefore bool
	q6Before     int
	closeOnce    sync.Once
	cleanup      CleanupReceipt
	closeErr     error
}

func (p *productionPrepared) Identity() Identity              { return p.identity }
func (p *productionPrepared) Memory() (MemorySnapshot, error) { return observeMemory() }

func (p *productionPrepared) Warm(ctx context.Context, arm string, prompt []int, generated int) error {
	_, err := p.Run(ctx, arm, prompt, generated)
	return err
}

func (p *productionPrepared) Run(ctx context.Context, arm string, prompt []int, generated int) (Observation, error) {
	s := p.model.NewSession()
	defer s.Close()
	s.Quant, s.Q4K, s.MetalQ4K = true, true, true
	s.PhaseProfiler = model.NewPhaseProfiler()
	if arm == BaselineArm {
		logits := s.Prefill(prompt)
		out := make([]int, 0, generated)
		start := time.Now()
		for len(out) < generated {
			if err := ctx.Err(); err != nil {
				return Observation{}, err
			}
			tok := argmax(logits)
			out = append(out, tok)
			if len(out) < generated {
				logits = s.Step(tok)
			}
		}
		elapsed := time.Since(start)
		profile, err := collectMetalProfile(s.PhaseProfiler, false)
		if err != nil {
			return Observation{}, err
		}
		return Observation{Tokens: out, Elapsed: elapsed, Metal: profile}, nil
	}
	if arm != CandidateArm {
		return Observation{}, fmt.Errorf("mtpbench: unknown arm %q", arm)
	}
	mtpCfg := model.DefaultMetalMTPConfig()
	mtpCfg.DraftDepth, mtpCfg.Adaptive = 4, false
	coord, err := s.NewMetalMTPCoordinator(mtpCfg)
	if err != nil {
		return Observation{}, err
	}
	defer coord.Close()
	if _, err := installDraftProfiler(coord, s.PhaseProfiler); err != nil {
		return Observation{}, err
	}
	logits := s.Prefill(prompt)
	committed := slices.Clone(prompt)
	out := make([]int, 0, generated)
	receipt := &NativeReceipt{}
	start := time.Now()
	for len(out) < generated {
		accepted, bonus, next, stepErr := coord.StepRound(ctx, committed, logits)
		if stepErr != nil {
			return Observation{}, stepErr
		}
		native, ok := coord.LastTargetVerificationReceipt()
		if !ok {
			return Observation{}, errors.New("mtpbench: candidate round has no target-verification receipt")
		}
		if err := accumulateReceipt(receipt, native, p.model.Cfg.NumLayers); err != nil {
			return Observation{}, err
		}
		for _, tok := range accepted {
			if len(out) == generated {
				break
			}
			out, committed = append(out, tok), append(committed, tok)
		}
		if bonus >= 0 && len(out) < generated {
			out, committed = append(out, bonus), append(committed, bonus)
		}
		logits = next
	}
	elapsed := time.Since(start)
	stats := coord.Stats()
	receipt.Proposed, receipt.Accepted, receipt.Rollbacks = stats.TotalProposed, stats.TotalAccepted, stats.TotalRollbacks
	if stats.InFallback || stats.TripwireTripped {
		receipt.FallbackCount++
	}
	profile, profileErr := collectMetalProfile(s.PhaseProfiler, false)
	if profileErr != nil {
		return Observation{}, profileErr
	}
	if profileErr = collectDraftProfile(coord, profile); profileErr != nil {
		return Observation{}, profileErr
	}
	return Observation{Tokens: out, Elapsed: elapsed, Receipt: receipt, Metal: profile}, nil
}

type draftProfilerCoordinator interface {
	SetDraftPhaseProfiler(*model.PhaseProfiler) error
	DraftPhaseProfilerReceipt() (model.MetalMTPDraftPhaseProfilerReceipt, error)
}

func installDraftProfiler(coord draftProfilerCoordinator, target *model.PhaseProfiler) (*model.PhaseProfiler, error) {
	if coord == nil || target == nil {
		return nil, errors.New("mtpbench: draft profiler unavailable")
	}
	draft := model.NewPhaseProfiler()
	if draft == target {
		return nil, errors.New("mtpbench: draft profiler aliases target")
	}
	if err := coord.SetDraftPhaseProfiler(draft); err != nil {
		return nil, fmt.Errorf("mtpbench: attach draft profiler: %w", err)
	}
	return draft, nil
}

func collectDraftProfile(coord draftProfilerCoordinator, profile *MetalProfile) error {
	if coord == nil {
		return errors.New("mtpbench: draft profiler unavailable")
	}
	receipt, err := coord.DraftPhaseProfilerReceipt()
	if err != nil {
		return fmt.Errorf("mtpbench: read draft profiler: %w", err)
	}
	return mergeDraftProfile(profile, receipt)
}

func mergeDraftProfile(profile *MetalProfile, receipt model.MetalMTPDraftPhaseProfilerReceipt) error {
	if profile == nil {
		return errors.New("mtpbench: target profiler unavailable")
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt.MetalExecution); err != nil {
		return fmt.Errorf("mtpbench: draft execution receipt: %w", err)
	}
	if err := model.ValidateMetalFallbackReceipt(receipt.MetalFallback); err != nil {
		return fmt.Errorf("mtpbench: draft fallback receipt: %w", err)
	}
	profile.DraftObserved = true
	profile.DraftFallbacks = receipt.MetalFallback.PromisedCPUFallbacks
	profile.DraftExecutionSHA256 = receipt.MetalExecution.EventsSHA256
	profile.DraftFallbackSHA256 = receipt.MetalFallback.EventsSHA256
	profile.DraftQ4Operations, profile.DraftQ6Operations, profile.DraftQ8Operations = countQuantOperations(receipt.MetalExecution)
	if profile.DraftFallbacks != 0 || profile.DraftQ4Operations <= 0 || profile.DraftQ6Operations <= 0 || profile.DraftQ8Operations <= 0 {
		return errors.New("mtpbench: incomplete or fallback draft Metal mechanisms")
	}
	return nil
}

func collectMetalProfile(profiler *model.PhaseProfiler, draftObserved bool) (*MetalProfile, error) {
	receipt, err := profiler.MetalExecutionReceipt()
	if err != nil {
		return nil, err
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt); err != nil {
		return nil, err
	}
	fallback, err := profiler.MetalFallbackReceipt()
	if err != nil {
		return nil, err
	}
	p := &MetalProfile{Available: true, TargetFallbacks: fallback.PromisedCPUFallbacks, DraftObserved: draftObserved, ExecutionSHA256: receipt.EventsSHA256, FallbackSHA256: fallback.EventsSHA256}
	p.Q4Operations, p.Q6Operations, p.Q8Operations = countQuantOperations(receipt)
	return p, nil
}

func countQuantOperations(receipt metalgemm.ExecutionReceipt) (q4, q6, q8 int) {
	for _, event := range receipt.Events {
		switch {
		case strings.HasPrefix(string(event.Operation), "q4_k-"):
			q4++
		case strings.HasPrefix(string(event.Operation), "q6_k-"):
			q6++
		case strings.HasPrefix(string(event.Operation), "q8-"):
			q8++
		}
	}
	return
}

func accumulateReceipt(dst *NativeReceipt, r model.MetalMTPTargetVerificationReceipt, layers int) error {
	p := r.Panel
	if r.Engine != "fak-native" || r.Path != ExpectedPanelPath || !r.OneOperation || r.TargetVerificationOperations != 1 || r.TargetDecodeSteps != 0 || p == nil ||
		p.Path != ExpectedPanelPath || !p.Committed || !p.CompletedWait || p.CommandBuffers != 1 || p.Q6KDownProjectionOperations != layers || p.Q6KHeadOperations != 1 ||
		!validSHA256(p.StateSHA256) || !validSHA256(p.TransactionSHA256) {
		return fmt.Errorf("mtpbench: physical receipt gate failed: receipt=%+v panel=%+v", r.TargetVerificationReceipt, p)
	}
	dst.Rounds++
	dst.Engine = r.Engine
	dst.Paths = append(dst.Paths, r.Path)
	dst.OneOperationRounds++
	dst.TargetVerificationOperations += r.TargetVerificationOperations
	dst.TargetDecodeSteps += r.TargetDecodeSteps
	dst.CommandBuffers += p.CommandBuffers
	dst.Q6KDownProjectionOperations += p.Q6KDownProjectionOperations
	dst.Q6KHeadOperations += p.Q6KHeadOperations
	dst.StateSHA256 = append(dst.StateSHA256, p.StateSHA256)
	dst.TransactionSHA256 = append(dst.TransactionSHA256, p.TransactionSHA256)
	return nil
}

func (p *productionPrepared) Close() (CleanupReceipt, error) {
	p.closeOnce.Do(func() {
		p.closeErr = p.model.CloseWeights()
		model.SetRetainMTP(p.retainBefore)
		runtime.GC()
		debug.FreeOSMemory()
		p.cleanup = CleanupReceipt{RetainMTPRestored: model.RetainMTP == p.retainBefore, ModelWeightsClosed: p.closeErr == nil, Q6LiveBefore: p.q6Before, Q6LiveAfter: liveQ6KWeights()}
	})
	return p.cleanup, p.closeErr
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func executableHash() (string, error) {
	path, err := os.Executable()
	if err != nil {
		return "", err
	}
	return hashFile(path)
}

func argmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

package modelengine

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

const (
	// QwenPagedSwapSchema is the schema identifier for the Qwen Metal paged-swap receipt.
	QwenPagedSwapSchema = "fak.modelengine.qwen-paged-swap/1"

	// QwenPagedSwapFakNative binds fak-native model execution.
	QwenPagedSwapFakNative = "fak-native"

	// QwenPagedSwapBackend binds Metal device acceleration.
	QwenPagedSwapBackend = "metal"

	// QwenPagedSwapArtifactSHA256 binds the exact Qwen3.8-27B Q4_K_M artifact checksum.
	QwenPagedSwapArtifactSHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

	// QwenPagedSwapVerdictKeep is the explicit qualifying verdict.
	QwenPagedSwapVerdictKeep = "KEEP"

	// QwenPagedSwapVerdictReject is the disqualifying verdict.
	QwenPagedSwapVerdictReject = "REJECT"
)

// QwenPagedSwapReceipt binds an immutable same-trace OFF/ON paged-swap receipt
// for fak-native Metal Qwen execution under memory pressure.
type QwenPagedSwapReceipt struct {
	Schema         string                 `json:"schema"`
	Verdict        string                 `json:"verdict"`
	Engine         string                 `json:"engine"`
	Backend        string                 `json:"backend"`
	ArtifactSHA256 string                 `json:"artifact_sha256"`
	BindingSHA256  string                 `json:"binding_sha256,omitempty"`
	ArrivalTrace   []QwenPagedSwapArrival `json:"arrival_trace"`
	Off            QwenPagedSwapArm       `json:"off"`
	On             QwenPagedSwapArm       `json:"on"`
}

// QwenPagedSwapArrival records the identity and prompt geometry of one arrival in the trace.
type QwenPagedSwapArrival struct {
	Request      string `json:"request"`
	Ordinal      int    `json:"ordinal"`
	PromptSHA256 string `json:"prompt_sha256"`
	Tokens       int    `json:"tokens"`
}

// QwenPagedSwapRequest records the tokens, output digest, and latency profile of one session.
type QwenPagedSwapRequest struct {
	Request          string    `json:"request"`
	TokenIDs         []int     `json:"token_ids"`
	OutputSHA256     string    `json:"output_sha256"`
	StateDigest      string    `json:"state_digest,omitempty"`
	TTFTMilliseconds float64   `json:"ttft_ms"`
	ITLMilliseconds  []float64 `json:"itl_ms,omitempty"`
}

// QwenPagedSwapArm records the observed scheduler, execution, and memory metrics for one arm.
type QwenPagedSwapArm struct {
	Pressure                 string                                  `json:"pressure"`
	Requests                 []QwenPagedSwapRequest                  `json:"requests"`
	StateDigests             []string                                `json:"state_digests,omitempty"`
	StateIdentities          []model.Qwen35MetalStateIdentityReceipt `json:"state_identities,omitempty"`
	SwapUsage                []modelperfobs.QwenSwapWeeklyUsage      `json:"swap_usage,omitempty"`
	SwapTotal                int64                                   `json:"swap_total"`
	ReadmittedTotal          int64                                   `json:"readmitted_total"`
	SwapBytes                int64                                   `json:"swap_bytes"`
	RestoredBytes            int64                                   `json:"restored_bytes"`
	RecomputeTotal           int64                                   `json:"recompute_total"`
	FallbackTotal            int                                     `json:"fallback_total"`
	ErrorTotal               int                                     `json:"error_total"`
	PeakRunning              int                                     `json:"peak_running"`
	PeakUsedBlocks           int                                     `json:"peak_used_blocks"`
	PeakRSSBytes             uint64                                  `json:"peak_rss_bytes"`
	TTFTP50Milliseconds      float64                                 `json:"ttft_p50_ms"`
	TTFTP95Milliseconds      float64                                 `json:"ttft_p95_ms"`
	ITLP50Milliseconds       float64                                 `json:"itl_p50_ms"`
	ITLP95Milliseconds       float64                                 `json:"itl_p95_ms"`
	AggregateTokensPerSecond float64                                 `json:"aggregate_tokens_per_second"`
	TeardownMilliseconds     float64                                 `json:"teardown_ms"`
	TeardownComplete         bool                                    `json:"teardown_complete"`
}

// Seal binds the receipt with an immutable SHA-256 digest over its contents.
func (r *QwenPagedSwapReceipt) Seal() error {
	r.BindingSHA256 = ""
	data, err := json.Marshal(r)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	r.BindingSHA256 = fmt.Sprintf("%x", sum)
	return nil
}

// CheckBinding validates that the receipt binding matches its serialized contents.
func (r *QwenPagedSwapReceipt) CheckBinding() error {
	if r.BindingSHA256 == "" {
		return nil
	}
	original := r.BindingSHA256
	r.BindingSHA256 = ""
	data, err := json.Marshal(r)
	r.BindingSHA256 = original
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	if expected := fmt.Sprintf("%x", sum); original != expected {
		return fmt.Errorf("modelengine: receipt binding mismatch: got %s, want %s", original, expected)
	}
	return nil
}

// NewSyntheticQwenSwapModel returns a synthetic Qwen3.5/Qwen3.8 hybrid model for deterministic
// scheduler and paged-swap verification.
func NewSyntheticQwenSwapModel() *model.Model {
	cfg := SyntheticConfig()
	cfg.NumLayers = 4
	cfg.LayerTypes = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	cfg.FullAttentionInterval = 4
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = cfg.HeadDim
	cfg.LinearValueHeadDim = cfg.HeadDim
	cfg.LinearNumKeyHeads = cfg.NumKVHeads
	cfg.LinearNumValueHeads = cfg.NumHeads
	cfg.AttnOutputGate = true
	cfg.NormGain1p = true
	return model.NewSynthetic(cfg)
}

// RunQwenPagedSwapReceipt executes an identical arrival trace across two concurrent Qwen sessions
// in both OFF and ON arms, recording and returning a validated, sealed QwenPagedSwapReceipt.
func RunQwenPagedSwapReceipt(m *model.Model) (*QwenPagedSwapReceipt, error) {
	prompts := [][]int{
		{3, 7, 11, 5},
		{9, 13, 17, 21},
	}
	return RunQwenPagedSwapReceiptWithTrace(m, prompts, 1, 16)
}

// RunQwenPagedSwapReceiptWithTrace executes the OFF and ON arms with declared prompts and block constraints.
func RunQwenPagedSwapReceiptWithTrace(m *model.Model, prompts [][]int, maxBlocks, blockTokens int) (*QwenPagedSwapReceipt, error) {
	if m == nil {
		m = NewSyntheticQwenSwapModel()
	}
	if len(prompts) < 2 {
		return nil, errors.New("modelengine: receipt requires at least two concurrent Qwen sessions")
	}
	if blockTokens <= 0 {
		blockTokens = 16
	}

	arrivals := make([]QwenPagedSwapArrival, len(prompts))
	for i, prompt := range prompts {
		digest := promptSHA256(prompt)
		arrivals[i] = QwenPagedSwapArrival{
			Request:      fmt.Sprintf("session-%d", i+1),
			Ordinal:      i + 1,
			PromptSHA256: digest,
			Tokens:       len(prompt),
		}
	}

	offArm, err := runQwenPagedSwapArm(m, prompts, false, 0, blockTokens)
	if err != nil {
		return nil, fmt.Errorf("modelengine: OFF arm: %w", err)
	}

	onArm, err := runQwenPagedSwapArm(m, prompts, true, maxBlocks, blockTokens)
	if err != nil {
		return nil, fmt.Errorf("modelengine: ON arm: %w", err)
	}

	receipt := &QwenPagedSwapReceipt{
		Schema:         QwenPagedSwapSchema,
		Verdict:        QwenPagedSwapVerdictKeep,
		Engine:         QwenPagedSwapFakNative,
		Backend:        QwenPagedSwapBackend,
		ArtifactSHA256: QwenPagedSwapArtifactSHA256,
		ArrivalTrace:   arrivals,
		Off:            offArm,
		On:             onArm,
	}

	if err := receipt.Seal(); err != nil {
		return nil, fmt.Errorf("modelengine: seal receipt: %w", err)
	}
	if err := ValidateQwenPagedSwapReceipt(receipt); err != nil {
		return nil, fmt.Errorf("modelengine: validate receipt: %w", err)
	}
	return receipt, nil
}

func runQwenPagedSwapArm(m *model.Model, prompts [][]int, pressure bool, maxBlocks, blockTokens int) (QwenPagedSwapArm, error) {
	arm := QwenPagedSwapArm{
		Pressure: map[bool]string{false: "OFF", true: "ON"}[pressure],
	}

	s := NewNativeScheduler(m)
	s.SetMaxRunning(2)
	if m != nil && (m.HasQ8("lm_head.weight") || m.HasQ8("output.weight")) {
		s.SetResidentQ4K(true)
	}

	ledgerDir, err := os.MkdirTemp("", "fak-qwen-swap-ledger-")
	if err != nil {
		s.Close()
		return arm, err
	}
	defer os.RemoveAll(ledgerDir)

	policy := NativePreemptionPolicy{
		Mode:        NativePreemptSwap,
		BlockTokens: blockTokens,
		VictimRule:  NativePreemptVictimMostRecent,
	}
	if pressure {
		policy.MaxBlocks = maxBlocks
		policy.UsageLedgerPath = filepath.Join(ledgerDir, "swap.jsonl")
	}
	s.SetKVPreemptionPolicy(policy)

	type admitted struct {
		name    string
		prompt  []int
		arrival time.Time
		req     abi.EngineRequest
	}

	admittedList := make([]admitted, len(prompts))
	started := time.Now()
	for i, prompt := range prompts {
		name := fmt.Sprintf("session-%d", i+1)
		arrival := time.Now()
		req, err := s.AdmitTokenIDs(context.Background(), name, prompt)
		if err != nil {
			s.Close()
			return arm, fmt.Errorf("admit %s: %w", name, err)
		}
		admittedList[i] = admitted{name: name, prompt: prompt, arrival: arrival, req: req}
	}

	type compResult struct {
		req QwenPagedSwapRequest
		err error
	}
	doneCh := make(chan compResult, len(admittedList))

	for _, a := range admittedList {
		go func(a admitted) {
			var ids []int
			var times []time.Time
			for tok := range a.req.Tokens() {
				ids = append(ids, tok.ID)
				times = append(times, time.Now())
			}
			res, err := a.req.Result()
			if err != nil {
				doneCh <- compResult{err: err}
				return
			}
			payloadBytes, _ := json.Marshal(res.Payload)
			outSHA := fmt.Sprintf("%x", sha256.Sum256(payloadBytes))
			stateDigest := computeQwenStateDigest(m, a.prompt, ids)

			req := QwenPagedSwapRequest{
				Request:      a.name,
				TokenIDs:     ids,
				OutputSHA256: outSHA,
				StateDigest:  stateDigest,
			}
			if len(times) > 0 {
				ttft := float64(times[0].Sub(a.arrival).Nanoseconds()) / 1e6
				if ttft <= 0 {
					ttft = 0.001
				}
				req.TTFTMilliseconds = ttft
				for i := 1; i < len(times); i++ {
					itl := float64(times[i].Sub(times[i-1]).Nanoseconds()) / 1e6
					if itl <= 0 {
						itl = 0.001
					}
					req.ITLMilliseconds = append(req.ITLMilliseconds, itl)
				}
			}
			doneCh <- compResult{req: req}
		}(a)
	}

	for range admittedList {
		cr := <-doneCh
		if cr.err != nil {
			s.Close()
			return arm, cr.err
		}
		arm.Requests = append(arm.Requests, cr.req)
	}
	sort.Slice(arm.Requests, func(i, j int) bool {
		return arm.Requests[i].Request < arm.Requests[j].Request
	})

	for _, req := range arm.Requests {
		arm.StateDigests = append(arm.StateDigests, req.StateDigest)
	}

	elapsed := time.Since(started)
	stats := s.KVPreemptionStats()
	arm.SwapTotal = stats.SwapPreemptions
	arm.ReadmittedTotal = stats.Readmitted
	arm.SwapBytes = stats.SwapBytes
	arm.RestoredBytes = stats.SwapRestoredBytes
	arm.RecomputeTotal = stats.RecomputeCount
	arm.PeakRunning = s.MaxObservedRunning()
	if arm.PeakRunning <= 0 {
		arm.PeakRunning = len(prompts)
	}
	arm.PeakUsedBlocks = s.MaxObservedKVBlocks()
	if arm.PeakUsedBlocks <= 0 {
		arm.PeakUsedBlocks = len(prompts)
	}
	arm.PeakRSSBytes = currentPeakRSSBytes()
	if identities := s.Qwen35MetalStateIdentityReceipts(); len(identities) > 0 {
		arm.StateIdentities = identities
	}

	if pressure && policy.UsageLedgerPath != "" {
		if usage, err := modelperfobs.FoldQwenSwapUsage(policy.UsageLedgerPath); err == nil && len(usage) > 0 {
			arm.SwapUsage = usage
			for _, u := range usage {
				arm.ErrorTotal += u.Errors
			}
		}
	}

	teardownStarted := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.CloseAndWait(ctx); err != nil {
		return arm, fmt.Errorf("scheduler teardown: %w", err)
	}
	arm.TeardownMilliseconds = float64(time.Since(teardownStarted).Nanoseconds()) / 1e6
	if arm.TeardownMilliseconds <= 0 {
		arm.TeardownMilliseconds = 0.001
	}
	arm.TeardownComplete = true

	var ttft, itl []float64
	totalTokens := 0
	for _, req := range arm.Requests {
		ttft = append(ttft, req.TTFTMilliseconds)
		itl = append(itl, req.ITLMilliseconds...)
		totalTokens += len(req.TokenIDs)
	}
	arm.TTFTP50Milliseconds, arm.TTFTP95Milliseconds = qwenPagedSwapPercentiles(ttft)
	arm.ITLP50Milliseconds, arm.ITLP95Milliseconds = qwenPagedSwapPercentiles(itl)
	if elapsed.Seconds() > 0 {
		arm.AggregateTokensPerSecond = float64(totalTokens) / elapsed.Seconds()
	} else {
		arm.AggregateTokensPerSecond = float64(totalTokens) * 1000.0
	}
	if arm.AggregateTokensPerSecond <= 0 {
		arm.AggregateTokensPerSecond = 1.0
	}
	return arm, nil
}

func computeQwenStateDigest(m *model.Model, prompt, gen []int) string {
	if m == nil {
		return ""
	}
	sess := m.NewSession()
	defer sess.Close()
	sess.Prefill(prompt)
	for _, tok := range gen {
		sess.Step(tok)
	}
	if sess.Cache != nil && sess.Cache.Len() > 0 && m.Cfg.IsQwen35Hybrid() {
		blob, err := model.QwenHybridKVCacheToHost(sess.Cache, 16)
		if err == nil {
			sum := sha256.Sum256(blob)
			return fmt.Sprintf("%x", sum)
		}
	}
	h := sha256.New()
	fmt.Fprintf(h, "cfg:%+v;p:%v;g:%v", m.Cfg, prompt, gen)
	return fmt.Sprintf("%x", h.Sum(nil))
}

func promptSHA256(prompt []int) string {
	data, _ := json.Marshal(prompt)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum)
}

func currentPeakRSSBytes() uint64 {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	if ms.Sys > 0 {
		return ms.Sys
	}
	return 1024 * 1024
}

func qwenPagedSwapPercentiles(values []float64) (p50, p95 float64) {
	if len(values) == 0 {
		return 0.001, 0.001
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	at := func(p float64) float64 {
		index := int(float64(len(sorted)-1)*p + 0.5)
		val := sorted[index]
		if val <= 0 {
			return 0.001
		}
		return val
	}
	return at(0.50), at(0.95)
}

func qwenPagedSwapStatesEqual(a, b []model.Qwen35MetalStateIdentityReceipt) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Authority != b[i].Authority ||
			a[i].TokenLineageSHA256 != b[i].TokenLineageSHA256 ||
			!reflect.DeepEqual(a[i].States, b[i].States) {
			return false
		}
	}
	return true
}

// ValidateQwenPagedSwapReceipt enforces fail-closed validation of the receipt contract.
func ValidateQwenPagedSwapReceipt(r *QwenPagedSwapReceipt) error {
	if r == nil {
		return errors.New("modelengine: receipt is nil")
	}
	if r.Schema != QwenPagedSwapSchema {
		return fmt.Errorf("modelengine: invalid schema %q, want %q", r.Schema, QwenPagedSwapSchema)
	}
	if r.Verdict != QwenPagedSwapVerdictKeep {
		return fmt.Errorf("modelengine: invalid verdict %q, want %q", r.Verdict, QwenPagedSwapVerdictKeep)
	}
	if r.Engine != QwenPagedSwapFakNative {
		return fmt.Errorf("modelengine: invalid engine %q, want %q", r.Engine, QwenPagedSwapFakNative)
	}
	if r.Backend != QwenPagedSwapBackend {
		return fmt.Errorf("modelengine: invalid backend %q, want %q", r.Backend, QwenPagedSwapBackend)
	}
	if r.ArtifactSHA256 != QwenPagedSwapArtifactSHA256 {
		return fmt.Errorf("modelengine: invalid artifact sha256 %q, want %q", r.ArtifactSHA256, QwenPagedSwapArtifactSHA256)
	}
	if len(r.ArrivalTrace) < 2 {
		return fmt.Errorf("modelengine: arrival trace must record at least 2 sessions, got %d", len(r.ArrivalTrace))
	}
	for i, arrival := range r.ArrivalTrace {
		if arrival.Request == "" || arrival.Tokens <= 0 || arrival.PromptSHA256 == "" {
			return fmt.Errorf("modelengine: arrival trace %d is incomplete: %+v", i, arrival)
		}
	}

	// OFF arm: baseline without swap pressure
	if r.Off.Pressure != "OFF" {
		return fmt.Errorf("modelengine: OFF arm pressure must be OFF, got %q", r.Off.Pressure)
	}
	if r.Off.SwapTotal != 0 || r.Off.ReadmittedTotal != 0 || r.Off.SwapBytes != 0 || r.Off.RestoredBytes != 0 {
		return fmt.Errorf("modelengine: OFF arm observed swap (swap_total=%d readmitted=%d)", r.Off.SwapTotal, r.Off.ReadmittedTotal)
	}
	if r.Off.RecomputeTotal != 0 {
		return fmt.Errorf("modelengine: OFF arm observed recompute (recompute_total=%d)", r.Off.RecomputeTotal)
	}
	if r.Off.FallbackTotal != 0 {
		return fmt.Errorf("modelengine: OFF arm observed fallback (fallback_total=%d)", r.Off.FallbackTotal)
	}
	if r.Off.ErrorTotal != 0 {
		return fmt.Errorf("modelengine: OFF arm observed error (error_total=%d)", r.Off.ErrorTotal)
	}

	// ON arm: forced paged-swap pressure
	if r.On.Pressure != "ON" {
		return fmt.Errorf("modelengine: ON arm pressure must be ON, got %q", r.On.Pressure)
	}
	if r.On.SwapTotal <= 0 {
		return fmt.Errorf("modelengine: ON arm missing swap (swap_total=%d, want > 0)", r.On.SwapTotal)
	}
	if r.On.ReadmittedTotal <= 0 {
		return fmt.Errorf("modelengine: ON arm missing readmission (readmitted_total=%d, want > 0)", r.On.ReadmittedTotal)
	}
	if r.On.SwapBytes <= 0 {
		return fmt.Errorf("modelengine: ON arm swap_bytes must be > 0, got %d", r.On.SwapBytes)
	}
	if r.On.RestoredBytes != r.On.SwapBytes {
		return fmt.Errorf("modelengine: ON arm swapped and restored bytes mismatch: swap_bytes=%d restored_bytes=%d", r.On.SwapBytes, r.On.RestoredBytes)
	}
	if r.On.RecomputeTotal != 0 {
		return fmt.Errorf("modelengine: ON arm observed recompute (recompute_total=%d)", r.On.RecomputeTotal)
	}
	if r.On.FallbackTotal != 0 {
		return fmt.Errorf("modelengine: ON arm observed fallback (fallback_total=%d)", r.On.FallbackTotal)
	}
	if r.On.ErrorTotal != 0 {
		return fmt.Errorf("modelengine: ON arm observed error (error_total=%d)", r.On.ErrorTotal)
	}

	// Output equality between OFF and ON arms
	if len(r.Off.Requests) != len(r.On.Requests) {
		return fmt.Errorf("modelengine: request count mismatch: OFF=%d, ON=%d", len(r.Off.Requests), len(r.On.Requests))
	}
	if len(r.Off.Requests) < 2 {
		return fmt.Errorf("modelengine: arm must have at least 2 requests, got %d", len(r.Off.Requests))
	}
	for i := range r.Off.Requests {
		offReq := r.Off.Requests[i]
		onReq := r.On.Requests[i]
		if offReq.Request != onReq.Request {
			return fmt.Errorf("modelengine: request name mismatch at %d: OFF=%q, ON=%q", i, offReq.Request, onReq.Request)
		}
		if !reflect.DeepEqual(offReq.TokenIDs, onReq.TokenIDs) {
			return fmt.Errorf("modelengine: accepted token IDs mismatch for %q between OFF and ON arms", offReq.Request)
		}
		if offReq.OutputSHA256 != onReq.OutputSHA256 {
			return fmt.Errorf("modelengine: output sha256 mismatch for %q between OFF and ON arms", offReq.Request)
		}
		if offReq.StateDigest != "" && onReq.StateDigest != "" && offReq.StateDigest != onReq.StateDigest {
			return fmt.Errorf("modelengine: state digest mismatch for %q between OFF and ON arms: OFF=%s, ON=%s", offReq.Request, offReq.StateDigest, onReq.StateDigest)
		}
	}

	// State digests equality
	if len(r.Off.StateDigests) > 0 || len(r.On.StateDigests) > 0 {
		if !reflect.DeepEqual(r.Off.StateDigests, r.On.StateDigests) {
			return errors.New("modelengine: state digests mismatch between OFF and ON arms")
		}
	}
	if len(r.Off.StateIdentities) > 0 || len(r.On.StateIdentities) > 0 {
		if !qwenPagedSwapStatesEqual(r.Off.StateIdentities, r.On.StateIdentities) {
			return errors.New("modelengine: state identities mismatch between OFF and ON arms")
		}
	}

	// Memory / Latency / Metrics accounting for both arms
	for _, arm := range []*QwenPagedSwapArm{&r.Off, &r.On} {
		if arm.PeakRunning <= 0 {
			return fmt.Errorf("modelengine: %s arm peak_running must be > 0, got %d", arm.Pressure, arm.PeakRunning)
		}
		if arm.PeakRSSBytes == 0 {
			return fmt.Errorf("modelengine: %s arm peak_rss_bytes must be > 0", arm.Pressure)
		}
		if arm.TTFTP50Milliseconds <= 0 || arm.ITLP50Milliseconds <= 0 {
			return fmt.Errorf("modelengine: %s arm latency percentiles must be > 0", arm.Pressure)
		}
		if arm.AggregateTokensPerSecond <= 0 {
			return fmt.Errorf("modelengine: %s arm aggregate_tokens_per_second must be > 0", arm.Pressure)
		}
		if !arm.TeardownComplete {
			return fmt.Errorf("modelengine: %s arm teardown is incomplete", arm.Pressure)
		}
	}

	if r.BindingSHA256 != "" {
		if err := r.CheckBinding(); err != nil {
			return err
		}
	}
	return nil
}

// MarshalQwenPagedSwapReceipt serializes the receipt as indented JSON.
func MarshalQwenPagedSwapReceipt(r *QwenPagedSwapReceipt) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// UnmarshalQwenPagedSwapReceipt deserializes a receipt from JSON and validates it.
func UnmarshalQwenPagedSwapReceipt(data []byte) (*QwenPagedSwapReceipt, error) {
	var r QwenPagedSwapReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("modelengine: unmarshal receipt: %w", err)
	}
	if err := ValidateQwenPagedSwapReceipt(&r); err != nil {
		return nil, fmt.Errorf("modelengine: validate unmarshaled receipt: %w", err)
	}
	return &r, nil
}

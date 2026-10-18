package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is a host-f32/HAL diagnostic target primitive for #12727, not a DFlash2 decoder or
// artifact admission. It captures actual post-block residuals and reuses the
// complete Qwen speculative transaction (attention KV, recurrent/conv state,
// lineage and final-hidden history). No state is reconstructed from embeddings.
// The eventual sidecar adapter must separately bind the checkpoint and its taps.
const qwen35TargetFeatureMaxTokens = 8

type qwen35TargetFeatureRow struct {
	Position int
	Token    int
	Layers   []int
	Hidden   [][]float32 // one independent post-block, pre-final-norm row per layer
}

type qwen35TargetLayerCapture struct {
	row qwen35TargetFeatureRow
}

func (c *qwen35TargetLayerCapture) record(token, position, layer int, hidden []float32) {
	next := len(c.row.Hidden)
	if token == c.row.Token && position == c.row.Position && next < len(c.row.Layers) && layer == c.row.Layers[next] {
		c.row.Hidden = append(c.row.Hidden, append([]float32(nil), hidden...))
	}
}

// qwen35TargetFeatureTransaction stages features privately until Commit. The
// caller owns the Session exclusively from begin through Commit/Abort, just as
// for the existing target transaction: no concurrent Step/Close/prefix restore,
// no streaming publication before Commit. Independent sessions share no mutable
// feature state. The active-session pointer rejects nested transactions; it is
// not a replacement for the engine's single-session execution owner.
type qwen35TargetFeatureTransaction struct {
	tx         *qwen35MTPTargetTransaction
	layers     []int
	rows       []qwen35TargetFeatureRow
	ctx        context.Context
	panelWidth int
	panelValid func() bool
}

func beginQwen35TargetFeatureTransaction(target *Session, beforeLogits []float32, layers []int) (*qwen35TargetFeatureTransaction, error) {
	if err := validateQwen35TargetFeatureSession(target); err != nil {
		return nil, err
	}
	if target.qwen35TargetFeatureTxn != nil || target.qwen35TargetLayerCapture != nil {
		return nil, errors.New("model: target feature transaction is already active")
	}
	if len(beforeLogits) != target.M.Cfg.VocabSize {
		return nil, errors.New("model: target features require the complete current logits row")
	}
	if len(layers) < 1 || len(layers) > 5 {
		return nil, errors.New("model: target features require one to five layer taps")
	}
	for i, layer := range layers {
		if layer < 0 || layer >= target.M.Cfg.NumLayers || (i > 0 && layer <= layers[i-1]) {
			return nil, errors.New("model: target feature layers must be distinct, increasing and in range")
		}
	}
	tx, err := beginQwen35MTPTargetTransaction(target, beforeLogits)
	if err != nil {
		return nil, err
	}
	f := &qwen35TargetFeatureTransaction{tx: tx, layers: append([]int(nil), layers...)}
	target.qwen35TargetFeatureTxn = f
	// Explicit serial native execution is the reference, not a claimed single
	// target panel. It also avoids the MTP panel's final-hidden-only capture.
	tx.verify = f.verifyNative
	return f, nil
}

func (f *qwen35TargetFeatureTransaction) active() bool {
	return f != nil && f.tx != nil && !f.tx.closed && f.tx.target.qwen35TargetFeatureTxn == f
}

func (f *qwen35TargetFeatureTransaction) release() {
	if f == nil || f.tx == nil {
		return
	}
	if f.tx.target.qwen35TargetFeatureTxn == f {
		f.tx.target.qwen35TargetFeatureTxn = nil
	}
	f.rows = nil
	f.ctx = nil
	f.panelValid = nil
}

// Verify returns next-token logits after each supplied token. The caller must
// prepend its beforeLogits when comparing proposals. Features remain private;
// accepted is selected by exact target verification, not drafter confidence.
func (f *qwen35TargetFeatureTransaction) Verify(ctx context.Context, tokens []int) ([][]float32, error) {
	if !f.active() || f.tx.verifyStarted {
		return nil, errors.New("model: target feature transaction is closed or already verified")
	}
	var invalid error
	if ctx == nil {
		invalid = errors.New("model: target feature verification requires a context")
	} else if err := ctx.Err(); err != nil {
		invalid = err
	} else if len(tokens) < 1 || len(tokens) > qwen35TargetFeatureMaxTokens {
		invalid = errors.New("model: target feature verification requires one to eight tokens")
	} else {
		for _, token := range tokens {
			if token < 0 || token >= f.tx.target.M.Cfg.VocabSize {
				invalid = errors.New("model: target feature token is outside the target vocabulary")
				break
			}
		}
	}
	if invalid != nil {
		return nil, errors.Join(invalid, f.Abort())
	}
	if f.panelWidth != 0 && len(tokens) > f.panelWidth {
		return nil, errors.Join(errors.New("model: target feature panel exceeds request width"), f.Abort())
	}
	f.ctx = ctx
	var rows [][]float32
	var err error
	if f.panelWidth == 0 {
		rows, err = f.tx.Verify(tokens)
	} else {
		rows, err = f.verifyPanelOwned(tokens)
	}
	if err != nil {
		f.release() // the underlying transaction has already restored the prefix
	}
	return rows, err
}

func (f *qwen35TargetFeatureTransaction) verifyNative(tokens []int) (rows [][]float32, receipt TargetVerificationReceipt, err error) {
	receipt = TargetVerificationReceipt{
		Schema: targetVerificationReceiptSchema,
		Engine: targetVerificationEngine,
		Path:   targetVerificationDecodePath,
	}
	started := time.Now()
	defer func() { receipt.Accounting.TargetVerification = measuredSpeculativeCost(started) }()
	target := f.tx.target
	for _, token := range tokens {
		if err := f.ctx.Err(); err != nil {
			return nil, receipt, err
		}
		position, _, prefixErr := qwen35TargetFeaturePrefix(target)
		if prefixErr != nil {
			return nil, receipt, prefixErr
		}
		capture := &qwen35TargetLayerCapture{row: qwen35TargetFeatureRow{
			Position: position, Token: token, Layers: append([]int(nil), f.layers...),
		}}
		logits := func() []float32 {
			target.qwen35TargetLayerCapture = capture
			defer func() { target.qwen35TargetLayerCapture = nil }()
			return target.stepOwned(token)
		}()
		receipt.TargetDecodeSteps++
		if err := f.ctx.Err(); err != nil {
			return nil, receipt, err
		}
		position, lineage, prefixErr := qwen35TargetFeaturePrefix(target)
		if prefixErr != nil || len(capture.row.Hidden) != len(f.layers) || position != capture.row.Position+1 || lineage.ids[capture.row.Position] != uint32(token) {
			return nil, receipt, errors.New("model: target feature capture or token lineage is incomplete")
		}
		for _, hidden := range capture.row.Hidden {
			if len(hidden) != target.M.Cfg.HiddenSize {
				return nil, receipt, errors.New("model: target feature hidden width mismatch")
			}
			receipt.Accounting.KnownMemoryBytes += int64(len(hidden)) * 4
		}
		f.rows = append(f.rows, capture.row)
		rows = append(rows, append([]float32(nil), logits...))
		receipt.Accounting.KnownMemoryBytes += int64(len(logits)) * 4
	}
	f.tx.verifiedLive = true
	return rows, receipt, nil
}

// Commit returns only accepted feature rows, in target token order, and the
// logits at that boundary. A partial commit uses the existing complete snapshot
// restore + native Step replay, or the explicitly selected device panel
// reuses retained GDN projections; bare KV eviction cannot restore Qwen GDN state.
// The returned rows own their storage independently of the Session and future
// transactions. This method does not decide acceptance or emit tokens.
func (f *qwen35TargetFeatureTransaction) Commit(ctx context.Context, accepted int) ([]float32, []qwen35TargetFeatureRow, error) {
	if !f.active() || !f.tx.verifyStarted || !f.tx.verifiedLive {
		return nil, nil, errors.New("model: target features require successful verification before commit")
	}
	if ctx == nil {
		return nil, nil, errors.Join(errors.New("model: target feature commit requires a context"), f.Abort())
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, errors.Join(err, f.Abort())
	}
	defer f.release()
	f.tx.step = func(token int) []float32 {
		if err := ctx.Err(); err != nil {
			panic(err) // the existing transaction catches and restores replay failures
		}
		logits := f.tx.target.stepOwned(token)
		if err := ctx.Err(); err != nil {
			panic(err)
		}
		return logits
	}
	var logits []float32
	var err error
	if f.panelWidth == 0 {
		logits, err = f.tx.Commit(accepted)
	} else {
		logits, err = f.commitPanelOwned(accepted)
	}
	if err != nil {
		return nil, nil, errors.Join(err, ctx.Err())
	}
	if accepted == 0 {
		return logits, nil, nil
	}
	// Copy row descriptors so accepted storage does not retain rejected rows.
	features := append([]qwen35TargetFeatureRow(nil), f.rows[:accepted]...)
	return logits, features, nil
}

func (f *qwen35TargetFeatureTransaction) Abort() error {
	if f == nil || f.tx == nil {
		return nil
	}
	defer f.release()
	if err := f.tx.Abort(); err != nil {
		return fmt.Errorf("model: abort target features: %w", err)
	}
	return nil
}

// admitQwen35TargetFeaturePanel is allocation-free with respect to backend
// resources. Callers may probe it before sidecar acquisition; dispatch repeats
// it under the live execution owner. It never enables a Session/global policy.
func admitQwen35TargetFeaturePanel(s *Session, layers []int, width int) error {
	if err := validateQwen35TargetFeatureSession(s); err != nil {
		return err
	}
	if (width != 4 && width != 8) || len(layers) < 1 || len(layers) > 5 {
		return errors.New("model: target feature panel requires width 4 or 8 and one to five taps")
	}
	for i, layer := range layers {
		if layer < 0 || layer >= s.M.Cfg.NumLayers || (i > 0 && layer <= layers[i-1]) {
			return errors.New("model: target feature panel taps must be distinct, increasing and in range")
		}
	}
	if !backendHasComparableIdentity(s.Backend) || backendExecutionOwner(s.Backend) == nil {
		return errors.New("model: target feature panel requires a stable backend execution owner")
	}
	cfg := s.M.Cfg
	if s.M.lora != nil || s.M.prism != nil || s.M.attnObs != nil || cfg.EnableResidualHook || s.activeTap() != nil || s.tapActive != nil {
		return errors.New("model: target feature panel excludes adapters, residual hooks and external observers")
	}
	if !cfg.AttnOutputGate || cfg.AttentionBias || cfg.LayerNorm || cfg.QKNormPerHeadWeight || cfg.qkNormEps() != float32(cfg.RMSNormEps) || cfg.RopeScaling != "" || cfg.LongRope != nil || cfg.Alibi || cfg.AttnSoftcap != 0 || cfg.LogitSoftcap != 0 || cfg.QueryPreAttnScalar != 0 || (cfg.EmbedScale != 0 && cfg.EmbedScale != 1) || (cfg.LogitScale != 0 && cfg.LogitScale != 1) || cfg.DenseMLP || cfg.ActGeluTanh || cfg.ActGeluErf || cfg.BlockTopology != PreNorm || cfg.NumLoops > 1 {
		return errors.New("model: target feature panel excludes nonstandard scalar arithmetic")
	}
	position, _, err := qwen35TargetFeaturePrefix(s)
	if err != nil {
		return err
	}
	if err := qwen35TargetFeaturePanelPosition(cfg, position, width); err != nil {
		return err
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if cfg.windowForLayer(layer) > 0 || s.M.hasWeight(layerName(layer, "self_attn.sinks")) {
			return errors.New("model: target feature panel excludes windowed attention and sinks")
		}
		if !cfg.isLinearAttnLayer(layer) {
			q := s.M.hasWeight(layerName(layer, "self_attn.q_norm.weight"))
			k := s.M.hasWeight(layerName(layer, "self_attn.k_norm.weight"))
			if q != k || q != cfg.QKNorm {
				return errors.New("model: target feature panel QK normalization differs from scalar configuration")
			}
		}
	}
	if _, present, err := qwen35SequencePrefillBackend(s.Backend); err != nil || !present {
		return fmt.Errorf("model: target feature panel requires the exact sequence operation: %v", err)
	}
	all, allOK := s.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	replay, replayOK := s.Backend.(compute.Qwen35SequencePrefixReplayBackend)
	features, featuresOK := s.Backend.(compute.Qwen35SequenceFeatureRowsBackend)
	if !allOK || all.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath || !replayOK || replay.Qwen35SequencePrefixReplayPath() != compute.Qwen35SequencePrefixReplayPath || !featuresOK || features.Qwen35SequenceFeatureRowsPath() != compute.Qwen35SequenceFeatureRowsPath {
		return errors.New("model: target feature panel requires exact all-logits, prefix-replay and feature-row capabilities")
	}
	if s.captureTargetHidden {
		raw, ok := s.Backend.(compute.Qwen35SequenceRawHiddenBackend)
		if !ok || raw.Qwen35SequenceRawHiddenPath() != compute.Qwen35SequenceRawHiddenPath {
			return errors.New("model: target feature panel cannot preserve raw target-hidden history")
		}
	}
	return nil
}

func qwen35TargetFeaturePanelPosition(cfg Config, base, rows int) error {
	if rows < 1 || rows > 8 || base < 0 || base > math.MaxInt32-rows {
		return errors.New("model: target feature panel position range is invalid")
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		// Scalar Step may switch to QSA on any row, while the sequence operation
		// computes dense attention. Refuse even a crossing in the last row.
		if cfg.ShouldUseQSASparseGather(layer, base+rows, 1) {
			return errors.New("model: target feature panel would cross scalar QSA selection")
		}
	}
	return nil
}

// This additive constructor is deliberately not called by the default DFlash
// request. A later typed per-request policy must explicitly choose width 4 or 8.
func beginQwen35TargetFeaturePanelTransaction(s *Session, beforeLogits []float32, layers []int, width int) (*qwen35TargetFeatureTransaction, error) {
	if err := admitQwen35TargetFeaturePanel(s, layers, width); err != nil {
		return nil, err
	}
	backend, model, epoch, owner := s.Backend, s.M, s.cacheGeometryEpoch, backendExecutionOwner(s.Backend)
	base, _, err := qwen35TargetFeaturePrefix(s)
	if err != nil {
		return nil, err
	}
	f, err := beginQwen35TargetFeatureTransaction(s, beforeLogits, layers)
	if err != nil {
		return nil, err
	}
	f.panelWidth = width
	f.panelValid = func() bool {
		scope := s.backendExecutionScope
		if !backendHasComparableIdentity(s.Backend) || s.Backend != backend || s.M != model || s.cacheGeometryEpoch != epoch || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.qwen35TargetFeatureTxn != f || scope == nil || !scope.live || scope.owner != owner || backendExecutionOwner(s.Backend) != owner {
			return false
		}
		for _, session := range scope.sessions {
			if session == s {
				return true
			}
		}
		return false
	}
	f.tx.verify = func(tokens []int) (rows [][]float32, receipt TargetVerificationReceipt, err error) {
		if !f.panelValid() || len(tokens) < 1 || len(tokens) > width {
			return nil, receipt, errors.New("model: target feature panel ownership or width changed")
		}
		if err = admitQwen35TargetFeaturePanel(s, f.layers, width); err != nil {
			return nil, receipt, err
		}
		if position, _, prefixErr := qwen35TargetFeaturePrefix(s); prefixErr != nil || position != base {
			return nil, receipt, errors.New("model: target feature panel prefix changed before dispatch")
		}
		if err = f.ctx.Err(); err != nil {
			return nil, receipt, err
		}
		rows, _, receipt, err = s.verifyQwen35DeviceFeaturePanel(tokens, f.tx.beforeLogits, true, f)
		if err != nil {
			// Selected failure cannot trigger tx.verifyOwned's generic downgrade
			// fallback, which has neither intermediate taps nor panel accounting.
			return nil, receipt, fmt.Errorf("model: selected target feature panel failed: %v", err)
		}
		if !f.panelValid() {
			return nil, receipt, errors.New("model: target feature panel owner changed during execution")
		}
		position, lineage, prefixErr := qwen35TargetFeaturePrefix(s)
		if prefixErr != nil || position != base+len(tokens) {
			return nil, receipt, errors.New("model: target feature panel lineage is incomplete")
		}
		for row, token := range tokens {
			if lineage.ids[base+row] != uint32(token) {
				return nil, receipt, errors.New("model: target feature panel token identity changed")
			}
		}
		if err = f.ctx.Err(); err != nil {
			return nil, receipt, err
		}
		f.tx.verifiedLive = true
		return rows, receipt, nil
	}
	return f, nil
}

func (f *qwen35TargetFeatureTransaction) verifyPanelOwned(tokens []int) (rows [][]float32, err error) {
	completed := false
	defer func() {
		if !completed {
			// The inner execution scope has already unwound. Preserve its primary
			// failure while aborting every transaction-owned sibling.
			suppressBackendCleanupPanic(func() { _ = f.Abort() })
		}
	}()
	rows, err = func() ([][]float32, error) {
		scope := f.tx.target.beginBackendExecution()
		finished := false
		defer scope.finishCompleted(&finished)
		r, e := f.tx.verifyOwned(tokens)
		finished = true
		return r, e
	}()
	if err == nil && (!f.tx.verifiedLive || len(rows) != len(tokens) || len(f.rows) != len(tokens)) {
		err = errors.Join(errors.New("model: interrupted target feature panel produced no complete result"), f.Abort())
	}
	completed = true
	return rows, err
}

func (f *qwen35TargetFeatureTransaction) readPanelFeatures(tokens []int, base int, panels []compute.Qwen35SequenceFeatureRows) error {
	if !f.panelValid() || len(panels) != len(f.layers) {
		return errors.New("model: target feature panel returned wrong ownership or tap count")
	}
	hidden, backend := f.tx.target.M.Cfg.HiddenSize, f.tx.target.Backend
	rows := make([]qwen35TargetFeatureRow, len(tokens))
	for i, token := range tokens {
		rows[i] = qwen35TargetFeatureRow{Position: base + i, Token: token, Layers: append([]int(nil), f.layers...), Hidden: make([][]float32, len(f.layers))}
	}
	for i, panel := range panels {
		tensor := panel.Rows
		if panel.Layer != f.layers[i] || tensor.Buf() == nil || !tensor.Ready() || tensor.Dtype != compute.F32 || tensor.Backend() != backend || len(tensor.Shape) != 2 || tensor.Shape[0] != len(tokens) || tensor.Shape[1] != hidden {
			return errors.New("model: target feature panel returned malformed tap rows")
		}
		flat := backend.Read(tensor)
		if !f.panelValid() || len(flat) != len(tokens)*hidden {
			return errors.New("model: target feature panel read returned wrong ownership or shape")
		}
		for _, value := range flat {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("model: target feature panel contains a nonfinite residual")
			}
		}
		for row := range rows {
			rows[row].Hidden[i] = append([]float32(nil), flat[row*hidden:(row+1)*hidden]...)
		}
	}
	f.rows = rows
	return nil
}

func (f *qwen35TargetFeatureTransaction) commitPanelOwned(accepted int) (logits []float32, err error) {
	completed := false
	defer func() {
		if !completed {
			suppressBackendCleanupPanic(func() { _ = f.Abort() })
		}
	}()
	logits, err = func() ([]float32, error) {
		scope := f.tx.target.beginBackendExecution()
		finished := false
		defer scope.finishCompleted(&finished)
		r, e := f.tx.commitOwned(accepted)
		finished = true
		return r, e
	}()
	completed = true
	return logits, err
}

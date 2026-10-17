package model

import (
	"context"
	"errors"
	"fmt"
	"time"
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
	tx     *qwen35MTPTargetTransaction
	layers []int
	rows   []qwen35TargetFeatureRow
	ctx    context.Context
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
	f.ctx = ctx
	rows, err := f.tx.Verify(tokens)
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
// restore + native Step replay; bare KV eviction cannot restore Qwen GDN state.
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
	logits, err := f.tx.Commit(accepted)
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

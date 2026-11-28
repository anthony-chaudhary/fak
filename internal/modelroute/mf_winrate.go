// mf_winrate.go — the MATRIX-FACTORIZATION WIN-RATE ROUTER: score candidate
// routes by a predicted win rate and select the top one.
//
// BORROWED CONTRACT (ADAPT, not a port): lm-sys/RouteLLM @
// 2024-08-10, so this is a design reference, never a live dependency). The
// contract adapted here is the matrix-factorization scorer in
// routellm/routers/matrix_factorization/model.py: the per-model embedding
// table P (num_models x dim), a 2-norm-normalized model embedding, a projected
// prompt embedding, and the classifier logit
//
//	logit = W . (normalize(P[model]) * prompt_proj)
//
// (elementwise product, then a single linear readout). The pairwise win rate
// is the sigmoid of the logit difference between two models:
//
//	winrate(a, b) = sigmoid(logit(a) - logit(b))
//
// which is exactly MFModel.pred_win_rate. Re-expressed in idiomatic Go against
// this package's Subject vocabulary; no upstream code block is copied. The
// OpenAI text-embedding call is replaced by an injected prompt embedding so the
// package stays pure (stdlib only, no I/O). Ticket:
package modelroute

import (
	"errors"
	"fmt"
	"math"
)

// Closed refusal vocabulary for the matrix-factorization win-rate router.
// Callers match these with errors.Is; the wrapped detail after ": " is
// diagnostic only.
var (
	// ErrMFCandidates is an empty candidate set: there is no route to select.
	ErrMFCandidates = errors.New("modelroute: no candidate routes to score")
	// ErrMFDimMismatch is a prompt embedding whose length does not equal the
	// model-embedding dimension, so the elementwise product is undefined.
	ErrMFDimMismatch = errors.New("modelroute: prompt embedding dimension does not match model embedding dimension")
	// ErrMFUnknownModel is a candidate whose model id has no row in the
	// learned model-embedding table.
	ErrMFUnknownModel = errors.New("modelroute: candidate model id has no learned embedding")
	// ErrMFZeroNorm is a model embedding of zero 2-norm, which cannot be
	// normalized (the source normalizes P[model] before the product).
	ErrMFZeroNorm = errors.New("modelroute: model embedding has zero 2-norm")
)

// MFWinRateRouter is the matrix-factorization win-rate scorer. It holds the
// learned model-embedding table P (one row per model id, each of length dim)
// and the classifier readout W (length dim). Given a prompt embedding it
// scores each candidate route by its predicted win-rate logit and selects the
// top one. It is pure: no I/O, no goroutines, deterministic.
type MFWinRateRouter struct {
	// ModelEmbed is the learned per-model embedding table: model id ->
	// embedding of length dim. This is the source's P (nn.Embedding).
	ModelEmbed map[string][]float64
	// Classifier is the readout W (length dim) applied to the elementwise
	// product of the normalized model embedding and the prompt embedding.
	// This is the source's classifier (a single bias-free Linear).
	Classifier []float64
}

// dim returns the embedding dimension, or 0 if the table is empty.
func (r *MFWinRateRouter) dim() int {
	for _, v := range r.ModelEmbed {
		return len(v)
	}
	return 0
}

// logit is the source's forward() for one model: normalize the model
// embedding, take the elementwise product with the prompt embedding, and read
// out a single logit with the classifier. It fails loud on a dimension
// mismatch, an unknown model, or a zero-norm embedding.
func (r *MFWinRateRouter) logit(model string, prompt []float64) (float64, error) {
	emb, ok := r.ModelEmbed[model]
	if !ok {
		return 0, fmt.Errorf("%w: %q", ErrMFUnknownModel, model)
	}
	if len(emb) != len(prompt) {
		return 0, fmt.Errorf("%w: model %q has dim %d, prompt has dim %d", ErrMFDimMismatch, model, len(emb), len(prompt))
	}
	// 2-norm of the model embedding (the source's F.normalize(p=2, dim=1)).
	norm := 0.0
	for _, x := range emb {
		norm += x * x
	}
	if norm == 0 {
		return 0, fmt.Errorf("%w: model %q", ErrMFZeroNorm, model)
	}
	inv := 1.0 / math.Sqrt(norm)
	// logit = sum_i W[i] * (emb[i]/norm) * prompt[i]
	sum := 0.0
	for i := range emb {
		sum += r.Classifier[i] * (emb[i] * inv) * prompt[i]
	}
	return sum, nil
}

// Score returns the predicted win-rate logit for one candidate model on the
// given prompt embedding. Higher is better.
func (r *MFWinRateRouter) Score(model string, prompt []float64) (float64, error) {
	return r.logit(model, prompt)
}

// WinRate returns the predicted probability that model a beats model b on the
// prompt, mirroring MFModel.pred_win_rate: sigmoid(logit(a) - logit(b)).
func (r *MFWinRateRouter) WinRate(a, b string, prompt []float64) (float64, error) {
	la, err := r.logit(a, prompt)
	if err != nil {
		return 0, err
	}
	lb, err := r.logit(b, prompt)
	if err != nil {
		return 0, err
	}
	return sigmoid(la - lb), nil
}

// Select scores every candidate route by its predicted win-rate logit and
// returns the model id of the top route. Ties break toward the earlier
// candidate (stable argmax). It fails loud on an empty candidate set or any
// candidate that cannot be scored.
func (r *MFWinRateRouter) Select(candidates []string, prompt []float64) (string, error) {
	if len(candidates) == 0 {
		return "", ErrMFCandidates
	}
	best := ""
	bestLogit := math.Inf(-1)
	for _, c := range candidates {
		l, err := r.logit(c, prompt)
		if err != nil {
			return "", err
		}
		if l > bestLogit {
			bestLogit = l
			best = c
		}
	}
	return best, nil
}

// sigmoid is the logistic function, the source's torch.sigmoid.
func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

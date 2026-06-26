//go:build fakwiremodel

package wirescreen

import (
	"context"
	"strings"
	"sync"
)

func init() {
	// Register under "model" so FAK_WIRE_REDACT=model resolves this redactor in the
	// tagged build. The default build never compiles this file, preserving the inert
	// pure-Go catalog.
	RegisterRedactor("model", modelRedactor{})
}

// modelRedactor is the model-backed pre-send redactor. It includes the deterministic
// piiRedactor floor first, then adds model-proposed spans from the shared fakwiremodel
// model/tokenizer loader. Selecting FAK_WIRE_REDACT=model therefore cannot weaken the
// existing floor: model failure degrades to the same deterministic spans.
type modelRedactor struct{}

func (modelRedactor) Name() string { return "model" }

func (modelRedactor) Propose(ctx context.Context, body []byte, tool string) []Span {
	if len(body) == 0 {
		return nil
	}
	spans := piiRedactor{}.Propose(ctx, body, tool)
	if b := currentModelRedactionBackend(); b != nil {
		spans = append(spans, b.ProposeRedactions(ctx, body, tool)...)
	}
	return coalesce(spans, len(body))
}

type modelRedactionBackend interface {
	ProposeRedactions(ctx context.Context, body []byte, tool string) []Span
}

var (
	modelRedactionMu      sync.RWMutex
	modelRedactionCurrent modelRedactionBackend = llmRedactionBackend{}
)

func currentModelRedactionBackend() modelRedactionBackend {
	modelRedactionMu.RLock()
	defer modelRedactionMu.RUnlock()
	return modelRedactionCurrent
}

// llmRedactionBackend is the real tagged backend. It reuses ensureModel(), loadTokenizer(),
// and the YES/NO verbalizer from model_screener.go; when no checkpoint is configured it
// returns nil and the modelRedactor degrades to piiRedactor-only behavior.
type llmRedactionBackend struct{}

func (llmRedactionBackend) ProposeRedactions(ctx context.Context, body []byte, tool string) []Span {
	m, tok, vrb, ok := ensureModel()
	if !ok || m == nil || tok == nil {
		return nil
	}
	cands := redactionCandidates(body)
	if len(cands) == 0 {
		return nil
	}
	spans := make([]Span, 0, len(cands))
	for _, c := range cands {
		if err := ctx.Err(); err != nil {
			break
		}
		if classifyPrompted(m, tok, vrb, redactionPrompt(body[c.Start:c.End], tool)) {
			spans = append(spans, Span{Start: c.Start, End: c.End, Kind: "model_secret"})
		}
	}
	return spans
}

const (
	modelRedactionCandidateCap   = 320
	modelRedactionCandidateLimit = 8
)

// redactionCandidates bounds the model work by handing it suspicious lines/sentences, not
// arbitrary bulk. The model is the final proposer for these coarse spans; the deterministic
// floor still handles exact secret shapes independently.
func redactionCandidates(body []byte) []Span {
	var out []Span
	start := 0
	flush := func(end int) {
		if len(out) >= modelRedactionCandidateLimit {
			return
		}
		for start < end && isSpaceByte(body[start]) {
			start++
		}
		for end > start && isSpaceByte(body[end-1]) {
			end--
		}
		if start >= end {
			return
		}
		text := strings.ToLower(string(body[start:end]))
		if !hasRedactionHint(text) {
			return
		}
		if end-start > modelRedactionCandidateCap {
			end = start + modelRedactionCandidateCap
		}
		out = append(out, Span{Start: start, End: end, Kind: "model_secret"})
	}
	for i, b := range body {
		switch b {
		case '\n', '\r':
			flush(i)
			start = i + 1
		case '.', '!', '?':
			flush(i + 1)
			start = i + 1
		}
	}
	flush(len(body))
	return out
}

func hasRedactionHint(s string) bool {
	for _, h := range []string{
		"api key", "auth", "bearer", "credential", "email", "oauth", "password",
		"private key", "secret", "social security", "ssn", "token",
	} {
		if strings.Contains(s, h) {
			return true
		}
	}
	return false
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func redactionPrompt(candidate []byte, tool string) string {
	text := string(candidate)
	toolLine := "an unspecified tool"
	if t := strings.TrimSpace(tool); t != "" {
		toolLine = t
	}
	const sys = "You are a precise security redaction classifier. You read one bounded " +
		"tool/message span and decide whether it contains private credentials, API keys, " +
		"authentication tokens, passwords, private keys, personal identifiers, or other " +
		"secret material that should be redacted before leaving the machine. Ordinary " +
		"non-secret text is not redaction-worthy. Answer with a single word: YES or NO."
	return "<|im_start|>system\n" + sys + "<|im_end|>\n" +
		"<|im_start|>user\n" +
		"The span came from " + toolLine + ":\n" +
		"<span>\n" + text + "\n</span>\n" +
		"Should this span be redacted? Answer YES or NO." +
		"<|im_end|>\n<|im_start|>assistant\n"
}

var _ Redactor = modelRedactor{}

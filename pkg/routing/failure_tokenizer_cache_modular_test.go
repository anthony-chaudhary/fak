package routing

import (
	"errors"
	"testing"
)

// failure_tokenizer_cache_modular_test.go — regression for the 2026-11-17
// incident: Modular's serving instance answered a valid request for a known
// model with a 400 whose body is the Go error chain
// "failed to translate request: tokenizing request: getting tokenizer for
// model \"RadixArk/Qwen3.8-27B-NVFP4\": tokenizer cache full (cap=1):
// refusing model \"RadixArk/Qwen3.8-27B-NVFP4\"" (witnessed live in the router
// journal, status 400, class PAYLOAD_REJECTED/fatal, caller opencode-desktop;
// the same model path returned 200 before, after, and on a same-day probe, so
// the 400 was the instance's transient resource guard, not a caller bug).
// Before the signature the body named no rung, so the 400 fell to the
// bare-400 default and read PAYLOAD_REJECTED/FATAL: the walk ended instead of
// failing over to a peer replica (or retrying, the slot cleared ~5s later).
// The sibling regression is failure_context_overflow_modular_test.go
// (ticket router-modular-overflow-misclass-20261108): same bug class, a
// Modular 400 that must not read as a caller bug.

// modularTokenizerCacheBody is the verbatim upstream 400 body opencode saw on
// 2026-11-17.
const modularTokenizerCacheBody = `failed to translate request: tokenizing request: getting tokenizer for model "RadixArk/Qwen3.8-27B-NVFP4": tokenizer cache full (cap=1): refusing model "RadixArk/Qwen3.8-27B-NVFP4"`

func TestClassifyModularTokenizerCacheFull400(t *testing.T) {
	got := Classify(400, modularTokenizerCacheBody)
	if got.Class != FailureUnavailable {
		t.Fatalf("Classify(400, %q).Class = %q, want %q", modularTokenizerCacheBody, got.Class, FailureUnavailable)
	}
	if want := FailureUnavailable.DefaultAction(); got.Action != want {
		t.Fatalf("Classify(400, %q).Action = %q, want %q", modularTokenizerCacheBody, got.Action, want)
	}
	if !got.Class.Transient() {
		t.Fatalf("Classify(400, %q).Transient = false, want true (the slot cleared and the same model served 200 minutes later)", modularTokenizerCacheBody)
	}
}

// The signature must claim the body on its own, so the entry is not dead
// weight if the surrounding Go error chain is ever reworded upstream.
func TestClassifyModularTokenizerCacheSignatureClaimsAlone(t *testing.T) {
	got := Classify(400, "tokenizer cache full (cap=1)")
	if got.Class != FailureUnavailable || got.Action != ActionFailoverNext {
		t.Fatalf("Classify(400, %q) = class %q action %q, want %q/%q", "tokenizer cache full (cap=1)", got.Class, got.Action, FailureUnavailable, ActionFailoverNext)
	}
}

// The stream-error string form (the shape opencode's drain surfaces) must
// classify through ClassifyError too, not just the HTTP body arm.
func TestClassifyErrorModularTokenizerCacheFull(t *testing.T) {
	got := ClassifyError(errors.New("stream error: " + modularTokenizerCacheBody))
	if got.Class != FailureUnavailable || got.Action != ActionFailoverNext {
		t.Fatalf("ClassifyError(stream error).Class/Action = %q/%q, want %q/%q", got.Class, got.Action, FailureUnavailable, ActionFailoverNext)
	}
}

// Guard against over-widening: a model-NAME rejection is a different wall and
// must NOT be claimed by the cache signature. "refusing model" alone names no
// cache wording, so it keeps its own verdict (the bare-400 default); "unknown
// model" is the model-missing rung. Neither reads UPSTREAM_UNAVAILABLE.
func TestClassifyRefusingModelWithoutCacheSignatureStaysDefault(t *testing.T) {
	for _, body := range []string{
		`refusing model "RadixArk/Qwen3.8-27B-NVFP4"`,
		`unknown model "RadixArk/Qwen3.8-27B-NVFP4"`,
	} {
		got := Classify(400, body)
		if got.Class == FailureUnavailable {
			t.Fatalf("Classify(400, %q) = %q, want not %q (the cache signature over-widened to a model-name rejection)", body, got.Class, FailureUnavailable)
		}
	}
}

package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// NativeCompactionObservation describes one applied typed-message cut consumed by
// a successful native completion. It contains no prompt or restore payload.
// Token counts are estimates; shared regions do not establish KV cache identity.
type NativeCompactionObservation struct {
	DroppedMessages                         int
	PreBytes, PostBytes, ShedBytes          int
	PreSHA256, PostSHA256                   string
	PreEstimatedTokens, PostEstimatedTokens int
	TokenMethod, SerializationMethod        string
	PrefixBytes, SuffixBytes                int
	PrePrefixSHA256, PostPrefixSHA256       string
	PreSuffixSHA256, PostSuffixSHA256       string
	ProofFailure                            string
}

type nativeCompactionObserverKey struct{}

// WithNativeCompactionObserver composes request-local observers in registration
// order. Observers run synchronously after successful native generation and must
// return promptly. Prompt encoding alone never publishes an observation.
func WithNativeCompactionObserver(ctx context.Context, observe func(NativeCompactionObservation)) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if observe == nil {
		return ctx
	}
	previous := nativeCompactionObserver(ctx)
	if previous != nil {
		next := observe
		observe = func(value NativeCompactionObservation) { previous(value); next(value) }
	}
	return context.WithValue(ctx, nativeCompactionObserverKey{}, observe)
}

func nativeCompactionObserver(ctx context.Context) func(NativeCompactionObservation) {
	if ctx == nil {
		return nil
	}
	observe, _ := ctx.Value(nativeCompactionObserverKey{}).(func(NativeCompactionObservation))
	return observe
}

func observeNativeCompaction(ctx context.Context, value *NativeCompactionObservation) {
	if value == nil || value.DroppedMessages <= 0 {
		return
	}
	if observe := nativeCompactionObserver(ctx); observe != nil {
		observe(*value)
	}
}

func nativeCompactionHash(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func captureNativeCompaction(before, after []Message, dropped int) *NativeCompactionObservation {
	value := &NativeCompactionObservation{DroppedMessages: dropped,
		TokenMethod: "typed_message_estimate", SerializationMethod: "typed_messages_json"}
	pre, preErr := json.Marshal(before)
	post, postErr := json.Marshal(after)
	if preErr != nil || postErr != nil {
		value.ProofFailure = "JSON_ENCODING_FAILED"
		return value
	}
	if len(pre) <= len(post) {
		value.ProofFailure = "NONPOSITIVE_BYTE_DELTA"
		return value
	}
	var preMessages, postMessages []json.RawMessage
	if json.Unmarshal(pre, &preMessages) != nil || json.Unmarshal(post, &postMessages) != nil {
		value.ProofFailure = "JSON_ENCODING_FAILED"
		return value
	}
	prefix := 0
	for prefix < len(preMessages) && prefix < len(postMessages) && bytes.Equal(preMessages[prefix], postMessages[prefix]) {
		prefix++
	}
	suffix := 0
	for suffix < len(preMessages)-prefix && suffix < len(postMessages)-prefix &&
		bytes.Equal(preMessages[len(preMessages)-1-suffix], postMessages[len(postMessages)-1-suffix]) {
		suffix++
	}
	// Include the array bracket and inter-message commas only for nonempty
	// whole-message regions. Empty regions hash the empty byte sequence.
	p, s := 0, 0
	if prefix > 0 {
		p = prefix
		for _, message := range preMessages[:prefix] {
			p += len(message)
		}
	}
	if suffix > 0 {
		s = suffix
		for _, message := range preMessages[len(preMessages)-suffix:] {
			s += len(message)
		}
	}
	value.PreBytes, value.PostBytes, value.ShedBytes = len(pre), len(post), len(pre)-len(post)
	value.PreSHA256, value.PostSHA256 = nativeCompactionHash(pre), nativeCompactionHash(post)
	value.PreEstimatedTokens, value.PostEstimatedTokens = EstimateMessagesTokens(before), EstimateMessagesTokens(after)
	value.PrefixBytes, value.SuffixBytes = p, s
	value.PrePrefixSHA256, value.PostPrefixSHA256 = nativeCompactionHash(pre[:p]), nativeCompactionHash(post[:p])
	value.PreSuffixSHA256, value.PostSuffixSHA256 = nativeCompactionHash(pre[len(pre)-s:]), nativeCompactionHash(post[len(post)-s:])
	return value
}

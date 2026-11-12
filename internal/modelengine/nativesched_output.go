// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
package modelengine

import "github.com/anthony-chaudhary/fak/internal/abi"

// newNativeRequestOutbox isolates the selected native syscall producer from a
// delayed Tokens consumer. The upstream analogue (vLLM at
// 975dca5bb5db302077674cfa9afe851ee700ad73) keeps publication request-local
// rather than awaiting its reader.
//
// This is bounded by the actual native request budget, not a guessed queue
// size: stageDeferredToken stops at EOS or genTokens, and retireDeferredToken
// publishes each staged ID once. Both local and imported lanes start with zero
// generated tokens; preemption retains that count and the same channel. The
// producer therefore cannot fill this outbox and need one more slot. Native
// events have no Text payload, so this does not retain arbitrary text buffers.
//
// No relay goroutine owns a second terminal edge. The existing serialized
// finish closes the stream and releases the real session/KV on success, error,
// cancellation or shutdown; buffered IDs remain readable after that close.
// Consumers must still obey EngineRequest's drain-or-cancel contract. A caller
// that abandons a selected stream may allow the remaining bounded model work
// to complete instead of applying the historical delivery backpressure.
func newNativeRequestOutbox() chan abi.EngineToken {
	capacity := 1
	if nativeOutputIsolationEnabled {
		capacity = genTokens
	}
	return make(chan abi.EngineToken, capacity)
}

package tokenizer

import (
	"strconv"
	"strings"
	"sync"
)

const (
	encodeMemoPayloadLimit = 1 << 20
	encodeMemoEntryLimit   = 1024
)

// EncodeWork measures ordinary text submitted to pre-tokenization and BPE.
// The canonical added-token scan still examines the full input on every call.
// Direct nil-memo, SPM and no-added-token fallbacks report zero counters.
type EncodeWork struct {
	OrdinaryBytesEncoded int
	OrdinaryBytesReused  int
	CachedPayloadBytes   int
}

// EncodeMemo retains bounded immutable encodings of ordinary runs from the last
// successful tokenizer encode, even if later request admission or inference fails.
// Its zero value is ready to use. One memo is safe to share
// across calls, but interleaved conversations may replace each other's entries.
// Logical key/ID payload is capped at 1 MiB with at most 1024 map entries. While
// an encode is in flight the two generations can retain up to 2 MiB payload.
// These are not heap bounds: map overhead and normal output storage are additional.
// It relies on Tokenizer's existing immutable-after-construction contract.
type EncodeMemo struct {
	mu        sync.Mutex
	tokenizer *Tokenizer
	previous  map[string]encodedRun
}

type encodedRun struct {
	text string
	ids  []int
}

// Encode preserves Encode's complete added-token recognition and longest-match
// behavior. Only an exact ordinary run, already bounded by that recognition, may
// reuse IDs. It never joins separately encoded arbitrary string segments.
// SPM remains on its original path; EncodeSegments does not use this memo.
func (m *EncodeMemo) Encode(t *Tokenizer, text string) ([]int, EncodeWork, error) {
	return m.encode(t, text, true)
}

// EncodeTransient reuses retained ordinary runs without replacing the memo's
// successful full-request generation. Auxiliary prefix measurements can reuse
// the same work without evicting the conversation's remaining runs.
// CachedPayloadBytes describes this call's temporary runs, not a published cache.
func (m *EncodeMemo) EncodeTransient(t *Tokenizer, text string) ([]int, EncodeWork, error) {
	return m.encode(t, text, false)
}

func (m *EncodeMemo) encode(t *Tokenizer, text string, publish bool) ([]int, EncodeWork, error) {
	if t == nil {
		return nil, EncodeWork{}, ErrEncodeUnsupported
	}
	if m == nil {
		ids, err := t.Encode(text)
		return ids, EncodeWork{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if t.spm != nil || len(t.addedByContent) == 0 {
		ids, err := t.Encode(text)
		if err == nil && publish {
			m.tokenizer, m.previous = t, nil
		}
		return ids, EncodeWork{}, err
	}
	run := &encodeRunMemo{next: make(map[string]encodedRun)}
	if m.tokenizer == t {
		run.previous = m.previous
	}
	ids, err := t.encode(text, run)
	if err != nil {
		return nil, run.work, err
	}
	if publish {
		m.tokenizer, m.previous = t, run.next
	}
	return ids, run.work, nil
}

type encodeRunMemo struct {
	previous map[string]encodedRun
	next     map[string]encodedRun
	work     EncodeWork
}

func (m *encodeRunMemo) lookup(text string) ([]int, bool) {
	if m == nil {
		return nil, false
	}
	entry, ok := m.next[text]
	if !ok {
		entry, ok = m.previous[text]
	}
	if !ok {
		m.work.OrdinaryBytesEncoded += len(text)
		return nil, false
	}
	m.work.OrdinaryBytesReused += len(text)
	m.retain(entry)
	return entry.ids, true
}

func (m *encodeRunMemo) remember(text string, ids []int) {
	if m == nil || !m.canRetain(text, len(ids)) {
		return
	}
	// Clone the run, not its backing full prompt. Returned result IDs must never
	// alias retained storage, and tiny keys must not retain oversized inputs.
	m.retain(encodedRun{strings.Clone(text), append([]int(nil), ids...)})
}

func (m *encodeRunMemo) canRetain(text string, ids int) bool {
	if _, exists := m.next[text]; exists || len(m.next) >= encodeMemoEntryLimit {
		return false
	}
	remaining := encodeMemoPayloadLimit - m.work.CachedPayloadBytes
	return len(text) <= remaining && ids <= (remaining-len(text))/(strconv.IntSize/8)
}

func (m *encodeRunMemo) retain(entry encodedRun) {
	if !m.canRetain(entry.text, len(entry.ids)) {
		return
	}
	m.next[entry.text] = entry
	m.work.CachedPayloadBytes += len(entry.text) + len(entry.ids)*(strconv.IntSize/8)
}

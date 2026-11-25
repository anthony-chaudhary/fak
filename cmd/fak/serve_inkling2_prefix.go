package main

import (
	"errors"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// A checkpoint keeps only what the next horizon tokens would overwrite and relies
// on its own session's earlier cache rows, so one session holds one conversation's
// prefix: each prefix-cache slot is a whole resident session.
const (
	inkling2PrefixSlotsMax        = 8
	inkling2PrefixCacheMiBDefault = 4096
)

// inkling2PrefixCacheFromEnv reads FAK_INKLING2_PREFIX_SLOTS (resident sessions,
// default 1) and FAK_INKLING2_PREFIX_CACHE_MIB (what more than one slot may hold
// together, default 4096); an unparsable or out-of-range value keeps its default.
func inkling2PrefixCacheFromEnv(lookup func(string) (string, bool)) (slots int, capBytes int64) {
	slots, capBytes = 1, int64(inkling2PrefixCacheMiBDefault)<<20
	if v, ok := lookup("FAK_INKLING2_PREFIX_SLOTS"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 1 {
			slots = min(n, inkling2PrefixSlotsMax)
		}
	}
	if v, ok := lookup("FAK_INKLING2_PREFIX_CACHE_MIB"); ok {
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil && n > 0 && n <= 1<<30 {
			capBytes = n << 20
		}
	}
	return slots, capBytes
}

// inkling2HeldBytes mirrors the resident session's meter of the device memory it holds.
type inkling2HeldBytes interface{ HeldBytes() int64 }

// inkling2PrefixSlot is one resident session and the prompt its checkpoint covers.
type inkling2PrefixSlot struct {
	sess    ggufload.Inkling2LoopSession
	hasCkpt bool
	ckpt    []int
	used    uint64 // engine clock at the slot's last request
}

// prefixHit is the slot whose checkpoint is the longest proper prefix of prompt.
func (e *inkling2LoopEngine) prefixHit(prompt []int) *inkling2PrefixSlot {
	var best *inkling2PrefixSlot
	for _, sl := range e.cache {
		if sl.hasCkpt && len(sl.ckpt) < len(prompt) && (best == nil || len(sl.ckpt) > len(best.ckpt)) && inkling2HasPrefix(prompt, sl.ckpt) {
			best = sl
		}
	}
	return best
}

// prefixMeter sums what the slots hold; ok is false when a slot cannot say.
func (e *inkling2LoopEngine) prefixMeter() (held, largest int64, ok bool) {
	for _, sl := range e.cache {
		m, metered := sl.sess.(inkling2HeldBytes)
		if !metered {
			return 0, 0, false
		}
		b := m.HeldBytes()
		held, largest = held+b, max(largest, b)
	}
	return held, largest, true
}

// prefixCanGrow reports whether a miss may open another slot: always into an empty
// cache, else only under the slot limit with every slot metered and room under the
// cap for one more slot the size of the largest.
func (e *inkling2LoopEngine) prefixCanGrow() bool {
	if len(e.cache) == 0 {
		return true
	}
	if e.noReuse || len(e.cache) >= max(e.slots, 1) {
		return false
	}
	held, largest, ok := e.prefixMeter()
	return ok && held+largest <= e.capBytes
}

// prefixVictim is the slot a miss runs on when the cache cannot grow: one holding
// no checkpoint, else the least recently used.
func (e *inkling2LoopEngine) prefixVictim() *inkling2PrefixSlot {
	var lru *inkling2PrefixSlot
	for _, sl := range e.cache {
		if !sl.hasCkpt {
			return sl
		}
		if lru == nil || sl.used < lru.used {
			lru = sl
		}
	}
	return lru
}

// prefixTrim closes least recently used slots, never the current one, while more
// than one slot holds more than the cap; a session grows after it opens (lazy
// prefill arenas, checkpoint rows), so the admission projection alone is not enough.
func (e *inkling2LoopEngine) prefixTrim() {
	for len(e.cache) > 1 {
		if held, _, ok := e.prefixMeter(); ok && held <= e.capBytes {
			return
		}
		var lru *inkling2PrefixSlot
		for _, sl := range e.cache {
			if sl != e.cur && (lru == nil || sl.used < lru.used) {
				lru = sl
			}
		}
		if lru == nil {
			return
		}
		_ = e.dropSlot(lru)
	}
}

func (e *inkling2LoopEngine) dropSlot(sl *inkling2PrefixSlot) error {
	if sl == nil {
		return nil
	}
	if e.cur == sl {
		e.cur = nil
	}
	for i, c := range e.cache {
		if c == sl {
			e.cache = append(e.cache[:i], e.cache[i+1:]...)
			break
		}
	}
	return sl.sess.Close()
}

// dropCurrent closes the slot of the request in flight, so a fault or a cancelled
// prefill never leaves a checkpoint claiming a prompt it did not finish.
func (e *inkling2LoopEngine) dropCurrent() error { return e.dropSlot(e.cur) }

func (e *inkling2LoopEngine) dropAllSlots() error {
	var errs []error
	for len(e.cache) > 0 {
		errs = append(errs, e.dropSlot(e.cache[0]))
	}
	e.cur = nil
	return errors.Join(errs...)
}

func (e *inkling2LoopEngine) prefixHeldBytes() int64 {
	held, _, _ := e.prefixMeter()
	return held
}

func (e *inkling2LoopEngine) prefixSlotCount() int { return len(e.cache) }

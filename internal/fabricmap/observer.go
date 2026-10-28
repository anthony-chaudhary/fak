package fabricmap

import (
	"sync"
)

// HopObserver observes one hop receipt after Execute has validated it as a
// witnessed, complete movement. Observers are additive telemetry: they never
// see failed or invalid hops, cannot influence the execution result, and a
// panicking observer is contained so a telemetry bug can never break data
// movement. The zero-value slice (no observers registered) keeps Execute
// exactly as fast as before the seam existed.
type HopObserver func(receipt HopReceipt)

// hopObserverSlot owns one registration. Functions are not comparable in
// Go, so unregistration marks the slot's tombstone instead of comparing
// identities; the slot itself is heap-stable for the process lifetime, so a
// captured pointer stays valid no matter how the slice reallocates.
type hopObserverSlot struct {
	fn      HopObserver
	removed bool
}

var (
	observerMu   sync.RWMutex
	hopObservers []*hopObserverSlot
)

// RegisterHopObserver adds obs to the process-wide observer set and returns
// an unregister function that removes it again. Registration is
// concurrency-safe; notification takes only a read lock, so hot executions
// pay no write-lock cost while observers are present. Tombstoned slots stay
// in the slice (they are skipped on notification); the set is process-lifetime
// telemetry plumbing, not a hot data structure.
func RegisterHopObserver(obs HopObserver) (unregister func()) {
	if obs == nil {
		return func() {}
	}
	slot := &hopObserverSlot{fn: obs}
	observerMu.Lock()
	hopObservers = append(hopObservers, slot)
	observerMu.Unlock()
	return func() {
		observerMu.Lock()
		defer observerMu.Unlock()
		slot.removed = true
	}
}

// observeHops notifies every registered observer about one validated receipt.
// Each observer runs under its own recover: one misbehaving observer is
// skipped for this hop without affecting execution or the other observers.
func observeHops(receipt HopReceipt) {
	observerMu.RLock()
	slots := hopObservers
	observerMu.RUnlock()
	for _, s := range slots {
		if s.removed {
			continue
		}
		fn := s.fn
		func() {
			defer func() { _ = recover() }()
			fn(receipt)
		}()
	}
}

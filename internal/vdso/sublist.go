package vdso

// sub is one coherence-/integrity-bus observer: a monotone id (for cancellation) and the
// callback fired with the bus event. The two buses (write Mutations, Revocations) differ
// only in the event type T, so both share this shape via subList[T].
type sub[T any] struct {
	id uint64
	fn func(T)
}

// subList[T] is the subscriber set behind one bus. It carries no lock of its own — every
// method is called with the owning VDSO's v.mu already held — so the snapshot/dispatch
// split that runs callbacks outside the lock stays the caller's responsibility.
type subList[T any] struct {
	subs []*sub[T]
}

// register appends a subscriber with a pre-allocated id (caller holds v.mu and owns the
// shared subSeq allocator so mutation and revocation ids share one space).
func (sl *subList[T]) register(id uint64, fn func(T)) {
	sl.subs = append(sl.subs, &sub[T]{id: id, fn: fn})
}

// remove drops the subscriber with the given id, if present (caller holds v.mu).
func (sl *subList[T]) remove(id uint64) {
	for i, s := range sl.subs {
		if s.id == id {
			sl.subs = append(sl.subs[:i], sl.subs[i+1:]...)
			break
		}
	}
}

// snapshot copies the current subscriber set so the caller can dispatch outside v.mu
// while letting a subscriber re-enter the vDSO (subscribe/cancel) without deadlock.
func (sl *subList[T]) snapshot() []*sub[T] {
	return append([]*sub[T](nil), sl.subs...)
}

// subscribe is the shared register-and-return-cancel body behind Subscribe and
// SubscribeRevocations. A nil fn is a no-op returning an inert cancel; otherwise the
// subscriber is added under v.mu with a fresh shared id, and the returned cancel removes
// it (also under v.mu, idempotently).
func subscribe[T any](v *VDSO, sl *subList[T], fn func(T)) (cancel func()) {
	if fn == nil {
		return func() {}
	}
	v.mu.Lock()
	v.subSeq++
	id := v.subSeq
	sl.register(id, fn)
	v.mu.Unlock()
	return func() {
		v.mu.Lock()
		sl.remove(id)
		v.mu.Unlock()
	}
}

package compute

import "fmt"

// These facts are private to the provider integration boundary. There is no
// physical provider in the heap constructor. In particular, callers cannot turn
// requested flags or keys into registration facts through DirectSlabConfig.
type directSlabRegistrationFacts struct {
	allocator        *DirectSlabAllocator
	domain           string
	base             uintptr
	length           uint64
	lkey             uint32
	deviceAccessible bool // provider attests the same address is device accessible
}

// A provider owns the actual registration and its domain lifetime. facts must
// report false after retirement and must not change while borrowed. The owner
// must keep the backing and domain registered until release, including any
// required GC/page pinning. A domain name is identity, not a hardware witness.
// release
// must not reenter the allocator. No driver or hardware implementation is added
// here; test-only owners exercise this software contract, not physical mapping.
type directSlabRegistrationOwner interface {
	facts() (directSlabRegistrationFacts, bool)
	release() error
}

// bindRegistration adopts an already-created provider registration before any
// allocation is exposed. On refusal, the caller still owns its registration.
func (d *DirectSlabAllocator) bindRegistration(owner directSlabRegistrationOwner) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.owner != nil || len(d.active) != 0 || owner == nil {
		return fmt.Errorf("compute: cannot bind slab registration in current state")
	}
	facts, live := owner.facts()
	if !live || !d.registrationMatchesLocked(facts) {
		return fmt.Errorf("compute: invalid slab provider registration")
	}
	d.registration = facts
	d.owner = owner
	return nil
}

func (d *DirectSlabAllocator) registrationMatchesLocked(facts directSlabRegistrationFacts) bool {
	// Full-slab coverage keeps suballocation ownership and range checks local.
	// Subtraction rejects address wrap without constructing an overflowing end.
	return facts.allocator == d && facts.domain != "" && facts.base == d.baseAddr &&
		facts.length == uint64(d.totalCapacity) && facts.length != 0 &&
		facts.length-1 <= uint64(^uintptr(0)-facts.base)
}

func (d *DirectSlabAllocator) registrationLiveLocked() bool {
	if d.closed || d.owner == nil {
		return false
	}
	facts, live := d.owner.facts()
	return live && facts == d.registration && d.registrationMatchesLocked(facts)
}

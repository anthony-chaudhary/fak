package eviction

// Policy is the interface that all eviction engines (WTinyLFU, SIEVE) satisfy.
type Policy interface {
	Access(keyHash uint64) bool
	Admit(keyHash uint64, keyLen uint16) []uint64
	Remove(keyHash uint64)
	Size() uint64
	EvictOne() bool // evict one entry; returns false if empty
}

// NewPolicy creates an eviction Policy by name.
// Supported names: "sieve", "wtinylfu" (default), "lru" (alias for wtinylfu).
func NewPolicy(name string, capacity uint64, onEvict func(keyHash uint64, keyLen uint16)) Policy {
	switch name {
	case "sieve":
		return NewSIEVE(capacity, onEvict)
	default: // "wtinylfu" or "lru"
		return NewWTinyLFU(capacity, onEvict)
	}
}

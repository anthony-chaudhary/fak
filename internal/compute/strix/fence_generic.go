//go:build !amd64

package strix

import (
	"sync/atomic"
)

var genericBarrierTarget uint64

func sfence() {
	atomic.StoreUint64(&genericBarrierTarget, 1)
}

func mfence() {
	atomic.AddUint64(&genericBarrierTarget, 1)
}

func clflushopt(addr uintptr) {
	atomic.StoreUint64(&genericBarrierTarget, uint64(addr))
}

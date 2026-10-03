package metalgemm

import (
	"os"
	"sync"
)

// KeepAlivePowerState records whether power discovery succeeded.
type KeepAlivePowerState struct{ Known, OnBattery, LowPowerMode bool }

// KeepAliveEnabled admits the explicit policy. Unknown power fails closed in auto.
func KeepAliveEnabled(policy string, power KeepAlivePowerState) bool {
	switch policy {
	case "on":
		return true
	case "auto":
		return power.Known && !power.OnBattery && !power.LowPowerMode
	default:
		return false
	}
}

type keepAliveController struct {
	mu      sync.Mutex
	holders int
	start   func() bool
	stop    func()
}

func newKeepAliveController(start func() bool, stop func()) *keepAliveController {
	return &keepAliveController{start: start, stop: stop}
}
func (c *keepAliveController) begin() func() {
	c.mu.Lock()
	if c.holders == 0 && !c.start() {
		c.mu.Unlock()
		return func() {}
	}
	c.holders++
	c.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.holders--
			if c.holders == 0 {
				c.stop()
			}
		})
	}
}

// KeepAliveReceipt is a bounded process snapshot; failed queues never resubmit.
type KeepAliveReceipt struct {
	Holders                     int     `json:"holders"`
	SubmittedBuffers            uint64  `json:"submitted_buffers"`
	InFlight                    int     `json:"in_flight"`
	Failed                      bool    `json:"failed"`
	FailureDomain               string  `json:"failure_domain,omitempty"`
	FailureCode                 int64   `json:"failure_code,omitempty"`
	FailureMessage              string  `json:"failure_message,omitempty"`
	Iterations                  uint32  `json:"iterations"`
	MaxCompletedGPUMilliseconds float64 `json:"max_completed_gpu_ms"`
}

var keepAliveNativeStart = func() bool { return false }
var keepAliveNativeStop = func() {}
var keepAliveNativePower = func() KeepAlivePowerState { return KeepAlivePowerState{} }
var keepAliveNativeState = func() KeepAliveReceipt { return KeepAliveReceipt{} }
var generationKeepAlive = newKeepAliveController(func() bool { return keepAliveNativeStart() }, func() { keepAliveNativeStop() })

// BeginKeepAlive keeps Metal awake within a balanced generation or token scope.
// FAK_METAL_KEEPALIVE=on|auto|off selects policy; the unqualified default is off.
func BeginKeepAlive() func() {
	policy := os.Getenv("FAK_METAL_KEEPALIVE")
	if policy == "" || policy == "off" {
		return func() {}
	}
	power := KeepAlivePowerState{}
	if policy == "auto" {
		power = keepAliveNativePower()
	}
	if !KeepAliveEnabled(policy, power) {
		return func() {}
	}
	return generationKeepAlive.begin()
}

// KeepAliveState reports activity without requiring a diagnostic flag.
func KeepAliveState() KeepAliveReceipt {
	generationKeepAlive.mu.Lock()
	defer generationKeepAlive.mu.Unlock()
	r := keepAliveNativeState()
	r.Holders = generationKeepAlive.holders
	return r
}

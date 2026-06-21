// events.go — the closed core EventKind vocabulary (additive). Emitters observe
// these lifecycle transitions; KPI taps, the vDSO cache-fill, stewards, and the
// self-labeling rung harvester all key off them. Numbers are drawn from the
// reserved EventsCore / EventsLabel blocks in registry.go.
package abi

const (
	EvSubmit     EventKind = iota // a call entered the kernel
	EvDecide                      // the adjudicator chain resolved a verdict
	EvDeny                        // a call was refused (verdict carries the reason)
	EvDispatch                    // an allowed call was dispatched to the engine
	EvComplete                    // an engine produced a result (Result is set)
	EvQuarantine                  // a result was held out of context by the MMU
	EvVDSOHit                     // a call was served locally by the vDSO
	// EvRungLabel lives in the EventsLabel block (>=128): a typed LabelRow rode the
	// event (the pre-flight self-labeling signal).
	EvRungLabel EventKind = 128
)

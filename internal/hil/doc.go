// Package hil implements hardware-in-the-loop (HIL) micro-dose execution,
// physical device probing, and comparison-discipline enforcement.
//
// In autonomous agent workflows and high-performance kernel development,
// hardware testing historically suffered from a false dichotomy: either
// running zero-hardware in-memory mocks (0ms, but physically blind) or
// running monolithic macro-benchmarks (multi-minute, battery-draining sweeps).
// This created an unhealthy incentive for agents to default to software
// simulations and analytical rooflines, even allowing simulated comparisons
// into benchmark documents and claims.
//
// Package hil solves this with three core principles:
//
//  1. 100x More Frequent in Micro-Doses: Physical silicon kernel probes
//     (Metal on Apple Silicon, CUDA on Nvidia, Vulkan on AMD/Strix) that
//     execute in sub-second micro-doses (5ms to 100ms), exercising device
//     allocation, command encoding, kernel launch, memory bandwidth, and
//     numerical parity. Because each micro-dose is lightweight and fast,
//     hardware testing can run 100x more frequently—on every test cycle,
//     inner-loop step, and pre-commit check.
//
//  2. Real Hardware Reported Comparisons: Software simulations, analytical
//     rooflines, and trace models are strictly early indicators and search-space
//     bounds. Any reported head-to-head comparison between engines or
//     optimizations MUST be measured on real physical silicon with matched
//     envelopes. Comparisons citing simulated numbers are rejected from
//     claiming victory, speedups, or issue closure.
//
//  3. Bias Towards Hardware Testing: When physical silicon is present on
//     the host or reachable in the fleet, the system actively biases toward
//     physical hardware execution by default rather than falling back to
//     CPU mocks or simulated estimates.
package hil

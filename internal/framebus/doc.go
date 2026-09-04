// Package framebus provides a high-throughput, thread-safe frame distribution
// bus for event streaming, telemetry frames, and inter-component messaging
// with configurable backpressure handling and fail-closed buffer semantics.
//
// Architecture:
// The Bus coordinates frame routing from concurrent publishers to one or more
// active subscriptions. Routing is determined by topic matching (exact, prefix,
// or wildcard), optional frame type restrictions, and user-provided predicate
// filters. Each subscription maintains an independent buffered channel and
// backpressure drop policy, isolating slow consumers from impacting publisher
// latency and peer subscriber throughput.
//
// Invariant: Frame delivery maintains strict FIFO ordering per subscriber channel.
// Invariant: Subscribers never receive frames after their subscription or the bus is closed.
// Invariant: Fail-closed buffer management rejects overflows without corrupting state.
//
// Contract:
//   - Publishers and subscribers operate concurrently with zero deadlocks across bus lifecycle transitions.
//   - Frame validation strictly rejects empty or malformed frame payloads before delivery routing.
//   - Backpressure policies (DropPolicyFailClosed, DropPolicyDropNewest, DropPolicyDropOldest, DropPolicyBlock) are deterministically executed per subscriber channel.
//
// Guard: Publish and subscribe operations fail closed upon bus closure or saturated buffers under fail-closed policies.
package framebus

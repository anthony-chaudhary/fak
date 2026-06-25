// Package agenttopo declares agent communication topology over comm.Group.
//
// A Topology is a named, validated DAG over a comm.Group. It records who may
// hand a result to whom, validates every edge endpoint against the group, refuses
// cycles, and preserves declaration order for each node's in- and out-neighbor
// lists. NeighborExchange exposes that fixed adjacency, and CombineIn folds a
// node's declared in-neighbor outputs through modelroute.Combine in declaration
// order instead of accepting an unstructured []Vote.
//
// Honesty caveat: this is NOT MPI. Nodes are agents or roles, not devices; an
// edge is a permitted result handoff, not a hardware link. The neighbor
// collective folds scalar/text agent results through modelroute, not tensors, and
// it is unrelated to the real tensor/device collectives in internal/model/
// dist_collective.go or internal/compute. It inherits no HPC latency,
// throughput, or progress guarantees.
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package agenttopo

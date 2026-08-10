// Package generationctl coordinates live generation epochs, steering directives,
// and compute handoffs.
//
// A trajectory is the durable unit of intent. A generation epoch is one
// provider/worker's contiguous decoding span inside that trajectory. Steering
// closes an epoch at a checkpoint and may open another; it does not pretend an
// already-sampled token can be edited in place.
//
// Tier: composer (3) - see internal/architest.
package generationctl

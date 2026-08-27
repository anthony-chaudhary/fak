// Package zaitask is a bounded Z.AI task runner. Its GLM-5.3-Flash route owns
// the hosted provider wire (including mandatory reasoning and SSE assembly),
// but never represents provider execution as fak-native inference.
//
// Tier: integrator (4) - see internal/architest. This package may import only
// packages whose tier is <= 4; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package zaitask

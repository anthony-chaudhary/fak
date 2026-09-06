// Package residency manages multi-model weight residency, hosting prefill-warm
// *model.Model handles under a shared weight-byte budget with LRU eviction. It
// composes internal/polymodel.Pool for budget and eviction policies, binding each
// admitted descriptor to the in-kernel weights it governs and returning evicted
// weight handles when space is reclaimed.
//
// Tier: mechanism (3) — see internal/architest.
package residency

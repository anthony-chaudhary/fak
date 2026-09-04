// Package modelsrc provides model source URL resolution, transport registries,
// and random-access readers for local and remote model artifacts.
//
// Invariant: model source resolution is fail-closed and bounded.
// Schemes without registered openers, malformed URLs, or inaccessible files are
// rejected immediately with explicit errors rather than silent fallbacks.
//
// Contract: all registered openers must return a valid io.ReaderAt and an accurate
// byte size, and must fail closed on missing or irregular file descriptors.
//
// Guard: remote model downloads enforce bounded connect, TLS, and response header
// timeouts to prevent hangs on stalled peers while allowing streaming model bodies.
package modelsrc

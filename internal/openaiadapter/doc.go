// Package openaiadapter provides the migration-critical OpenAI wire subset for one authenticated app.
//
// Invariant: OpenAI adapter conversions are fail-closed and preserve request integrity.
// Guard: Requests with invalid app tokens, unmapped model aliases, or unsupported response formats
// are rejected deterministically with typed error payloads before downstream execution.
// Guard: Network listeners fail-closed by restricting address binding strictly to loopback interfaces or unix sockets.
package openaiadapter

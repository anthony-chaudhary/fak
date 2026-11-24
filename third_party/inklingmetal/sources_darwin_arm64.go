//go:build cgo

// Package inklingmetalsources makes the C++ and Metal inputs visible to the Go build cache.
package inklingmetalsources

import "embed"

// cgo only tracks native files directly in the package directory. This dependency
// invalidates its compiled operators when an included vendor file changes.
//
//go:embed runtime
var sources embed.FS

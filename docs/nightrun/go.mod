// docs/nightrun is a SEPARATE Go module boundary, not a package tree: it holds
// no .go files at all, only the overnight-run ledgers (cache-value.jsonl,
// trajctl.jsonl, fanout-reuse.jsonl) and their notes. The boundary exists so
// the Go module proxy can publish the root module at all — `modzip` skips any
// directory containing a go.mod, and the proxy refuses a module whose source
// tree exceeds 500 MiB (golang.org/x/mod/zip.MaxZipFile). This tree is ~6 MiB
// of append-only ledgers, so it grows monotonically.
//
// Nothing imports this path; the ledgers are read as data by
// internal/cachevaluereport and friends via a relative path, which is
// unaffected by the module boundary. If a Go package is ever needed here, give
// it its own module rather than deleting this file.
module github.com/anthony-chaudhary/fak/docs/nightrun

go 1.26.0

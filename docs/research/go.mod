// docs/research is a SEPARATE Go module boundary, not a package tree: it holds
// no .go files at all, only the research corpus (studies, inventories, JSONL
// ledgers). The boundary exists so the Go module proxy can publish the root
// module at all — `modzip` skips any directory containing a go.mod, and the
// proxy refuses a module whose source tree exceeds 500 MiB
// (golang.org/x/mod/zip.MaxZipFile). This corpus alone is ~101 MiB and grows
// every release, so without the boundary the root module crosses the ceiling
// and every external `go get` / `go install` fails with
// "module source tree too large".
//
// Nothing imports this path; the corpus is read as data (docs_read, ledgers).
// If a Go package is ever needed here, give it its own module rather than
// deleting this file.
module github.com/anthony-chaudhary/fak/docs/research

go 1.26.0

// docs/notes is a SEPARATE Go module boundary, not a package tree: it holds no
// .go files at all, only dated engineering notes. The boundary exists so the Go
// module proxy can publish the root module at all — `modzip` skips any
// directory containing a go.mod, and the proxy refuses a module whose source
// tree exceeds 500 MiB (golang.org/x/mod/zip.MaxZipFile). This tree is ~9 MiB
// and grows with every note, so it is part of the same growth class as
// docs/research.
//
// Nothing imports this path; the notes are read as data. If a Go package is
// ever needed here, give it its own module rather than deleting this file.
module github.com/anthony-chaudhary/fak/docs/notes

go 1.26.0

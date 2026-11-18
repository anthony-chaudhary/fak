// visuals is a SEPARATE Go module boundary, not a package tree: it holds no
// .go files at all, only media assets (GIFs, MP4s, hero images) referenced by
// the README and the docs site. The boundary exists so the Go module proxy can
// publish the root module at all — `modzip` skips any directory containing a
// go.mod, and the proxy refuses a module whose source tree exceeds 500 MiB
// (golang.org/x/mod/zip.MaxZipFile). This tree is ~85 MiB of static media, so
// without the boundary it eats a sixth of the entire module budget.
//
// Nothing imports this path; the assets are served as files. If a Go package
// is ever needed here, give it its own module rather than deleting this file.
module github.com/anthony-chaudhary/fak/visuals

go 1.26.0

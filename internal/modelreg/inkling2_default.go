package modelreg

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

// MacDefaultAlias is the first-run default model on Apple Silicon hosts whose
// unified memory fits it; DefaultAlias stays the default everywhere else.
const MacDefaultAlias = "inkling2-mini"

// macDefaultArtifactBytes is the exact size of the pinned MacDefaultAlias GGUF.
// A local copy is only reused when its size matches. Tests shrink it.
var macDefaultArtifactBytes int64 = 13_631_779_296

// Companion is a set of files a model artifact expects in a directory next to
// it, fetched from their own pinned repo revision.
type Companion struct {
	Dir      string   // directory name beside the artifact
	Repo     string   // "<owner>/<repo>"
	Revision string   // full commit sha
	Files    []string // repo-relative files placed in Dir
	GOOS     string   // only fetched on this platform
	GOARCH   string
}

// companions is keyed by the catalog target that expects them, so a registry
// override pointing an alias elsewhere brings none.
var companions = map[string][]Companion{
	Catalog[MacDefaultAlias]: {{
		Dir:      "Inkling-2-Mini-DFlash2",
		Repo:     "incoai/Inkling-2-Mini-DFlash2",
		Revision: "4f9088ea66c21f2b99ed657d1941c306b03e84d2",
		Files:    []string{"model.safetensors", "config.json"},
		GOOS:     "darwin",
		GOARCH:   "arm64",
	}},
}

// Companions returns the companion artifacts the resolved target uri expects on
// the given platform.
func Companions(uri, goos, goarch string) []Companion {
	var out []Companion
	for _, c := range companions[uri] {
		if c.GOOS == goos && c.GOARCH == goarch {
			out = append(out, c)
		}
	}
	return out
}

// FetchCompanions places every companion of uri beside artifactPath, the local
// file uri was fetched to. A companion directory that already exists is kept.
func FetchCompanions(ctx context.Context, uri, artifactPath string, progress io.Writer) error {
	return fetchCompanions(ctx, hfhub.NewClient(), Companions(uri, runtime.GOOS, runtime.GOARCH), artifactPath, progress)
}

func fetchCompanions(ctx context.Context, client *hfhub.Client, list []Companion, artifactPath string, progress io.Writer) error {
	for _, c := range list {
		dst := filepath.Join(filepath.Dir(artifactPath), c.Dir)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		cached := make([]string, len(c.Files))
		for i, file := range c.Files {
			p, err := client.Download(ctx, hfhub.Ref{Repo: c.Repo, Revision: c.Revision, File: file}, progress)
			if err != nil {
				return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
			}
			cached[i] = p
		}
		// Loaders treat the directory's existence as the whole companion, so it
		// appears only once every file is in place.
		staging := dst + ".partial"
		if err := os.RemoveAll(staging); err != nil {
			return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
		}
		if err := os.MkdirAll(staging, 0o755); err != nil {
			return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
		}
		for i, file := range c.Files {
			link := filepath.Join(staging, filepath.FromSlash(file))
			if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
				return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
			}
			if err := os.Symlink(cached[i], link); err != nil {
				if err = os.Link(cached[i], link); err != nil {
					return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
				}
			}
		}
		if err := os.Rename(staging, dst); err != nil {
			return fmt.Errorf("modelreg: companion %s: %w", c.Dir, err)
		}
	}
	return nil
}

// LocalMacDefaultArtifact finds an already-downloaded copy of the pinned
// MacDefaultAlias GGUF outside the hub cache, directly in a model search
// directory or one level below it, so a host that holds the file does not fetch
// it again. uri must be the catalog target; an overridden alias is never matched.
func LocalMacDefaultArtifact(uri string) (string, bool) {
	if uri != Catalog[MacDefaultAlias] {
		return "", false
	}
	ref, err := hfhub.ParseURI(uri)
	if err != nil {
		return "", false
	}
	base := filepath.Base(filepath.FromSlash(ref.File))
	// The repo publishes the file as ".gguf_file"; local copies usually drop that.
	names := []string{base, strings.TrimSuffix(base, "_file")}
	match := func(p string) bool {
		fi, err := os.Stat(p)
		return err == nil && fi.Mode().IsRegular() && fi.Size() == macDefaultArtifactBytes
	}
	for _, dir := range DefaultModelSearchDirs() {
		for _, name := range names {
			if p := filepath.Join(dir, name); match(p) {
				return p, true
			}
			nested, _ := filepath.Glob(filepath.Join(dir, "*", name))
			for _, p := range nested {
				if match(p) {
					return p, true
				}
			}
		}
	}
	return "", false
}

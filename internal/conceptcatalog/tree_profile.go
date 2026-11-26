package conceptcatalog

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// This is an explicit, versioned export contract, not a freshness receipt.
// Default/unknown profiles retain the full export; a qualified tree is exported
// sparsely, keeping bytes only for the corpus the scorecard reads.
const conceptTreeProfileEnv = "FAK_CONCEPT_TREE_PROFILE"
const conceptTreeProfileV1 = "isolated-v1"
const conceptTreeProfileV2 = "isolated-v2"
const conceptTreeProfileV3 = "isolated-v3"
const conceptScriptSHA = "f03ebadc6443c4f4a0533118ebb007930b0e70218f0bf3c9230f4fd77d786259"
const conceptToolsSHA = "62313b76d15080e1d912367e6a29170fc5dd71489522e446d10719331a65ddb2"
const conceptToolsV2SHA = "4312ceeba9394b80c3f4145dc4afe413d3eb4b085513b9fc7ae93b0b80caeade"
const conceptToolsV3SHA = "54a7277caa27eb516064e7d3a088dd68074b5ba45742c89efedb68c760e36e63"

// A materialized tree carries its execution contract explicitly. Sparse roots
// must never be handed to the ordinary generator without this descriptor.
type conceptGitTree struct {
	root    string
	profile *conceptTreeProfile
}
type conceptTreeProfile struct{ tree string }

func conceptDigest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func conceptFileDigest(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Each explicit profile admits only its independently qualified tools surface.
// In particular, v2 does not widen v1 or turn an unknown selector into a profile.
func conceptProfilePins(profile string) (tools string, selected bool) {
	switch profile {
	case conceptTreeProfileV1:
		return conceptToolsSHA, true
	case conceptTreeProfileV2:
		return conceptToolsV2SHA, true
	case conceptTreeProfileV3:
		return conceptToolsV3SHA, true
	default:
		return "", false
	}
}

// The sparse export is a property of the qualified tree alone: generation runs
// in-process, so no interpreter or host runtime takes part in admission.
func qualifyConceptTreeProfile(root, tree string) (*conceptTreeProfile, error) {
	toolsSHA, selected := conceptProfilePins(os.Getenv(conceptTreeProfileEnv))
	if !selected || runtime.GOOS != "linux" {
		return nil, nil
	}
	return qualifyConceptTree(root, tree, toolsSHA, conceptScriptSHA)
}

// qualifyConceptTree admits tree only when its tools surface and archived
// script match the pins. Missing/changed content is ineligible, while failure
// to inspect the tree is fatal.
func qualifyConceptTree(root, tree, toolsSHA, scriptSHA string) (*conceptTreeProfile, error) {
	listing, e := git(root, "ls-tree", "-rz", tree, "--", "tools")
	if e != nil {
		return nil, e
	}
	surface, scriptPresent, e := conceptToolsSurface(listing)
	if e != nil {
		return nil, e
	}
	if !scriptPresent || surface != toolsSHA {
		return nil, nil
	}
	script, e := git(root, "show", tree+":tools/concept_disambiguation_scorecard.py")
	if e != nil {
		return nil, e
	}
	if conceptDigest(script) != scriptSHA {
		return nil, nil
	}
	return &conceptTreeProfile{tree: tree}, nil
}

// conceptToolsSurface digests a `git ls-tree -rz -- tools` listing, leaving out
// the script and the catalog data files, whose content the pins cover separately
// or the scorecard reads as its corpus.
func conceptToolsSurface(listing []byte) (digest string, scriptPresent bool, err error) {
	var surface []byte
	for _, row := range strings.Split(string(listing), "\x00") {
		if row == "" {
			continue
		}
		_, path, ok := strings.Cut(row, "\t")
		if !ok {
			return "", false, fmt.Errorf("profile tree entry")
		}
		if path == "tools/concept_disambiguation_scorecard.py" {
			scriptPresent = true
			continue
		}
		if strings.HasPrefix(path, "tools/concept_disambiguation_scorecard.data/") && strings.Count(path, "/") == 2 && strings.HasSuffix(path, ".json") {
			continue
		}
		surface = append(surface, []byte(row)...)
		surface = append(surface, 0)
	}
	return conceptDigest(surface), scriptPresent, nil
}

func conceptProfileNeedsBytes(name string, size int64) bool {
	p := filepath.ToSlash(name)
	if p == "tools/concept_disambiguation_scorecard.py" || p == "docs/cli-reference.md" || p == GeneratedReadme || p == GeneratedIndex {
		return true
	}
	if strings.HasPrefix(p, "tools/concept_disambiguation_scorecard.data/") && strings.Count(p, "/") == 2 && strings.HasSuffix(p, ".json") {
		return true
	}
	if size > 512*1024 {
		return false
	}
	switch p {
	case "CLAIMS.md", "ARCHITECTURE.md", "README.md", "AGENTS.md", "INDEX.md", "GLOSSARY.md":
		return true
	}
	for _, part := range strings.Split(p, "/") {
		switch part {
		case ".git", ".cache", "node_modules", "testdata", "_registry", "__pycache__", ".pytest_cache", ".ruff_cache", "vendor", ".dispatch-runs", ".goal-runs":
			return false
		}
	}
	return ((strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/")) && strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go")) || (strings.HasPrefix(p, "docs/") && strings.HasSuffix(p, ".md"))
}

func extractConceptProfileTar(archive io.Reader, tmp string) error {
	tr := tar.NewReader(archive)
	for {
		h, e := tr.Next()
		if e == io.EOF {
			return nil
		}
		if e != nil {
			return e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive path %q", h.Name)
		}
		dst := filepath.Join(tmp, clean)
		if h.FileInfo().IsDir() {
			if e = os.MkdirAll(dst, 0755); e != nil {
				return e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e = os.MkdirAll(filepath.Dir(dst), 0755); e != nil {
			return e
		}
		f, e := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return e
		}
		var copyErr error
		if conceptProfileNeedsBytes(clean, h.Size) {
			_, copyErr = io.Copy(f, tr)
		} else {
			_, copyErr = io.Copy(io.Discard, tr)
			if copyErr == nil {
				copyErr = f.Truncate(h.Size)
			}
		}
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
}

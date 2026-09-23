package validate

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// pathList is a repeatable owned-path flag.
type pathList []string

func (p *pathList) String() string { return strings.Join(*p, ",") }

func (p *pathList) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("empty --mine")
	}
	*p = append(*p, v)
	return nil
}

type ciPreflightFailure struct {
	Step   string   `json:"step"`
	Detail string   `json:"detail,omitempty"`
	Files  []string `json:"files,omitempty"`
}

func short(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

func resolveRootWithin(ctx context.Context, explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func extractCommittedTip(repo, object string) (string, error) {
	return committedtree.Extract(repo, object)
}

type goPkg struct {
	ImportPath string
	Name       string
	Dir        string
	Module     *struct{ Path, Dir string }

	GoFiles           []string
	CgoFiles          []string
	CFiles            []string
	CXXFiles          []string
	MFiles            []string
	HFiles            []string
	FFiles            []string
	SFiles            []string
	SwigFiles         []string
	SwigCXXFiles      []string
	SysoFiles         []string
	TestGoFiles       []string
	XTestGoFiles      []string
	EmbedFiles        []string
	TestEmbedFiles    []string
	XTestEmbedFiles   []string
	IgnoredGoFiles    []string
	IgnoredOtherFiles []string

	Imports      []string
	TestImports  []string
	XTestImports []string
}

func parseGoList(r io.Reader) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	fileToPkg = map[string]string{}
	edges = map[string][]string{}
	var modPath, modDir string
	dec := json.NewDecoder(r)
	for {
		var p goPkg
		if decErr := dec.Decode(&p); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return nil, nil, 0, fmt.Errorf("parsing go list json: %w", decErr)
		}
		if p.Module != nil && modPath == "" {
			modPath, modDir = p.Module.Path, p.Module.Dir
		}
		total++
		if p.Dir != "" && modDir != "" {
			if rel, relErr := filepath.Rel(modDir, p.Dir); relErr == nil {
				relSlash := filepath.ToSlash(rel)
				for _, group := range [][]string{
					p.GoFiles, p.CgoFiles, p.CFiles, p.CXXFiles, p.MFiles, p.HFiles,
					p.FFiles, p.SFiles, p.SwigFiles, p.SwigCXXFiles, p.SysoFiles,
					p.TestGoFiles, p.XTestGoFiles, p.EmbedFiles, p.TestEmbedFiles,
					p.XTestEmbedFiles, p.IgnoredGoFiles, p.IgnoredOtherFiles,
				} {
					for _, f := range group {
						key := f
						if relSlash != "." {
							key = relSlash + "/" + f
						}
						fileToPkg[key] = p.ImportPath
					}
				}
			}
		}
		var deps []string
		for _, group := range [][]string{p.Imports, p.TestImports, p.XTestImports} {
			for _, imp := range group {
				if modPath != "" && strings.HasPrefix(imp, modPath) {
					deps = append(deps, imp)
				}
			}
		}
		if len(deps) > 0 {
			edges[p.ImportPath] = deps
		}
	}
	return fileToPkg, edges, total, nil
}

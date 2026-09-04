package buildwitness

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ReleaseTarget records an operating system and architecture target for release cross-compilation.
type ReleaseTarget struct {
	GOOS   string `json:"goos"`
	GOARCH string `json:"goarch"`
}

func (t ReleaseTarget) String() string {
	return t.GOOS + "/" + t.GOARCH
}

// Variant describes one supported build-tag combination in the taxonomy.
type Variant struct {
	Name                  string `json:"name"`
	Tags                  string `json:"tags"`
	CGOEnabled            string `json:"cgo_enabled"`
	DefaultInclusion      string `json:"default_inclusion"`
	TargetOSArch          string `json:"target_os_arch"`
	ToolchainRequirements string `json:"toolchain_requirements"`
	Constraint            string `json:"constraint"`
	Class                 string `json:"class"`
	Gate                  string `json:"gate"`
	Witness               string `json:"witness"`
}

// IsAdvisory returns true if the variant gate allows failure (e.g. wip_ fences).
func (v Variant) IsAdvisory() bool {
	return v.Gate == "advisory"
}

// IsPureGo returns true if CGO is disabled for this variant.
func (v Variant) IsPureGo() bool {
	return v.CGOEnabled == "0"
}

// VariantManifest contains the full build-variant taxonomy and targets.
type VariantManifest struct {
	Schema         string          `json:"schema"`
	Doc            []string        `json:"doc,omitempty"`
	ReleaseTargets []ReleaseTarget `json:"release_targets"`
	Variants       []Variant       `json:"variants"`
}

// LoadVariantManifest resolves and parses the build variants manifest.
func LoadVariantManifest(root string) (*VariantManifest, string, error) {
	candidates := []string{
		filepath.Join(root, "docs", "build-variants.json"),
		filepath.Join(root, ".github", "build-variants.json"),
	}
	for _, candidate := range candidates {
		data, err := os.ReadFile(candidate)
		if err == nil {
			var m VariantManifest
			if err := json.Unmarshal(data, &m); err != nil {
				return nil, candidate, fmt.Errorf("parse %s: %w", candidate, err)
			}
			return &m, candidate, nil
		}
	}
	return nil, "", fmt.Errorf("build-variants manifest not found in %v", candidates)
}

// PureGoVariants returns all variants where CGOEnabled is "0".
func (m *VariantManifest) PureGoVariants() []Variant {
	var out []Variant
	for _, v := range m.Variants {
		if v.IsPureGo() {
			out = append(out, v)
		}
	}
	return out
}

// FindVariant returns the variant with the given name, or nil if not found.
func (m *VariantManifest) FindVariant(name string) *Variant {
	for i := range m.Variants {
		if strings.EqualFold(m.Variants[i].Name, name) {
			return &m.Variants[i]
		}
	}
	return nil
}

// MatrixCompilePlan encapsulates the exact command and environment to compile-check one matrix entry.
type MatrixCompilePlan struct {
	Variant Variant       `json:"variant"`
	Target  ReleaseTarget `json:"target"`
	Package string        `json:"package"`
	OutPath string        `json:"out_path"`
	Command []string      `json:"command"`
	Env     []string      `json:"env"`
}

// PlanCompile creates a compile plan for a variant and target.
func PlanCompile(pkg string, variant Variant, target ReleaseTarget, outPath string) MatrixCompilePlan {
	if pkg == "" {
		pkg = TargetPackage
	}
	if outPath == "" {
		outPath = NullDevice()
	}

	args := []string{"build"}
	if strings.TrimSpace(variant.Tags) != "" {
		args = append(args, "-tags", variant.Tags)
	}
	args = append(args, "-o", outPath, pkg)

	cgo := variant.CGOEnabled
	if cgo == "" {
		cgo = "0"
	}

	env := []string{
		"GOOS=" + target.GOOS,
		"GOARCH=" + target.GOARCH,
		"CGO_ENABLED=" + cgo,
	}

	return MatrixCompilePlan{
		Variant: variant,
		Target:  target,
		Package: pkg,
		OutPath: outPath,
		Command: args,
		Env:     env,
	}
}

// RunCompileCheck executes the plan using the go toolchain.
func RunCompileCheck(goBin, root string, plan MatrixCompilePlan) (string, error) {
	if goBin == "" {
		var err error
		goBin, err = exec.LookPath("go")
		if err != nil {
			return "", fmt.Errorf("go toolchain not on PATH: %w", err)
		}
	}
	cmd := exec.Command(goBin, plan.Command...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), plan.Env...)

	out, err := cmd.CombinedOutput()
	return string(out), err
}

// GHAMatrixEntry represents an item in GitHub Actions matrix include list.
type GHAMatrixEntry struct {
	Target   string `json:"target"`
	GOOS     string `json:"goos"`
	GOARCH   string `json:"goarch"`
	Variant  string `json:"variant"`
	Tags     string `json:"tags"`
	Advisory bool   `json:"advisory"`
}

// BuildGHAMatrix produces the include list for GitHub Actions matrix consumption.
func BuildGHAMatrix(manifest *VariantManifest, pureOnly bool) map[string][]GHAMatrixEntry {
	var includes []GHAMatrixEntry
	variants := manifest.Variants
	if pureOnly {
		variants = manifest.PureGoVariants()
	}

	for _, v := range variants {
		for _, t := range manifest.ReleaseTargets {
			// If target_os_arch is specific and not "all", only include matching targets
			if v.TargetOSArch != "" && v.TargetOSArch != "all" {
				if v.TargetOSArch != t.String() {
					continue
				}
			}
			includes = append(includes, GHAMatrixEntry{
				Target:   t.String(),
				GOOS:     t.GOOS,
				GOARCH:   t.GOARCH,
				Variant:  v.Name,
				Tags:     v.Tags,
				Advisory: v.IsAdvisory(),
			})
		}
	}
	return map[string][]GHAMatrixEntry{"include": includes}
}

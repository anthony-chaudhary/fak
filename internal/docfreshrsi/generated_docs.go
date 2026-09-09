package docfreshrsi

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/markerblock"
	"github.com/anthony-chaudhary/fak/internal/milestonedoc"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/internal/supportmaturityscore"
)

// GeneratedDocStatus is the per-generated-document health and audit record.
type GeneratedDocStatus struct {
	ID        string `json:"id"`
	Path      string `json:"path"`
	Fresh     bool   `json:"fresh"`
	Refreshed bool   `json:"refreshed,omitempty"`
	Detail    string `json:"detail,omitempty"`
	Error     string `json:"error,omitempty"`
}

// GeneratedDocEntry registers a generated document with deterministic check and refresh actions.
type GeneratedDocEntry struct {
	ID          string
	Path        string
	Description string
	Check       func(root string) (fresh bool, detail string, err error)
	Refresh     func(root string) (refreshed bool, detail string, err error)
}

// DefaultGeneratedDocs returns the canonical registry of generated documentation in fak.
func DefaultGeneratedDocs() []GeneratedDocEntry {
	return []GeneratedDocEntry{
		{
			ID:          "milestone-status",
			Path:        "docs/milestones/STATUS.md",
			Description: "milestone-climb maturity ladder block rendered from covmatrix",
			Check: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "milestones", "STATUS.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						return true, "not applicable (docs/milestones/STATUS.md not present)", nil
					}
					return false, fmt.Sprintf("read error: %v", err), err
				}
				if !milestonedoc.Fresh(string(b)) {
					return false, "milestone-climb block drifted from live covmatrix grid", nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "milestones", "STATUS.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						if mkErr := os.MkdirAll(filepath.Dir(p), 0755); mkErr != nil {
							return false, "", mkErr
						}
						initial := milestonedoc.Block() + "\n"
						if wErr := os.WriteFile(p, []byte(initial), 0644); wErr != nil {
							return false, "", wErr
						}
						return true, "created docs/milestones/STATUS.md", nil
					}
					return false, "", err
				}
				spliced, err := milestonedoc.Splice(string(b))
				if err != nil {
					return false, "", err
				}
				if spliced == string(b) {
					return false, "already up to date", nil
				}
				if err := os.WriteFile(p, []byte(spliced), 0644); err != nil {
					return false, "", err
				}
				return true, "refreshed milestone-climb block", nil
			},
		},
		{
			ID:          "hardware-matrix",
			Path:        "docs/HARDWARE-MATRIX.md",
			Description: "hardware support-maturity matrix block rendered from covmatrix",
			Check: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "HARDWARE-MATRIX.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						return true, "not applicable (docs/HARDWARE-MATRIX.md not present)", nil
					}
					return false, fmt.Sprintf("read error: %v", err), err
				}
				block, ok := supportmaturityscore.ExtractMatrixBlock(string(b))
				if !ok {
					return false, "support-maturity matrix markers not found", nil
				}
				expected := supportmaturityscore.MatrixBlock()
				if block != expected {
					return false, "support-maturity matrix block drifted from live covmatrix grid", nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "HARDWARE-MATRIX.md")
				b, err := os.ReadFile(p)
				if err != nil {
					return false, "", err
				}
				spliced, err := supportmaturityscore.SpliceMatrixBlock(string(b))
				if err != nil {
					return false, "", err
				}
				if spliced == string(b) {
					return false, "already up to date", nil
				}
				if err := os.WriteFile(p, []byte(spliced), 0644); err != nil {
					return false, "", err
				}
				return true, "refreshed support-maturity matrix", nil
			},
		},
		{
			ID:          "scoreboard-debt",
			Path:        "docs/scoreboard-debt.md",
			Description: "scoreboard-debt reference table generated from scorecard baseline",
			Check: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "scoreboard-debt.md")
				if _, err := os.Stat(p); os.IsNotExist(err) {
					return true, "not applicable (docs/scoreboard-debt.md not present)", nil
				}
				fresh, detail, err := scorecardpane.CheckScoreboardDebtDoc(root)
				if err != nil {
					return false, fmt.Sprintf("check error: %v", err), err
				}
				if !fresh {
					return false, detail, nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "scoreboard-debt.md")
				if _, err := os.Stat(p); os.IsNotExist(err) {
					return false, "not applicable (docs/scoreboard-debt.md not present)", nil
				}
				written, err := scorecardpane.WriteScoreboardDebtDoc(root)
				if err != nil {
					return false, "", err
				}
				if !written {
					return false, "already up to date", nil
				}
				return true, "refreshed scoreboard-debt doc", nil
			},
		},
		{
			ID:          "workflow-branch-audit",
			Path:        "docs/ci/workflow-branch-audit.md",
			Description: "branch/tag role audit report generated from .github/workflows/*.yml",
			Check: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "ci", "workflow-branch-audit.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						return true, "not applicable (docs/ci/workflow-branch-audit.md not present)", nil
					}
					return false, fmt.Sprintf("read error: %v", err), err
				}
				const begin = "<!-- BEGIN workflow-branch-audit"
				const end = "<!-- END workflow-branch-audit -->"
				block, ok := markerblock.Extract(string(b), begin, end)
				if !ok {
					return false, "workflow-branch-audit markers not found", nil
				}
				if strings.Contains(block, "STALE DRIFT") {
					return false, "workflow-branch-audit block drifted", nil
				}
				// If fak-dev or go toolchain is available, run exact check
				fakDev, err := exec.LookPath("fak-dev")
				if err == nil {
					cmd := exec.Command(fakDev, "workflow-audit", "--check-doc")
					cmd.Dir = root
					if out, err := cmd.CombinedOutput(); err != nil {
						return false, strings.TrimSpace(string(out)), nil
					}
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				fakDev, err := exec.LookPath("fak-dev")
				if err == nil {
					cmd := exec.Command(fakDev, "workflow-audit", "--write-doc")
					cmd.Dir = root
					out, err := cmd.CombinedOutput()
					return err == nil, strings.TrimSpace(string(out)), err
				}
				goExe, err := exec.LookPath("go")
				if err == nil {
					cmd := exec.Command(goExe, "run", "./cmd/fak-dev", "workflow-audit", "--write-doc")
					cmd.Dir = root
					out, err := cmd.CombinedOutput()
					return err == nil, strings.TrimSpace(string(out)), err
				}
				return false, "fak-dev or go toolchain needed to regenerate workflow-branch-audit", nil
			},
		},
		{
			ID:          "serve-wiring",
			Path:        "docs/serve-config.md",
			Description: "serve-wiring status block in serve-config.md",
			Check: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "serve-config.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						return true, "not applicable (docs/serve-config.md not present)", nil
					}
					return false, fmt.Sprintf("read error: %v", err), err
				}
				const begin = "<!-- BEGIN serve-wiring"
				const end = "<!-- END serve-wiring -->"
				if _, ok := markerblock.Extract(string(b), begin, end); !ok {
					return false, "serve-wiring marker block not found", nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				p := filepath.Join(root, "docs", "serve-config.md")
				b, err := os.ReadFile(p)
				if err != nil {
					if os.IsNotExist(err) {
						return false, "not applicable (docs/serve-config.md not present)", nil
					}
					return false, "", err
				}
				const begin = "<!-- BEGIN serve-wiring"
				const end = "<!-- END serve-wiring -->"
				if _, ok := markerblock.Extract(string(b), begin, end); !ok {
					return false, "serve-wiring markers not found", nil
				}
				return false, "serve-wiring markers verified intact", nil
			},
		},
		{
			ID:          "structured-data",
			Path:        "docs/_includes/head-custom.html",
			Description: "JSON-LD structured data and breadcrumbs in head-custom.html, FAQ.md, and index.md",
			Check: func(root string) (bool, string, error) {
				script := filepath.Join(root, "tools", "gen_structured_data.py")
				if _, err := os.Stat(script); os.IsNotExist(err) {
					return true, "not applicable (tools/gen_structured_data.py not present)", nil
				}
				ok, out, err := runPythonTool(root, "tools/gen_structured_data.py", "--check")
				if err != nil {
					return false, fmt.Sprintf("failed running gen_structured_data.py: %v (%s)", err, out), err
				}
				if !ok {
					return false, out, nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				script := filepath.Join(root, "tools", "gen_structured_data.py")
				if _, err := os.Stat(script); os.IsNotExist(err) {
					return false, "not applicable (tools/gen_structured_data.py not present)", nil
				}
				ok, out, err := runPythonTool(root, "tools/gen_structured_data.py")
				if err != nil {
					return false, fmt.Sprintf("failed running gen_structured_data.py: %v (%s)", err, out), err
				}
				return ok, out, nil
			},
		},
		{
			ID:          "llms-full",
			Path:        "llms-full.txt",
			Description: "llms-full.txt inlined documentation corpus generated from llms.txt",
			Check: func(root string) (bool, string, error) {
				script := filepath.Join(root, "tools", "gen_llms_full.py")
				if _, err := os.Stat(script); os.IsNotExist(err) {
					return true, "not applicable (tools/gen_llms_full.py not present)", nil
				}
				ok, out, err := runPythonTool(root, "tools/gen_llms_full.py", "--check")
				if err != nil {
					return false, fmt.Sprintf("failed running gen_llms_full.py: %v (%s)", err, out), err
				}
				if !ok {
					return false, out, nil
				}
				return true, "fresh", nil
			},
			Refresh: func(root string) (bool, string, error) {
				script := filepath.Join(root, "tools", "gen_llms_full.py")
				if _, err := os.Stat(script); os.IsNotExist(err) {
					return false, "not applicable (tools/gen_llms_full.py not present)", nil
				}
				ok, out, err := runPythonTool(root, "tools/gen_llms_full.py")
				if err != nil {
					return false, fmt.Sprintf("failed running gen_llms_full.py: %v (%s)", err, out), err
				}
				return ok, out, nil
			},
		},
	}
}

func runPythonTool(root, scriptRel string, args ...string) (bool, string, error) {
	scriptPath := filepath.Join(root, filepath.FromSlash(scriptRel))
	if _, err := os.Stat(scriptPath); err != nil {
		return false, fmt.Sprintf("%s does not exist", scriptRel), err
	}

	py, err := exec.LookPath("python")
	if err != nil {
		py, err = exec.LookPath("python3")
	}
	if err != nil {
		return false, "python/python3 not found on PATH", err
	}

	cmdArgs := append([]string{scriptPath}, args...)
	cmd := exec.Command(py, cmdArgs...)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	outputStr := strings.TrimSpace(out.String())
	return runErr == nil, outputStr, nil
}

// AuditGeneratedDocs audits every registered generated document in the repository.
func AuditGeneratedDocs(root string) ([]GeneratedDocStatus, bool) {
	docs := DefaultGeneratedDocs()
	statuses := make([]GeneratedDocStatus, 0, len(docs))
	allFresh := true
	for _, d := range docs {
		fresh, detail, err := d.Check(root)
		status := GeneratedDocStatus{
			ID:     d.ID,
			Path:   d.Path,
			Fresh:  fresh,
			Detail: detail,
		}
		if err != nil {
			status.Error = err.Error()
		}
		if !fresh {
			allFresh = false
		}
		statuses = append(statuses, status)
	}
	return statuses, allFresh
}

// RefreshGeneratedDocs refreshes all stale generated documents across the repository.
func RefreshGeneratedDocs(root string) ([]GeneratedDocStatus, error) {
	docs := DefaultGeneratedDocs()
	statuses := make([]GeneratedDocStatus, 0, len(docs))
	for _, d := range docs {
		refreshed, detail, err := d.Refresh(root)
		status := GeneratedDocStatus{
			ID:        d.ID,
			Path:      d.Path,
			Refreshed: refreshed,
			Detail:    detail,
		}
		if err != nil {
			status.Error = err.Error()
		}
		// Re-check freshness
		fresh, checkDetail, _ := d.Check(root)
		status.Fresh = fresh
		if !fresh && status.Detail == "" {
			status.Detail = checkDetail
		}
		statuses = append(statuses, status)
	}
	return statuses, nil
}

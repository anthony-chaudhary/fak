package devcmd

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// overrideCompanionRoots is used in unit tests to simulate missing or custom companion roots.
var overrideCompanionRoots func() (string, string)

// ResolveCompanionRoots discovers the fak root and the companion fak-private root.
func ResolveCompanionRoots() (string, string) {
	if overrideCompanionRoots != nil {
		return overrideCompanionRoots()
	}
	fakRoot := findFakRepoRoot()
	privRoot := ResolveCompanionRoot(fakRoot)
	return fakRoot, privRoot
}

// ResolveCompanionRoot resolves the companion fak-private repository root.
// It checks in deterministic order:
// 1. FAK_PRIVATE_ROOT environment variable
// 2. go.work inspection in fakRoot, parent, or cwd
// 3. Sibling ../fak-private
// 4. Parent directory tree discovery
func ResolveCompanionRoot(fakRoot string) string {
	// 1. Environment variable FAK_PRIVATE_ROOT
	if env := strings.TrimSpace(os.Getenv("FAK_PRIVATE_ROOT")); env != "" {
		if isDir(env) {
			return filepath.Clean(env)
		}
	}

	// 2. go.work inspection
	if priv := findPrivateInGoWork(fakRoot); priv != "" {
		return priv
	}

	// 3. Sibling ../fak-private relative to fakRoot
	if fakRoot != "" {
		candidate := filepath.Join(fakRoot, "..", "fak-private")
		if isPrivateRepoRoot(candidate) {
			return filepath.Clean(candidate)
		}
	}

	// Also check sibling relative to cwd
	if isPrivateRepoRoot(filepath.Join("..", "fak-private")) {
		if abs, err := filepath.Abs(filepath.Join("..", "fak-private")); err == nil {
			return abs
		}
		return filepath.Clean(filepath.Join("..", "fak-private"))
	}

	// 4. Parent directory discovery: walk up from fakRoot (or cwd)
	startDir := fakRoot
	if startDir == "" {
		startDir, _ = os.Getwd()
	}
	for d := startDir; d != ""; {
		candidate := filepath.Join(d, "fak-private")
		if isPrivateRepoRoot(candidate) {
			return filepath.Clean(candidate)
		}
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}

	return ""
}

func findPrivateInGoWork(fakRoot string) string {
	var candidates []string
	if fakRoot != "" {
		candidates = append(candidates, filepath.Join(fakRoot, "go.work"))
		candidates = append(candidates, filepath.Join(fakRoot, "..", "go.work"))
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(cwd, "go.work"))
		candidates = append(candidates, filepath.Join(cwd, "..", "go.work"))
	}

	for _, workPath := range candidates {
		b, err := os.ReadFile(workPath)
		if err != nil {
			continue
		}
		workDir := filepath.Dir(workPath)
		inUseBlock := false
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if idx := strings.Index(line, "//"); idx >= 0 {
				line = strings.TrimSpace(line[:idx])
			}
			if line == "" {
				continue
			}
			if inUseBlock {
				if strings.HasPrefix(line, ")") {
					inUseBlock = false
					continue
				}
				entry := strings.Trim(line, `"'`+" \t\r")
				cand := filepath.Clean(filepath.Join(workDir, entry))
				if isPrivateRepoRoot(cand) {
					return cand
				}
			} else if strings.HasPrefix(line, "use (") || line == "use (" {
				inUseBlock = true
			} else if strings.HasPrefix(line, "use ") {
				rest := strings.TrimSpace(strings.TrimPrefix(line, "use "))
				entry := strings.Trim(rest, `"'`+" \t\r")
				cand := filepath.Clean(filepath.Join(workDir, entry))
				if isPrivateRepoRoot(cand) {
					return cand
				}
			}
		}
	}
	return ""
}

func isPrivateRepoRoot(dir string) bool {
	if dir == "" {
		return false
	}
	fi, err := os.Stat(dir)
	if err != nil || !fi.IsDir() {
		return false
	}
	if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
		if strings.Contains(string(b), "fak-private") {
			return true
		}
	}
	if fi, err := os.Stat(filepath.Join(dir, "cmd", "fak-boundary")); err == nil && fi.IsDir() {
		return true
	}
	if fi, err := os.Stat(filepath.Join(dir, "platform")); err == nil && fi.IsDir() {
		return true
	}
	return false
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func isBoundaryCheckInvocation(argv []string) bool {
	if len(argv) == 0 {
		return false
	}
	verb := argv[0]
	if verb == "check" {
		return true
	}
	if strings.HasPrefix(verb, "-") && verb != "-h" && verb != "--help" {
		return true
	}
	return false
}

func hasFlag(args []string, flags ...string) bool {
	for _, a := range args {
		for _, f := range flags {
			if a == f || strings.HasPrefix(a, f+"=") {
				return true
			}
		}
	}
	return false
}

func findPython() (string, error) {
	if p, err := exec.LookPath("python"); err == nil {
		return p, nil
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("neither python nor python3 found in PATH")
}

// RunCompanionGate proxies to cmd/fak-boundary in the companion root.
// Invokes: go -C <privRoot> run ./cmd/fak-boundary <argv> --fak-dir <fakRoot> --private-dir <privRoot>
// Passes through stdout, stderr, and exit code.
// If companion root is not found, prints an actionable diagnostic:
// "fak-dev gate: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling."
// and returns exit code 1 (or 0 with notice if --help).
func RunCompanionGate(stdout, stderr io.Writer, argv []string) int {
	fakRoot, privRoot := ResolveCompanionRoots()
	isHelp := false
	for _, a := range argv {
		if a == "--help" || a == "-h" || a == "help" {
			isHelp = true
			break
		}
	}

	if privRoot == "" {
		if isHelp {
			fmt.Fprintln(stdout, "fak-dev gate: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
			writeGateHelp(stdout)
			return 0
		}
		fmt.Fprintln(stderr, "fak-dev gate: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
		return 1
	}

	subArgs := []string{"-C", privRoot, "run", "./cmd/fak-boundary"}

	if isBoundaryCheckInvocation(argv) {
		if argv[0] == "check" {
			subArgs = append(subArgs, "check")
			if !hasFlag(argv, "--fak-dir", "-fak-dir") && fakRoot != "" {
				subArgs = append(subArgs, "--fak-dir", fakRoot)
			}
			if !hasFlag(argv, "--private-dir", "-private-dir") {
				subArgs = append(subArgs, "--private-dir", privRoot)
			}
			subArgs = append(subArgs, argv[1:]...)
		} else {
			if !hasFlag(argv, "--fak-dir", "-fak-dir") && fakRoot != "" {
				subArgs = append(subArgs, "--fak-dir", fakRoot)
			}
			if !hasFlag(argv, "--private-dir", "-private-dir") {
				subArgs = append(subArgs, "--private-dir", privRoot)
			}
			subArgs = append(subArgs, argv...)
		}
	} else {
		subArgs = append(subArgs, argv...)
	}

	cmd := exec.Command("go", subArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(stderr, "fak-dev gate: failed to execute companion boundary: %v\n", err)
		return 1
	}
	return 0
}

func writeGateHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev gate <subcommand> [flags] [args]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "5-Gate Import Encapsulation and Placement Pre-Commit Gate (proxied to companion cmd/fak-boundary).")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  check               Run automated 5-gate import encapsulation and placement verification")
	fmt.Fprintln(w, "  query <intent...>   Search the 5-Gate IP Taxonomy by intent or concept")
	fmt.Fprintln(w, "  explain <path>      Explain repository placement and gate classification for a file or package path")
	fmt.Fprintln(w, "  adjudicate-serving  Adjudicate model serving placement and generate split contracts")
	fmt.Fprintln(w, "  version             Display version and build identity")
}

// RunCompanionProvenance proxies to cmd/fak-sync provenance in the companion root.
// Invokes: go -C <privRoot> run ./cmd/fak-sync provenance <argv> --dir <fakRoot>
// Passes through stdout, stderr, and exit code.
// Handles missing companion gracefully.
func RunCompanionProvenance(stdout, stderr io.Writer, argv []string) int {
	fakRoot, privRoot := ResolveCompanionRoots()
	isHelp := false
	for _, a := range argv {
		if a == "--help" || a == "-h" || a == "help" {
			isHelp = true
			break
		}
	}

	if privRoot == "" {
		if isHelp {
			fmt.Fprintln(stdout, "fak-dev provenance: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
			writeProvenanceHelp(stdout)
			return 0
		}
		fmt.Fprintln(stderr, "fak-dev provenance: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
		return 1
	}

	subArgs := []string{"-C", privRoot, "run", "./cmd/fak-sync", "provenance"}

	if len(argv) > 0 && argv[0] == "audit" {
		subArgs = append(subArgs, "audit")
		if !hasFlag(argv, "--dir", "-dir") && fakRoot != "" {
			subArgs = append(subArgs, "--dir", fakRoot)
		}
		subArgs = append(subArgs, argv[1:]...)
	} else {
		subArgs = append(subArgs, argv...)
	}

	cmd := exec.Command("go", subArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(stderr, "fak-dev provenance: failed to execute companion provenance: %v\n", err)
		return 1
	}
	return 0
}

func writeProvenanceHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev provenance <subcommand> [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Cryptographic context separation, dual-context audit, and provenance minting (proxied to companion cmd/fak-sync provenance).")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  audit         Audit a Git commit or range for dual-context separation")
	fmt.Fprintln(w, "  mint          Generate cryptographic provenance receipts and commit trailers")
	fmt.Fprintln(w, "  status        Display rule invariants, separation criteria, and closed vocabulary")
}

// RunCompanionAuditLeak proxies to python <privRoot>/tools/scrub_public_copy.py.
// Invokes: python <privRoot>/tools/scrub_public_copy.py --audit-staged --root <fakRoot> (or full working tree if --all).
// Passes through stdout, stderr, and exit code.
func RunCompanionAuditLeak(stdout, stderr io.Writer, argv []string) int {
	fakRoot, privRoot := ResolveCompanionRoots()
	isHelp := false
	for _, a := range argv {
		if a == "--help" || a == "-h" || a == "help" {
			isHelp = true
			break
		}
	}

	if privRoot == "" {
		if isHelp {
			fmt.Fprintln(stdout, "fak-dev audit-leak: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
			writeAuditLeakHelp(stdout)
			return 0
		}
		fmt.Fprintln(stderr, "fak-dev audit-leak: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
		return 1
	}

	if isHelp {
		writeAuditLeakHelp(stdout)
		return 0
	}

	pythonBin, err := findPython()
	if err != nil {
		fmt.Fprintln(stderr, "fak-dev audit-leak: python or python3 executable not found in PATH")
		return 1
	}

	scriptPath := filepath.Join(privRoot, "tools", "scrub_public_copy.py")
	if _, err := os.Stat(scriptPath); err != nil {
		if fakRoot != "" {
			candidate := filepath.Join(fakRoot, "tools", "scrub_public_copy.py")
			if _, err := os.Stat(candidate); err == nil {
				scriptPath = candidate
			}
		}
	}

	isAll := false
	var filteredArgs []string
	for _, a := range argv {
		if a == "--all" || a == "-all" {
			isAll = true
			continue
		}
		if a == "--staged" || a == "-staged" {
			continue
		}
		filteredArgs = append(filteredArgs, a)
	}

	var pyArgs []string
	pyArgs = append(pyArgs, scriptPath)

	if isAll {
		targetScript := scriptPath
		if fakRoot != "" {
			fakScript := filepath.Join(fakRoot, "tools", "scrub_public_copy.py")
			if _, err := os.Stat(fakScript); err == nil {
				targetScript = fakScript
			}
		}
		pyArgs[0] = targetScript
		pyArgs = append(pyArgs, "--audit-tree")
	} else {
		pyArgs = append(pyArgs, "--audit-staged")
	}

	if !hasFlag(filteredArgs, "--root", "-root") && fakRoot != "" {
		pyArgs = append(pyArgs, "--root", fakRoot)
	}
	pyArgs = append(pyArgs, filteredArgs...)

	cmd := exec.Command(pythonBin, pyArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(stderr, "fak-dev audit-leak: failed to execute leak audit: %v\n", err)
		return 1
	}
	return 0
}

func writeAuditLeakHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev audit-leak [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Scan repository content for secret leaks and private cluster tokens (proxied to companion tools/scrub_public_copy.py).")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fmt.Fprintln(w, "  --staged      Scan staged additions (default)")
	fmt.Fprintln(w, "  --all         Scan full working tree")
}

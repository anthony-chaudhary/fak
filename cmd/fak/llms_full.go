package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type llmsFullResult struct {
	OK           bool     `json:"ok"`
	Check        bool     `json:"check"`
	SourceCommit string   `json:"source_commit"`
	Included     []string `json:"included_overlays"`
	Excluded     []string `json:"excluded_dirty_paths"`
	DriftCause   string   `json:"drift_cause,omitempty"`
	Output       string   `json:"output,omitempty"`
	Detail       string   `json:"detail,omitempty"`
}

func cmdLLMSFull(args []string) {
	code := runLLMSFull(args, os.Stdout, os.Stderr)
	if code != 0 {
		os.Exit(code)
	}
}

func runLLMSFull(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("llms-full", flag.ContinueOnError)
	fs.SetOutput(stderr)
	check := fs.Bool("check", false, "check isolated llms-full.txt drift without writing")
	ref := fs.String("ref", "HEAD", "committed source ref")
	jsonOut := fs.Bool("json", false, "emit machine-readable provenance")
	var mines pathList
	fs.Var(&mines, "mine", "explicit caller-owned path to overlay (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak llms-full: unexpected arguments")
		return 2
	}
	root := strings.TrimSpace(os.Getenv("FAK_LLMS_FULL_ROOT"))
	if root == "" {
		root = discoverRepoRoot()
	}
	if root == "" {
		fmt.Fprintln(stderr, "fak llms-full: not in a git repository")
		return 2
	}
	normalized, err := normalizeMinePaths(root, mines)
	if err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 2
	}
	sha, err := gitOutput(root, "rev-parse", "--verify", *ref+"^{commit}")
	if err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 2
	}
	sha = strings.TrimSpace(sha)
	dir, err := extractCommittedTip(root, sha)
	if err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 1
	}
	defer os.RemoveAll(dir)

	baseFresh, _ := runLLMSGenerator(dir, true)
	if err := overlayMinePaths(root, dir, normalized); err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 1
	}
	excluded, err := dirtyPathsExcluding(root, normalized)
	if err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 1
	}
	res := llmsFullResult{Check: *check, SourceCommit: sha, Included: normalized, Excluded: excluded}
	if *check {
		ok, detail := runLLMSGenerator(dir, true)
		res.OK, res.Detail = ok, detail
		if !ok {
			switch {
			case len(normalized) > 0 && baseFresh:
				res.DriftCause = "owned_inputs"
			case len(normalized) > 0:
				res.DriftCause = "committed_and_owned_inputs"
			default:
				res.DriftCause = "committed_tip"
			}
		}
		writeLLMSFullResult(stdout, res, *jsonOut)
		if !ok {
			return 1
		}
		return 0
	}
	ok, detail := runLLMSGenerator(dir, false)
	if !ok {
		res.Detail = detail
		writeLLMSFullResult(stderr, res, *jsonOut)
		return 1
	}
	src := filepath.Join(dir, "llms-full.txt")
	dst := filepath.Join(root, "llms-full.txt")
	b, err := os.ReadFile(src)
	if err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 1
	}
	if current, readErr := os.ReadFile(dst); readErr == nil && bytes.Contains(current, []byte("\r\n")) {
		b = bytes.ReplaceAll(b, []byte("\n"), []byte("\r\n"))
	}
	if err = os.WriteFile(dst, b, 0644); err != nil {
		fmt.Fprintln(stderr, "fak llms-full:", err)
		return 1
	}
	res.OK = true
	res.Output = "llms-full.txt"
	writeLLMSFullResult(stdout, res, *jsonOut)
	return 0
}

func runLLMSGenerator(root string, check bool) (bool, string) {
	python := "python3"
	if filepath.Separator == '\\' {
		python = "python"
	}
	args := []string{filepath.Join(root, "tools", "gen_llms_full.py"), "--root", root}
	if check {
		args = append(args, "--check")
	}
	cmd := exec.Command(python, args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	detail := strings.TrimSpace(string(out))
	return err == nil, detail
}

func dirtyPathsExcluding(root string, owned []string) ([]string, error) {
	set := map[string]bool{}
	for _, p := range owned {
		set[p] = true
	}
	found := map[string]bool{}
	for _, args := range [][]string{{"diff", "--name-only"}, {"diff", "--cached", "--name-only"}, {"ls-files", "--others", "--exclude-standard"}} {
		out, err := gitOutput(root, args...)
		if err != nil {
			return nil, err
		}
		for _, p := range strings.Split(out, "\n") {
			p = filepath.ToSlash(strings.TrimSpace(p))
			if p != "" && !set[p] {
				found[p] = true
			}
		}
	}
	out := make([]string, 0, len(found))
	for p := range found {
		out = append(out, p)
	}
	sort.Strings(out)
	return out, nil
}

func gitOutput(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, er bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &er
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(er.String()))
	}
	return out.String(), nil
}
func writeLLMSFullResult(w io.Writer, r llmsFullResult, j bool) {
	if j {
		_ = json.NewEncoder(w).Encode(r)
		return
	}
	verdict := "OK"
	if !r.OK {
		verdict = "OUT OF DATE"
	}
	fmt.Fprintf(w, "llms-full %s — source %s, %d overlay(s), %d excluded dirty path(s)\n", verdict, r.SourceCommit[:12], len(r.Included), len(r.Excluded))
	if r.DriftCause != "" {
		fmt.Fprintln(w, "  drift cause:", r.DriftCause)
	}
	if r.Detail != "" {
		fmt.Fprintln(w, "  detail:", r.Detail)
	}
	if r.Output != "" {
		fmt.Fprintln(w, "  wrote:", r.Output)
	}
}

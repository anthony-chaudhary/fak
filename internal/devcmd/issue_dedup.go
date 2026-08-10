package devcmd

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuededup"
)

// runIssueDedup is the retrospective backlog duplicate census — the read-only
// complement of the write-time near-duplicate gate. It builds a body-aware index
// over the open backlog (title + title+body simhash), clusters near-twins with
// per-pair evidence, and emits a ranked merge/close proposal report for the issue
// gardener. It never writes to GitHub: the confirm-before-closing-as-dup
// discipline stands, so the census only proposes.
//
// By default it reads the live backlog via `gh issue list --state open --json
// number,title,body,labels`; --from-issues PATH|- reads a cached gh array instead
// (hermetic and offline-safe). Exit 0 on a valid report, exit 2 on bad flags or
// input, exit 1 on a gh/encode failure.
func runIssueDedup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue dedup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the machine-readable census report instead of markdown")
	fromIssues := fs.String("from-issues", "", "read a cached `gh issue list --json number,title,body,labels` array from a file or '-' (stdin) instead of a live gh fetch")
	limit := fs.Int("limit", 500, "cap the live gh issue list fetch at N open issues")
	threshold := fs.Float64("threshold", 0, "pairwise cosine floor to link two issues (0 = census default)")
	topK := fs.Int("topk", 0, "neighbor fan-out per issue before clustering (0 = census default)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak-dev issue dedup: unexpected argument(s); see fak-dev issue dedup --help")
		return 2
	}

	var raw []byte
	if *fromIssues != "" {
		b, err := readIssueDedupInput(*fromIssues)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue dedup: %v\n", err)
			return 2
		}
		raw = b
	} else {
		b, err := fetchIssueDedupBacklog(*limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue dedup: %v\n", err)
			return 1
		}
		raw = b
	}

	issues, err := issuededup.ParseBacklog(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue dedup: %v\n", err)
		return 2
	}

	rep := issuededup.Census(issues, *threshold, *topK)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak-dev issue dedup")
	}
	fmt.Fprint(stdout, issuededup.RenderCensus(rep))
	return 0
}

// readIssueDedupInput reads a cached gh issue array from a file path or '-'
// (stdin). ParseBacklog does the BOM tolerance and JSON validation.
func readIssueDedupInput(source string) ([]byte, error) {
	if source == "-" {
		return io.ReadAll(os.Stdin)
	}
	abs, err := filepath.Abs(source)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(abs)
}

// fetchIssueDedupBacklog shells to gh for the open backlog on the census axes
// (title + body + labels), read-only. The whole path stays offline-safe: cached
// reads in via --from-issues, ranked proposals out, no gh write anywhere.
func fetchIssueDedupBacklog(limit int) ([]byte, error) {
	if limit <= 0 {
		limit = 500
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "open",
		"--limit", fmt.Sprint(limit), "--json", "number,title,body,labels")
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("gh issue list failed: %w (%s)", err, strings.TrimSpace(string(b)))
	}
	return b, nil
}

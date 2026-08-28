package devcmd

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/studylink"
)

type studyLinkOperations struct {
	build        func(studylink.BuildOptions) (studylink.Ledger, studylink.Summary, error)
	writeLedger  func(string, studylink.Ledger) error
	writeSummary func(string, studylink.Summary) error
	validate     func(studylink.ValidateOptions) error
}

var defaultStudyLinkOperations = studyLinkOperations{
	build:        studylink.Build,
	writeLedger:  studylink.WriteLedger,
	writeSummary: studylink.WriteSummary,
	validate:     studylink.ValidateFiles,
}

func RunStudyLink(stdout, stderr io.Writer, args []string) int {
	return runStudyLinkWithOperations(stdout, stderr, args, defaultStudyLinkOperations)
}

func runStudyLinkWithOperations(stdout, stderr io.Writer, args []string, ops studyLinkOperations) int {
	if len(args) == 0 {
		studyLinkUsage(stderr)
		return 2
	}
	switch args[0] {
	case "build":
		return runStudyLinkBuild(stdout, stderr, args[1:], ops)
	case "validate":
		return runStudyLinkValidate(stdout, stderr, args[1:], ops)
	default:
		studyLinkUsage(stderr)
		return 2
	}
}

func runStudyLinkBuild(stdout, stderr io.Writer, args []string, ops studyLinkOperations) int {
	fs := flag.NewFlagSet("study-link build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	indexPath := fs.String("index", "", "compact vLLM cluster index path (required)")
	forgePath := fs.String("forge", "", "complete FAK forge corpus path (required)")
	adjacencyPath := fs.String("adjacency", "", "related-system adjacency manifest path (required)")
	repoPath := fs.String("repo", "", "FAK repository root (required)")
	outPath := fs.String("out", "", "output ledger JSON path (required)")
	summaryPath := fs.String("summary", "", "output human summary path (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || anyStudyLinkPathMissing(*indexPath, *forgePath, *adjacencyPath, *repoPath, *outPath, *summaryPath) {
		fmt.Fprintln(stderr, "usage: fak study-link build --index PATH --forge PATH --adjacency PATH --repo PATH --out PATH --summary PATH")
		return 2
	}

	ledger, summary, err := ops.build(studylink.BuildOptions{
		IndexPath:     *indexPath,
		ForgePath:     *forgePath,
		AdjacencyPath: *adjacencyPath,
		RepoRoot:      *repoPath,
	})
	if err != nil {
		fmt.Fprintf(stderr, "study-link: build: %v\n", err)
		return 1
	}
	if err := ops.writeLedger(*outPath, ledger); err != nil {
		fmt.Fprintf(stderr, "study-link: write ledger: %v\n", err)
		return 1
	}
	if err := ops.writeSummary(*summaryPath, summary); err != nil {
		fmt.Fprintf(stderr, "study-link: write summary: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "built study-link ledger %s and summary %s\n", *outPath, *summaryPath)
	return 0
}

func runStudyLinkValidate(stdout, stderr io.Writer, args []string, ops studyLinkOperations) int {
	fs := flag.NewFlagSet("study-link validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledgerPath := fs.String("ledger", "", "study-link ledger path (required)")
	indexPath := fs.String("index", "", "compact vLLM cluster index path (required)")
	forgePath := fs.String("forge", "", "complete FAK forge corpus path (required)")
	adjacencyPath := fs.String("adjacency", "", "related-system adjacency manifest path (required)")
	repoPath := fs.String("repo", "", "FAK repository root (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || anyStudyLinkPathMissing(*ledgerPath, *indexPath, *forgePath, *adjacencyPath, *repoPath) {
		fmt.Fprintln(stderr, "usage: fak study-link validate --ledger PATH --index PATH --forge PATH --adjacency PATH --repo PATH")
		return 2
	}

	if err := ops.validate(studylink.ValidateOptions{
		LedgerPath:    *ledgerPath,
		IndexPath:     *indexPath,
		ForgePath:     *forgePath,
		AdjacencyPath: *adjacencyPath,
		RepoRoot:      *repoPath,
	}); err != nil {
		fmt.Fprintf(stderr, "study-link: validate: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid study-link ledger %s\n", *ledgerPath)
	return 0
}

func anyStudyLinkPathMissing(paths ...string) bool {
	for _, path := range paths {
		if path == "" {
			return true
		}
	}
	return false
}

func studyLinkUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak study-link <build|validate> [flags]")
	fmt.Fprintln(w, "  build --index PATH --forge PATH --adjacency PATH --repo PATH --out PATH --summary PATH")
	fmt.Fprintln(w, "  validate --ledger PATH --index PATH --forge PATH --adjacency PATH --repo PATH")
}

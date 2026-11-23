package devcmd

// `fak-dev feature batch` answers several self-queries from one process. Each
// segment is the argv of a command that also runs on its own (`feature query ...`
// or `capabilities ...`), and its captured output is byte-for-byte what that
// command prints, so a caller that used to start one process per query can start
// one and split the results. Segments naming the same --root share a single
// catalog load.
//
//	fak-dev feature batch --json \
//	    query "landing bottleneck" --root A --json \
//	    --next query "landing bottleneck" --root B --json \
//	    --next capabilities --root B "ship a change"

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

const (
	featureBatchSchema    = "fak.feature-batch.v1"
	featureBatchSeparator = "--next"
)

type featureBatchResult struct {
	// Argv is the equivalent standalone fak-dev invocation, for a rerun.
	Argv     []string `json:"argv"`
	ExitCode int      `json:"exit_code"`
	Stdout   string   `json:"stdout"`
	Stderr   string   `json:"stderr,omitempty"`
}

type featureBatchOutput struct {
	Schema  string               `json:"schema"`
	Results []featureBatchResult `json:"results"`
}

func runFeatureBatch(stdout, stderr io.Writer, argv []string) int {
	asJSON := false
	for len(argv) > 0 {
		switch argv[0] {
		case "--json", "-json":
			asJSON = true
			argv = argv[1:]
			continue
		case "-h", "--help", "help":
			writeFeatureUsage(stdout)
			return 0
		}
		break
	}
	segments, err := splitFeatureBatch(argv)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev feature batch: %v\n", err)
		writeFeatureUsage(stderr)
		return 2
	}

	load := memoizedSelfQueryCatalogLoader(loadSelfQueryCatalog)
	out := featureBatchOutput{Schema: featureBatchSchema}
	exit := 0
	for _, segment := range segments {
		var segOut, segErr bytes.Buffer
		result := featureBatchResult{}
		switch segment[0] {
		case "query":
			result.Argv = append([]string{"feature"}, segment...)
			result.ExitCode = runFeatureQuery(&segOut, &segErr, segment[1:], load)
		case "capabilities":
			result.Argv = append([]string(nil), segment...)
			result.ExitCode = runCapabilities(&segOut, &segErr, segment[1:], load)
		}
		result.Stdout, result.Stderr = segOut.String(), segErr.String()
		if exit == 0 {
			exit = result.ExitCode
		}
		out.Results = append(out.Results, result)
	}

	if asJSON {
		if err := writeIndentedJSONNoEscape(stdout, out); err != nil {
			fmt.Fprintf(stderr, "fak-dev feature batch: encode json: %v\n", err)
			return 1
		}
		return exit
	}
	for _, result := range out.Results {
		header := fmt.Sprintf("=== fak-dev %s (exit %d) ===", quoteBatchArgv(result.Argv), result.ExitCode)
		fmt.Fprintln(stdout, header)
		io.WriteString(stdout, result.Stdout)
		if result.Stdout != "" && !strings.HasSuffix(result.Stdout, "\n") {
			fmt.Fprintln(stdout)
		}
		if result.Stderr != "" {
			fmt.Fprintln(stderr, header)
			io.WriteString(stderr, result.Stderr)
		}
	}
	return exit
}

// splitFeatureBatch cuts argv into segments at each separator and checks that every
// segment names a command the batch can run.
func splitFeatureBatch(argv []string) ([][]string, error) {
	var segments [][]string
	var current []string
	flush := func() error {
		if len(current) == 0 {
			return fmt.Errorf("empty segment (each segment is `query ...` or `capabilities ...`, separated by %s)", featureBatchSeparator)
		}
		if current[0] != "query" && current[0] != "capabilities" {
			return fmt.Errorf("segment starts with %q (want query or capabilities)", current[0])
		}
		segments = append(segments, current)
		current = nil
		return nil
	}
	for _, arg := range argv {
		if arg == featureBatchSeparator {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		current = append(current, arg)
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return segments, nil
}

// memoizedSelfQueryCatalogLoader loads each root once and replays the result,
// errors included, so a failing root fails every segment that names it the same way.
// The key is the root exactly as spelled: a catalog echoes its root string into every
// response, so two spellings of one directory must not share a catalog.
func memoizedSelfQueryCatalogLoader(load selfQueryCatalogLoader) selfQueryCatalogLoader {
	type loaded struct {
		cat *selfquery.Catalog
		err error
	}
	seen := map[string]loaded{}
	return func(root string) (*selfquery.Catalog, error) {
		if got, ok := seen[root]; ok {
			return got.cat, got.err
		}
		cat, err := load(root)
		seen[root] = loaded{cat: cat, err: err}
		return cat, err
	}
}

func quoteBatchArgv(argv []string) string {
	parts := make([]string, len(argv))
	for i, arg := range argv {
		if arg == "" || strings.ContainsAny(arg, " \t\"") {
			parts[i] = strconv.Quote(arg)
			continue
		}
		parts[i] = arg
	}
	return strings.Join(parts, " ")
}

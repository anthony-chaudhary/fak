package devcmd

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/studyclass"
	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

func RunStudyClassify(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak study-classify classify|validate|validate-index|schema [flags]")
		return 2
	}
	switch args[0] {
	case "classify":
		return runStudyClassifyClassify(stdout, stderr, args[1:])
	case "validate":
		return runStudyClassifyValidate(stdout, stderr, args[1:])
	case "validate-index":
		return runStudyClassifyValidateIndex(stdout, stderr, args[1:])
	case "schema":
		if len(args) != 1 {
			fmt.Fprintln(stderr, "usage: fak study-classify schema")
			return 2
		}
		if _, err := stdout.Write(studyclass.SchemaDocument()); err != nil {
			fmt.Fprintf(stderr, "study-classify: write schema: %v\n", err)
			return 1
		}
		return 0
	default:
		fmt.Fprintln(stderr, "usage: fak study-classify classify|validate|validate-index|schema [flags]")
		return 2
	}
}

func runStudyClassifyValidateIndex(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-classify validate-index", flag.ContinueOnError)
	fs.SetOutput(stderr)
	indexPath := fs.String("index", "", "compact classification index JSON path (required)")
	classificationPath := fs.String("classification", "", "full classification JSON path (required)")
	corpusPath := fs.String("corpus", "", "validated study-forge corpus path (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*indexPath) == "" || strings.TrimSpace(*classificationPath) == "" || strings.TrimSpace(*corpusPath) == "" {
		fmt.Fprintln(stderr, "usage: fak study-classify validate-index --index PATH --classification PATH --corpus PATH")
		return 2
	}
	indexFile, err := os.Open(*indexPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: open compact index: %v\n", err)
		return 1
	}
	index, readErr := studyclass.ReadCompactJSON(indexFile)
	closeErr := indexFile.Close()
	if readErr != nil {
		fmt.Fprintf(stderr, "study-classify: read compact index: %v\n", readErr)
		return 1
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "study-classify: close compact index: %v\n", closeErr)
		return 1
	}
	classificationFile, err := os.Open(*classificationPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: open classification: %v\n", err)
		return 1
	}
	output, readErr := studyclass.ReadJSON(classificationFile)
	closeErr = classificationFile.Close()
	if readErr != nil {
		fmt.Fprintf(stderr, "study-classify: read classification: %v\n", readErr)
		return 1
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "study-classify: close classification: %v\n", closeErr)
		return 1
	}
	corpus, err := readValidatedStudyCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: %v\n", err)
		return 1
	}
	rawSHA, err := studyClassFileSHA256(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: hash corpus: %v\n", err)
		return 1
	}
	if err := studyclass.ValidateAgainst(output, corpus, rawSHA); err != nil {
		fmt.Fprintf(stderr, "study-classify: invalid classification: %v\n", err)
		return 1
	}
	if err := studyclass.ValidateCompactAgainst(index, output); err != nil {
		fmt.Fprintf(stderr, "study-classify: invalid compact index: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid compact study classification: records=%d clusters=%d\n", index.Summary.RecordCount, index.Summary.ClusterCount)
	return 0
}

func runStudyClassifyClassify(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-classify classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpusPath := fs.String("corpus", "", "validated study-forge corpus path (required)")
	outPath := fs.String("out", "", "full per-record classification output path (required)")
	indexPath := fs.String("index-out", "", "compact cluster index output path (required)")
	relatedLimit := fs.Int("related-limit", studyclass.DefaultRelatedSampleLimit, "maximum related identities retained per compact cluster")
	jsonOutput := fs.Bool("json", false, "print the machine-readable summary")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*corpusPath) == "" || strings.TrimSpace(*outPath) == "" || strings.TrimSpace(*indexPath) == "" || *relatedLimit < 0 || filepath.Clean(*outPath) == filepath.Clean(*indexPath) {
		fmt.Fprintln(stderr, "usage: fak study-classify classify --corpus PATH --out PATH --index-out PATH [--related-limit N] [--json]")
		return 2
	}

	corpus, err := readValidatedStudyCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: %v\n", err)
		return 1
	}
	rawSHA, err := studyClassFileSHA256(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: hash corpus: %v\n", err)
		return 1
	}
	output, err := studyclass.Classify(corpus, rawSHA)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: classify: %v\n", err)
		return 1
	}
	index, err := studyclass.Compact(output, *relatedLimit)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: compact: %v\n", err)
		return 1
	}
	if err := studyClassWriteAtomic(*outPath, func(w io.Writer) error { return studyclass.WriteJSON(w, output) }); err != nil {
		fmt.Fprintf(stderr, "study-classify: write full output: %v\n", err)
		return 1
	}
	if err := studyClassWriteAtomic(*indexPath, func(w io.Writer) error { return studyclass.WriteCompactJSON(w, index) }); err != nil {
		fmt.Fprintf(stderr, "study-classify: write compact index: %v\n", err)
		return 1
	}

	if *jsonOutput {
		if err := json.NewEncoder(stdout).Encode(output.Summary); err != nil {
			fmt.Fprintf(stderr, "study-classify: write summary: %v\n", err)
			return 1
		}
		return 0
	}
	renderStudyClassSummary(stdout, output.Summary)
	return 0
}

func runStudyClassifyValidate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-classify validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	classificationPath := fs.String("classification", "", "full classification JSON path (required)")
	corpusPath := fs.String("corpus", "", "validated study-forge corpus path (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*classificationPath) == "" || strings.TrimSpace(*corpusPath) == "" {
		fmt.Fprintln(stderr, "usage: fak study-classify validate --classification PATH --corpus PATH")
		return 2
	}

	classificationFile, err := os.Open(*classificationPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: open classification: %v\n", err)
		return 1
	}
	output, readErr := studyclass.ReadJSON(classificationFile)
	closeErr := classificationFile.Close()
	if readErr != nil {
		fmt.Fprintf(stderr, "study-classify: read classification: %v\n", readErr)
		return 1
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "study-classify: close classification: %v\n", closeErr)
		return 1
	}
	corpus, err := readValidatedStudyCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: %v\n", err)
		return 1
	}
	rawSHA, err := studyClassFileSHA256(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-classify: hash corpus: %v\n", err)
		return 1
	}
	if err := studyclass.ValidateAgainst(output, corpus, rawSHA); err != nil {
		fmt.Fprintf(stderr, "study-classify: invalid classification: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid study classification: records=%d clusters=%d\n", output.Summary.RecordCount, output.Summary.ClusterCount)
	return 0
}

func readValidatedStudyCorpus(path string) (studyforge.Corpus, error) {
	corpus, err := studyforge.Read(path)
	if err != nil {
		return studyforge.Corpus{}, fmt.Errorf("read corpus: %w", err)
	}
	if err := studyforge.Validate(corpus); err != nil {
		return studyforge.Corpus{}, fmt.Errorf("validate corpus: %w", err)
	}
	// studyforge.Read owns the corpus compatibility and semantic contract. The
	// CLI adds a closed-field pass because classification artifacts must not bind
	// silently to corpus fields this version does not understand.
	f, err := os.Open(path)
	if err != nil {
		return studyforge.Corpus{}, fmt.Errorf("open corpus for strict decode: %w", err)
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		zr, err := gzip.NewReader(f)
		if err != nil {
			return studyforge.Corpus{}, fmt.Errorf("open gzip corpus for strict decode: %w", err)
		}
		defer zr.Close()
		r = zr
	}
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var strict studyforge.Corpus
	if err := dec.Decode(&strict); err != nil {
		return studyforge.Corpus{}, fmt.Errorf("strict-decode corpus: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return studyforge.Corpus{}, fmt.Errorf("strict-decode corpus: %w", err)
	}
	return corpus, nil
}

func studyClassFileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func studyClassWriteAtomic(path string, write func(io.Writer) error) (err error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".study-classify-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		if err != nil {
			_ = os.Remove(tmpPath)
		}
	}()
	if err = tmp.Chmod(0600); err != nil {
		return err
	}
	if err = write(tmp); err != nil {
		return err
	}
	if err = tmp.Sync(); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func renderStudyClassSummary(w io.Writer, summary studyclass.Summary) {
	fmt.Fprintf(w, "records: %d\nclusters: %d\n", summary.RecordCount, summary.ClusterCount)
	renderStudyClassCounts(w, "source", summary.BySource)
	renderStudyClassCounts(w, "disposition", summary.ByDisposition)
	renderStudyClassCounts(w, "mechanism", summary.ByMechanism)
	renderStudyClassCounts(w, "state", summary.ByState)
	renderStudyClassCounts(w, "confidence", summary.ByConfidence)
}

func renderStudyClassCounts(w io.Writer, label string, counts map[string]int) {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(w, "%s.%s: %d\n", label, key, counts[key])
	}
}

package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/experiments"
)

func cmdExperiments(argv []string) { os.Exit(runExperiments(os.Stdout, os.Stderr, argv)) }

func runExperiments(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		experimentsUsage(stderr)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "list", "ls":
		return experimentsList(stdout, stderr, rest)
	case "query":
		return experimentsQuery(stdout, stderr, rest)
	case "register":
		return experimentsRegister(stdout, stderr, rest)
	case "record":
		return experimentsRecord(stdout, stderr, rest)
	case "lookup":
		return experimentsLookup(stdout, stderr, rest)
	case "-h", "--help", "help":
		experimentsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak experiments: unknown subcommand %q\n", sub)
		experimentsUsage(stderr)
		return 2
	}
}

func experimentsUsage(w io.Writer) {
	fmt.Fprint(w, `fak experiments — researcher experiment registry: view what's running where and check for collisions

usage:
  fak experiments list [--host <host>]              list all registered experiments
  fak experiments query --model <m> --backend <b>  query a (model, backend) cell for prior runs
  fak experiments register --id <id> --owner <o> --host <h> --models <m1,m2> --backends <b1,b2> [--artifact <path>]
                                            register a new experiment
  fak experiments record --file <receipt.json> [--ledger <path>] [--json]
                                            append a scoped result receipt
  fak experiments lookup --file <identity.json> [--ledger <path>] [--json]
                                            look up an exact hypothesis + identity

Start here:
  fak experiments list                              see all experiments across all hosts
  fak experiments query --model qwen2.5-1.5b --backend cpu  check for existing runs
`)
}

func experimentsRecord(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("experiments record", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var input, ledger string
	var asJSON bool
	fs.StringVar(&input, "file", "", "receipt JSON file")
	fs.StringVar(&ledger, "ledger", "", "receipt ledger path")
	fs.BoolVar(&asJSON, "json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if input == "" {
		fmt.Fprintln(stderr, "fak experiments record: --file is required")
		return 2
	}
	var receipt experiments.Receipt
	if err := decodeExperimentsJSONFile(input, &receipt); err != nil {
		fmt.Fprintf(stderr, "fak experiments record: %v\n", err)
		return 2
	}
	if receipt.Schema == "" {
		receipt.Schema = experiments.ReceiptSchema
	}
	if receipt.EvidenceClass == "" {
		receipt.EvidenceClass = experiments.EvidenceClassScreening
	}
	if receipt.RecordedAt == "" {
		receipt.RecordedAt = time.Now().UTC().Format(time.RFC3339)
	}
	if ledger == "" {
		ledger = filepath.Join(repoRoot(), experiments.ReceiptLedgerRel)
	}
	if err := experiments.AppendReceipt(ledger, receipt); err != nil {
		fmt.Fprintf(stderr, "fak experiments record: %v\n", err)
		return 2
	}
	if asJSON {
		emitExperimentsJSON(stdout, receipt)
		return 0
	}
	fmt.Fprintf(stdout, "Recorded receipt %s (%s)\n", receipt.ID, receipt.Verdict)
	fmt.Fprintln(stdout, "  Evidence: screening (claim eligible: false)")
	fmt.Fprintf(stdout, "  Hypothesis: %s\n", receipt.Hypothesis)
	fmt.Fprintf(stdout, "  Scope: %s\n", receipt.Scope)
	fmt.Fprintf(stdout, "  Revision: %s\n", receipt.Revision)
	fmt.Fprintf(stdout, "  Environment: %s\n", receipt.Environment)
	fmt.Fprintf(stdout, "  Environment digest: %s\n", receipt.EnvironmentDigest)
	fmt.Fprintf(stdout, "  Artifact digest: %s\n", receipt.ArtifactDigest)
	fmt.Fprintf(stdout, "  Next action: %s\n", receipt.NextAction)
	return 0
}

func experimentsLookup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("experiments lookup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var input, ledger string
	var asJSON bool
	fs.StringVar(&input, "file", "", "identity JSON file")
	fs.StringVar(&ledger, "ledger", "", "receipt ledger path")
	fs.BoolVar(&asJSON, "json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if input == "" {
		fmt.Fprintln(stderr, "fak experiments lookup: --file is required")
		return 2
	}
	var identity experiments.ReceiptIdentity
	if err := decodeExperimentsJSONFile(input, &identity); err != nil {
		fmt.Fprintf(stderr, "fak experiments lookup: %v\n", err)
		return 2
	}
	if err := identity.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak experiments lookup: %v\n", err)
		return 2
	}
	if ledger == "" {
		ledger = filepath.Join(repoRoot(), experiments.ReceiptLedgerRel)
	}
	receipts, err := experiments.ReadReceiptLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak experiments lookup: %v\n", err)
		return 1
	}
	result := experiments.LookupReceipt(receipts, identity)
	if asJSON {
		emitExperimentsJSON(stdout, result)
		return 0
	}
	renderExperimentLookup(stdout, result)
	return 0
}

func decodeExperimentsJSONFile(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if dec.Decode(&struct{}{}) != io.EOF {
		return errors.New("input must contain exactly one JSON object")
	}
	return nil
}

func renderExperimentLookup(w io.Writer, result experiments.LookupResult) {
	switch result.Status {
	case experiments.LookupExact:
		r := result.Receipt
		fmt.Fprintf(w, "Exact receipt %s\n", r.ID)
		if result.MeasuredLoss {
			fmt.Fprintln(w, "  Verdict: lost (measured loss within receipt scope)")
		} else {
			fmt.Fprintf(w, "  Verdict: %s (not a measured loss)\n", r.Verdict)
		}
		fmt.Fprintln(w, "  Evidence: screening (claim eligible: false)")
		fmt.Fprintf(w, "  Scope: %s\n", r.Scope)
		fmt.Fprintf(w, "  Revision: %s\n", r.Revision)
		fmt.Fprintf(w, "  Environment: %s\n", r.Environment)
		fmt.Fprintf(w, "  Environment digest: %s\n", r.EnvironmentDigest)
		fmt.Fprintf(w, "  Artifact digest: %s\n", r.ArtifactDigest)
		fmt.Fprintf(w, "  Next action: %s\n", r.NextAction)
	case experiments.LookupIdentityMismatch:
		fmt.Fprintf(w, "Identity mismatch: %d receipt(s) share the hypothesis, but none match all identity fields.\n", result.Matches)
		fmt.Fprintln(w, "No measured loss applies to this identity.")
	default:
		fmt.Fprintln(w, "No receipt found for this hypothesis.")
	}
}

type experimentsFlags struct {
	host     string
	model    string
	backend  string
	asJSON   bool
	registry string
	id       string
	owner    string
	models   string
	backends string
	artifact string
}

func parseExperimentsFlags(name string, fs *flag.FlagSet, argv []string) (*experimentsFlags, error) {
	f := &experimentsFlags{}
	fs.StringVar(&f.host, "host", "", "filter by host")
	fs.StringVar(&f.model, "model", "", "model to query")
	fs.StringVar(&f.backend, "backend", "", "backend to query")
	fs.StringVar(&f.registry, "registry", "", "registry path (default: <root>/experiments/registry.jsonl)")
	fs.BoolVar(&f.asJSON, "json", false, "emit machine-readable JSON")
	if name == "register" {
		fs.StringVar(&f.id, "id", "", "experiment ID")
		fs.StringVar(&f.owner, "owner", "", "experiment owner")
		fs.StringVar(&f.models, "models", "", "comma-separated models")
		fs.StringVar(&f.backends, "backends", "", "comma-separated backends")
		fs.StringVar(&f.artifact, "artifact", "", "artifact path")
	}
	if err := fs.Parse(argv); err != nil {
		return nil, err
	}
	return f, nil
}

func (f *experimentsFlags) root() string {
	return repoRoot()
}

func (f *experimentsFlags) registryPath(root string) string {
	if f.registry != "" {
		return f.registry
	}
	return filepath.Join(root, experiments.ExperimentLedgerRel)
}

func experimentsList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("experiments list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseExperimentsFlags("list", fs, argv)
	if err != nil {
		return 2
	}
	root := f.root()
	all := experiments.ReadAllLedgers(root)
	if f.host != "" {
		var filtered []experiments.Experiment
		for _, e := range all {
			if e.Host == f.host {
				filtered = append(filtered, e)
			}
		}
		all = filtered
	}
	if f.asJSON {
		emitExperimentsJSON(stdout, all)
		return 0
	}
	renderExperimentsList(stdout, all)
	return 0
}

func experimentsQuery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("experiments query", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseExperimentsFlags("query", fs, argv)
	if err != nil {
		return 2
	}
	if f.model == "" || f.backend == "" {
		fmt.Fprintf(stderr, "fak experiments query: --model and --backend are required\n")
		return 2
	}
	root := f.root()
	all := experiments.ReadAllLedgers(root)
	overlaps := experiments.FindOverlaps(all, []string{f.model}, []string{f.backend})
	if f.asJSON {
		emitExperimentsJSON(stdout, overlaps)
		return 0
	}
	renderExperimentsQuery(stdout, f.model, f.backend, overlaps)
	return 0
}

func experimentsRegister(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("experiments register", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseExperimentsFlags("register", fs, argv)
	if err != nil {
		return 2
	}
	if f.id == "" || f.owner == "" {
		fmt.Fprintf(stderr, "fak experiments register: --id and --owner are required\n")
		return 2
	}
	if f.models == "" || f.backends == "" {
		fmt.Fprintf(stderr, "fak experiments register: --models and --backends are required\n")
		return 2
	}
	root := f.root()
	registryPath := f.registryPath(root)
	host, _ := os.Hostname()
	if f.host != "" {
		host = f.host
	}
	exp := experiments.Experiment{
		ID:           f.id,
		Owner:        f.owner,
		Host:         host,
		Models:       strings.Split(f.models, ","),
		Backends:     strings.Split(f.backends, ","),
		Started:      time.Now().UTC().Format(time.RFC3339),
		ArtifactPath: f.artifact,
	}
	for i := range exp.Models {
		exp.Models[i] = strings.TrimSpace(exp.Models[i])
	}
	for i := range exp.Backends {
		exp.Backends[i] = strings.TrimSpace(exp.Backends[i])
	}
	if err := experiments.WriteLedgerLine(registryPath, exp); err != nil {
		fmt.Fprintf(stderr, "fak experiments register: %v\n", err)
		return 1
	}
	all := experiments.ReadAllLedgers(root)
	overlaps := experiments.FindOverlaps(all, exp.Models, exp.Backends)
	if len(overlaps) > 0 {
		fmt.Fprintf(stdout, "Registered experiment %s\n", exp.ID)
		fmt.Fprintf(stdout, "WARNING: Found %d overlapping experiment(s) on same (model, backend) cells:\n", len(overlaps))
		for _, o := range overlaps {
			fmt.Fprintf(stdout, "  - %s (owner=%s, host=%s, started=%s)\n", o.ID, o.Owner, o.Host, o.Started)
		}
		return 0
	}
	fmt.Fprintf(stdout, "Registered experiment %s\n", exp.ID)
	return 0
}

func renderExperimentsList(w io.Writer, exps []experiments.Experiment) {
	if len(exps) == 0 {
		fmt.Fprintln(w, "No experiments registered.")
		return
	}
	fmt.Fprintf(w, "Registered experiments (%d):\n\n", len(exps))
	sort.Slice(exps, func(i, j int) bool {
		return exps[i].Started > exps[j].Started
	})
	for _, e := range exps {
		fmt.Fprintf(w, "ID: %s\n", e.ID)
		fmt.Fprintf(w, "  Owner: %s\n", e.Owner)
		fmt.Fprintf(w, "  Host: %s\n", e.Host)
		fmt.Fprintf(w, "  Models: %s\n", strings.Join(e.Models, ", "))
		fmt.Fprintf(w, "  Backends: %s\n", strings.Join(e.Backends, ", "))
		fmt.Fprintf(w, "  Started: %s\n", e.Started)
		if e.ArtifactPath != "" {
			fmt.Fprintf(w, "  Artifact: %s\n", e.ArtifactPath)
		}
		fmt.Fprintln(w)
	}
}

func renderExperimentsQuery(w io.Writer, model, backend string, overlaps []experiments.Experiment) {
	fmt.Fprintf(w, "Querying cell (model=%s, backend=%s):\n", model, backend)
	if len(overlaps) == 0 {
		fmt.Fprintln(w, "  No prior runs found.")
		return
	}
	fmt.Fprintf(w, "  Found %d prior run(s):\n\n", len(overlaps))
	sort.Slice(overlaps, func(i, j int) bool {
		return overlaps[i].Started > overlaps[j].Started
	})
	for _, e := range overlaps {
		fmt.Fprintf(w, "  ID: %s\n", e.ID)
		fmt.Fprintf(w, "    Owner: %s\n", e.Owner)
		fmt.Fprintf(w, "    Host: %s\n", e.Host)
		fmt.Fprintf(w, "    Started: %s\n", e.Started)
		if e.ArtifactPath != "" {
			fmt.Fprintf(w, "    Artifact: %s\n", e.ArtifactPath)
		}
		fmt.Fprintln(w)
	}
}

func emitExperimentsJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

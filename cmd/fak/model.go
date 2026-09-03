package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

const modelUsage = "usage: fak model <plan|load|pull|ls|provider-contracts|qwen38-ladder|canary-gate|canary-run|acceptance-gate|acceptance-run|acceptance-refold|readiness-inventory> ...\n" +
	"  fak model plan         choose a Qwen3.8-27B load from goals and declared capacity without downloading\n" +
	"  fak model load <ref>   resolve a model ref (alias | hf://… | path) to a cached file path\n" +
	"  fak model pull <ref>   download a model ref into the local cache (alias-aware)\n" +
	"  fak model ls           list known model aliases and which are cached locally\n" +
	"  fak model provider-contracts inspect canonical provider behavior and provenance (--json for complete records)\n" +
	"  fak model qwen38-ladder prove concepts cheaply, then promote to exact Qwen3.8-27B evidence\n" +
	"  " + modelCanaryGateSynopsis + "  fold exact-model observations into PROMOTE/ROLLBACK/HOLD\n" +
	"  fak model canary-run  own one guarded Darwin local-model canary transaction\n" +
	"  fak model incumbent   own the preserved Qwen3.6 incumbent service identity (#9714): render | preflight | install\n" +
	"  fak model acceptance-gate evaluate a versioned exact-model capability report\n" +
	"  fak model acceptance-run execute a latest-generation exact-model campaign (older generations require a named exception)\n" +
	"  fak model acceptance-refold replay immutable raw streams with the current parser\n" +
	"  fak model readiness-inventory join exact-ID acceptance provenance into readiness rows\n"

// cmdModel handles `fak model <subcommand>`: load (resolve a ref to a cached path),
// pull (download by alias/hf:// into the cache), ls (list the alias registry +
// local-cache status), canary-gate (make an exact-model rollout decision), and
// acceptance-gate (evaluate exact-model capability evidence).
// pull/ls give fak the Ollama-style run-by-name surface;
// top-level `fak pull` / `fak ls` are thin aliases for the latter two.
func cmdModel(args []string) {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, modelUsage)
		os.Exit(2)
	}
	switch args[0] {
	case "plan":
		os.Exit(runModelPlan(os.Stdout, os.Stderr, args[1:]))
	case "load":
		cmdModelLoad(args[1:])
	case "pull":
		cmdModelPull(args[1:])
	case "ls", "list":
		cmdModelLs(args[1:])
	case "provider-contracts":
		os.Exit(runModelProviderContracts(os.Stdout, os.Stderr, args[1:]))
	case "qwen38-ladder":
		os.Exit(runModelQwen38Ladder(os.Stdout, os.Stderr, args[1:]))
	case "canary-gate":
		os.Exit(runModelCanaryGate(os.Stdout, os.Stderr, args[1:]))
	case "canary-run":
		os.Exit(runModelCanaryRun(os.Stdout, os.Stderr, args[1:]))
	case "incumbent":
		cmdModelIncumbent(args[1:])
	case "acceptance-gate":
		os.Exit(runModelAcceptanceGate(os.Stdout, os.Stderr, args[1:]))
	case "acceptance-run":
		os.Exit(runModelAcceptanceRun(os.Stdout, os.Stderr, args[1:]))
	case "acceptance-refold":
		os.Exit(runModelAcceptanceRefold(os.Stdout, os.Stderr, args[1:]))
	case "acceptance-fixture":
		os.Exit(runModelAcceptanceFixture(os.Stdin, os.Stdout, os.Stderr))
	case "readiness-inventory":
		os.Exit(runModelReadinessInventory(os.Stdout, os.Stderr, args[1:]))
	case "-h", "--help", "help":
		fmt.Fprint(os.Stderr, modelUsage)
	default:
		fmt.Fprintf(os.Stderr, "fak model: unknown subcommand %q\n", args[0])
		os.Exit(2)
	}
}

// cmdModelPull is `fak pull <ref>` / `fak model pull <ref>`: resolve an alias to its
// target (a bare hf:// URI or path passes through), download it into the local cache
// (cache-hit if already present), and print the local path. The Ollama `pull` analog.
// Progress goes to stderr so the printed path on stdout stays scriptable.
func cmdModelPull(args []string) {
	fs := flag.NewFlagSet("model pull", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak pull <alias | hf://owner/repo[@rev]/file | path>")
		fmt.Fprintln(os.Stderr, "  run `fak ls` to see known aliases")
		fs.PrintDefaults()
	}
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	resolveAndFetchModelRef(fs.Arg(0), "pull", 1)
}

// cmdModelLs is `fak ls` / `fak model ls`: list the merged alias registry (embedded
// catalog + the user's registry.json) with each alias's target and whether it is
// already downloaded locally. The Ollama `list` analog. --json emits the rows.
func cmdModelLs(args []string) {
	fs := flag.NewFlagSet("model ls", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit the alias rows as JSON")
	_ = fs.Parse(args)

	reg, err := modelreg.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak ls: %v\n", err)
		os.Exit(1)
	}
	entries := reg.Entries()
	if *asJSON {
		emitJSON(entries)
		return
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tCODING\tCACHED\tSIZE\tSOURCE\tTARGET")
	for _, e := range entries {
		cached, size := "no", "-"
		if e.Cached() {
			cached, size = "yes", humanBytes(e.SizeBytes)
		}
		// CODING marks the curated tool-call-capable coding models (#1058) — the ones
		// `fak guard --local`/`--gguf -- claude` can actually drive a coding loop with.
		coding := "-"
		if e.Coding {
			coding = "yes"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", e.Name, coding, cached, size, e.Source, e.Target)
	}
	_ = tw.Flush()
}

// cmdModelLoad downloads (or cache-hits) the file a model ref names and prints its
// local path. The ref is alias-aware: a friendly name (`smollm2`) resolves through
// the registry to its hf:// target, while a bare hf:// URI or an existing path passes
// through. HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) authorizes gated repos; FAK_MODELS_DIR
// overrides the cache root. Progress goes to stderr so the printed path on stdout stays
// scriptable: GGUF=$(fak model load smollm2).
func cmdModelLoad(args []string) {
	fs := flag.NewFlagSet("model load", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak model load <alias | hf://owner/repo[@rev]/file | path>")
		fs.PrintDefaults()
	}
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	resolveAndFetchModelRef(fs.Arg(0), "model load", 2)
}

// resolveAndFetchModelRef resolves a model ref (alias-aware), short-circuits a ref that is
// already a local path, otherwise downloads the hf:// URI and prints the resulting local path.
// label names the command in messages ("pull" / "model load"); notFoundExit is the exit code
// used when the ref is neither an hf:// URI nor an existing path.
func blockedModelMessage(arg, label string) (string, bool) {
	blocked, ok := modelreg.Blocked(arg)
	if !ok {
		return "", false
	}
	return fmt.Sprintf("fak %s: model %q is unavailable: %s; stage a provenance-verified %s (%d bytes, sha256 %s) and pass its local path directly",
		label, arg, blocked.Reason, blocked.Filename, blocked.Bytes, blocked.SHA256), true
}

func resolveAndFetchModelRef(arg, label string, notFoundExit int) {
	if message, blocked := blockedModelMessage(arg, label); blocked {
		fmt.Fprintln(os.Stderr, message)
		os.Exit(1)
	}
	ref, expanded := modelreg.Resolve(arg)
	if expanded {
		fmt.Fprintf(os.Stderr, "fak %s: %s → %s\n", label, arg, ref)
	}
	if !hfhub.IsURI(ref) {
		if _, err := os.Stat(ref); err == nil {
			fmt.Println(ref)
			return
		}
		fmt.Fprintf(os.Stderr, "fak %s: %q is not a known alias, an hf:// URI, or an existing path\n", label, arg)
		os.Exit(notFoundExit)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	path, err := hfhub.FetchURI(ctx, ref, os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak %s: %v\n", label, err)
		os.Exit(1)
	}
	fmt.Println(path)
}

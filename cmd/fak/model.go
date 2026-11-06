package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

const modelUsage = "usage: fak model <fit|plan|load|pull|quantize|ls|provider-contracts|qwen38-ladder|canary-gate|canary-run|acceptance-gate|acceptance-run|acceptance-refold|readiness-inventory|preflight|trace-stub> ...\n" +
	"  fak model fit compose|plan|check|compare declared multi-device memory fit; no deployment\n" +
	"  fak model plan         choose a Qwen3.8-27B load from goals and declared capacity without downloading\n" +
	"  fak model load <ref>   resolve a model ref (alias | hf://… | path) to a cached file path\n" +
	"  fak model pull <ref>   download a model ref into the local cache (alias-aware)\n" +
	"  fak model quantize     convert local F32 GGUF matrices to q4_k/q5_k/q6_k; new output only\n" +
	"  fak model ls           list known model aliases and which are cached locally\n" +
	"  fak model provider-contracts inspect canonical provider behavior and provenance (--json for complete records)\n" +
	"  fak model qwen38-ladder prove concepts cheaply, then promote to exact Qwen3.8-27B evidence\n" +
	"  " + modelCanaryGateSynopsis + "  fold exact-model observations into PROMOTE/ROLLBACK/HOLD\n" +
	"  fak model canary-run  own one guarded Darwin local-model canary transaction\n" +
	"  fak model incumbent   own the preserved Qwen3.6 incumbent service identity (#9714): render | preflight | install\n" +
	"  fak model acceptance-gate evaluate a versioned exact-model capability report\n" +
	"  fak model acceptance-run execute a latest-generation exact-model campaign (older generations require a named exception)\n" +
	"  fak model acceptance-refold replay immutable raw streams with the current parser\n" +
	"  fak model readiness-inventory join exact-ID acceptance provenance into readiness rows\n" +
	"  fak model preflight    list every unread GGUF key, HF-config mismatch, missing/orphan tensor and unsupported dtype off the header\n" +
	"  fak model trace-stub   inspect embedding and output rows through diagnostic identity layers; never generate tokens\n"

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
	case "fit":
		os.Exit(runModelFit(os.Stdin, os.Stdout, os.Stderr, args[1:]))
	case "plan":
		os.Exit(runModelPlan(os.Stdout, os.Stderr, args[1:]))
	case "quantize":
		os.Exit(runModelQuantize(os.Stdout, os.Stderr, args[1:]))
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
	case "preflight":
		os.Exit(runModelPreflight(os.Stdout, os.Stderr, args[1:]))
	case "trace-stub":
		os.Exit(runModelTraceStub(os.Stdout, os.Stderr, args[1:]))
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

// ErrUnknownModelType is the closed-contract sentinel for the pre-transfer
// capability gate: a ref whose declared model family is absent from the
// compiled capability table (modelCapabilities) is refused before a single
// byte is transferred. Adapted from TensorFold's pre-transfer gate
// (zig/src/cli/pull.zig:69 @9e8c368b0b4c2c7892a0c9f0f302d7c164016209).
var ErrUnknownModelType = errors.New("model type not supported by this build")

// unknownModelTypeError carries the offending family token on the sentinel so
// the refusal can name the requested type without enumerating registry
// contents (the rejection probe: the message must not list known types).
type unknownModelTypeError struct{ family string }

func (e *unknownModelTypeError) Error() string {
	return fmt.Sprintf("model type %q is not supported by this build", e.family)
}

func (e *unknownModelTypeError) Unwrap() error { return ErrUnknownModelType }

// modelCapability is one row of the compiled capability table: a model family
// this binary can load, the weight formats its loader accepts, and the native
// flags and env knobs that steer it. The table is the SINGLE source both the
// pre-transfer gate (modelTypeGate) and the serve-fit capability report
// (serveCapabilityReport) consult, so a listed family is admitted everywhere
// and an unlisted one is refused everywhere — the two gates cannot disagree.
// Rows are static and sorted by Family, so the report is deterministic for a
// given build (TensorFold cli_capabilities.zig:20 doc shape: version/chip/
// backends/families -> weight formats, native flags, env).
type modelCapability struct {
	Family  string   // family token matched against the alias name
	Formats []string // weight formats the loader accepts for this family
	Flags   []string // native CLI flags that apply
	Env     []string // env knobs that apply
}

// modelCapabilities is the compiled family/format table. It mirrors the
// embedded alias catalog's families exactly, so every registered alias
// proceeds through the gate unchanged while an alias-declared family outside
// this set is refused before transfer.
var modelCapabilities = []modelCapability{
	{Family: "glm-5.3-flash", Formats: []string{"bf16", "fp8"}},
	{Family: "glm53", Formats: []string{"bf16", "fp8"}},
	{Family: "ornith", Formats: []string{"fp8", "q4_k_m"}, Flags: []string{"--gguf"}, Env: []string{"FAK_Q4K"}},
	{Family: "qwen2.5", Formats: []string{"q4_k_m", "q8_0"}, Flags: []string{"--gguf"}, Env: []string{"FAK_Q4K"}},
	{Family: "qwen2.5-coder", Formats: []string{"q4_k_m"}, Flags: []string{"--gguf"}, Env: []string{"FAK_Q4K"}},
	{Family: "qwen3.8", Formats: []string{"fp8", "q4_k_m", "ud_q2_k_xl"}, Flags: []string{"--gguf"}, Env: []string{"FAK_Q4K", "FAK_STREAM_Q4K"}},
	{Family: "qwen38", Formats: []string{"fp8", "q4_k_m", "ud_q2_k_xl"}, Flags: []string{"--gguf"}, Env: []string{"FAK_Q4K", "FAK_STREAM_Q4K"}},
	{Family: "smollm2", Formats: []string{"q8_0"}, Flags: []string{"--gguf"}},
}

// modelFamilyOfAlias extracts the family token the pre-transfer gate matches
// against the compiled table: the alias with its ":tag" suffix stripped
// ("qwen38:27b" -> "qwen38"), mirroring modelreg's "default" normalization so
// `fak pull default` gates on the default alias's family.
func modelFamilyOfAlias(arg string) string {
	a := strings.TrimSpace(arg)
	if strings.EqualFold(a, "default") {
		a = modelreg.DefaultAlias
	}
	if i := strings.IndexByte(a, ':'); i > 0 {
		a = a[:i]
	}
	return a
}

// modelFamilyListed reports whether token matches a compiled capability row:
// exact family equality, or a dash-separated extension of a listed family
// ("qwen3.8-70b-q4_k_m" extends "qwen3.8"). Longest family wins, so
// "qwen2.5-coder" matches its own row rather than "qwen2.5".
func modelFamilyListed(token string) bool {
	best := 0
	for _, c := range modelCapabilities {
		if token == c.Family || strings.HasPrefix(token, c.Family+"-") {
			if len(c.Family) > best {
				best = len(c.Family)
			}
		}
	}
	return best > 0
}

// modelTypeGate is the pre-transfer capability gate: it refuses (with
// ErrUnknownModelType) an alias-declared model family that the compiled
// capability table does not list, so an unknown type never reaches the
// downloader. expanded=false (a bare hf:// URI or local path) declares no
// type and passes through — its format is judged at load time, not here. It
// is the SAME table the serve-fit path reports (serveCapabilityReport), so
// admission and disclosure agree by construction.
func modelTypeGate(arg string, expanded bool) error {
	if !expanded {
		return nil
	}
	family := modelFamilyOfAlias(arg)
	if !modelFamilyListed(family) {
		return &unknownModelTypeError{family: family}
	}
	return nil
}

type modelTransferKind int

const (
	modelTransferBlocked modelTransferKind = iota
	modelTransferNotFound
	modelTransferLocal
	modelTransferFetch
)

// modelTransferPlan is the decision of planModelTransfer: which outcome the
// caller takes, the concrete ref (or local path) involved, the human-facing
// refusal text, and the process exit code. kind==modelTransferFetch means the
// ref cleared every check up to the byte transfer itself.
type modelTransferPlan struct {
	kind     modelTransferKind
	ref      string
	message  string
	exit     int
	expanded bool
}

// planModelTransfer performs every step of resolveAndFetchModelRef up to (but
// excluding) the byte transfer: the blocked-source refusal, alias expansion,
// the local-path short-circuit, and the pre-transfer capability gate. It moves
// no bytes, so a test can inspect the decision with a transfer spy standing in
// for modelTransfer.
func planModelTransfer(arg, label string, notFoundExit int) modelTransferPlan {
	if message, blocked := blockedModelMessage(arg, label); blocked {
		return modelTransferPlan{kind: modelTransferBlocked, message: message, exit: 1}
	}
	ref, expanded := modelreg.Resolve(arg)
	if !hfhub.IsURI(ref) {
		if _, err := os.Stat(ref); err == nil {
			return modelTransferPlan{kind: modelTransferLocal, ref: ref, expanded: expanded}
		}
		return modelTransferPlan{
			kind:     modelTransferNotFound,
			message:  fmt.Sprintf("fak %s: %q is not a known alias, an hf:// URI, or an existing path", label, arg),
			exit:     notFoundExit,
			expanded: expanded,
		}
	}
	if err := modelTypeGate(arg, expanded); err != nil {
		return modelTransferPlan{
			kind:     modelTransferBlocked,
			message:  fmt.Sprintf("fak %s: %v", label, err),
			exit:     1,
			expanded: expanded,
		}
	}
	return modelTransferPlan{kind: modelTransferFetch, ref: ref, expanded: expanded}
}

// modelTransfer is the byte-transfer seam behind the pull/load gate; production
// wraps hfhub.FetchURI. Tests replace it with a counting spy to witness that a
// refused model type never reaches the downloader.
var modelTransfer func(ctx context.Context, ref string, progress io.Writer) (string, error) = func(ctx context.Context, ref string, progress io.Writer) (string, error) {
	return hfhub.FetchURI(ctx, ref, progress)
}

func resolveAndFetchModelRef(arg, label string, notFoundExit int) {
	plan := planModelTransfer(arg, label, notFoundExit)
	// The blocked case never resolved an alias (plan.expanded stays false), so
	// only post-resolution outcomes carry the expansion arrow.
	if plan.expanded {
		fmt.Fprintf(os.Stderr, "fak %s: %s → %s\n", label, arg, plan.ref)
	}
	switch plan.kind {
	case modelTransferBlocked:
		fmt.Fprintln(os.Stderr, plan.message)
		os.Exit(plan.exit)
	case modelTransferNotFound:
		fmt.Fprintln(os.Stderr, plan.message)
		os.Exit(plan.exit)
	case modelTransferLocal:
		fmt.Println(plan.ref)
		return
	default: // modelTransferFetch
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		path, err := modelTransfer(ctx, plan.ref, os.Stderr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak %s: %v\n", label, err)
			os.Exit(1)
		}
		fmt.Println(path)
	}
}

package main

// fused.go — the impure shell over internal/fusedturn: the executable form of the
// fused-agent-kernel thesis at the level of ONE TURN. It answers, concretely, the
// question the product name poses — can a single turn spawn BOTH a classical
// operation (a tool call, a git commit) AND a weight-based operation (a model
// forward) and have BOTH cross the same default-deny floor?
//
//	fak fused explain [--json]            the concept + a built-in fused demo turn,
//	                                      classified and adjudicated through a REAL kernel
//	fak fused classify [--file f] [--json] classify a proposed turn's ops into their
//	                                      concept-families and fold the fused summary
//	fak fused run [--json]                submit and reap a no-key fused turn through
//	                                      two real EngineDriver routes
//
// `classify` is pure (no floor); `explain` runs the demo batch through a real
// kernel.Kernel whose tiny policy allows one benign op of each family and denies a
// destructive one — so the output SHOWS both families governed by one kernel. `run`
// goes one rung lower: it dispatches both families through Submit -> Reap ->
// Engine.Complete with deterministic local engines.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sync"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func cmdFused(argv []string) { os.Exit(runFused(os.Stdout, os.Stderr, argv)) }

func runFused(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fusedUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "explain":
		return runFusedExplain(stdout, stderr, argv[1:])
	case "classify":
		return runFusedClassify(stdout, stderr, argv[1:])
	case "run":
		return runFusedRun(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fusedUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak fused: unknown subcommand %q\n", argv[0])
		fusedUsage(stderr)
		return 2
	}
}

// fusedOpSpec is one op in a --file batch: a tool name, an optional engine route,
// and an optional declared concept-family. An omitted/unknown class stays
// ClassUnknown (fail-closed) — the CLI never guesses a family from the name.
type fusedOpSpec struct {
	Tool   string `json:"tool"`
	Engine string `json:"engine,omitempty"`
	Class  string `json:"class,omitempty"` // "classical" | "weight" | "" (undeclared)
}

type fusedBatchFile struct {
	Ops []fusedOpSpec `json:"ops"`
}

const (
	fusedRunClassicalEngineID = "fak-fused-run-classical"
	fusedRunWeightEngineID    = "fak-fused-run-weight"
)

var fusedRunRegisterEnginesOnce sync.Once

// fusedRunEngine is the deterministic no-key/no-GPU engine behind `fak fused run`.
// It still implements the real abi.EngineDriver seam, and WeightBearing tells the
// live classifier whether its route is a model-forward family or a classical tool.
type fusedRunEngine struct {
	id     string
	weight bool
}

func (e fusedRunEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body := fusedRunEngineBody(c, e.weight)
	return &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: inlineFusedRef([]byte(body)),
		Meta: map[string]string{
			"engine":         e.id,
			"weight_bearing": fmt.Sprintf("%t", e.weight),
		},
	}, nil
}

func (e fusedRunEngine) Caps() []abi.Capability { return nil }
func (e fusedRunEngine) WeightBearing() bool    { return e.weight }

func fusedRunEngineBody(c *abi.ToolCall, weight bool) string {
	args := string(c.Args.Inline)
	if args == "" {
		args = "{}"
	}
	if weight {
		return fmt.Sprintf("weight result: %s handled prompt %s", c.Tool, args)
	}
	return fmt.Sprintf("classical result: %s handled fixture %s", c.Tool, args)
}

func inlineFusedRef(b []byte) abi.Ref {
	return abi.Ref{
		Kind:   abi.RefInline,
		Inline: append([]byte(nil), b...),
		Len:    int64(len(b)),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}
}

func ensureFusedRunEngines() {
	fusedRunRegisterEnginesOnce.Do(func() {
		abi.RegisterEngine(fusedRunClassicalEngineID, fusedRunEngine{id: fusedRunClassicalEngineID})
		abi.RegisterEngine(fusedRunWeightEngineID, fusedRunEngine{id: fusedRunWeightEngineID, weight: true})
	})
}

// specToCall builds a declared abi.ToolCall from a spec: it tags the declared
// family so Classify reads it back authoritatively (or leaves it undeclared).
func specToCall(s fusedOpSpec) *abi.ToolCall {
	c := &abi.ToolCall{Tool: s.Tool, Engine: s.Engine}
	switch s.Class {
	case "classical":
		fusedturn.Tag(c, fusedturn.ClassClassical)
	case "weight":
		fusedturn.Tag(c, fusedturn.ClassWeight)
	}
	return c
}

// demoTurn is the built-in fused turn: a benign classical op, a benign weight-based
// op, and a destructive classical op — the three the `explain` floor governs so the
// output shows one kernel allowing across families and refusing across families.
func demoTurn() []*abi.ToolCall {
	return []*abi.ToolCall{
		fusedturn.Classical("read_file", abi.Ref{}),               // classic layer: a deterministic tool
		fusedturn.Weight("glm-5.2", "chat_completion", abi.Ref{}), // model layer: a weight-based forward
		fusedturn.Classical("rm_rf", abi.Ref{}),                   // classic layer: a destructive op the floor refuses
	}
}

// demoFloor is the tiny capability floor the demo adjudicates against: it allows one
// benign op of EACH family and denies everything else (default-deny), so a reader
// sees the same kernel govern both concept-families uniformly.
func demoFloor() *kernel.Kernel {
	floor := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{
		"read_file":       true,
		"chat_completion": true,
	}})
	return kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{floor}))
}

func fusedRunTurn() []*abi.ToolCall {
	ensureFusedRunEngines()
	calls := []*abi.ToolCall{
		{Tool: "read_fixture", Engine: fusedRunClassicalEngineID, Args: inlineFusedRef([]byte(`{"fixture":"ticket-42"}`))},
		{Tool: "mini_infer", Engine: fusedRunWeightEngineID, Args: inlineFusedRef([]byte(`{"prompt":"classify ticket-42"}`))},
	}
	for _, c := range calls {
		fusedturn.Tag(c, fusedturn.ClassifyResolved(c))
	}
	return calls
}

func fusedRunFloor() *kernel.Kernel {
	floor := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{
		"read_fixture": true,
		"mini_infer":   true,
	}})
	return kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{floor}))
}

func runFusedClassify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fused classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "JSON batch of proposed ops ({\"ops\":[{\"tool\",\"engine\",\"class\"}]}); default: the built-in demo turn")
	asJSON := fs.Bool("json", false, "emit the fused summary as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	var calls []*abi.ToolCall
	if *file != "" {
		b, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintf(stderr, "fak fused classify: %v\n", err)
			return 2
		}
		var batch fusedBatchFile
		if err := json.Unmarshal(b, &batch); err != nil {
			fmt.Fprintf(stderr, "fak fused classify: parse %s: %v\n", *file, err)
			return 2
		}
		for _, s := range batch.Ops {
			calls = append(calls, specToCall(s))
		}
	} else {
		calls = demoTurn()
	}

	ft := fusedturn.Fuse(calls)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, ft.Summary(), "fak fused classify")
	}
	renderFusedTurn(stdout, ft)
	return 0
}

func runFusedExplain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fused explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the demo classification + per-op verdicts as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ft := fusedturn.Fuse(demoTurn())
	rows := ft.Adjudicate(context.Background(), demoFloor())
	gov := fusedturn.GovernedFamilies(rows)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"summary":  ft.Summary(),
			"verdicts": rows,
			"governed": governedNames(gov),
		}, "fak fused explain")
	}

	fmt.Fprintln(stdout, "fak fused — one turn spawns BOTH classical and weight-based operations, on one floor")
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "The kernel treats every op uniformly: a classical tool and a weight-based model")
	fmt.Fprintln(stdout, "are both engines behind one path (ToolCall -> adjudicate -> dispatch -> engine).")
	fmt.Fprintln(stdout, "That uniformity IS the fusion. This turn spans both families and crosses one floor:")
	fmt.Fprintln(stdout)

	renderFusedVerdicts(stdout, rows)

	fmt.Fprintf(stdout, "\nfused: %v (classical=%d weight=%d unknown=%d)\n",
		ft.Fused(), ft.Classical(), ft.Weight(), ft.Unknown())
	fmt.Fprintf(stdout, "governed families (all crossed the SAME kernel): %v\n", governedNames(gov))
	fmt.Fprintln(stdout, "\nclassify your own turn: fak fused classify --file <ops.json> --json")
	return 0
}

type fusedRunReport struct {
	Summary     fusedturn.Summary `json:"summary"`
	EngineCalls int64             `json:"engine_calls"`
	Ops         []fusedRunOp      `json:"ops"`
}

type fusedRunOp struct {
	Tool    string `json:"tool"`
	Engine  string `json:"engine"`
	Class   string `json:"class"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason,omitempty"`
	Status  string `json:"status"`
	Result  string `json:"result"`
}

func runFusedRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fused run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the submitted/reaped fused turn as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ctx := context.Background()
	ft := fusedturn.Fuse(fusedRunTurn())
	k := fusedRunFloor()
	rows := make([]fusedRunOp, 0, len(ft.Ops))
	for _, op := range ft.Ops {
		h, v := k.Submit(ctx, op.Call)
		r, err := k.Reap(ctx, h)
		if err != nil {
			fmt.Fprintf(stderr, "fak fused run: reap %s: %v\n", op.Tool, err)
			return 1
		}
		result, err := fusedRunPayload(ctx, r)
		if err != nil {
			fmt.Fprintf(stderr, "fak fused run: resolve %s result: %v\n", op.Tool, err)
			return 1
		}
		reason := abi.ReasonName(v.Reason)
		if v.Kind == abi.VerdictAllow {
			reason = ""
		}
		rows = append(rows, fusedRunOp{
			Tool:    op.Tool,
			Engine:  op.Call.Engine,
			Class:   op.Class.String(),
			Verdict: fusedVerdictName(v.Kind),
			Reason:  reason,
			Status:  fusedStatusName(r.Status),
			Result:  result,
		})
	}
	report := fusedRunReport{
		Summary:     ft.Summary(),
		EngineCalls: k.Counters().EngineCalls,
		Ops:         rows,
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak fused run")
	}
	renderFusedRun(stdout, report)
	return 0
}

func fusedRunPayload(ctx context.Context, r *abi.Result) (string, error) {
	if r == nil {
		return "", nil
	}
	if r.Payload.Kind == abi.RefInline {
		return string(r.Payload.Inline), nil
	}
	b, err := abi.ActiveResolver().Resolve(ctx, r.Payload)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func renderFusedTurn(w io.Writer, ft fusedturn.FusedTurn) {
	s := ft.Summary()
	fmt.Fprintf(w, "fused turn — %d op(s): classical=%d weight=%d unknown=%d  fused=%v\n\n",
		s.Ops, s.Classical, s.Weight, s.Unknown, s.Fused)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  #\tTOOL\tFAMILY")
	for i, o := range ft.Ops {
		fmt.Fprintf(tw, "  %d\t%s\t%s\n", i, o.Tool, o.Class)
	}
	_ = tw.Flush()
	if !s.Fused {
		fmt.Fprintln(w, "\nnot a fused turn — it does not span both families (a normal turn)")
	}
}

func renderFusedVerdicts(w io.Writer, rows []fusedturn.AdjudicatedOp) {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  #\tTOOL\tFAMILY\tVERDICT\tREASON")
	for i, r := range rows {
		reason := r.Reason
		if r.Kind == "allow" {
			reason = "-"
		}
		fmt.Fprintf(tw, "  %d\t%s\t%s\t%s\t%s\n", i, r.Tool, r.Class, r.Kind, reason)
	}
	_ = tw.Flush()
}

func renderFusedRun(w io.Writer, report fusedRunReport) {
	s := report.Summary
	fmt.Fprintln(w, "fak fused run - Submit -> Reap -> Engine.Complete for one fused turn")
	fmt.Fprintf(w, "fused: %v (classical=%d weight=%d unknown=%d engine_calls=%d)\n\n",
		s.Fused, s.Classical, s.Weight, s.Unknown, report.EngineCalls)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  #\tTOOL\tENGINE\tFAMILY\tVERDICT\tSTATUS\tRESULT")
	for i, r := range report.Ops {
		fmt.Fprintf(tw, "  %d\t%s\t%s\t%s\t%s\t%s\t%s\n",
			i, r.Tool, r.Engine, r.Class, r.Verdict, r.Status, r.Result)
	}
	_ = tw.Flush()
}

func governedNames(gov []fusedturn.OpClass) []string {
	out := make([]string, len(gov))
	for i, c := range gov {
		out[i] = c.String()
	}
	return out
}

func fusedVerdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "allow"
	case abi.VerdictDeny:
		return "deny"
	case abi.VerdictDefer:
		return "defer"
	case abi.VerdictTransform:
		return "transform"
	case abi.VerdictQuarantine:
		return "quarantine"
	case abi.VerdictRequireWitness:
		return "require-witness"
	case abi.VerdictIndeterminate:
		return "indeterminate"
	default:
		return "other"
	}
}

func fusedStatusName(s abi.Status) string {
	switch s {
	case abi.StatusOK:
		return "ok"
	case abi.StatusError:
		return "error"
	case abi.StatusPending:
		return "pending"
	default:
		return "unknown"
	}
}

func fusedUsage(w io.Writer) {
	fmt.Fprint(w, `fak fused - one turn spawns BOTH classical and weight-based operations, on one floor

  fak fused explain [--json]             the concept + a built-in fused demo turn,
                                         classified and adjudicated through a REAL kernel
  fak fused classify [--file f] [--json] classify a proposed turn's ops into their
                                         concept-families and fold the fused summary
  fak fused run [--json]                 submit and reap a no-key fused turn through
                                         two registered EngineDriver routes

A CLASSICAL op is a deterministic-effect tool call (bash, git commit, a lease, a verify);
a WEIGHT-based op is a model forward (an inference, an ensemble member, an expert dispatch).
The fused agent kernel routes BOTH through one default-deny floor, so a single turn can
interleave them. "classify" names each op's family and reports whether the turn is FUSED
(spans both); "explain" also adjudicates the demo turn so you see one kernel govern both;
"run" proves dispatch by returning real Engine.Complete payloads for both families.
`)
}

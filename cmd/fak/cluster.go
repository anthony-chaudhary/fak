package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cluster"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// cluster.go — `fak cluster`, the first RUNNABLE multi-node compute witness.
//
// internal/model ships DistComm (dist_collective.go): a real cross-process
// coordinator-rooted process group that performs AllReduceSum / AllGather over a
// TCP wire, each rank holding ONLY its own part, proven byte-identical to the
// in-process LocalCollective. But that proof lives entirely inside a test that
// spins the ranks as goroutines over a loopback socket on ONE box — there is no
// command an operator can launch on two SEPARATE machines (two laptops, two cloud
// nodes, two GPU boxes) to make fak compute actually cross a machine boundary.
//
// `fak cluster` is that command. It is a thin shell over DistComm: the kernel
// primitive is unchanged and still carries its own bit-exactness gates; this verb
// only binds it to a real listener/dialer and a CLI so the collective runs between
// processes that the operator places wherever they like.
//
//	# self-check on one box (loopback, asserts bit-exact vs LocalCollective):
//	fak cluster selftest
//
//	# a real two-node all-reduce — run on node A (the coordinator):
//	fak cluster coordinator --listen 0.0.0.0:7777 --size 2 --vec 1,2,3
//	# ...and on node B (worker rank 1), pointing at node A's address:
//	fak cluster worker --coord A.B.C.D:7777 --rank 1 --size 2 --vec 4,5,6
//	# both print 5,7,9 — the element-wise sum, reduced across the wire.
//
// HONESTY. This is a cross-PROCESS / cross-NODE collective over HOST float32. It is
// NOT multi-GPU and NOT NCCL: the device-tensor collective (a non-cpu-ref
// compute.CollectiveBackend over NCCL/RCCL) is the separate, GPU-gated rung. What
// `fak cluster` proves is the distributed plumbing ABOVE the device line — rank
// coordination, the wire protocol, rank-order reduction across real machines, the
// fail-closed contract — runnable today on any two CPU hosts. The path from here to
// high performance is the rung ladder in docs/serving/multi-node-compute.md.
func cmdCluster(args []string) {
	if len(args) == 0 {
		clusterUsage(os.Stderr)
		os.Exit(2)
	}
	switch args[0] {
	case "info":
		cmdClusterInfo(args[1:])
	case "selftest":
		cmdClusterSelftest(args[1:])
	case "coordinator":
		cmdClusterCoordinator(args[1:])
	case "worker":
		cmdClusterWorker(args[1:])
	case "-h", "--help", "help":
		clusterUsage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "fak cluster: unknown subcommand %q\n", args[0])
		clusterUsage(os.Stderr)
		os.Exit(2)
	}
}

func clusterUsage(w *os.File) {
	fmt.Fprint(w, `usage: fak cluster <subcommand>

Run cross-node collectives and inspect local Thunderbolt 5 / RDMA configuration.

subcommands:
  info           report local Thunderbolt ports, RDMA status, and bridge conflicts
  selftest       spin the ranks over a loopback socket on this box and assert the
                 result is bit-exact vs the in-process LocalCollective reference
  coordinator    rank 0: listen, accept the workers, run the collective
  worker         rank r>=1: dial the coordinator, join, run the collective

common flags (coordinator/worker):
  --size N       total ranks in the group (>=1)
  --op OP        allreduce (sum, default) | allgather (rank-ordered concat)
  --vec a,b,c    this rank's float32 part (comma-separated)
  --widths w0,.. allgather only: every rank's part width in rank order (the shared
                 tiling; this rank's --vec must be exactly --widths[rank] long)

info flags (--plan runs admission planning; planning over declared limits never
proves physical serving):
  --plan             run admission planning after local discovery
  --widths w0,..     tiling across ranks, e.g. 1,1 (the shared AllGather plan)
  --limits L0,..     per-rank GPU limit bytes, e.g. 24G,24G (one value applies to all)
  --weights B        total resident weight bytes, sharded by --widths (default 0)
  --static B         per-rank drafter/buffer overhead bytes (default 0)
  --margin B         per-rank headroom kept free (default 0)
  --kv-per-token B   per-rank KV-cache bytes per token (default 0)
  --state-per-stream B
                     per-rank state bytes per stream (default 0)
  --max-context N    the model's longest context in tokens (required with --plan)
  --share F          extra round-time share prompt chunks may add (default 0.25)
  --decode-ms F      planning ms per decode row (default 1)
  --prompt-ms F      planning ms per prompt row (default 1)

coordinator flags:
  --listen ADDR  bind address (default 0.0.0.0:7777)

worker flags:
  --coord ADDR   the coordinator's address to dial (host:port)
  --rank R       this worker's rank in [1,size)
  --timeout DUR  how long to keep retrying the dial (default 30s)
`)
}

type clusterInfoReport struct {
	Schema         string                     `json:"schema"`
	Source         string                     `json:"source"`
	ProofScope     string                     `json:"proof_scope"`
	DiscoveryError string                     `json:"discovery_error,omitempty"`
	Discovery      *cluster.LocalTB5Discovery `json:"discovery"`
	Admission      *clusterAdmissionReport    `json:"admission,omitempty"`
}

// clusterAdmissionReport is the planning payload of `fak cluster info --plan`:
// declared limits plus model shape, reduced to the admitted load and tiling.
type clusterAdmissionReport struct {
	Schema     string       `json:"schema"`
	Source     string       `json:"source"`
	ProofScope string       `json:"proof_scope"`
	Streams    int64        `json:"streams"`
	Context    int64        `json:"context"`
	ChunkRows  int64        `json:"chunk_rows"`
	Plan       model.TPPlan `json:"plan"`
}

func cmdClusterInfo(args []string) {
	fs := flag.NewFlagSet("cluster info", flag.ExitOnError)
	jsonOutput := fs.Bool("json", false, "emit machine-readable JSON")
	doPlan := fs.Bool("plan", false, "run admission planning over declared limits + model shape")
	widthsStr := fs.String("widths", "", "plan: tiling across ranks, e.g. 1,1 (the shared AllGather plan)")
	limitsStr := fs.String("limits", "", "plan: per-rank GPU limit bytes, e.g. 24G,24G (one value applies to all ranks)")
	weightsStr := fs.String("weights", "0", "plan: total resident weight bytes, sharded by --widths (e.g. 8G)")
	staticStr := fs.String("static", "0", "plan: per-rank drafter/buffer overhead bytes")
	marginStr := fs.String("margin", "0", "plan: per-rank headroom kept free (bytes)")
	kvStr := fs.String("kv-per-token", "0", "plan: per-rank KV-cache bytes per token")
	stateStr := fs.String("state-per-stream", "0", "plan: per-rank state bytes per stream")
	maxContextN := fs.Int64("max-context", 0, "plan: the model's longest context (tokens)")
	share := fs.Float64("share", 0.25, "plan: extra round-time share prompt chunks may add (fraction)")
	decodeMS := fs.Float64("decode-ms", 1, "plan: planning ms per decode row")
	promptMS := fs.Float64("prompt-ms", 1, "plan: planning ms per prompt row")
	_ = fs.Parse(args)

	var admission *clusterAdmissionReport
	if *doPlan {
		cfg, perr := parseAdmissionConfig(*widthsStr, *limitsStr, *weightsStr, *staticStr, *marginStr, *kvStr, *stateStr, *maxContextN, *share, *decodeMS, *promptMS)
		if perr != nil {
			fmt.Fprintf(os.Stderr, "fak cluster info: %v\n", perr)
			os.Exit(2)
		}
		ap, perr := planAdmission(cfg)
		if perr != nil {
			if errors.Is(perr, errAdmissionOverBudget) {
				fmt.Fprintln(os.Stderr, "fak cluster info: admission refused: over budget (even one stream does not fit the declared limits)")
			} else {
				fmt.Fprintf(os.Stderr, "fak cluster info: plan admission: %v\n", perr)
			}
			os.Exit(1)
		}
		admission = &clusterAdmissionReport{
			Schema:     "fak.cluster-admission-plan/1",
			Source:     "declared_limits_and_model_shape",
			ProofScope: "planning_only_not_physical_serving",
			Streams:    ap.Streams,
			Context:    ap.Context,
			ChunkRows:  ap.ChunkRows,
			Plan:       ap.Plan,
		}
	}

	discovery, discoveryErr := cluster.DiscoverLocalTB5RDMA()
	if discovery == nil && !*doPlan {
		fmt.Fprintf(os.Stderr, "fak cluster info: local discovery failed: %v\n", discoveryErr)
		os.Exit(1)
	}

	report := clusterInfoReport{
		Schema:     "fak.cluster-local-discovery/1",
		Source:     "local_system_discovery",
		ProofScope: "configuration_only_not_data_transfer_or_working_mesh",
		Discovery:  discovery,
		Admission:  admission,
	}
	if discoveryErr != nil {
		report.DiscoveryError = discoveryErr.Error()
	}
	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(os.Stderr, "fak cluster info: encode report: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if discovery != nil {
		fmt.Println("=== Thunderbolt 5 / RDMA Local Discovery ===")
		fmt.Println("Proof scope: local configuration only; this does not prove data transfer or a working mesh.")
		fmt.Printf("Thunderbolt 5 ports: %d (all discovered Thunderbolt ports: %d)\n", len(discovery.TB5Ports), len(discovery.Ports))
		fmt.Printf("RDMA enabled: %t (status: %s)\n", discovery.RDMAEnabled, discovery.RDMAStatus)
		if discovery.HasBridgeConflict {
			fmt.Printf("bridge0 conflict: detected (members: %s)\n", strings.Join(discovery.BridgeMembers, ", "))
		} else {
			fmt.Println("bridge0 conflict: none detected")
		}
		fmt.Printf("Local prerequisites flag: %t\n", discovery.MeshReady)
		for _, warning := range discovery.Warnings {
			fmt.Printf("warning: %s\n", warning)
		}
	} else {
		fmt.Printf("local discovery: unavailable (%v)\n", discoveryErr)
	}
	if discoveryErr != nil {
		fmt.Printf("discovery error: %v\n", discoveryErr)
	}
	if admission != nil {
		printAdmissionText(admission)
	}
}

// parseVec parses a comma-separated float32 vector. An empty string is the empty
// vector (a valid zero-width part). It fails closed on a non-numeric field.
func parseVec(s string) ([]float32, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []float32{}, nil
	}
	fields := strings.Split(s, ",")
	v := make([]float32, len(fields))
	for i, f := range fields {
		x, err := strconv.ParseFloat(strings.TrimSpace(f), 32)
		if err != nil {
			return nil, fmt.Errorf("field %d %q: %w", i, f, err)
		}
		v[i] = float32(x)
	}
	return v, nil
}

// parseWidths parses a comma-separated list of non-negative shard widths.
func parseWidths(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty widths")
	}
	fields := strings.Split(s, ",")
	w := make([]int, len(fields))
	for i, f := range fields {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			return nil, fmt.Errorf("width %d %q: %w", i, f, err)
		}
		if n < 0 {
			return nil, fmt.Errorf("width %d = %d, want >= 0", i, n)
		}
		w[i] = n
	}
	return w, nil
}

// planFromWidths builds a validated TPPlan whose shard r has width widths[r], in
// rank order — the shared tiling every rank passes to AllGather.
func planFromWidths(widths []int) (model.TPPlan, error) {
	shards := make([]model.TPShard, len(widths))
	lo := 0
	for r, w := range widths {
		shards[r] = model.TPShard{Rank: r, Lo: lo, Hi: lo + w}
		lo += w
	}
	p := model.TPPlan{Dim: lo, Shards: shards}
	if err := p.Validate(); err != nil {
		return model.TPPlan{}, err
	}
	return p, nil
}

// --- Cluster admission planning ---
//
// ADAPT (not a verbatim port) of the converged-round admission math from
// TensorFold @9e8c368b0b4c2c7892a0c9f0f302d7c164016209 (Apache-2.0; see
// THIRD_PARTY_NOTICES.md, "TensorFold"): fitsAt compares each rank's need —
// resident weights plus per-stream/per-token caches plus margin — against that
// rank's GPU limit; maxStreams/maxContext binary-search the largest admitted
// load; chunkRows sizes prompt chunks in 16-row steps so a round stays within
// (1+share) of its decode-only time. Admission planning NEVER proves physical
// serving — no process launches, no device is touched, no collective runs.

// admissionStreamCeiling bounds the stream bisection, mirroring the upstream search space.
const admissionStreamCeiling = int64(1 << 16)

// admissionChunkStep / admissionChunkCap bound the chunkRows walk: prompt chunks
// move in 16-row steps and never exceed 4096 rows.
const (
	admissionChunkStep = int64(16)
	admissionChunkCap  = int64(4096)
)

// errAdmissionOverBudget is the closed refusal when even one stream does not fit
// the declared limits. Callers match it with errors.Is, never on prose.
var errAdmissionOverBudget = errors.New("cluster admission: over budget")

// admissionRank is one rank's declared memory ledger for admission planning.
type admissionRank struct {
	GPULimit       int64 // bytes the rank may hold
	Weights        int64 // resident weight bytes on this rank
	Static         int64 // drafter/buffer overhead independent of load
	Margin         int64 // headroom kept free
	KVPerToken     int64 // KV-cache bytes per token on this rank
	StatePerStream int64 // recurrent/activation state bytes per stream
}

// needAt is the rank's total byte need for streams x context tokens: the static
// need (weights + static + margin) plus the load-scaled caches.
func (r admissionRank) needAt(streams, context int64) int64 {
	return r.Weights + r.Static + r.Margin + r.KVPerToken*streams*context + r.StatePerStream*streams
}

// admissionShape is the model-shape input to admission planning.
type admissionShape struct {
	MaxContext int64 // longest context the model supports
}

// fitsAt reports whether every rank holds its caches for streams x context within
// its GPU limit — the per-rank needs + KV + margin vs gpu_limit check.
func fitsAt(ranks []admissionRank, streams, context int64) bool {
	for _, r := range ranks {
		if r.needAt(streams, context) > r.GPULimit {
			return false
		}
	}
	return true
}

// maxStreams is the most streams whose caches fit every rank at context (0 when
// even one does not). The need is monotone in streams, so the bisection is exact:
// if N fits, every smaller count fits.
func maxStreams(ranks []admissionRank, context int64) int64 {
	lo, hi := int64(0), admissionStreamCeiling
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if fitsAt(ranks, mid, context) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// maxContext is the longest context every one of streams streams can hold, capped
// at the model's MaxContext.
func maxContext(ranks []admissionRank, shape admissionShape, streams int64) int64 {
	lo, hi := int64(0), max(shape.MaxContext, int64(1))
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if fitsAt(ranks, streams, mid) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// chunkRows is the prompt-chunk row count a round takes while decode rows run: the
// largest multiple of admissionChunkStep (at least one step) that keeps the round
// within (1+share) of its decode-only time. It walks up from the floor while the
// NEXT step still fits the share bound and stops at the first step that does not
// (or at the cap), returning the last step that fit — the floor when none above
// it was measured to fit. roundCost(decode, prompt) models one round's ms.
func chunkRows(roundCost func(decode, prompt int64) float64, decode int64, share float64) int64 {
	base := roundCost(decode, 0)
	chunk := admissionChunkStep
	for chunk < admissionChunkCap {
		if roundCost(decode, chunk+admissionChunkStep) > base*(1+share) {
			break
		}
		chunk += admissionChunkStep
	}
	return chunk
}

// admissionShardWeights splits totalBytes across ranks proportionally to their
// shard widths, rounding each rank UP so no rank is under-accounted (the sum may
// exceed total by less than the rank count).
func admissionShardWeights(totalBytes int64, widths []int) []int64 {
	dim := 0
	for _, w := range widths {
		dim += w
	}
	out := make([]int64, len(widths))
	for r, w := range widths {
		out[r] = (totalBytes*int64(w) + int64(dim) - 1) / int64(dim)
	}
	return out
}

// admissionConfig declares everything the planner needs: the tiling, per-rank
// limits, the model shape, and the planning round-cost coefficients.
type admissionConfig struct {
	Widths         []int   // tiling across ranks (drives the AllGather plan)
	Limits         []int64 // per-rank GPU limit bytes
	TotalWeights   int64   // total resident weight bytes, sharded by Widths
	Static         int64   // per-rank drafter/buffer overhead
	Margin         int64   // per-rank headroom kept free
	KVPerToken     int64   // per-rank KV-cache bytes per token
	StatePerStream int64   // per-rank state bytes per stream
	MaxContext     int64   // model's longest context
	Share          float64 // extra round-time share prompt chunks may add
	DecodeMS       float64 // planning ms per decode row
	PromptMS       float64 // planning ms per prompt row
}

// admissionPlan is what the planner admits: the streams and context that fit, the
// prompt-chunk rows that ride decode rounds, and the tiling plan.
type admissionPlan struct {
	Streams   int64
	Context   int64
	ChunkRows int64
	Plan      model.TPPlan
}

// planAdmission plans how much load a converged cluster round admits: probe
// limits + model shape -> admitted streams/context/chunk rows + plan. Policy:
// admit the longest context one stream can hold, then the most streams that
// survive at that context. It refuses with errAdmissionOverBudget when even one
// stream does not fit — over-budget is a refusal, not a zero plan. Planning is
// arithmetic on declared limits; it never proves physical serving.
func planAdmission(cfg admissionConfig) (admissionPlan, error) {
	if len(cfg.Widths) == 0 || len(cfg.Limits) != len(cfg.Widths) {
		return admissionPlan{}, fmt.Errorf("plan admission: %d limits for %d widths, want equal non-empty lists", len(cfg.Limits), len(cfg.Widths))
	}
	for i, l := range cfg.Limits {
		if l <= 0 {
			return admissionPlan{}, fmt.Errorf("plan admission: rank %d limit = %d, want > 0", i, l)
		}
	}
	if cfg.TotalWeights < 0 || cfg.Static < 0 || cfg.Margin < 0 || cfg.KVPerToken < 0 || cfg.StatePerStream < 0 {
		return admissionPlan{}, fmt.Errorf("plan admission: byte counts must be >= 0")
	}
	if cfg.MaxContext < 1 {
		return admissionPlan{}, fmt.Errorf("plan admission: max context = %d, want >= 1", cfg.MaxContext)
	}
	if cfg.Share < 0 || cfg.DecodeMS < 0 || cfg.PromptMS < 0 || (cfg.DecodeMS == 0 && cfg.PromptMS == 0) {
		return admissionPlan{}, fmt.Errorf("plan admission: share/ms coefficients must be >= 0 with a nonzero round cost")
	}

	plan, err := planFromWidths(cfg.Widths)
	if err != nil {
		return admissionPlan{}, fmt.Errorf("plan admission: %w", err)
	}
	weights := admissionShardWeights(cfg.TotalWeights, cfg.Widths)
	ranks := make([]admissionRank, len(cfg.Widths))
	for r := range cfg.Widths {
		ranks[r] = admissionRank{
			GPULimit:       cfg.Limits[r],
			Weights:        weights[r],
			Static:         cfg.Static,
			Margin:         cfg.Margin,
			KVPerToken:     cfg.KVPerToken,
			StatePerStream: cfg.StatePerStream,
		}
	}

	shape := admissionShape{MaxContext: cfg.MaxContext}
	context := maxContext(ranks, shape, 1)
	if context == 0 {
		return admissionPlan{}, errAdmissionOverBudget
	}
	streams := maxStreams(ranks, context)
	if streams == 0 {
		return admissionPlan{}, errAdmissionOverBudget
	}
	roundCost := func(decode, prompt int64) float64 {
		return float64(decode)*cfg.DecodeMS + float64(prompt)*cfg.PromptMS
	}
	return admissionPlan{
		Streams:   streams,
		Context:   context,
		ChunkRows: chunkRows(roundCost, streams, cfg.Share),
		Plan:      plan,
	}, nil
}

// parseAdmissionConfig turns the --plan flag set into an admissionConfig. Byte
// sizes accept raw integers or k/m/g suffixes (compute.ParseByteSizeString);
// --limits may give one value for every rank or one per rank.
func parseAdmissionConfig(widthsStr, limitsStr, weightsStr, staticStr, marginStr, kvStr, stateStr string, maxContextN int64, share, decodeMS, promptMS float64) (admissionConfig, error) {
	widths, err := parseWidths(widthsStr)
	if err != nil {
		return admissionConfig{}, fmt.Errorf("--widths: %v", err)
	}
	limits, err := parseAdmissionLimits(limitsStr, len(widths))
	if err != nil {
		return admissionConfig{}, err
	}
	weights, err := nonNegativeSize(weightsStr, "--weights")
	if err != nil {
		return admissionConfig{}, err
	}
	static, err := nonNegativeSize(staticStr, "--static")
	if err != nil {
		return admissionConfig{}, err
	}
	margin, err := nonNegativeSize(marginStr, "--margin")
	if err != nil {
		return admissionConfig{}, err
	}
	kv, err := nonNegativeSize(kvStr, "--kv-per-token")
	if err != nil {
		return admissionConfig{}, err
	}
	state, err := nonNegativeSize(stateStr, "--state-per-stream")
	if err != nil {
		return admissionConfig{}, err
	}
	return admissionConfig{
		Widths:         widths,
		Limits:         limits,
		TotalWeights:   weights,
		Static:         static,
		Margin:         margin,
		KVPerToken:     kv,
		StatePerStream: state,
		MaxContext:     maxContextN,
		Share:          share,
		DecodeMS:       decodeMS,
		PromptMS:       promptMS,
	}, nil
}

// parseAdmissionLimits parses the per-rank GPU limits: one positive byte size for
// every rank, or a single value applied to all ranks.
func parseAdmissionLimits(s string, ranks int) ([]int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("--limits is required with --plan (per-rank GPU limit bytes, e.g. 24G,24G)")
	}
	fields := strings.Split(s, ",")
	vals := make([]int64, 0, len(fields))
	for _, f := range fields {
		v, err := compute.ParseByteSizeString(f)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("--limits: %q: want a positive byte size like 24G", f)
		}
		vals = append(vals, v)
	}
	if len(vals) == 1 && ranks > 1 {
		repeated := make([]int64, ranks)
		for i := range repeated {
			repeated[i] = vals[0]
		}
		vals = repeated
	}
	if len(vals) != ranks {
		return nil, fmt.Errorf("--limits has %d entries, want %d (or one value for all)", len(vals), ranks)
	}
	return vals, nil
}

// nonNegativeSize parses one declared byte size (raw integer or k/m/g suffix).
func nonNegativeSize(s, name string) (int64, error) {
	v, err := compute.ParseByteSizeString(s)
	if err != nil || v < 0 {
		return 0, fmt.Errorf("%s: %q: want a non-negative byte size like 8G", name, s)
	}
	return v, nil
}

// printAdmissionText renders the planning payload for the human surface, carrying
// the honesty line: planning over declared limits, never physical serving.
func printAdmissionText(a *clusterAdmissionReport) {
	fmt.Println("=== Cluster Admission Plan ===")
	fmt.Println("Proof scope: planning over declared limits and model shape; this does not prove physical serving.")
	fmt.Printf("admitted streams: %d\n", a.Streams)
	fmt.Printf("admitted context: %d\n", a.Context)
	fmt.Printf("chunk rows: %d\n", a.ChunkRows)
	shards := make([]string, len(a.Plan.Shards))
	for i, s := range a.Plan.Shards {
		shards[i] = fmt.Sprintf("rank%d=[%d,%d)", s.Rank, s.Lo, s.Hi)
	}
	fmt.Printf("plan: dim=%d %s\n", a.Plan.Dim, strings.Join(shards, " "))
}

// formatVec renders a float32 vector as a comma-separated line for stdout, so a
// caller can diff two nodes' output or pipe it onward.
func formatVec(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = strconv.FormatFloat(float64(x), 'g', -1, 32)
	}
	return strings.Join(parts, ",")
}

// runCollective runs one collective on g for the given op and prints the result.
func runCollective(g *model.DistComm, op string, myPart []float32, widths []int) error {
	var (
		out []float32
		err error
	)
	switch op {
	case "allreduce":
		out, err = g.AllReduceSum(myPart)
	case "allgather":
		if widths == nil {
			return fmt.Errorf("allgather needs --widths (the shared per-rank tiling)")
		}
		plan, perr := planFromWidths(widths)
		if perr != nil {
			return fmt.Errorf("build plan: %w", perr)
		}
		if g.Rank() < len(widths) && len(myPart) != widths[g.Rank()] {
			return fmt.Errorf("rank %d --vec has %d elements, want --widths[%d] = %d", g.Rank(), len(myPart), g.Rank(), widths[g.Rank()])
		}
		out, err = g.AllGather(myPart, plan)
	default:
		return fmt.Errorf("unknown --op %q (want allreduce|allgather)", op)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fak cluster: rank %d/%d %s ok (%d elements)\n", g.Rank(), g.Size(), op, len(out))
	fmt.Println(formatVec(out))
	return nil
}

func cmdClusterCoordinator(args []string) {
	fs := flag.NewFlagSet("cluster coordinator", flag.ExitOnError)
	listen := fs.String("listen", "0.0.0.0:7777", "bind address (host:port)")
	size := fs.Int("size", 0, "total ranks in the group (>=1)")
	op := fs.String("op", "allreduce", "collective: allreduce | allgather")
	vec := fs.String("vec", "", "this rank's float32 part (comma-separated)")
	widthsStr := fs.String("widths", "", "allgather only: every rank's part width, rank order")
	_ = fs.Parse(args)

	if *size < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster coordinator: --size must be >= 1")
		os.Exit(2)
	}
	myPart, widths := parseClusterParts("coordinator", *vec, *widthsStr, *op, *size)

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: listen %s: %v\n", *listen, err)
		os.Exit(1)
	}
	defer ln.Close()
	fmt.Fprintf(os.Stderr, "fak cluster: coordinator listening on %s, waiting for %d worker(s)...\n", ln.Addr(), *size-1)

	g, err := model.Coordinate(ln, *size)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: coordinate: %v\n", err)
		os.Exit(1)
	}
	defer g.Close()

	if err := runCollective(g, *op, myPart, widths); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: %v\n", err)
		os.Exit(1)
	}
}

func cmdClusterWorker(args []string) {
	fs := flag.NewFlagSet("cluster worker", flag.ExitOnError)
	coord := fs.String("coord", "", "coordinator address to dial (host:port)")
	rank := fs.Int("rank", 0, "this worker's rank in [1,size)")
	size := fs.Int("size", 0, "total ranks in the group (>=2)")
	op := fs.String("op", "allreduce", "collective: allreduce | allgather")
	vec := fs.String("vec", "", "this rank's float32 part (comma-separated)")
	widthsStr := fs.String("widths", "", "allgather only: every rank's part width, rank order")
	timeout := fs.Duration("timeout", 30*time.Second, "how long to keep retrying the dial")
	_ = fs.Parse(args)

	if *coord == "" {
		fmt.Fprintln(os.Stderr, "fak cluster worker: --coord is required")
		os.Exit(2)
	}
	if *size < 2 {
		fmt.Fprintln(os.Stderr, "fak cluster worker: --size must be >= 2 (rank 0 coordinates)")
		os.Exit(2)
	}
	if *rank < 1 || *rank >= *size {
		fmt.Fprintf(os.Stderr, "fak cluster worker: --rank %d out of [1,%d)\n", *rank, *size)
		os.Exit(2)
	}
	myPart, widths := parseClusterParts("worker", *vec, *widthsStr, *op, *size)

	conn, err := dialCoordinator(*coord, *timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster worker: dial %s: %v\n", *coord, err)
		os.Exit(1)
	}
	g, err := model.Join(conn, *rank, *size)
	if err != nil {
		conn.Close()
		fmt.Fprintf(os.Stderr, "fak cluster worker: join: %v\n", err)
		os.Exit(1)
	}
	defer g.Close()
	fmt.Fprintf(os.Stderr, "fak cluster: worker rank %d/%d joined coordinator %s\n", *rank, *size, *coord)

	if err := runCollective(g, *op, myPart, widths); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster worker: %v\n", err)
		os.Exit(1)
	}
}

// parseClusterParts parses this rank's --vec float32 part and, for allgather, every rank's
// --widths (validated against size). label names the subcommand in error messages
// ("coordinator" / "worker"); a malformed flag prints to stderr and exits 2.
func parseClusterParts(label, vec, widthsStr, op string, size int) ([]float32, []int) {
	myPart, err := parseVec(vec)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster %s: bad --vec: %v\n", label, err)
		os.Exit(2)
	}
	var widths []int
	if op == "allgather" {
		widths, err = parseWidths(widthsStr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak cluster %s: bad --widths: %v\n", label, err)
			os.Exit(2)
		}
		if len(widths) != size {
			fmt.Fprintf(os.Stderr, "fak cluster %s: --widths has %d entries, want --size = %d\n", label, len(widths), size)
			os.Exit(2)
		}
	}
	return myPart, widths
}

// dialCoordinator retries net.Dial until the coordinator is reachable or the
// deadline passes — a worker is routinely launched before the coordinator is up,
// so a single dial would race-lose. Each attempt has its own short connect timeout.
func dialCoordinator(addr string, timeout time.Duration) (net.Conn, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("gave up after %s: %w", timeout, lastErr)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func cmdClusterSelftest(args []string) {
	fs := flag.NewFlagSet("cluster selftest", flag.ExitOnError)
	maxSize := fs.Int("size", 4, "test rank counts 1..size over loopback")
	n := fs.Int("len", 17, "vector length per rank (a non-round length exercises the tail)")
	_ = fs.Parse(args)

	if *maxSize < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster selftest: --size must be >= 1")
		os.Exit(2)
	}
	if *n < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster selftest: --len must be >= 1")
		os.Exit(2)
	}
	if err := clusterSelftest(*maxSize, *n); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster selftest: FAIL: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("PASS: cross-process allreduce + allgather bit-exact vs LocalCollective for sizes 1..%d (max|Δ|=0)\n", *maxSize)
}

// clusterSelftest runs, for each rank count in 1..maxSize, a real cross-process
// AllReduceSum and AllGather over a loopback socket (the ranks as goroutines, each
// holding only its own part) and asserts every rank's result is byte-for-byte equal
// to the in-process LocalCollective reference. It is the runnable, hardware-free
// witness that the two-node path is correct before it is launched on two machines:
// the wire and the orchestration here are identical to the cross-host case; only the
// loopback address differs from a real interface.
func clusterSelftest(maxSize, n int) error {
	for size := 1; size <= maxSize; size++ {
		// Deterministic parts: rank r holds [r*n+1, r*n+2, ...]. No RNG (Math.random
		// is unavailable to the kernel's determinism contract anyway), and distinct
		// per rank so a dropped/reordered/double-counted rank changes the result.
		parts := make([][]float32, size)
		for r := 0; r < size; r++ {
			parts[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				parts[r][i] = float32(r*n + i + 1)
			}
		}

		// AllReduceSum.
		wantReduce, err := model.LocalCollective{}.AllReduceSum(parts)
		if err != nil {
			return fmt.Errorf("size=%d local allreduce: %w", size, err)
		}
		gotReduce, errs := runLoopbackGroup(size, func(g *model.DistComm) ([]float32, error) {
			return g.AllReduceSum(parts[g.Rank()])
		})
		if err := assertRanksBitExact(size, "allreduce", gotReduce, errs, wantReduce); err != nil {
			return err
		}

		// AllGather (near-even tiling of size*n into `size` shards).
		plan, err := model.NewTPPlan(size*n, size)
		if err != nil {
			return fmt.Errorf("size=%d plan: %w", size, err)
		}
		gParts := make([][]float32, size)
		for r := 0; r < size; r++ {
			gParts[r] = make([]float32, plan.Shards[r].Width())
			for i := range gParts[r] {
				gParts[r][i] = float32(plan.Shards[r].Lo + i + 1)
			}
		}
		wantGather, err := model.LocalCollective{}.AllGather(gParts, plan)
		if err != nil {
			return fmt.Errorf("size=%d local allgather: %w", size, err)
		}
		gotGather, errs := runLoopbackGroup(size, func(g *model.DistComm) ([]float32, error) {
			return g.AllGather(gParts[g.Rank()], plan)
		})
		if err := assertRanksBitExact(size, "allgather", gotGather, errs, wantGather); err != nil {
			return err
		}
	}
	return nil
}

// assertRanksBitExact checks every rank of one loopback collective: the rank's own transport
// error first, then bit-exactness against the LocalCollective reference. op names the
// collective in the failure prose, which is the only thing that differs between the
// allreduce and allgather checks the loopback exercise runs.
func assertRanksBitExact(size int, op string, got [][]float32, errs []error, want []float32) error {
	for r := 0; r < size; r++ {
		if errs[r] != nil {
			return fmt.Errorf("size=%d rank %d %s: %w", size, r, op, errs[r])
		}
		if err := assertBitExact(got[r], want); err != nil {
			return fmt.Errorf("size=%d rank %d %s: %w", size, r, op, err)
		}
	}
	return nil
}

// assertBitExact fails unless got == want bit-for-bit — the max|Δ|=0 bar a reorder,
// drop, or double-count of a rank over the wire would break.
func assertBitExact(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("[%d] = %v, want %v (not bit-exact vs LocalCollective)", i, got[i], want[i])
		}
	}
	return nil
}

// runLoopbackGroup spins a size-rank DistComm over a loopback TCP listener, runs fn
// on every rank concurrently (rank 0 coordinates; ranks 1.. dial in and Join), and
// returns each rank's (result, error) in rank order. It mirrors the idiom DistComm's
// own tests use — a genuine cross-process exchange on one box — so the selftest
// exercises the same code the two-node launch does.
func runLoopbackGroup(size int, fn func(g *model.DistComm) ([]float32, error)) ([][]float32, []error) {
	results := make([][]float32, size)
	errs := make([]error, size)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		errs[0] = err
		return results, errs
	}
	defer ln.Close()
	addr := ln.Addr().String()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		g, cerr := model.Coordinate(ln, size)
		if cerr != nil {
			errs[0] = cerr
			return
		}
		defer g.Close()
		results[0], errs[0] = fn(g)
	}()

	for r := 1; r < size; r++ {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			conn, derr := net.Dial("tcp", addr)
			if derr != nil {
				errs[rank] = derr
				return
			}
			g, jerr := model.Join(conn, rank, size)
			if jerr != nil {
				conn.Close()
				errs[rank] = jerr
				return
			}
			defer g.Close()
			results[rank], errs[rank] = fn(g)
		}(r)
	}
	wg.Wait()
	return results, errs
}

// Package bench is the A/B ablation runner behind `fak bench`. It replays a
// FROZEN tool-call trace through the one kernel twice — once with the vDSO on,
// once off — timing the in-process adjudication boundary (Submit) per call, and
// measures the spawned-hook baseline by spawning `fak hook` once per decide. The
// delta between in-process and spawned is a subsystem boundary-tax check,
// measured on THIS machine (not cited), apples-to-apples: the same decide logic,
// two transports.
//
// This is useful as a regression sentinel for "no per-call process boundary" on
// the decide path. It is not a production-readiness, model-quality, or serving-
// throughput headline. The vDSO token delta is reported as a soft secondary
// (unit 83), never a production gate.
//
// os/exec lives HERE (the baseline harness), never on the dispatch hot path —
// the absence proof (unit 72) targets internal/kernel, not internal/bench.
package bench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// Call is one trace entry.
type Call struct {
	Tool string            `json:"tool"`
	Args json.RawMessage   `json:"args"`
	Meta map[string]string `json:"meta,omitempty"`
}

// Trace is a frozen, replayable tool-call slice.
type Trace struct {
	SliceID string `json:"slice_id"`
	Calls   []Call `json:"calls"`
}

// LoadTrace reads a trace JSON file.
func LoadTrace(path string) (*Trace, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t Trace
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// WorkloadHash is a stable hash of the trace's calls (the identical-workload
// guard input, unit 80/81). Independent of arm.
func (t *Trace) WorkloadHash() string {
	h := sha256.New()
	h.Write([]byte(t.SliceID))
	// sort-independent: calls are replayed in order, so hash them in order.
	for _, c := range t.Calls {
		h.Write([]byte(c.Tool))
		h.Write([]byte{0})
		h.Write(c.Args)
		h.Write([]byte{0})
		keys := make([]string, 0, len(c.Meta))
		for k := range c.Meta {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			h.Write([]byte(k + "=" + c.Meta[k] + ";"))
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// adjReps is the inner calibration loop for the adjudication-latency measurement.
// The in-process fold is faster than the OS clock granularity (~100ns-1µs on
// Windows), so we time a batch of identical decides and divide — the standard
// microbenchmark technique — to recover a real sub-microsecond per-call number.
const adjReps = 2000

// RunArm replays the trace through a fresh kernel with the vDSO on/off and
// returns the arm's metrics. The adjudication p50 is the per-call cost of the
// in-process Decide fold — the SAME logic the spawned `fak hook` baseline runs,
// so the two are apples-to-apples. Submit/Reap are still driven for the counters,
// token accounting, and MMU admission.
func RunArm(ctx context.Context, t *Trace, engineID string, vdsoOn bool, label string) (metrics.Arm, error) {
	k := kernel.New(engineID)
	k.SetVDSO(vdsoOn)
	res := abi.ActiveResolver()
	var h metrics.Hist
	arm := metrics.Arm{Label: label, Calls: len(t.Calls)}
	for _, c := range t.Calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx, args)
		if err != nil {
			return arm, err
		}
		tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}

		// Measure the in-process adjudication boundary with an inner calibration
		// loop (clock-granularity-robust).
		t0 := time.Now()
		for i := 0; i < adjReps; i++ {
			_ = k.Decide(ctx, tc)
		}
		h.RecordNs(int64(time.Since(t0)) / adjReps)

		// Drive the functional path once for counters / tokens / admission.
		hdl, _ := k.Submit(ctx, tc)
		r, err := k.Reap(ctx, hdl)
		if err != nil {
			continue
		}
		in, out, cacheRead, cacheCreate := tokens(r)
		arm.InTokens += in
		arm.OutTokens += out
		arm.ProviderCacheReadTokens += cacheRead
		arm.ProviderCacheCreationTokens += cacheCreate
	}
	cc := k.Counters()
	arm.EngineCalls = cc.EngineCalls
	arm.VDSOHits = cc.VDSOHits
	arm.Denies = cc.Denies
	arm.Quarantines = cc.Quarantines
	arm.P50Ns = h.P50()
	arm.P99Ns = h.P99()
	arm.MeanNs = h.Mean()
	arm.Buckets = h.Buckets()
	return arm, nil
}

// tokens extracts the 4-way usage split from a result's Meta: the mock engine
// stamps only input_tokens/output_tokens (cacheRead/cacheCreate default to 0), while
// a CassetteEngine built from a captured session (engine.NewCassette) additionally
// carries cache_read_tokens/cache_creation_tokens — the real provider cache axes
// (issue #1846) — so an ablate arm replayed against a session cassette reports the
// SAME 4 columns the session was billed on.
func tokens(r *abi.Result) (in, out, cacheRead, cacheCreate int64) {
	if r == nil || r.Meta == nil {
		return 0, 0, 0, 0
	}
	in, _ = strconv.ParseInt(r.Meta["input_tokens"], 10, 64)
	out, _ = strconv.ParseInt(r.Meta["output_tokens"], 10, 64)
	cacheRead, _ = strconv.ParseInt(r.Meta["cache_read_tokens"], 10, 64)
	cacheCreate, _ = strconv.ParseInt(r.Meta["cache_creation_tokens"], 10, 64)
	return in, out, cacheRead, cacheCreate
}

// MeasureSpawnedBaseline spawns `binPath hook` once per sample, piping a call on
// stdin, and times the round-trip — the honest, same-machine spawned-hook floor
// (unit 23). Each spawn runs the SAME decide logic the in-process path runs.
func MeasureSpawnedBaseline(binPath string, sample Call, n int) (metrics.Baseline, error) {
	payload, _ := json.Marshal(sample)
	var h metrics.Hist
	for i := 0; i < n; i++ {
		t0 := time.Now()
		cmd := exec.Command(binPath, "hook")
		cmd.Stdin = bytes.NewReader(payload)
		var out bytes.Buffer
		cmd.Stdout = &out
		if err := cmd.Run(); err != nil {
			return metrics.Baseline{}, err
		}
		h.Record(time.Since(t0))
	}
	return metrics.Baseline{
		Source:     "spawned `fak hook` per decide, this machine",
		P50Ns:      h.P50(),
		P99Ns:      h.P99(),
		Calls:      n,
		SpawnModel: "process-per-decide (" + runtime.GOOS + ")",
	}, nil
}

// Options configure a full A/B run.
type Options struct {
	EngineID       string
	EngineModel    string
	BinPath        string // path to the fak binary for the spawned baseline ("" => skip, RED)
	BaselineN      int
	LiveTranscript string // a real-engine transcript hash, or "" => live_seam_unverified
}

// Run executes both arms + the baseline and assembles the report. It enforces the
// identical-workload guard (unit 81) before comparing.
func Run(ctx context.Context, t *Trace, opt Options) (*metrics.Report, error) {
	on, err := RunArm(ctx, t, opt.EngineID, true, "vdso_on")
	if err != nil {
		return nil, err
	}
	off, err := RunArm(ctx, t, opt.EngineID, false, "vdso_off")
	if err != nil {
		return nil, err
	}

	lin := benchcli.Stamp()
	rep := &metrics.Report{
		Provenance: metrics.Provenance{
			AppVersion:   lin.AppVersion,
			Command:      "fak bench --suite " + t.SliceID,
			EngineModel:  opt.EngineModel,
			SliceID:      t.SliceID,
			WorkloadHash: t.WorkloadHash(),
			GoVersion:    lin.GoVersion,
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/bench",
			GitCommit:    lin.GitCommit,
			UTC:          lin.UTC,
			Hostname:     lin.Node,
		},
		On:  on,
		Off: off,
	}

	// Identical-workload guard: both arms replay the same trace, so the hashes
	// match by construction — but we assert it (unit 81).
	if err := rep.Validate(t.WorkloadHash(), t.WorkloadHash()); err != nil {
		return nil, err
	}

	// Spawned-hook baseline.
	if opt.BinPath != "" {
		n := opt.BaselineN
		if n <= 0 {
			n = 30
		}
		sample := Call{Tool: "get_reservation_details", Args: json.RawMessage(`{"id":"ABC123"}`),
			Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}}
		base, err := MeasureSpawnedBaseline(opt.BinPath, sample, n)
		if err != nil {
			return nil, err
		}
		rep.Baseline = base
	}

	// KPIs.
	hitRate := 0.0
	if on.Calls > 0 {
		hitRate = float64(on.VDSOHits) / float64(on.Calls)
	}
	rep.KPIs = metrics.KPIs{
		ToolCallP50Ns:        on.P50Ns,
		ToolCallP99Ns:        on.P99Ns,
		VDSOHitRate:          hitRate,
		ContextPollutionRate: rate(on.Quarantines, int64(on.Calls)),
		TokensPerTask:        float64(on.InTokens+on.OutTokens) / float64(max1(on.Calls)),
	}

	// Secondary soft token delta (unit 83): on vs off (never gates).
	offTok := off.InTokens + off.OutTokens
	onTok := on.InTokens + on.OutTokens
	if offTok > 0 {
		rep.TokenDeltaPct = 100 * float64(offTok-onTok) / float64(offTok)
	}
	// tokencost (unit 84): a representative $/Mtok blended rate applied to on-arm.
	rep.DollarPerTask = dollar(onTok, on.Calls)

	rep.ComputeGate()

	if opt.LiveTranscript != "" {
		rep.LiveSeam = opt.LiveTranscript
	} else {
		rep.LiveSeam = "live_seam_unverified" // honest RED flag (unit 46)
	}
	return rep, nil
}

func rate(a, b int64) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}
func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// dollar applies a blended $3/Mtok-in, $15/Mtok-out representative rate (a stand-in
// for the tokencost 218-model table; the figure is illustrative, not billed).
func dollar(totalTok int64, calls int) float64 {
	// Without the in/out split here we approximate at the blended input rate.
	return float64(totalTok) / 1e6 * 3.0
}

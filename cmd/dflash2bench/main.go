// Command dflash2bench compares exact committed token prefixes on one authenticated
// target. The DFlash2 side is a host-draft diagnostic, not an optimized GPU drafter.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const maxInputBytes = 1 << 20

type options struct {
	specPath, promptPath, backend, order string
	draftMode, targetPanel               string
	tokens                               int
	timeout                              time.Duration
}

type phase struct {
	Seconds      float64              `json:"seconds"`
	DeviceBefore []compute.DeviceWork `json:"device_before"`
	DeviceAfter  []compute.DeviceWork `json:"device_after"`
}

type arm struct {
	Verification             []model.DFlash2VerificationWork `json:"completed_target_verification"`
	DiagnosticOpen           phase                           `json:"diagnostic_gate_session_sidecar_open_and_verification"`
	Mode                     string                          `json:"mode"`
	InitialCacheTokens       int                             `json:"initial_cache_tokens"`
	PromptTokens             int                             `json:"prompt_tokens"`
	TokenIDs                 []int                           `json:"token_ids"`
	CommittedPositions       int                             `json:"committed_positions"`
	StopToken                *int                            `json:"stop_token,omitempty"`
	SessionOpen              phase                           `json:"session_open"`
	SidecarOpen              phase                           `json:"sidecar_open_and_full_verification"`
	Prefill                  phase                           `json:"prefill"`
	Decode                   phase                           `json:"decode_including_final_commit"`
	Finalize                 phase                           `json:"finalization"`
	RequestSeconds           float64                         `json:"request_seconds"`
	TTFTSeconds              *float64                        `json:"first_committed_token_seconds_from_request_start,omitempty"`
	Rounds                   int                             `json:"rounds"`
	DraftCalls               int                             `json:"draft_calls"`
	CommittedTargetAnchors   int                             `json:"committed_target_anchors"`
	RequestedDraftPositions  int                             `json:"requested_draft_positions"`
	ProposedDraftTokens      int                             `json:"proposed_draft_tokens"`
	VerifiedDraftTokens      int                             `json:"verified_draft_tokens"`
	CommittedDraftTokens     int                             `json:"committed_draft_tokens"`
	ProposalSeconds          float64                         `json:"draft_proposal_seconds_including_shared_target_embed_head"`
	VerifySeconds            float64                         `json:"target_verify_method_seconds"`
	PrepareSeconds           float64                         `json:"prepare_total_seconds"`
	PrepareOtherSeconds      float64                         `json:"prepare_residual_seconds"`
	CommitSeconds            float64                         `json:"commit_including_replay_and_draft_append_seconds"`
	CommittedTokensPerSecond *float64                        `json:"committed_tokens_per_decode_second,omitempty"`
	FallbackBefore           []compute.HostFallbackCount     `json:"host_fallback_before"`
	FallbackAfter            []compute.HostFallbackCount     `json:"host_fallback_after"`
	Error                    string                          `json:"error,omitempty"`
}

type report struct {
	Schema                  string                  `json:"schema"`
	Status                  string                  `json:"status"`
	StartedUTC              string                  `json:"started_utc"`
	Build                   *debug.BuildInfo        `json:"go_build_info"`
	GOOS                    string                  `json:"goos"`
	GOARCH                  string                  `json:"goarch"`
	CPUs                    int                     `json:"logical_cpus"`
	GOMAXPROCS              int                     `json:"gomaxprocs"`
	Q8Workers               int                     `json:"q8_workers"`
	Q4KWorkers              int                     `json:"q4k_workers"`
	PromptPath              string                  `json:"prompt_ids_path"`
	TimeoutSeconds          float64                 `json:"cooperative_timeout_seconds"`
	Backend                 string                  `json:"backend"`
	BackendTier             string                  `json:"backend_tier"`
	BackendClass            string                  `json:"backend_class"`
	BackendCaps             compute.Caps            `json:"backend_caps"`
	KernelProfileRegistered bool                    `json:"kernel_profile_registered"`
	Spec                    agent.DFlash2TargetSpec `json:"trusted_target_spec"`
	PromptSHA256            string                  `json:"prompt_ids_file_sha256"`
	TargetConfig            *model.Config           `json:"authenticated_target_config,omitempty"`
	TargetConfigSHA256      string                  `json:"target_config_sha256"`
	TargetEOS               int                     `json:"target_eos"`
	TargetEOSList           []int                   `json:"target_eos_list"`
	RequestedDraftMode      string                  `json:"requested_draft_mode"`
	RequestedTargetPanel    string                  `json:"requested_target_panel"`
	Order                   string                  `json:"order"`
	RequestedTokens         int                     `json:"requested_tokens"`
	TargetLoad              phase                   `json:"target_authentication_and_load"`
	TargetClose             phase                   `json:"target_retirement"`
	Arms                    []arm                   `json:"arms"`
	ExactTokenParity        bool                    `json:"exact_token_parity"`
	PreliminaryDecodeRatio  *float64                `json:"preliminary_plain_seconds_over_draft_seconds,omitempty"`
	PreliminaryRequestRatio *float64                `json:"preliminary_plain_request_seconds_over_draft_request_seconds,omitempty"`
	Notes                   []string                `json:"notes"`
	Error                   string                  `json:"error,omitempty"`
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "prepare" {
		ctx, cancelSignal := signal.NotifyContext(context.Background(), os.Interrupt)
		code := runPrepareCLI(ctx, os.Args[2:], os.Stdout, os.Stderr)
		cancelSignal()
		if code != 0 {
			os.Exit(code)
		}
		return
	}
	var o options
	flag.StringVar(&o.specPath, "target-spec", "", "trusted JSON DFlash2TargetSpec: ordered shard identities, independent digest, tokenizer/renderer IDs, sidecar path")
	flag.StringVar(&o.promptPath, "prompt-ids", "", "JSON integer array of already rendered prompt token IDs")
	flag.StringVar(&o.backend, "backend", "", "required exact registered backend name; never falls back")
	flag.StringVar(&o.order, "order", "plain-first", "plain-first or draft-first; one preliminary pair")
	flag.IntVar(&o.tokens, "tokens", 64, "maximum emitted/committed tokens per arm (1..4096)")
	flag.DurationVar(&o.timeout, "timeout", 30*time.Minute, "cooperative whole-operation timeout (does not preempt a blocked device callback)")
	flag.StringVar(&o.draftMode, "draft-mode", "host", "host; resident-mlp explicitly refuses pending physical admission")
	flag.StringVar(&o.targetPanel, "target-panel", "serial", "serial, panel4 or panel8; selected native panel never silently falls back")
	flag.Parse()
	ctx, cancelSignal := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancelSignal()
	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()
	r, err := run(ctx, o)
	if err != nil {
		r.Status = "FAILED"
		r.Error = err.Error()
	}
	if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err != nil {
		os.Exit(1)
	}
}

func readJSON(path string, dst any) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	b, readErr := io.ReadAll(io.LimitReader(f, maxInputBytes+1))
	closeErr := f.Close()
	if err = errors.Join(readErr, closeErr); err != nil {
		return nil, err
	}
	if len(b) > maxInputBytes {
		return nil, errors.New("input JSON exceeds 1 MiB")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err = d.Decode(dst); err != nil {
		return nil, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return nil, errors.New("input must contain exactly one JSON value")
	}
	return b, nil
}

func beginPhase() (time.Time, []compute.DeviceWork) {
	return time.Now(), compute.DeviceWorkSnapshot()
}
func endPhase(start time.Time, before []compute.DeviceWork) phase {
	return phase{Seconds: time.Since(start).Seconds(), DeviceBefore: before, DeviceAfter: compute.DeviceWorkSnapshot()}
}
func run(ctx context.Context, o options) (r report, err error) {
	r = report{Schema: "fak-dflash2-diagnostic/1", Status: "INCOMPLETE", StartedUTC: time.Now().UTC().Format(time.RFC3339Nano),
		GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, CPUs: runtime.NumCPU(), GOMAXPROCS: runtime.GOMAXPROCS(0), Q8Workers: model.Q8DecodeWorkers(), Q4KWorkers: model.Q4KDecodeWorkers(), PromptPath: o.promptPath, TimeoutSeconds: o.timeout.Seconds(), Order: o.order, RequestedTokens: o.tokens,
		Notes: []string{
			"One preliminary order-sensitive pair; reverse the order in a separate run before interpreting a speed difference.",
			"Fresh session/cache=0 in both arms; no warmup. Shared model/device/page caches can warm between arms.",
			"Prerendered prompt IDs; no template rendering, string-stop handling, HTTP or client-stream timing.",
			"Serial verification uses native Step/replay; explicit panel4/panel8 requires exact device capabilities and reports actual completed verification receipts. Bootstrap remains serial.",
			"DFlash2 draft is host BF16 streaming math with native target embedding/head and five 5120-wide target tap readbacks; this is not a device-optimized drafter.",
			"Each DFlash2 request opens and authenticates all 3848817896 sidecar bytes. Request TTFT and total include this cost; decode throughput excludes it.",
			"Proposed draft tokens exclude the mandatory exact target anchor. Verified and committed draft tokens are separate; no bonus token is counted.",
			"Decode includes committing every emitted token, including the final token, on both arms. EOS/stop IDs are excluded before commit.",
			"Device counters are cumulative snapshots from existing readers, not kernel timings. Missing readers mean unavailable; SubmitWaitObserved=false remains unavailable.",
			"Cancellation is cooperative. Use an external resource supervisor and hard deadline on the admitted hardware runner.",
		}}
	r.Build, _ = debug.ReadBuildInfo()
	selection, selectionErr := benchmarkSelection(o)
	if selectionErr != nil {
		return r, selectionErr
	}
	r.RequestedDraftMode, r.RequestedTargetPanel = o.draftMode, o.targetPanel
	if o.backend == "" || o.specPath == "" || o.promptPath == "" || o.tokens < 1 || o.tokens > 4096 ||
		(o.order != "plain-first" && o.order != "draft-first") || o.timeout <= 0 || o.timeout > 2*time.Hour {
		return r, errors.New("required: --backend, --target-spec, --prompt-ids; bounded tokens, order and timeout")
	}
	if _, err = readJSON(o.specPath, &r.Spec); err != nil {
		return r, fmt.Errorf("target spec: %w", err)
	}
	var prompt []int
	promptBytes, err := readJSON(o.promptPath, &prompt)
	if err != nil {
		return r, fmt.Errorf("prompt: %w", err)
	}
	sum := sha256.Sum256(promptBytes)
	r.PromptSHA256 = hex.EncodeToString(sum[:])
	if len(prompt) == 0 || len(prompt) > 8192 {
		return r, errors.New("prompt must have 1..8192 token IDs")
	}
	for _, id := range prompt {
		if id < 0 {
			return r, errors.New("negative prompt token")
		}
	}
	be, ok := compute.Lookup(o.backend)
	if !ok || be == nil || be.Name() != o.backend {
		return r, fmt.Errorf("backend %q unavailable (registered: %v)", o.backend, compute.Registered())
	}
	if !be.Caps().DeviceMemory || !be.Caps().UploadDtype {
		return r, errors.New("diagnostic requires the explicitly selected quantized device backend")
	}
	r.Backend, r.BackendTier, r.BackendClass, r.BackendCaps = be.Name(), be.Tier(), be.Class().String(), be.Caps()
	r.KernelProfileRegistered = compute.KernelProfileReaderRegistered()
	start, before := beginPhase()
	target, err := agent.NewDFlash2TargetWithOptions(ctx, r.Spec, be, selection)
	r.TargetLoad = endPhase(start, before)
	if err != nil {
		return r, err
	}
	// This standalone process is the sole backend client. The plain arm uses
	// a direct Session; the draft arm holds the target diagnostic domain lease. Keep target/readers alive
	// until every arm retires normally. Returned errors cause main to exit;
	// interrupted callbacks cannot continue to another arm. Do not force-close
	// backing. A hard external deadline also covers Goexit/background goroutines.
	m := target.Model()
	cfg := m.Cfg
	if err = recordTargetConfig(&r, cfg); err != nil {
		return r, err
	}
	for _, id := range prompt {
		if id >= cfg.VocabSize {
			return r, errors.New("prompt token exceeds authenticated vocabulary")
		}
	}
	stops := agent.StopIDs(target.Tokenizer(), cfg)
	for id := range cfg.VocabSize {
		if cfg.IsEOS(id) {
			stops[id] = true
		}
	}
	order := []string{"plain", "dflash2"}
	if o.order == "draft-first" {
		slices.Reverse(order)
	}
	for _, mode := range order {
		if err = ctx.Err(); err != nil {
			return r, err
		}
		if err = target.ValidateBinding(m, target.Tokenizer(), be, true, false); err != nil {
			return r, err
		}
		a, runErr := runArm(ctx, mode, target, m, be, prompt, stops, o.tokens, selection)
		if runErr != nil {
			a.Error = runErr.Error()
		}
		r.Arms = append(r.Arms, a)
		if runErr != nil {
			return r, runErr
		}
	}
	start, before = beginPhase()
	err = target.Close()
	r.TargetClose = endPhase(start, before)
	if err != nil {
		return r, fmt.Errorf("target retirement: %w", err)
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	a, b := r.Arms[0], r.Arms[1]
	if a.CommittedPositions != b.CommittedPositions || !slices.Equal(a.TokenIDs, b.TokenIDs) {
		return r, errors.New("exact committed token parity failed; no speed ratio")
	}
	r.ExactTokenParity = true
	if a.Mode != "plain" {
		a, b = b, a
	}
	if len(a.TokenIDs) > 0 && a.Decode.Seconds > 0 && b.Decode.Seconds > 0 {
		ratio := a.Decode.Seconds / b.Decode.Seconds
		r.PreliminaryDecodeRatio = &ratio
		ratio = a.RequestSeconds / b.RequestSeconds
		r.PreliminaryRequestRatio = &ratio
	}
	r.Status = "PRELIMINARY_PAIR_COMPLETE"
	return r, nil
}

func recordTargetConfig(r *report, cfg model.Config) error {
	digest, err := targetConfigIdentitySHA256(cfg)
	if err != nil {
		return err
	}
	r.TargetConfig = &cfg
	r.TargetConfigSHA256 = digest
	r.TargetEOS = cfg.EOSTokenID
	r.TargetEOSList = slices.Clone(cfg.EOSTokenIDs)
	return nil
}

func benchmarkSelection(o options) (agent.DFlash2TargetOptions, error) {
	var selection agent.DFlash2TargetOptions
	switch o.draftMode {
	case "", "host":
	case "resident-mlp":
		return selection, agent.ErrDFlash2ResidentAdmissionUnavailable
	default:
		return selection, errors.New("unknown --draft-mode")
	}
	switch o.targetPanel {
	case "", "serial":
	case "panel4":
		selection.Request.TargetPanel = model.DFlash2TargetPanel4
	case "panel8":
		selection.Request.TargetPanel = model.DFlash2TargetPanel8
	default:
		return selection, errors.New("unknown --target-panel")
	}
	return selection, nil
}

func runArm(ctx context.Context, mode string, target *agent.DFlash2Target, m *model.Model, be compute.Backend, prompt []int, stops map[int]bool, limit int, selection agent.DFlash2TargetOptions) (a arm, err error) {
	a = arm{Mode: mode, PromptTokens: len(prompt), TokenIDs: make([]int, 0, limit), FallbackBefore: compute.HostFallbackCounts()}
	requestStart := time.Now()
	start, before := beginPhase()
	if mode == "plain" {
		s, openErr := m.NewBackendSessionChecked(be)
		a.SessionOpen = endPhase(start, before)
		if openErr != nil {
			return a, openErr
		}
		s.Q4K = true
		a.InitialCacheTokens = s.SequencePosition()
		if a.InitialCacheTokens != 0 {
			return a, errors.New("fresh session has a nonempty cache")
		}
		err = plainBody(ctx, s, prompt, stops, limit, requestStart, &a)
		start, before = beginPhase()
		s.Close()
		a.Finalize = endPhase(start, before)
	} else {
		q, openErr := target.OpenDiagnosticRequest(ctx)
		a.DiagnosticOpen = endPhase(start, before)
		if openErr != nil {
			return a, openErr
		}
		// The owning constructor requires a fresh zero-position target; native
		// committed position is checked after bootstrap/generation before retirement.
		panelWidth := 0
		if selection.Request.TargetPanel == model.DFlash2TargetPanel4 {
			panelWidth = 4
		}
		if selection.Request.TargetPanel == model.DFlash2TargetPanel8 {
			panelWidth = 8
		}
		err = draftBody(ctx, q, prompt, stops, limit, panelWidth, requestStart, &a)
		start, before = beginPhase()
		err = errors.Join(err, q.Close())
		a.Finalize = endPhase(start, before)
	}
	a.RequestSeconds = time.Since(requestStart).Seconds()
	a.FallbackAfter = compute.HostFallbackCounts()
	if err == nil {
		if len(a.TokenIDs) > 0 && a.Decode.Seconds > 0 {
			tps := float64(len(a.TokenIDs)) / a.Decode.Seconds
			a.CommittedTokensPerSecond = &tps
		}
	}
	return a, err
}

func greedy(logits []float32, vocab int) (int, error) {
	if len(logits) != vocab {
		return 0, errors.New("incomplete target logits")
	}
	best := 0
	for i, x := range logits {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return 0, errors.New("nonfinite target logits")
		}
		if x > logits[best] {
			best = i
		}
	}
	return best, nil
}
func markFirst(a *arm, start time.Time) {
	if a.TTFTSeconds == nil && len(a.TokenIDs) > 0 {
		elapsed := time.Since(start).Seconds()
		a.TTFTSeconds = &elapsed
	}
}
func plainBody(ctx context.Context, s *model.Session, prompt []int, stops map[int]bool, limit int, requestStart time.Time, a *arm) error {
	start, before := beginPhase()
	logits := s.Prefill(prompt)
	a.Prefill = endPhase(start, before)
	if err := ctx.Err(); err != nil {
		return err
	}
	token, err := greedy(logits, s.M.Cfg.VocabSize)
	if err != nil {
		return err
	}
	start, before = beginPhase()
	for len(a.TokenIDs) < limit {
		if err = ctx.Err(); err != nil {
			return err
		}
		if stops[token] {
			id := token
			a.StopToken = &id
			break
		}
		next, host := s.StepGreedy(token)
		if err = ctx.Err(); err != nil {
			return err
		}
		a.TokenIDs = append(a.TokenIDs, token)
		markFirst(a, requestStart)
		if len(a.TokenIDs) == limit {
			break
		}
		if host != nil {
			next, err = greedy(host, s.M.Cfg.VocabSize)
			if err != nil {
				return err
			}
		}
		if next < 0 || next >= s.M.Cfg.VocabSize {
			return errors.New("device argmax outside vocabulary")
		}
		token = next
	}
	a.Decode = endPhase(start, before)
	a.CommittedPositions = s.SequencePosition()
	if a.CommittedPositions != len(prompt)+len(a.TokenIDs) {
		return errors.New("plain committed position mismatch")
	}
	return nil
}
func draftBody(ctx context.Context, q *agent.DFlash2DiagnosticRequest, prompt []int, stops map[int]bool, limit, panelWidth int, requestStart time.Time, a *arm) error {
	start, before := beginPhase()
	err := q.Bootstrap(ctx, prompt)
	a.Prefill = endPhase(start, before)
	if err != nil {
		return err
	}
	start, before = beginPhase()
	for len(a.TokenIDs) < limit {
		maximum := min(8, limit-len(a.TokenIDs))
		if panelWidth != 0 {
			maximum = min(maximum, panelWidth)
		}
		prepareStart := time.Now()
		round, timing, err := q.PrepareTimed(ctx, maximum)
		prepareSeconds := time.Since(prepareStart).Seconds()
		if err != nil {
			return err
		}
		if !timing.Complete {
			return errors.New("incomplete Prepare timing")
		}
		work := timing.Verification
		if work.Path == "" || work.Rows != 1+timing.ProposedDraftTokens || work.ConfiguredPanelWidth != panelWidth {
			return errors.New("completed verification receipt disagrees with selected boundary")
		}
		if panelWidth != 0 {
			if !work.OneOperation || work.Operations != 1 || work.DecodeSteps != 0 {
				return errors.New("selected panel did not execute one native verification operation")
			}
		} else if work.OneOperation || work.Operations != 0 || work.DecodeSteps != work.Rows {
			return errors.New("serial verification receipt disagrees with actual steps")
		}
		verified := round.Tokens()
		if len(verified) < 1 || len(verified) > maximum {
			return errors.New("invalid verified prefix")
		}
		a.Verification = append(a.Verification, timing.Verification)
		a.Rounds++
		if timing.ProposalRan {
			a.DraftCalls++
		}
		a.RequestedDraftPositions += maximum - 1
		a.ProposedDraftTokens += timing.ProposedDraftTokens
		a.VerifiedDraftTokens += len(verified) - 1
		a.ProposalSeconds += timing.Proposal.Seconds()
		a.VerifySeconds += timing.Verify.Seconds()
		a.PrepareSeconds += prepareSeconds
		a.PrepareOtherSeconds += prepareSeconds - timing.Proposal.Seconds() - timing.Verify.Seconds()
		chosen := len(verified)
		for i, id := range verified {
			if stops[id] {
				chosen = i
				stop := id
				a.StopToken = &stop
				break
			}
		}
		commitStart := time.Now()
		committed, err := round.Commit(ctx, chosen)
		a.CommitSeconds += time.Since(commitStart).Seconds()
		if err != nil {
			return err
		}
		if !slices.Equal(committed, verified[:chosen]) {
			return errors.New("committed prefix differs from prepared selection")
		}
		if err = ctx.Err(); err != nil {
			return err
		}
		a.TokenIDs = append(a.TokenIDs, committed...)
		a.CommittedDraftTokens += max(0, len(committed)-1)
		if len(committed) > 0 {
			a.CommittedTargetAnchors++
		}
		markFirst(a, requestStart)
		if a.StopToken != nil {
			break
		}
	}
	a.Decode = endPhase(start, before)
	a.CommittedPositions, err = q.CommittedPosition(ctx)
	if err != nil {
		return err
	}
	if a.CommittedPositions != len(prompt)+len(a.TokenIDs) {
		return errors.New("draft committed position mismatch")
	}
	return nil
}

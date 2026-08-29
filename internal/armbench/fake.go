package armbench

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strings"
	"time"
)

// FakeProvider is the deterministic spine provider. It exists so the whole
// runner — pairing, ordering, resume, evidence fencing, rollup — can be proven
// end to end with no network, no key, and no model, and so that the exact code
// path a live provider will take is the one already under test.
//
// Determinism is total: every number it returns is a pure function of
// (arm id, arm kind, capabilities, task id, trial, prompt hash, model snapshot).
// Two runs of the same manifest therefore produce byte-identical ledgers, which
// is what makes the resume and identity proofs checkable rather than flaky.
//
// The shape of the synthetic numbers is deliberately NOT flattering: the
// fak_passthrough arm costs a little more wall time than baseline (plumbing is
// not free), and the capability arm's saving is on OUTPUT tokens while its input
// tokens rise slightly — the exact pattern a blended token column would hide.
type FakeProvider struct {
	// SetupWallMS is charged once per non-baseline arm. Zero means the arms are
	// setup-free, which is recorded as an honest zero.
	SetupWallMS float64
	// OmitRawResponse, when non-empty, makes the provider return an empty raw
	// response for that arm id. It is the negative fixture for the
	// MISSING_RAW_EVIDENCE fence — a provider that reports usage with nothing
	// behind it.
	OmitRawResponse string
}

var (
	_ Provider        = (*FakeProvider)(nil)
	_ ArmSetup        = (*FakeProvider)(nil)
	_ ReceiptProvider = (*FakeProvider)(nil)
)

// SetupArm charges the declared one-time setup to every arm that installs a
// treatment; the untreated baseline installs nothing and pays nothing.
func (p *FakeProvider) SetupArm(_ context.Context, arm Arm) (SetupCost, error) {
	if arm.Kind == ArmBaseline || p.SetupWallMS == 0 {
		return SetupCost{Note: "no setup required for " + string(arm.Kind)}, nil
	}
	return SetupCost{
		WallMS:  p.SetupWallMS,
		Tokens:  400,
		CostUSD: 0.0004,
		Note:    "synthetic one-time install for " + arm.ID,
	}, nil
}

// Complete returns the deterministic synthetic trial.
func (p *FakeProvider) Complete(_ context.Context, req Request) (Response, error) {
	seed := hashSeed(req.ArmID, string(req.ArmKind), strings.Join(req.Capabilities, ","), req.TaskID, fmt.Sprint(req.Trial), req.PromptHash, req.Model.Snapshot)
	jitter := float64(seed%17) / 10.0

	inTok, outTok := 1200, 900
	wall := 900.0
	cache := CacheCounters{Misses: 1}
	switch req.ArmKind {
	case ArmBaseline:
		// Defaults above define the untreated control explicitly.
	case ArmUpstreamTreatment:
		// The comparator's treatment: fewer output tokens, more system prompt in.
		inTok, outTok = 1340, 520
		wall = 780
	case ArmFakPassthrough:
		// Plumbing costs something and saves nothing. Charging it here is the
		// whole point of having a passthrough arm.
		inTok, outTok = 1215, 900
		wall = 935
		cache = CacheCounters{WriteTokens: 1200, Misses: 1}
	case ArmFakCapability:
		inTok, outTok = 1260, 610
		wall = 700
		cache = CacheCounters{ReadTokens: 1100, Hits: 1}
	}
	text := fmt.Sprintf("answer(%s/%s/t%d)", req.ArmID, req.TaskID, req.Trial)

	rawReq, err := json.Marshal(req)
	if err != nil {
		return Response{}, err
	}
	resp := Response{
		RawRequest: string(rawReq),
		Text:       text,
		Usage: Usage{
			InputTokens:  inTok,
			OutputTokens: outTok,
			CostUSD:      float64(inTok)*0.000003 + float64(outTok)*0.000015,
		},
		Latency: Latency{
			WallMS:              wall + jitter,
			TTFTMS:              wall/6 + jitter,
			TTFTAvailable:       true,
			InterTokenMS:        wall / float64(outTok),
			InterTokenAvailable: true,
		},
		Cache: cache,
	}
	if p.OmitRawResponse == req.ArmID {
		return resp, nil // raw response deliberately absent — the fence must catch it
	}
	rawResp, err := json.Marshal(map[string]any{"text": text, "usage": resp.Usage, "stop_reason": "end_turn"})
	if err != nil {
		return Response{}, err
	}
	resp.RawResponse = string(rawResp)
	inputTokens := float64(resp.Usage.InputTokens)
	outputTokens := float64(resp.Usage.OutputTokens)
	costUSD := resp.Usage.CostUSD
	cacheReadTokens := float64(resp.Cache.ReadTokens)
	cacheWriteTokens := float64(resp.Cache.WriteTokens)
	cacheHits := float64(resp.Cache.Hits)
	cacheMisses := float64(resp.Cache.Misses)
	resp.Accounting, err = ReconcileAccounting([]AccountingSource{{
		Authority: AuthorityProviderAggregate,
		Artifact:  ArtifactFor("fake://raw-response/"+req.ArmID+"/"+req.TaskID+"/"+fmt.Sprint(req.Trial), rawResp),
		Coverage:  AccountingCoverage{Scope: "provider_response", Observed: 1, Expected: 1},
		Values: AccountingValues{
			InputTokens: &inputTokens, OutputTokens: &outputTokens, CostUSD: &costUSD,
			CacheReadTokens: &cacheReadTokens, CacheWriteTokens: &cacheWriteTokens,
			CacheHits: &cacheHits, CacheMisses: &cacheMisses,
		},
	}})
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

// CompleteWithReceipt gives the synthetic provider a synthetic but fully
// deterministic process envelope. Real process-backed adapters report their
// observed child timestamps and exit/reap result through the same seam.
func (p *FakeProvider) CompleteWithReceipt(ctx context.Context, req Request) (Response, LaunchReceipt, error) {
	resp, err := p.Complete(ctx, req)
	seed := hashSeed(req.ManifestIdentity, req.ArmID, req.TaskID, fmt.Sprint(req.Trial), fmt.Sprint(req.Position))
	started := time.Date(2026, 8, 13, 0, 0, 0, 0, time.UTC).Add(time.Duration(seed%86_400_000) * time.Millisecond)
	wall := resp.Latency.WallMS
	if wall < 0 {
		wall = 0
	}
	ended := started.Add(time.Duration(wall * float64(time.Millisecond)))
	exitCode := 0
	if err != nil {
		exitCode = 1
	}
	return resp, LaunchReceipt{
		StartedAt: started.Format(time.RFC3339Nano),
		EndedAt:   ended.Format(time.RFC3339Nano),
		WallMS:    wall, ExitCode: exitCode,
		Reaped: true, ReapOutcome: "synthetic_provider_returned",
	}, err
}

// FakeGrader is the deterministic spine grader. It grades on the synthetic
// answer's shape and always records its raw judgment, so the evidence fence has
// something real to check.
type FakeGrader struct {
	// FailArm, when non-empty, marks every trial of that arm as failing the
	// judge. It is the fixture that proves a token saving with a correctness
	// loss is visible in the rollup rather than absorbed into the mean.
	FailArm string
}

var _ Grader = (*FakeGrader)(nil)

// Grade returns the deterministic verdict for one completed trial.
func (g *FakeGrader) Grade(_ context.Context, req Request, resp Response) (Judgment, error) {
	pass := strings.Contains(resp.Text, req.TaskID) && req.ArmID != g.FailArm
	score := 0.0
	if pass {
		score = 1.0
	}
	raw, err := json.Marshal(map[string]any{
		"task": req.TaskID, "arm": req.ArmID, "trial": req.Trial,
		"pass": pass, "checker": "fake/contains-task-id",
	})
	if err != nil {
		return Judgment{}, err
	}
	reason := "answer names the task"
	if !pass {
		reason = "answer did not satisfy the checker"
	}
	return Judgment{Pass: pass, Score: score, Reason: reason, RawJudgment: string(raw)}, nil
}

func hashSeed(parts ...string) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, p := range parts {
		binary.LittleEndian.PutUint64(buf[:], uint64(len(p)))
		_, _ = h.Write(buf[:])
		_, _ = h.Write([]byte(p))
	}
	return h.Sum64()
}

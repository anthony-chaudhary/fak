package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// devicePanelTestBackend is an injected device-sequence backend whose numerical
// oracle is the ordinary serial model. It still mutates the live backend KV and
// recurrent tensors, so transaction tests exercise the production snapshot and
// restore ownership rather than a cacheless verifier stub.
type devicePanelTestBackend struct {
	*sequencePrefillBackend
	target               *Session
	maxWeightBufferBytes int64
	afterSequence        func()
	failCloneAt          int
}

func newDevicePanelTestBackend(m *Model) *devicePanelTestBackend {
	base := newSequencePrefillBackend(m)
	base.reference.captureTargetHidden = true
	return &devicePanelTestBackend{sequencePrefillBackend: base}
}

func (b *devicePanelTestBackend) Name() string          { return "test-device" }
func (b *devicePanelTestBackend) Qwen35GDNPath() string { return Qwen35GDNVulkanPath }
func (b *devicePanelTestBackend) Qwen35SequenceAllLogitsPath() string {
	return compute.Qwen35SequenceAllLogitsPath
}
func (b *devicePanelTestBackend) MaxWeightBufferBytes() int64 { return b.maxWeightBufferBytes }

func (b *devicePanelTestBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	if b.failCloneAt > 0 && b.cloneCalls+1 >= b.failCloneAt {
		b.cloneCalls++
		return compute.Tensor{}, errors.New("injected tensor clone failure")
	}
	return b.recordingQwen35Backend.CloneTensor(t)
}

func (b *devicePanelTestBackend) syncReference() {
	if b.target == nil || b.reference.Cache.Len() == b.target.halKV.Len() {
		return
	}
	history := make([]int, len(b.target.halLineage.ids))
	for i, id := range b.target.halLineage.ids {
		history[i] = int(id)
	}
	b.reference.Close()
	b.reference = b.model.NewSession()
	b.reference.captureTargetHidden = true
	if len(history) > 0 {
		b.reference.Prefill(history)
	}
}

func (b *devicePanelTestBackend) copyLinearState(layer int, dst compute.Qwen35SequenceState) {
	src := b.reference.Cache.linear.layers[layer]
	conv := b.Backend.Read(dst.Conv)
	recurrent := b.Backend.Read(dst.Recurrent)
	conv = conv[:0]
	for _, row := range src.conv {
		conv = append(conv, row...)
	}
	copy(b.Backend.Read(dst.Conv), conv)
	recurrent = recurrent[:0]
	for _, row := range src.recurrent {
		recurrent = append(recurrent, row...)
	}
	copy(b.Backend.Read(dst.Recurrent), recurrent)
}

func (b *devicePanelTestBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	b.calls++
	b.requests = append(b.requests, req)
	if b.err != nil {
		return compute.Qwen35SequencePrefillResult{}, b.err
	}
	b.syncReference()
	if b.reference.Cache.Len() != req.StartPos {
		return compute.Qwen35SequencePrefillResult{}, errors.New("test device reference does not match request prefix")
	}

	rows := make([]float32, 0, len(req.TokenIDs)*b.model.Cfg.VocabSize)
	var lastHidden []float32
	for _, token := range req.TokenIDs {
		rows = append(rows, b.reference.Step(token)...)
		var err error
		lastHidden, err = b.reference.TargetHiddenAt(b.reference.Cache.Len() - 1)
		if err != nil {
			return compute.Qwen35SequencePrefillResult{}, err
		}
	}

	width := req.NumKVHeads * req.HeadDim
	for token := range req.TokenIDs {
		pos := req.StartPos + token
		for layer, spec := range req.Layers {
			if spec.Linear {
				continue
			}
			lo, hi := pos*width, (pos+1)*width
			kvLayer := qwen35HALKVLayer(b.model.Cfg, layer)
			raw := compute.NewF32(b.Backend, []int{width}, b.reference.Cache.Kraw[layer][lo:hi])
			key := compute.NewF32(b.Backend, []int{width}, b.reference.Cache.K[layer][lo:hi])
			value := compute.NewF32(b.Backend, []int{width}, b.reference.Cache.V[layer][lo:hi])
			req.KV.AppendKV(kvLayer, raw, key, value, pos)
		}
	}
	for layer, spec := range req.Layers {
		if spec.Linear {
			b.copyLinearState(layer, req.States[layer])
		}
	}

	result := compute.Qwen35SequencePrefillResult{
		LastHidden: compute.NewF32(b.Backend, []int{req.Hidden}, lastHidden),
		Tokens:     len(req.TokenIDs),
	}
	if req.NeedLogits {
		last := rows[len(rows)-b.model.Cfg.VocabSize:]
		result.Logits = compute.NewF32(b.Backend, []int{b.model.Cfg.VocabSize}, last)
	}
	if req.NeedAllLogits {
		result.LogitsRows = compute.NewF32(b.Backend, []int{len(req.TokenIDs), b.model.Cfg.VocabSize}, rows)
	}
	if b.afterSequence != nil {
		b.afterSequence()
	}
	return result, nil
}

func (b *devicePanelTestBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	b.syncReference()
	layer := b.linearLayers[b.gdnCalls%len(b.linearLayers)]
	b.gdnCalls++
	xn := append([]float32(nil), b.Backend.Read(normalizedInput)...)
	out := b.reference.linearAttnStep(layer, xn, residentKernel{b.model})
	b.copyLinearState(layer, compute.Qwen35SequenceState{Conv: convState, Recurrent: recurrentState})
	return compute.NewF32(b.Backend, []int{b.model.Cfg.HiddenSize}, out), convState, recurrentState, nil
}

func newDevicePanelTestSession(t *testing.T, m *Model) (*Session, *devicePanelTestBackend) {
	t.Helper()
	backend := newDevicePanelTestBackend(m)
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	backend.target = target
	target.captureTargetHidden = false
	t.Cleanup(target.Close)
	return target, backend
}

func assertDeviceStateMatchesSerial(t *testing.T, target *Session, serial *Session, tokens []int) {
	t.Helper()
	snapshot, err := target.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	host, err := snapshot.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()

	wantKV := compute.KVHostSnapshot{
		Config: halKVConfig(serial.M.Cfg),
		Pos:    append([]int(nil), serial.Cache.pos...),
		K:      make([][]float32, serial.M.Cfg.NumLayers),
		KRaw:   make([][]float32, serial.M.Cfg.NumLayers),
		V:      make([][]float32, serial.M.Cfg.NumLayers),
	}
	for layer := 0; layer < serial.M.Cfg.NumLayers; layer++ {
		if serial.M.Cfg.isLinearAttnLayer(layer) {
			continue
		}
		compact := qwen35HALKVLayer(serial.M.Cfg, layer)
		wantKV.K[compact] = append([]float32(nil), serial.Cache.K[layer]...)
		wantKV.KRaw[compact] = append([]float32(nil), serial.Cache.Kraw[layer]...)
		wantKV.V[compact] = append([]float32(nil), serial.Cache.V[layer]...)
	}
	if !reflect.DeepEqual(host.kv, wantKV) {
		t.Fatal("device attention KV differs from serial target state")
	}
	for layer := range serial.Cache.linear.layers {
		if !serial.M.Cfg.isLinearAttnLayer(layer) {
			continue
		}
		got := host.qwen35.layers[layer]
		want := serial.Cache.linear.layers[layer]
		if !reflect.DeepEqual(got.conv.data, flattenRows(want.conv)) || !reflect.DeepEqual(got.recurrent.data, flattenRows(want.recurrent)) {
			t.Fatalf("device recurrent state differs at layer %d", layer)
		}
	}
	if _, err := target.VerifyTokenLineage(tokens); err != nil {
		t.Fatalf("device token lineage: %v", err)
	}
}

func flattenRows(rows [][]float32) []float32 {
	var out []float32
	for _, row := range rows {
		out = append(out, row...)
	}
	return out
}

func TestQwen35DeviceVerificationReturnsEveryRowAndAdvancesLiveState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	target, backend := newDevicePanelTestSession(t, m)
	serial := m.NewSession()
	serial.captureTargetHidden = true
	defer serial.Close()
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}
	boundary := target.Prefill(prefix)
	serial.Prefill(prefix)
	calls := backend.calls

	rows, receipt, err := target.verifyQwen35DevicePanel(draft, boundary)
	if err != nil {
		t.Fatal(err)
	}
	if backend.calls != calls+1 || !backend.requests[len(backend.requests)-1].NeedAllLogits {
		t.Fatalf("sequence calls=%d want=%d with all-logit request", backend.calls, calls+1)
	}
	if len(rows) != len(draft) {
		t.Fatalf("logit rows=%d want=%d", len(rows), len(draft))
	}
	for i, token := range draft {
		assertFloat32BitsEqual(t, "device verification row", rows[i], serial.Step(token))
	}
	if receipt.Path != targetVerificationQwen38DevicePanelPath || !receipt.OneOperation || receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 {
		t.Fatalf("device verification receipt=%+v", receipt)
	}
	assertDeviceStateMatchesSerial(t, target, serial, append(append([]int(nil), prefix...), draft...))
}

func TestQwen35DeviceVerificationOutputHeadAdmissionIsAtomic(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	draft := []int{3, 7, 11, 5}
	boundary := make([]float32, m.Cfg.VocabSize)

	t.Run("oversized-head-declines-before-device-mutation", func(t *testing.T) {
		target, backend := newDevicePanelTestSession(t, m)
		backend.maxWeightBufferBytes = 1
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		beforeLineageIDs := append([]uint32(nil), target.halLineage.ids...)
		beforeLineageFault := target.halLineage.fault
		calls, uploads, steps := backend.calls, len(backend.classes), target.halStep

		rows, receipt, err := target.verifyQwen35DevicePanel(draft, boundary)
		var downgrade *TargetVerificationDowngradeError
		if !errors.As(err, &downgrade) || !errors.Is(err, ErrTargetVerificationDowngrade) {
			t.Fatalf("error=%v, want typed target-verification downgrade", err)
		}
		if rows != nil || receipt.TargetVerificationOperations != 0 || receipt.OneOperation {
			t.Fatalf("declined rows=%v receipt=%+v", rows, receipt)
		}
		if backend.calls != calls || len(backend.classes) != uploads || target.halStep != steps || target.halKV.Len() != 0 {
			t.Fatalf("decline mutated backend: calls=%d/%d uploads=%d/%d steps=%d/%d kv=%d", backend.calls, calls, len(backend.classes), uploads, target.halStep, steps, target.halKV.Len())
		}
		if !reflect.DeepEqual(target.halLineage.ids, beforeLineageIDs) || target.halLineage.fault != beforeLineageFault {
			t.Fatal("decline mutated target token lineage")
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
	})

	t.Run("head-at-cap-remains-one-operation", func(t *testing.T) {
		target, backend := newDevicePanelTestSession(t, m)
		backend.maxWeightBufferBytes = int64(m.Cfg.VocabSize * m.Cfg.HiddenSize * compute.F32.Bytes())
		rows, receipt, err := target.verifyQwen35DevicePanel(draft, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != len(draft) || backend.calls != 1 || !receipt.OneOperation || receipt.TargetVerificationOperations != 1 {
			t.Fatalf("valid capped verification rows=%d calls=%d receipt=%+v", len(rows), backend.calls, receipt)
		}
	})
}

func TestVerifyGreedyDeviceDraftRejectsWrongFirstTokenFromBoundaryWithoutDeviceWork(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	target, backend := newDevicePanelTestSession(t, m)
	boundary := make([]float32, m.Cfg.VocabSize)
	boundary[7] = 10
	draft := []int{3, 5, 7, 11}
	before, err := target.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer before.Close()
	calls, uploads, reads, clones, steps := backend.calls, len(backend.classes), backend.readCalls, backend.cloneCalls, target.halStep

	got, err := target.VerifyGreedyDeviceDraft(context.Background(), draft, boundary)
	if err != nil {
		t.Fatal(err)
	}
	if got.Accepted != nil || got.Correction != 7 || !reflect.DeepEqual(got.NextLogits, boundary) || got.TargetLogits != nil || got.CommittedPrefixTokens != 0 {
		t.Fatalf("boundary rejection=%+v, want correction 7 with unchanged boundary and prefix", got)
	}
	r := got.Receipt
	if r.Path != targetVerificationBoundaryRejectPath || r.DraftTokens != len(draft) || r.AcceptedTokens != 0 || r.RejectedTokens != len(draft) || r.TargetVerificationOperations != 0 || r.TargetDecodeSteps != 0 || r.OneOperation || r.DowngradeReason != "" {
		t.Fatalf("boundary rejection receipt=%+v", r)
	}
	if backend.calls != calls || len(backend.classes) != uploads || backend.readCalls != reads || backend.cloneCalls != clones || target.halStep != steps || target.halKV.Len() != 0 {
		t.Fatalf("boundary rejection touched device: calls=%d/%d uploads=%d/%d reads=%d/%d clones=%d/%d steps=%d/%d kv=%d", backend.calls, calls, len(backend.classes), uploads, backend.readCalls, reads, backend.cloneCalls, clones, target.halStep, steps, target.halKV.Len())
	}
	after, err := target.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer after.Close()
	assertPrefixSnapshotsEqual(t, after, before)

	invalid := []struct {
		name     string
		draft    []int
		boundary []float32
	}{
		{name: "short-boundary", draft: draft, boundary: boundary[:len(boundary)-1]},
		{name: "nonfinite-boundary", draft: draft, boundary: func() []float32 {
			values := append([]float32(nil), boundary...)
			values[0] = float32(math.NaN())
			return values
		}()},
		{name: "invalid-draft-token", draft: []int{m.Cfg.VocabSize, 5}, boundary: boundary},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			result, err := target.VerifyGreedyDeviceDraft(context.Background(), tc.draft, tc.boundary)
			if err == nil {
				t.Fatalf("invalid input bypassed validation: %+v", result)
			}
			if result.Receipt.Path == targetVerificationBoundaryRejectPath {
				t.Fatalf("invalid input received boundary-reject receipt: %+v", result.Receipt)
			}
		})
	}
}

func TestVerifyGreedyDeviceDraftNormalizesHALMetadataToCommittedPrefix(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	first := 7
	boundary := make([]float32, m.Cfg.VocabSize)
	boundary[first] = 10
	oracle := m.NewSession()
	second := argmaxF32(oracle.Step(first))
	oracle.Close()
	wrongSecond := (second + 1) % m.Cfg.VocabSize

	for _, tc := range []struct {
		name         string
		draft        []int
		wantAccepted int
	}{
		{name: "full", draft: []int{first, second}, wantAccepted: 2},
		{name: "partial", draft: []int{first, wrongSecond}, wantAccepted: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target, _ := newDevicePanelTestSession(t, m)
			if target.halStep != 0 || target.halLogitsWarm {
				t.Fatalf("initial metadata step=%d warm=%t", target.halStep, target.halLogitsWarm)
			}
			got, err := target.VerifyGreedyDeviceDraft(context.Background(), tc.draft, boundary)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Accepted) != tc.wantAccepted || target.halStep != tc.wantAccepted || !target.halLogitsWarm {
				t.Fatalf("accepted=%d step=%d warm=%t, want %d/%d/true", len(got.Accepted), target.halStep, target.halLogitsWarm, tc.wantAccepted, tc.wantAccepted)
			}
		})
	}

	t.Run("cancellation-restores-cold-metadata", func(t *testing.T) {
		target, backend := newDevicePanelTestSession(t, m)
		ctx, cancel := context.WithCancel(context.Background())
		backend.afterSequence = cancel
		got, err := target.VerifyGreedyDeviceDraft(ctx, []int{first, second}, boundary)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v result=%+v, want context cancellation", err, got)
		}
		if target.halStep != 0 || target.halLogitsWarm || target.halKV.Len() != 0 {
			t.Fatalf("cancellation retained speculative metadata/state: step=%d warm=%t kv=%d", target.halStep, target.halLogitsWarm, target.halKV.Len())
		}
	})

	t.Run("partial-and-cancellation-preserve-warm-prefix-metadata", func(t *testing.T) {
		prefix := []int{3, 7, 11, 5}
		newWarmTarget := func(t *testing.T) (*Session, *devicePanelTestBackend, []float32, []int, int) {
			t.Helper()
			target, backend := newDevicePanelTestSession(t, m)
			boundary := target.Prefill(prefix)
			warmToken := argmaxF32(boundary)
			boundary = target.Step(warmToken)
			committed := append(append([]int(nil), prefix...), warmToken)
			if target.halStep == 0 || !target.halLogitsWarm {
				t.Fatalf("warm-prefix metadata step=%d warm=%t, want nonzero/true", target.halStep, target.halLogitsWarm)
			}
			return target, backend, boundary, committed, target.halStep
		}

		target, _, warmBoundary, committedPrefix, beforeStep := newWarmTarget(t)
		warmFirst := argmaxF32(warmBoundary)
		serial := m.NewSession()
		serial.Prefill(committedPrefix)
		warmSecond := argmaxF32(serial.Step(warmFirst))
		serial.Close()
		partial, err := target.VerifyGreedyDeviceDraft(context.Background(), []int{warmFirst, (warmSecond + 1) % m.Cfg.VocabSize}, warmBoundary)
		if err != nil {
			t.Fatal(err)
		}
		if len(partial.Accepted) != 1 || target.halStep != beforeStep+1 || !target.halLogitsWarm {
			t.Fatalf("warm partial accepted=%d step=%d warm=%t, want 1/%d/true", len(partial.Accepted), target.halStep, target.halLogitsWarm, beforeStep+1)
		}

		cancelTarget, cancelBackend, cancelBoundary, cancelPrefix, cancelBeforeStep := newWarmTarget(t)
		cancelFirst := argmaxF32(cancelBoundary)
		ctx, cancel := context.WithCancel(context.Background())
		cancelBackend.afterSequence = cancel
		if _, err := cancelTarget.VerifyGreedyDeviceDraft(ctx, []int{cancelFirst, (cancelFirst + 1) % m.Cfg.VocabSize}, cancelBoundary); !errors.Is(err, context.Canceled) {
			t.Fatalf("warm cancellation error=%v, want context cancellation", err)
		}
		if cancelTarget.halStep != cancelBeforeStep || !cancelTarget.halLogitsWarm || cancelTarget.halKV.Len() != len(cancelPrefix) {
			t.Fatalf("warm cancellation metadata/state step=%d warm=%t kv=%d, want %d/true/%d", cancelTarget.halStep, cancelTarget.halLogitsWarm, cancelTarget.halKV.Len(), cancelBeforeStep, len(cancelPrefix))
		}
	})
}

func TestQwen35DeviceTargetTransactionAcceptRollbackAndFailureAtomicity(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}
	for _, accepted := range []int{len(draft), 2, 0} {
		t.Run(itoa(accepted)+"-accepted", func(t *testing.T) {
			target, _ := newDevicePanelTestSession(t, m)
			serial := m.NewSession()
			serial.captureTargetHidden = true
			defer func() { serial.Close() }()
			boundary := target.Prefill(prefix)
			serial.Prefill(prefix)
			tx, err := beginQwen35MTPTargetTransaction(target, boundary)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := tx.Verify(draft)
			if err != nil {
				t.Fatal(err)
			}
			for i, token := range draft {
				assertFloat32BitsEqual(t, "transaction verification row", rows[i], serial.Step(token))
			}
			serial.Close()
			serial = m.NewSession()
			serial.captureTargetHidden = true
			serial.Prefill(prefix)
			for _, token := range draft[:accepted] {
				serial.Step(token)
			}
			steps := 0
			tx.step = func(token int) []float32 {
				steps++
				return target.Step(token)
			}
			if _, err := tx.Commit(accepted); err != nil {
				t.Fatal(err)
			}
			wantSteps := accepted
			if accepted == len(draft) {
				wantSteps = 0
			}
			if steps != wantSteps {
				t.Fatalf("accepted=%d replay steps=%d want=%d", accepted, steps, wantSteps)
			}
			assertDeviceStateMatchesSerial(t, target, serial, append(append([]int(nil), prefix...), draft[:accepted]...))
		})
	}

	t.Run("cancellation-after-device-mutation", func(t *testing.T) {
		target, _ := newDevicePanelTestSession(t, m)
		boundary := target.Prefill(prefix)
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		verify := tx.verify
		tx.verify = func(ids []int) ([][]float32, TargetVerificationReceipt, error) {
			rows, receipt, err := verify(ids)
			if err != nil {
				return rows, receipt, err
			}
			return nil, receipt, context.Canceled
		}
		if _, err := tx.Verify(draft); !errors.Is(err, context.Canceled) {
			t.Fatalf("verification error=%v want context cancellation", err)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
	})

	t.Run("commit-preservation-clone-failure-restores-pre-round-state", func(t *testing.T) {
		target, backend := newDevicePanelTestSession(t, m)
		boundary := target.Prefill(prefix)
		warmToken := argmaxF32(boundary)
		boundary = target.Step(warmToken)
		beforeStep, beforeWarm := target.halStep, target.halLogitsWarm
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatal(err)
		}
		backend.failCloneAt = backend.cloneCalls + 1
		if _, err := tx.Commit(2); err == nil || !strings.Contains(err.Error(), "injected tensor clone failure") {
			t.Fatalf("commit error=%v, want injected preservation clone failure", err)
		}
		backend.failCloneAt = 0
		if target.halStep != beforeStep || target.halLogitsWarm != beforeWarm {
			t.Fatalf("clone failure metadata step/warm=%d/%t, want %d/%t", target.halStep, target.halLogitsWarm, beforeStep, beforeWarm)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
	})
}

func assertPrefixSnapshotsEqual(t *testing.T, got, want *PrefixSnapshot) {
	t.Helper()
	gotHost, err := got.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer gotHost.Close()
	wantHost, err := want.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer wantHost.Close()
	if !reflect.DeepEqual(gotHost.cache, wantHost.cache) || !reflect.DeepEqual(gotHost.kv, wantHost.kv) ||
		!reflect.DeepEqual(gotHost.qwen35.layers, wantHost.qwen35.layers) ||
		!reflect.DeepEqual(got.targetHidden, want.targetHidden) || !reflect.DeepEqual(got.targetHiddenTokens, want.targetHiddenTokens) {
		t.Fatal("device prefix snapshot changed across rollback")
	}
}

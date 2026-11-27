package modelloadplan

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

func findCandidate(t *testing.T, p Plan, id string) Candidate {
	t.Helper()
	for _, c := range p.Candidates {
		if c.ID == id {
			return c
		}
	}
	t.Fatalf("candidate %q not found in plan", id)
	return Candidate{}
}

// fak-test:runtime fast est=50ms lane=default
func TestBuildChargesKVAtRequestedContext(t *testing.T) {
	reqSmall := Request{
		Setup:         "personal",
		Goal:          "quality",
		LocalPolicy:   "require",
		Memory:        "unified",
		DeviceBytes:   28 << 30,
		DiskBytes:     28 << 30,
		ContextTokens: 8192,
	}
	pSmall, err := Build(reqSmall)
	if err != nil {
		t.Fatal(err)
	}
	q6Small := findCandidate(t, pSmall, "local/q6_k")
	if q6Small.KVBytes != 8192*32768 {
		t.Fatalf("KVBytes(8192) = %d, want %d", q6Small.KVBytes, 8192*32768)
	}
	if q6Small.FixedSessionBytes != DefaultFixedSessionBytes {
		t.Fatalf("FixedSessionBytes = %d, want %d", q6Small.FixedSessionBytes, DefaultFixedSessionBytes)
	}
	if q6Small.RuntimeBytes != q6Small.WeightsBytes+q6Small.OverheadBytes+q6Small.KVBytes {
		t.Fatalf("RuntimeBytes = %d, want Weights+Overhead+KV = %d", q6Small.RuntimeBytes, q6Small.WeightsBytes+q6Small.OverheadBytes+q6Small.KVBytes)
	}
	if !q6Small.Fits {
		t.Fatalf("expected Q6_K to fit at 8192 tokens on 28 GiB, reasons=%v", q6Small.Reasons)
	}

	// Increasing requested context to 131072 adds ~4 GiB KV, so Q6_K no longer fits 28 GiB
	// and Build selects a smaller quant (Q5_K_M) that genuinely fits at 131072 tokens.
	reqLarge := reqSmall
	reqLarge.ContextTokens = 131072
	pLarge, err := Build(reqLarge)
	if err != nil {
		t.Fatal(err)
	}
	q6Large := findCandidate(t, pLarge, "local/q6_k")
	if q6Large.KVBytes != 131072*32768 {
		t.Fatalf("KVBytes(131072) = %d, want %d", q6Large.KVBytes, 131072*32768)
	}
	if q6Large.Fits || q6Large.FitGrade != FitGradeTooTight {
		t.Fatalf("expected Q6_K to be rejected as too_tight at 131072 on 28 GiB, got fits=%v grade=%q", q6Large.Fits, q6Large.FitGrade)
	}
	if pLarge.Selected == nil || pLarge.Selected.Quantization != "Q5_K_M" {
		t.Fatalf("selected = %#v, want Q5_K_M", pLarge.Selected)
	}
	if pLarge.Selected.EffectiveContextTokens != 131072 || pLarge.Selected.KVBytes != 131072*32768 {
		t.Fatalf("selected did not charge requested context 131072: eff=%d kv=%d", pLarge.Selected.EffectiveContextTokens, pLarge.Selected.KVBytes)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestBuildUsableContextFormulaAndNativeCap(t *testing.T) {
	p, err := Build(Request{
		Setup:       "personal",
		Goal:        "quality",
		LocalPolicy: "require",
		Memory:      "unified",
		DeviceBytes: 27 << 30,
		DiskBytes:   27 << 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	q6 := findCandidate(t, p, "local/q6_k")
	admitted := int64(math.Floor(float64(int64(27<<30)) * MarginalMaxUtilization))
	wantUsable := int((admitted - q6.WeightsBytes - q6.OverheadBytes) / 32768)
	if q6.UsableContextTokens != wantUsable {
		t.Fatalf("UsableContextTokens = %d, want %d", q6.UsableContextTokens, wantUsable)
	}
	if q6.UsableContextTokens <= 0 || q6.UsableContextTokens >= DefaultNativeContextTokens {
		t.Fatalf("expected 0 < usable < native, got %d", q6.UsableContextTokens)
	}

	// Exact boundary: UsableContextTokens fits (marginal), while UsableContextTokens+1 is too_tight.
	pAtBoundary, err := Build(Request{
		LocalPolicy:   "require",
		Memory:        "unified",
		DeviceBytes:   27 << 30,
		DiskBytes:     27 << 30,
		ContextTokens: q6.UsableContextTokens,
	})
	if err != nil {
		t.Fatal(err)
	}
	q6At := findCandidate(t, pAtBoundary, "local/q6_k")
	if !q6At.Fits || q6At.FitGrade == FitGradeTooTight {
		t.Fatalf("expected exact UsableContextTokens (%d) to fit, got fits=%v grade=%q reasons=%v", q6.UsableContextTokens, q6At.Fits, q6At.FitGrade, q6At.Reasons)
	}

	pOverBoundary, err := Build(Request{
		LocalPolicy:   "require",
		Memory:        "unified",
		DeviceBytes:   27 << 30,
		DiskBytes:     27 << 30,
		ContextTokens: q6.UsableContextTokens + 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	q6Over := findCandidate(t, pOverBoundary, "local/q6_k")
	if q6Over.Fits || q6Over.FitGrade != FitGradeTooTight {
		t.Fatalf("expected UsableContextTokens+1 (%d) to fail as too_tight, got fits=%v grade=%q", q6.UsableContextTokens+1, q6Over.Fits, q6Over.FitGrade)
	}

	// Large pool caps UsableContextTokens at NativeContextTokens and clamps over-native requests.
	pBig, err := Build(Request{
		LocalPolicy:   "require",
		Memory:        "unified",
		DeviceBytes:   141 << 30,
		DiskBytes:     141 << 30,
		ContextTokens: 524288,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4Big := findCandidate(t, pBig, "local/q4_k_m")
	if q4Big.UsableContextTokens != DefaultNativeContextTokens {
		t.Fatalf("UsableContextTokens on 141 GiB = %d, want native cap %d", q4Big.UsableContextTokens, DefaultNativeContextTokens)
	}
	if q4Big.RequestedContextTokens != 524288 || q4Big.EffectiveContextTokens != DefaultNativeContextTokens {
		t.Fatalf("requested/effective = %d/%d, want 524288/%d", q4Big.RequestedContextTokens, q4Big.EffectiveContextTokens, DefaultNativeContextTokens)
	}
	if !q4Big.ContextClamped || q4Big.ContextClampReason != "clamped_to_native" {
		t.Fatalf("clamp = (%v, %q), want (true, clamped_to_native)", q4Big.ContextClamped, q4Big.ContextClampReason)
	}
	if q4Big.KVBytes != int64(DefaultNativeContextTokens)*32768 {
		t.Fatalf("KVBytes after native clamp = %d, want %d", q4Big.KVBytes, int64(DefaultNativeContextTokens)*32768)
	}

	// Unknown and tiny pools fail closed to 0 usable context.
	pUnknown, err := Build(Request{Setup: "shared"})
	if err != nil {
		t.Fatal(err)
	}
	q4Unk := findCandidate(t, pUnknown, "local/q4_k_m")
	if q4Unk.UsableContextTokens != 0 || q4Unk.FitGrade != FitGradeUnknown || q4Unk.MaxContextFor(1) != 0 {
		t.Fatalf("unknown pool candidate = usable:%d grade:%q max:%d", q4Unk.UsableContextTokens, q4Unk.FitGrade, q4Unk.MaxContextFor(1))
	}

	pTiny, err := Build(Request{LocalPolicy: "require", DeviceBytes: 8 << 30, DiskBytes: 8 << 30})
	if err != nil {
		t.Fatal(err)
	}
	iq2Tiny := findCandidate(t, pTiny, "local/iq2_s")
	if iq2Tiny.UsableContextTokens != 0 || iq2Tiny.FitGrade != FitGradeTooTight || iq2Tiny.MaxContextFor(1) != 0 {
		t.Fatalf("tiny pool candidate = usable:%d grade:%q max:%d", iq2Tiny.UsableContextTokens, iq2Tiny.FitGrade, iq2Tiny.MaxContextFor(1))
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestContextClampPropagatesEffectiveContextAndWalkVerdictAgree(t *testing.T) {
	// With AllowContextClamp=true, Q6_K on 27 GiB cannot meet 131072 tokens, so it clamps
	// to UsableContextTokens and propagates that effective context into KVBytes, RuntimeBytes,
	// PoolUtilization, and FitGrade (fixing llmfit's fit.rs:842 dropped-context defect).
	pClamp, err := Build(Request{
		Setup:             "personal",
		Goal:              "quality",
		LocalPolicy:       "require",
		Memory:            "unified",
		DeviceBytes:       27 << 30,
		DiskBytes:         27 << 30,
		ContextTokens:     131072,
		AllowContextClamp: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	q6 := findCandidate(t, pClamp, "local/q6_k")
	if !q6.ContextClamped || q6.ContextClampReason != "clamped_to_usable_context" {
		t.Fatalf("Q6_K clamp = (%v, %q), want (true, clamped_to_usable_context)", q6.ContextClamped, q6.ContextClampReason)
	}
	if q6.EffectiveContextTokens != q6.UsableContextTokens || q6.EffectiveContextTokens >= q6.RequestedContextTokens {
		t.Fatalf("Q6_K effective=%d usable=%d requested=%d", q6.EffectiveContextTokens, q6.UsableContextTokens, q6.RequestedContextTokens)
	}
	wantKV := int64(q6.EffectiveContextTokens) * 32768
	if q6.KVBytes != wantKV {
		t.Fatalf("Q6_K KVBytes = %d, want %d (must propagate clamped context)", q6.KVBytes, wantKV)
	}
	if !q6.Fits || q6.FitGrade == FitGradeTooTight {
		t.Fatalf("clamped Q6_K should fit with aligned grade, got fits=%v grade=%q", q6.Fits, q6.FitGrade)
	}
	// Unclamped candidates that satisfy the full 131072 context (Q4_K_M, Q5_K_M) must
	// rank ahead of clamped Q6_K even under Goal=quality.
	if pClamp.Selected == nil || pClamp.Selected.Quantization != "Q5_K_M" || pClamp.Selected.ContextClamped {
		t.Fatalf("selected = %#v, want unclamped Q5_K_M", pClamp.Selected)
	}

	// Verify walk/admission (Fits) and verdict (FitGrade) agree across every headroom threshold,
	// specifically refusing ratios in (0.98, 1.00] instead of admitting and grading TooTight.
	q4Base := int64(16464440224) + int64(16464440224)/12 + (2 << 30) + DefaultFixedSessionBytes + 8192*32768
	cases := []struct {
		name      string
		pool      int64
		wantFits  bool
		wantGrade string
	}{
		{name: "perfect", pool: int64(float64(q4Base) / 0.50), wantFits: true, wantGrade: FitGradePerfect},
		{name: "good", pool: int64(float64(q4Base) / 0.75), wantFits: true, wantGrade: FitGradeGood},
		{name: "marginal", pool: int64(float64(q4Base) / 0.95), wantFits: true, wantGrade: FitGradeMarginal},
		{name: "between_098_and_100_must_refuse", pool: int64(float64(q4Base) / 0.99), wantFits: false, wantGrade: FitGradeTooTight},
		{name: "over_pool", pool: q4Base - 1, wantFits: false, wantGrade: FitGradeTooTight},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := Build(Request{
				LocalPolicy:   "require",
				Memory:        "unified",
				DeviceBytes:   tc.pool,
				DiskBytes:     tc.pool,
				ContextTokens: 8192,
			})
			if err != nil {
				t.Fatal(err)
			}
			cand := findCandidate(t, plan, "local/q4_k_m")
			if cand.Fits != tc.wantFits || cand.FitGrade != tc.wantGrade {
				t.Fatalf("util=%.4f fits=%v grade=%q, want fits=%v grade=%q", cand.PoolUtilization, cand.Fits, cand.FitGrade, tc.wantFits, tc.wantGrade)
			}
		})
	}

	// Off-device host offload caps FitGrade at Good even when total utilization <= 0.60.
	pSplit, err := Build(Request{
		LocalPolicy:   "require",
		Memory:        "split",
		DeviceBytes:   12 << 30,
		HostBytes:     100 << 30,
		DiskBytes:     64 << 30,
		ContextTokens: 8192,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4Split := findCandidate(t, pSplit, "local/q4_k_m")
	if !q4Split.Fits || q4Split.HostBytes <= 0 {
		t.Fatalf("expected split offload fit, fits=%v host=%d", q4Split.Fits, q4Split.HostBytes)
	}
	if q4Split.PoolUtilization > PerfectMaxUtilization {
		t.Fatalf("precondition failed: util=%v > %v", q4Split.PoolUtilization, PerfectMaxUtilization)
	}
	if q4Split.FitGrade != FitGradeGood {
		t.Fatalf("split offload grade = %q, want %q (capped below perfect)", q4Split.FitGrade, FitGradeGood)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestMaxContextForUsersAndGradedHeadroom(t *testing.T) {
	p, err := Build(Request{
		Setup:       "personal",
		Goal:        "balanced",
		LocalPolicy: "require",
		Memory:      "unified",
		DeviceBytes: 64 << 30,
		DiskBytes:   64 << 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4 := findCandidate(t, p, "local/q4_k_m")
	if q4.MaxContextFor(0) != 0 || q4.MaxContextFor(-1) != 0 {
		t.Fatalf("non-positive users should return 0, got %d/%d", q4.MaxContextFor(0), q4.MaxContextFor(-1))
	}
	if got, want := q4.MaxContextFor(1), q4.UsableContextTokens; got != want {
		t.Fatalf("MaxContextFor(1) = %d, want UsableContextTokens = %d", got, want)
	}

	prev := q4.MaxContextFor(1)
	for _, users := range []int{2, 4, 8, 16, 64} {
		b := q4.MaxContextForUsers(users)
		if b.MaxContextTokens != q4.MaxContextFor(users) {
			t.Fatalf("users=%d budget mismatch: %d != %d", users, b.MaxContextTokens, q4.MaxContextFor(users))
		}
		if b.MaxContextTokens > prev {
			t.Fatalf("users=%d context %d > prev %d (must be monotonically non-increasing)", users, b.MaxContextTokens, prev)
		}
		if b.PerfectContextTokens > b.GoodContextTokens || b.GoodContextTokens > b.MarginalContextTokens {
			t.Fatalf("users=%d graded ordering broken: perfect=%d good=%d marginal=%d", users, b.PerfectContextTokens, b.GoodContextTokens, b.MarginalContextTokens)
		}
		if b.MaxContextTokens > 0 {
			// Re-building with users and b.MaxContextTokens must fit cleanly.
			verifyPlan, err := Build(Request{
				LocalPolicy:   "require",
				Memory:        "unified",
				DeviceBytes:   64 << 30,
				DiskBytes:     64 << 30,
				Users:         users,
				ContextTokens: b.MaxContextTokens,
			})
			if err != nil {
				t.Fatal(err)
			}
			verifyCand := findCandidate(t, verifyPlan, "local/q4_k_m")
			if !verifyCand.Fits || verifyCand.FitGrade == FitGradeTooTight {
				t.Fatalf("users=%d at MaxContextTokens=%d did not fit: fits=%v grade=%q", users, b.MaxContextTokens, verifyCand.Fits, verifyCand.FitGrade)
			}
			if verifyCand.MaxUsers < users {
				t.Fatalf("users=%d duality failed: MaxUsers=%d", users, verifyCand.MaxUsers)
			}
		}
		prev = b.MaxContextTokens
	}
	if got := q4.MaxContextFor(20000); got != 0 {
		t.Fatalf("expected 20000 users to exhaust pool to 0 context, got %d", got)
	}

	// Verify JSON round-trip keeps MaxContextForUsers exact.
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Plan
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	q4Decoded := findCandidate(t, decoded, "local/q4_k_m")
	if q4Decoded.MaxContextForUsers(4) != q4.MaxContextForUsers(4) {
		t.Fatalf("JSON round-trip altered MaxContextForUsers(4): %+v vs %+v", q4Decoded.MaxContextForUsers(4), q4.MaxContextForUsers(4))
	}

	// Verify split offload blocks PerfectContextTokens when base weights exceed DeviceBytes.
	pSplit, err := Build(Request{
		LocalPolicy: "require",
		Memory:      "split",
		DeviceBytes: 12 << 30,
		HostBytes:   100 << 30,
		DiskBytes:   64 << 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4Split := findCandidate(t, pSplit, "local/q4_k_m")
	bSplit := q4Split.MaxContextForUsers(1)
	if bSplit.PerfectContextTokens != 0 || bSplit.GoodContextTokens == 0 {
		t.Fatalf("split offload graded budget = %+v, want Perfect=0 and Good>0", bSplit)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestContextLadderAndConfigDrivenMLAShape(t *testing.T) {
	// Config-driven MLA + compressed/windowed PerLayer shape (no hardcoded Qwen width or GDN state).
	ratio := make([]int, 43)
	window := make([]int, 43)
	idx := make([]int, 43)
	for l := 0; l < 43; l++ {
		window[l] = 128
		switch {
		case l < 2:
			ratio[l] = 0
		case l < 23:
			ratio[l] = 4
			idx[l] = 128
		default:
			ratio[l] = 128
		}
	}
	mlaShape := kvbudget.Shape{
		Kind:       kvbudget.MLA,
		Layers:     43,
		KVLoraRank: 512,
		PerLayer: &kvbudget.LayerProfile{
			Window:        window,
			CompressRatio: ratio,
			IndexHeadDim:  idx,
		},
	}
	quant := kvbudget.F32
	p, err := Build(Request{
		Setup:               "personal",
		LocalPolicy:         "require",
		Memory:              "unified",
		DeviceBytes:         32 << 30,
		DiskBytes:           32 << 30,
		ContextTokens:       131072,
		NativeContextTokens: 1048576,
		KVShape:             &mlaShape,
		KVQuant:             &quant,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4 := findCandidate(t, p, "local/q4_k_m")
	if q4.FixedSessionBytes != 0 {
		t.Fatalf("custom KVShape should default FixedSessionBytes=0, got %d", q4.FixedSessionBytes)
	}
	wantKV := int64(math.Ceil(mlaShape.KVBytesPerStream(131072, quant)))
	if q4.KVBytes != wantKV {
		t.Fatalf("MLA compressed KVBytes(131072) = %d, want %d", q4.KVBytes, wantKV)
	}
	if !q4.Fits || q4.UsableContextTokens <= 524288 || q4.UsableContextTokens >= 1048576 {
		t.Fatalf("expected compressed MLA shape on 32 GiB to fit 131K with 512K < usable < 1M: fits=%v usable=%d", q4.Fits, q4.UsableContextTokens)
	}
	// Verify exact non-linear compressed boundary against kvbudget.Shape directly.
	admitted := int64(math.Floor(float64(int64(32<<30)) * MarginalMaxUtilization))
	availKV := admitted - q4.WeightsBytes - q4.OverheadBytes
	kvAtUsable := int64(math.Ceil(mlaShape.KVBytesPerStream(q4.UsableContextTokens, quant)))
	kvOverUsable := int64(math.Ceil(mlaShape.KVBytesPerStream(q4.UsableContextTokens+1, quant)))
	if kvAtUsable > availKV || kvOverUsable <= availKV {
		t.Fatalf("non-linear MLA usable boundary inconsistent: usable=%d kvAt=%d kvOver=%d avail=%d", q4.UsableContextTokens, kvAtUsable, kvOverUsable, availKV)
	}
	if len(q4.ContextLadder) == 0 {
		t.Fatal("expected populated ContextLadder")
	}
	prevCtx := 0
	for _, rung := range q4.ContextLadder {
		if rung.ContextTokens <= prevCtx || rung.ContextTokens > 1048576 {
			t.Fatalf("ContextLadder out of ascending native-capped order: prev=%d cur=%d", prevCtx, rung.ContextTokens)
		}
		if rung.Fits != (rung.FitGrade != FitGradeTooTight && rung.FitGrade != FitGradeUnknown) {
			t.Fatalf("rung %d fits/grade disagree: fits=%v grade=%q", rung.ContextTokens, rung.Fits, rung.FitGrade)
		}
		prevCtx = rung.ContextTokens
	}

	// With 36 GiB, the same compressed MLA shape reaches full 1M native context.
	p36, err := Build(Request{
		Setup:               "personal",
		LocalPolicy:         "require",
		Memory:              "unified",
		DeviceBytes:         36 << 30,
		DiskBytes:           36 << 30,
		ContextTokens:       1048576,
		NativeContextTokens: 1048576,
		KVShape:             &mlaShape,
		KVQuant:             &quant,
	})
	if err != nil {
		t.Fatal(err)
	}
	q4At36 := findCandidate(t, p36, "local/q4_k_m")
	if !q4At36.Fits || q4At36.UsableContextTokens != 1048576 {
		t.Fatalf("expected compressed MLA shape on 36 GiB to reach 1M: fits=%v usable=%d", q4At36.Fits, q4At36.UsableContextTokens)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNormalizeRejectsInvalidContextInputs(t *testing.T) {
	negFixed := int64(-1)
	badShape := kvbudget.Shape{Kind: kvbudget.MHA, Layers: 16}
	badQuant := kvbudget.Quant{Name: "zero", BytesPerElem: 0}
	nanQuant := kvbudget.Quant{Name: "nan", BytesPerElem: math.NaN()}

	cases := []Request{
		{ContextTokens: -1},
		{Users: -1},
		{NativeContextTokens: -1},
		{FixedSessionBytes: &negFixed},
		{KVShape: &badShape},
		{KVQuant: &badQuant},
		{KVQuant: &nanQuant},
	}
	for i, req := range cases {
		if _, err := Build(req); err == nil {
			t.Errorf("case %d (%+v): expected error, got nil", i, req)
		}
	}
}

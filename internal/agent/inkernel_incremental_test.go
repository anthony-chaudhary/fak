package agent

// Adversarial tests for the frozen incremental-context contract (see
// .fak/runs/oss-port-p0-incremental-context/scratch/CONTRACT.md). Written from the
// specification only; the implementation file is intentionally not consulted.
//
// Every test function is named TestInKernel... so the acceptance witness
// `go test ./internal/agent/ -run TestInKernel -count=1` selects all of them.

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// noRemoved is the "source has no Removed renderer" sentinel for tests: the bool
// reports that no removal renderer exists, so the snapshot carries no removal text.
func noRemoved[A any](A) (string, bool) { return "", false }

// mustSource builds a PackedContextSource through the public constructor, failing
// the test on a constructor error.
func mustSource[A any](
	t *testing.T,
	key ContextKey,
	load func() (A, bool),
	baseline func(A) string,
	update func(prev, cur A) string,
	removed func(A) (string, bool),
) PackedContextSource {
	t.Helper()
	src, err := MakeContextSource(key, load, baseline, update, removed)
	if err != nil {
		t.Fatalf("MakeContextSource(%q) unexpectedly failed: %v", key, err)
	}
	return src
}

// TestInKernelInitializeBlocksOnUnavailable proves that a single unavailable
// source refuses the whole generation and names exactly that key.
func TestInKernelInitializeBlocksOnUnavailable(t *testing.T) {
	ok := mustSource(t, "ns/ok",
		func() (int, bool) { return 1, true },
		func(int) string { return "OK-BASE" },
		func(int, int) string { return "OK-UPD" },
		noRemoved[int],
	)
	bad := mustSource(t, "ns/bad",
		func() (int, bool) { return 0, false },
		func(int) string { return "BAD-BASE" },
		func(int, int) string { return "BAD-UPD" },
		noRemoved[int],
	)

	ctx, err := CombineIncrementalContext(MakeIncrementalContext(ok), MakeIncrementalContext(bad))
	if err != nil {
		t.Fatalf("CombineIncrementalContext: %v", err)
	}

	_, initErr := InitializeIncrementalContext(ctx)
	if initErr == nil {
		t.Fatal("InitializeIncrementalContext with an unavailable source: got nil error, want *InitializationBlockedError")
	}
	var blocked *InitializationBlockedError
	if !errors.As(initErr, &blocked) {
		t.Fatalf("error type = %T, want *InitializationBlockedError", initErr)
	}
	if len(blocked.Keys) != 1 || blocked.Keys[0] != ContextKey("ns/bad") {
		t.Fatalf("blocked keys = %v, want exactly [ns/bad]", blocked.Keys)
	}
	if !strings.Contains(blocked.Error(), "ns/bad") {
		t.Fatalf("blocked error %q does not name the unavailable key", blocked.Error())
	}
}

// TestInKernelReconcileEmitsOnlyChangedSource proves an incremental reconcile
// emits ONLY the changed source's text (never a full re-render) and advances the
// snapshot.
func TestInKernelReconcileEmitsOnlyChangedSource(t *testing.T) {
	alpha := 1
	srcA := mustSource(t, "ns/alpha",
		func() (int, bool) { return alpha, true },
		func(int) string { return "ALPHA-BASE" },
		func(int, int) string { return "ALPHA-DELTA" },
		noRemoved[int],
	)
	srcB := mustSource(t, "ns/beta",
		func() (int, bool) { return 2, true },
		func(int) string { return "BETA-BASE" },
		func(int, int) string { return "BETA-DELTA" },
		noRemoved[int],
	)
	ctx, err := CombineIncrementalContext(MakeIncrementalContext(srcA), MakeIncrementalContext(srcB))
	if err != nil {
		t.Fatalf("CombineIncrementalContext: %v", err)
	}

	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}
	if gen.Baseline != "ALPHA-BASE\n\nBETA-BASE" {
		t.Fatalf("baseline = %q, want registry-order join", gen.Baseline)
	}

	alpha = 9
	res, err := ReconcileIncrementalContext(ctx, gen.Snapshot)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileUpdated {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileUpdated)
	}
	if !strings.Contains(res.Text, "ALPHA-DELTA") {
		t.Fatalf("Text = %q, want it to contain the changed source's update", res.Text)
	}
	if strings.Contains(res.Text, "BETA-BASE") {
		t.Fatalf("Text = %q must NOT contain the unchanged source's baseline text", res.Text)
	}
	if strings.Contains(res.Text, "BETA-DELTA") {
		t.Fatalf("Text = %q must NOT contain the unchanged source's update", res.Text)
	}

	// Snapshot advanced for the changed key, preserved for the unchanged key.
	if bytes.Equal(res.Snapshot["ns/alpha"].Value, gen.Snapshot["ns/alpha"].Value) {
		t.Fatalf("snapshot for ns/alpha did not advance: %s", res.Snapshot["ns/alpha"].Value)
	}
	if !bytes.Equal(res.Snapshot["ns/beta"].Value, gen.Snapshot["ns/beta"].Value) {
		t.Fatalf("snapshot for ns/beta changed unexpectedly: got %s want %s",
			res.Snapshot["ns/beta"].Value, gen.Snapshot["ns/beta"].Value)
	}
}

// TestInKernelByteEqualValueUnchanged proves a byte-equal value reconciles as
// unchanged with empty text.
func TestInKernelByteEqualValueUnchanged(t *testing.T) {
	src := mustSource(t, "ns/same",
		func() (int, bool) { return 5, true },
		func(int) string { return "SAME-BASE" },
		func(int, int) string { return "SAME-UPD" },
		noRemoved[int],
	)
	ctx := MakeIncrementalContext(src)
	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}

	res, err := ReconcileIncrementalContext(ctx, gen.Snapshot)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileUnchanged {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileUnchanged)
	}
	if res.Text != "" {
		t.Fatalf("Text = %q, want empty for unchanged", res.Text)
	}
}

// TestInKernelUnavailableAfterAdmissionKeepsSnapshot proves that an unavailable
// source with an admitted snapshot is Unchanged and its snapshot is preserved
// (NOT removed).
func TestInKernelUnavailableAfterAdmissionKeepsSnapshot(t *testing.T) {
	avail := true
	src := mustSource(t, "ns/x",
		func() (int, bool) {
			if !avail {
				return 0, false
			}
			return 7, true
		},
		func(int) string { return "X-BASE" },
		func(int, int) string { return "X-UPD" },
		noRemoved[int],
	)
	ctx := MakeIncrementalContext(src)
	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}

	avail = false
	res, err := ReconcileIncrementalContext(ctx, gen.Snapshot)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileUnchanged {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileUnchanged)
	}
	if res.Text != "" {
		t.Fatalf("Text = %q, want empty", res.Text)
	}
	got, present := res.Snapshot["ns/x"]
	if !present {
		t.Fatal("admitted snapshot for ns/x was removed by a temporary unavailability")
	}
	if !bytes.Equal(got.Value, gen.Snapshot["ns/x"].Value) {
		t.Fatalf("preserved snapshot value = %s, want %s", got.Value, gen.Snapshot["ns/x"].Value)
	}
}

// TestInKernelRemovedWithoutRendererRequiresReplacement proves that dropping a
// source whose snapshot has no removal renderer requires a replacement
// generation.
func TestInKernelRemovedWithoutRendererRequiresReplacement(t *testing.T) {
	src := mustSource(t, "ns/dropme",
		func() (int, bool) { return 3, true },
		func(int) string { return "DROP-BASE" },
		func(int, int) string { return "DROP-UPD" },
		noRemoved[int],
	)
	ctx := MakeIncrementalContext(src)
	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}
	if gen.Snapshot["ns/dropme"].Removed != "" {
		t.Fatalf("snapshot removal text = %q, want empty when no renderer exists",
			gen.Snapshot["ns/dropme"].Removed)
	}

	res, err := ReconcileIncrementalContext(EmptyIncrementalContext(), gen.Snapshot)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileReplacementReady {
		t.Fatalf("Kind = %q, want %q (removal without a renderer must require replacement)",
			res.Kind, ReconcileReplacementReady)
	}
}

// TestInKernelRemovedWithRendererEmitsRemovalText proves that a source removed
// from the registry with a Removed renderer reconciles as Updated with exactly
// the removal text.
func TestInKernelRemovedWithRendererEmitsRemovalText(t *testing.T) {
	keep := mustSource(t, "ns/keep",
		func() (int, bool) { return 1, true },
		func(int) string { return "KEEP-BASE" },
		func(int, int) string { return "KEEP-UPD" },
		noRemoved[int],
	)
	gone := mustSource(t, "ns/gone",
		func() (int, bool) { return 2, true },
		func(int) string { return "GONE-BASE" },
		func(int, int) string { return "GONE-UPD" },
		func(int) (string, bool) { return "GONE-REMOVED", true },
	)
	ctx, err := CombineIncrementalContext(MakeIncrementalContext(keep), MakeIncrementalContext(gone))
	if err != nil {
		t.Fatalf("CombineIncrementalContext: %v", err)
	}
	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}
	if gen.Snapshot["ns/gone"].Removed != "GONE-REMOVED" {
		t.Fatalf("admitted snapshot removal text = %q, want GONE-REMOVED",
			gen.Snapshot["ns/gone"].Removed)
	}

	res, err := ReconcileIncrementalContext(MakeIncrementalContext(keep), gen.Snapshot)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileUpdated {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileUpdated)
	}
	if res.Text != "GONE-REMOVED" {
		t.Fatalf("Text = %q, want exactly the removal text", res.Text)
	}
}

// TestInKernelDuplicateKeysRejected proves duplicate keys are rejected at combine
// time with a typed error naming the key.
func TestInKernelDuplicateKeysRejected(t *testing.T) {
	first := mustSource(t, "ns/dup",
		func() (int, bool) { return 1, true },
		func(int) string { return "A" },
		func(int, int) string { return "A-UPD" },
		noRemoved[int],
	)
	second := mustSource(t, "ns/dup",
		func() (int, bool) { return 2, true },
		func(int) string { return "B" },
		func(int, int) string { return "B-UPD" },
		noRemoved[int],
	)

	_, err := CombineIncrementalContext(MakeIncrementalContext(first), MakeIncrementalContext(second))
	if err == nil {
		t.Fatal("CombineIncrementalContext with duplicate keys: got nil error, want *DuplicateContextKeyError")
	}
	var dup *DuplicateContextKeyError
	if !errors.As(err, &dup) {
		t.Fatalf("error type = %T, want *DuplicateContextKeyError", err)
	}
	if dup.Key != ContextKey("ns/dup") {
		t.Fatalf("duplicate key = %q, want ns/dup", dup.Key)
	}
}

// TestInKernelInvalidKeyRejected proves the key grammar and that MakeContextSource
// rejects keys the grammar refuses.
func TestInKernelInvalidKeyRejected(t *testing.T) {
	if ValidContextKey("NoSlash") {
		t.Fatal(`ValidContextKey("NoSlash") = true, want false`)
	}
	if !ValidContextKey("ns/ok") {
		t.Fatal(`ValidContextKey("ns/ok") = false, want true`)
	}
	if !ValidContextKey("a.b-c_d/e.f-g/h") {
		t.Fatal(`ValidContextKey("a.b-c_d/e.f-g/h") = false, want true`)
	}

	_, err := MakeContextSource[string]("NoSlash",
		func() (string, bool) { return "v", true },
		func(string) string { return "BASE" },
		func(string, string) string { return "UPD" },
		noRemoved[string],
	)
	if err == nil {
		t.Fatal("MakeContextSource with invalid key: got nil error, want non-nil")
	}
}

// TestInKernelEmptyBaselineRefused proves an empty baseline render is refused at
// initialize time.
func TestInKernelEmptyBaselineRefused(t *testing.T) {
	src, err := MakeContextSource("ns/emptybase",
		func() (int, bool) { return 1, true },
		func(int) string { return "" }, // invalid: empty baseline render
		func(int, int) string { return "UPD" },
		noRemoved[int],
	)
	if err != nil {
		t.Fatalf("MakeContextSource with valid key unexpectedly failed: %v", err)
	}

	_, initErr := InitializeIncrementalContext(MakeIncrementalContext(src))
	if initErr == nil {
		t.Fatal("InitializeIncrementalContext with an empty baseline: got nil error, want refusal")
	}
}

// TestInKernelReplaceUnavailableWithAdmissionBlocked proves replacement is
// blocked (never builds an incomplete baseline) when an unavailable source has an
// admitted snapshot.
func TestInKernelReplaceUnavailableWithAdmissionBlocked(t *testing.T) {
	avail := true
	src := mustSource(t, "ns/admitted",
		func() (int, bool) {
			if !avail {
				return 0, false
			}
			return 4, true
		},
		func(int) string { return "ADMITTED-BASE" },
		func(int, int) string { return "ADMITTED-UPD" },
		noRemoved[int],
	)
	ctx := MakeIncrementalContext(src)
	gen, err := InitializeIncrementalContext(ctx)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext: %v", err)
	}

	avail = false
	res, err := ReplaceIncrementalContext(ctx, gen.Snapshot)
	if err != nil {
		t.Fatalf("ReplaceIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileReplacementBlocked {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileReplacementBlocked)
	}
	if res.Generation.Baseline != "" || len(res.Generation.Snapshot) != 0 {
		t.Fatalf("blocked replacement populated a generation: %+v", res.Generation)
	}
}

// TestInKernelPlannerNilRegistryInert proves a bare planner with no registry set
// is a byte-for-byte no-op.
func TestInKernelPlannerNilRegistryInert(t *testing.T) {
	p := &InKernelPlanner{}

	text, invalidated, err := p.ApplyIncrementalContextUpdate()
	if err != nil {
		t.Fatalf("ApplyIncrementalContextUpdate: %v", err)
	}
	if text != "" || invalidated {
		t.Fatalf("nil registry: got text=%q invalidated=%v, want empty/false", text, invalidated)
	}

	gen, released, err := p.IncrementalContextGeneration()
	if err != nil {
		t.Fatalf("IncrementalContextGeneration: %v", err)
	}
	if released {
		t.Fatal("nil registry: IncrementalContextGeneration reported a released registry")
	}
	if gen.Baseline != "" || len(gen.Snapshot) != 0 {
		t.Fatalf("nil registry: generation = %+v, want zero", gen)
	}

	res, err := p.ReconcileIncrementalContextNow()
	if err != nil {
		t.Fatalf("ReconcileIncrementalContextNow: %v", err)
	}
	if res.Kind != ReconcileUnchanged {
		t.Fatalf("nil registry: Kind = %q, want %q", res.Kind, ReconcileUnchanged)
	}
}

// TestInKernelPlannerReconcileIdempotent proves the planner advances its stored
// snapshot on an update so the next reconcile is Unchanged.
func TestInKernelPlannerReconcileIdempotent(t *testing.T) {
	src := mustSource(t, "ns/count",
		func() (int, bool) { return 0, true },
		func(v int) string { return "count=" + itoa(v) },
		func(p, c int) string { return "count:" + itoa(p) + "->" + itoa(c) },
		noRemoved[int],
	)

	p := &InKernelPlanner{}
	p.SetIncrementalContext(MakeIncrementalContext(src))

	res1, err := p.ReconcileIncrementalContextNow()
	if err != nil {
		t.Fatalf("first ReconcileIncrementalContextNow: %v", err)
	}
	if res1.Kind != ReconcileUpdated {
		t.Fatalf("first reconcile Kind = %q, want %q", res1.Kind, ReconcileUpdated)
	}

	res2, err := p.ReconcileIncrementalContextNow()
	if err != nil {
		t.Fatalf("second ReconcileIncrementalContextNow: %v", err)
	}
	if res2.Kind != ReconcileUnchanged {
		t.Fatalf("second reconcile Kind = %q, want %q (snapshot must have advanced)",
			res2.Kind, ReconcileUnchanged)
	}

	// A settled snapshot means no further model-visible delta and no re-baseline.
	text, invalidated, err := p.ApplyIncrementalContextUpdate()
	if err != nil {
		t.Fatalf("ApplyIncrementalContextUpdate: %v", err)
	}
	if text != "" || invalidated {
		t.Fatalf("apply after idempotent reconcile: text=%q invalidated=%v, want empty/false", text, invalidated)
	}
}

// TestInKernelPlannerApplyEmitsDeltaOnFirstObservation proves a fresh planner
// surfaces the admitted baseline as the first model-visible delta and does not
// mark the resident context invalidated.
func TestInKernelPlannerApplyEmitsDeltaOnFirstObservation(t *testing.T) {
	src := mustSource(t, "ns/greeting",
		func() (string, bool) { return "hello", true },
		func(string) string { return "GREETING:hello" },
		func(p, c string) string { return "GREETING:" + p + "->" + c },
		noRemoved[string],
	)
	p := &InKernelPlanner{}
	p.SetIncrementalContext(MakeIncrementalContext(src))

	text, invalidated, err := p.ApplyIncrementalContextUpdate()
	if err != nil {
		t.Fatalf("ApplyIncrementalContextUpdate: %v", err)
	}
	if text != "GREETING:hello" {
		t.Fatalf("text = %q, want the admitted baseline delta", text)
	}
	if invalidated {
		t.Fatal("first observation must not invalidate the resident context")
	}
}

// TestInKernelPlannerApplyInvalidatesOnReplacement proves that when a required
// replacement occurs (a removed source with no Removed renderer), the planner
// reports invalidated=true so the resident context is re-baselined.
func TestInKernelPlannerApplyInvalidatesOnReplacement(t *testing.T) {
	src := mustSource(t, "ns/transient",
		func() (int, bool) { return 1, true },
		func(int) string { return "TRANSIENT-BASE" },
		func(int, int) string { return "TRANSIENT-UPD" },
		noRemoved[int],
	)
	p := &InKernelPlanner{}
	p.SetIncrementalContext(MakeIncrementalContext(src))

	// Admit the source first so its snapshot exists.
	if _, _, err := p.ApplyIncrementalContextUpdate(); err != nil {
		t.Fatalf("admission: %v", err)
	}

	// Drop the source: no Removed renderer => replacement required => invalidate.
	p.SetIncrementalContext(EmptyIncrementalContext())
	text, invalidated, err := p.ApplyIncrementalContextUpdate()
	if err != nil {
		t.Fatalf("replacement apply: %v", err)
	}
	if !invalidated {
		t.Fatal("replacement_ready must report invalidated=true")
	}
	if text != "" {
		t.Fatalf("text = %q, want empty on replacement (a full generation replaces it)", text)
	}
}

// TestInKernelDeterminismRemovedKeysSorted proves removed keys are processed in
// sorted order regardless of map iteration order.
func TestInKernelDeterminismRemovedKeysSorted(t *testing.T) {
	previous := ContextSnapshot{
		"ns/b": {Value: json.RawMessage("1"), Removed: "REMOVE-B"},
		"ns/a": {Value: json.RawMessage("2"), Removed: "REMOVE-A"},
	}

	res, err := ReconcileIncrementalContext(EmptyIncrementalContext(), previous)
	if err != nil {
		t.Fatalf("ReconcileIncrementalContext: %v", err)
	}
	if res.Kind != ReconcileUpdated {
		t.Fatalf("Kind = %q, want %q", res.Kind, ReconcileUpdated)
	}
	const want = "REMOVE-A\n\nREMOVE-B"
	if res.Text != want {
		t.Fatalf("Text = %q, want %q (removed keys must be sorted)", res.Text, want)
	}
}

// TestInKernelDeterminismRegistryOrder proves sources render in registry order,
// not map order.
func TestInKernelDeterminismRegistryOrder(t *testing.T) {
	a := mustSource(t, "ns/a",
		func() (int, bool) { return 1, true },
		func(int) string { return "A" },
		func(int, int) string { return "A-UPD" },
		noRemoved[int],
	)
	b := mustSource(t, "ns/b",
		func() (int, bool) { return 2, true },
		func(int) string { return "B" },
		func(int, int) string { return "B-UPD" },
		noRemoved[int],
	)

	ab, err := CombineIncrementalContext(MakeIncrementalContext(a), MakeIncrementalContext(b))
	if err != nil {
		t.Fatalf("CombineIncrementalContext(a,b): %v", err)
	}
	abGen, err := InitializeIncrementalContext(ab)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext(a,b): %v", err)
	}
	if abGen.Baseline != "A\n\nB" {
		t.Fatalf("registry order (a,b) baseline = %q, want %q", abGen.Baseline, "A\n\nB")
	}

	ba, err := CombineIncrementalContext(MakeIncrementalContext(b), MakeIncrementalContext(a))
	if err != nil {
		t.Fatalf("CombineIncrementalContext(b,a): %v", err)
	}
	baGen, err := InitializeIncrementalContext(ba)
	if err != nil {
		t.Fatalf("InitializeIncrementalContext(b,a): %v", err)
	}
	if baGen.Baseline != "B\n\nA" {
		t.Fatalf("registry order (b,a) baseline = %q, want %q", baGen.Baseline, "B\n\nA")
	}
}

// TestInKernelConfigIncrementalContextOptIn proves the construction-time
// IncrementalContext seam: the default config installs NO registry (released ==
// false), the opt-in config installs an EMPTY registry (released == true with a
// zero generation), that empty registry reconciles as Unchanged with empty text,
// and a bare &InKernelPlanner{} preserves the historical nil-registry inertness.
func TestInKernelConfigIncrementalContextOptIn(t *testing.T) {
	// 1. Default config: no registry installed.
	def := NewInKernelPlannerWithConfig(
		model.NewSynthetic(model.Config{}), nil, "incremental-default", true, nil, false,
		InKernelPlannerConfig{},
	)
	defGen, defReleased, err := def.IncrementalContextGeneration()
	if err != nil {
		t.Fatalf("default config IncrementalContextGeneration: %v", err)
	}
	if defReleased {
		t.Fatal("default config: released = true, want false (no registry installed)")
	}
	if defGen.Baseline != "" || len(defGen.Snapshot) != 0 {
		t.Fatalf("default config generation = %+v, want zero", defGen)
	}

	// 2. Opt-in config: an EMPTY registry is installed, reported as released with a
	// zero value generation.
	opt := NewInKernelPlannerWithConfig(
		model.NewSynthetic(model.Config{}), nil, "incremental-optin", true, nil, false,
		InKernelPlannerConfig{IncrementalContext: true},
	)
	optGen, optReleased, err := opt.IncrementalContextGeneration()
	if err != nil {
		t.Fatalf("opt-in config IncrementalContextGeneration: %v", err)
	}
	if !optReleased {
		t.Fatal("opt-in config: released = false, want true (empty registry installed)")
	}
	if optGen.Baseline != "" {
		t.Fatalf("opt-in config generation Baseline = %q, want empty", optGen.Baseline)
	}
	if len(optGen.Snapshot) != 0 {
		t.Fatalf("opt-in config generation Snapshot = %+v, want empty", optGen.Snapshot)
	}

	// 3. The empty opt-in registry is inert: reconcile reports Unchanged with no text.
	res, err := opt.ReconcileIncrementalContextNow()
	if err != nil {
		t.Fatalf("opt-in ReconcileIncrementalContextNow: %v", err)
	}
	if res.Kind != ReconcileUnchanged {
		t.Fatalf("opt-in empty registry Kind = %q, want %q", res.Kind, ReconcileUnchanged)
	}
	if res.Text != "" {
		t.Fatalf("opt-in empty registry Text = %q, want empty", res.Text)
	}

	// 4. Historical inertness is preserved: a bare planner reports no registry.
	bare := &InKernelPlanner{}
	bareGen, bareReleased, err := bare.IncrementalContextGeneration()
	if err != nil {
		t.Fatalf("bare planner IncrementalContextGeneration: %v", err)
	}
	if bareReleased {
		t.Fatal("bare planner: released = true, want false (historical nil-registry inertness)")
	}
	if bareGen.Baseline != "" || len(bareGen.Snapshot) != 0 {
		t.Fatalf("bare planner generation = %+v, want zero", bareGen)
	}
}

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
	"reflect"
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

// ---------------------------------------------------------------------------
// Layer-memo tests (TICKET-06, oss-port-p0-incremental-context).
//
// These are the ticket's own witness: the incrementally assembled prompt must be
// byte-identical to a full rebuild, and a change confined to one layer must
// re-render only that layer — naming the reuse path, not merely counting calls.
// ---------------------------------------------------------------------------

// newIncrementalContextPlanner builds a tiny in-kernel planner over the synthetic
// model and the byte-level probe tokenizer (the same fixtures the warm/prefix
// witnesses use). It needs no checkpoint, no GPU and no network.
func newIncrementalContextPlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	return &InKernelPlanner{
		m:       model.NewSynthetic(warmCfg()),
		modelID: "incremental-context",
		tok:     loadProbeTok(t),
	}
}

// incrementalContextTools is a one-tool surface so the Source layer carries a
// non-empty tool spec (the folded tool block is part of the Source fingerprint).
func incrementalContextTools() []ToolDef {
	return []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "lookup",
			Description: "look up a value",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}}}`),
		},
	}}
}

// TestInKernelIncrementalContextPromptMatchesFullRebuild is the ticket's named
// witness. Request 1 is cold (full rebuild). Request 2 appends ONE user turn: the
// memo must reuse the Source and Snapshot layers, re-render only the Registry
// layer, and produce a prompt (and token ids) byte-identical to a full rebuild.
// The reuse path is asserted explicitly — the delta render and the reassembled
// source+registry+delta+tail string — not by counting calls.
func TestInKernelIncrementalContextPromptMatchesFullRebuild(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}
	tools := incrementalContextTools()

	sys := Message{Role: RoleSystem, Content: "You are a careful helper."}
	user1 := Message{Role: RoleUser, Content: "first turn"}
	msgs1 := []Message{sys, user1}

	prompt1, ids1, stats1, err := p.RenderIncrementalContext(msgs1, tools, sp)
	if err != nil {
		t.Fatalf("request 1 RenderIncrementalContext: %v", err)
	}
	want1 := inKernelRenderPrompt(msgs1, tools, p.m.Cfg, sp)
	if prompt1 != want1 {
		t.Fatalf("request 1 prompt differs from the full rebuild:\n got %q\nwant %q", prompt1, want1)
	}
	wantIDs1, err := p.tok.Encode(want1)
	if err != nil {
		t.Fatalf("encode full rebuild 1: %v", err)
	}
	if !reflect.DeepEqual(ids1, wantIDs1) {
		t.Fatalf("request 1 ids differ from the full tokenization: got %v want %v", ids1, wantIDs1)
	}
	if stats1.FellBack {
		t.Fatal("request 1 (cold memo) must not report FellBack")
	}
	if !reflect.DeepEqual(stats1.Rendered, contextLayerAll()) {
		t.Fatalf("request 1 Rendered = %v, want all three layers", stats1.Rendered)
	}
	if len(stats1.Reused) != 0 {
		t.Fatalf("request 1 Reused = %v, want none on a cold memo", stats1.Reused)
	}

	user2 := Message{Role: RoleUser, Content: "second turn"}
	msgs2 := []Message{sys, user1, user2}

	prompt2, ids2, stats2, err := p.RenderIncrementalContext(msgs2, tools, sp)
	if err != nil {
		t.Fatalf("request 2 RenderIncrementalContext: %v", err)
	}
	want2 := inKernelRenderPrompt(msgs2, tools, p.m.Cfg, sp)
	if prompt2 != want2 {
		t.Fatalf("request 2 prompt differs from the full rebuild:\n got %q\nwant %q", prompt2, want2)
	}
	wantIDs2, err := p.tok.Encode(want2)
	if err != nil {
		t.Fatalf("encode full rebuild 2: %v", err)
	}
	if !reflect.DeepEqual(ids2, wantIDs2) {
		t.Fatalf("request 2 ids differ from the full tokenization: got %v want %v", ids2, wantIDs2)
	}
	if stats2.FellBack {
		t.Fatal("request 2 (pure append) must not report FellBack")
	}
	if !reflect.DeepEqual(stats2.Rendered, []ContextLayerKind{LayerRegistry}) {
		t.Fatalf("request 2 Rendered = %v, want exactly [registry]", stats2.Rendered)
	}
	if !reflect.DeepEqual(stats2.Reused, []ContextLayerKind{LayerSource, LayerSnapshot}) {
		t.Fatalf("request 2 Reused = %v, want [source snapshot]", stats2.Reused)
	}

	// Explicit reuse-path assertion: the assembled prompt is exactly the reused
	// Source layer + reused Registry layer + the re-rendered delta + the reused
	// Snapshot tail. This names the reuse path rather than counting calls.
	sourceText := renderTranscriptTools([]Message{sys}, tools)
	registryText := renderTranscriptTools([]Message{user1}, nil)
	deltaText := renderTranscriptTools([]Message{user2}, nil)
	transcript2 := renderTranscriptTools(msgs2, tools)
	tail, ok := strings.CutPrefix(want2, transcript2)
	if !ok {
		t.Fatalf("full rebuild does not start with the transcript; cannot derive the Snapshot tail")
	}
	if tail == "" {
		t.Fatal("Snapshot tail is empty; the reuse path would be vacuous")
	}
	reassembled := sourceText + registryText + deltaText + tail
	if reassembled != prompt2 {
		t.Fatalf("reassembled reuse path differs from the returned prompt:\n got %q\nwant %q", reassembled, prompt2)
	}
	if reassembled != want2 {
		t.Fatalf("reassembled reuse path differs from the full rebuild:\n got %q\nwant %q", reassembled, want2)
	}
}

// TestInKernelIncrementalContextFallsBackWhenSourceChanges proves a changed
// Source layer (a different system message) refuses the reuse path: FellBack is
// reported, nothing is reused, and the prompt is still the full rebuild.
func TestInKernelIncrementalContextFallsBackWhenSourceChanges(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}

	msgs1 := []Message{
		{Role: RoleSystem, Content: "system A"},
		{Role: RoleUser, Content: "hello"},
	}
	if _, _, _, err := p.RenderIncrementalContext(msgs1, nil, sp); err != nil {
		t.Fatalf("request 1 RenderIncrementalContext: %v", err)
	}

	msgs2 := []Message{
		{Role: RoleSystem, Content: "system B"},
		{Role: RoleUser, Content: "hello"},
		{Role: RoleUser, Content: "again"},
	}
	prompt2, ids2, stats2, err := p.RenderIncrementalContext(msgs2, nil, sp)
	if err != nil {
		t.Fatalf("request 2 RenderIncrementalContext: %v", err)
	}
	if !stats2.FellBack {
		t.Fatalf("changed Source layer: FellBack = false, want true (stats=%+v)", stats2)
	}
	if len(stats2.Reused) != 0 {
		t.Fatalf("changed Source layer: Reused = %v, want none", stats2.Reused)
	}
	want2 := inKernelRenderPrompt(msgs2, nil, p.m.Cfg, sp)
	if prompt2 != want2 {
		t.Fatalf("changed Source layer prompt differs from the full rebuild:\n got %q\nwant %q", prompt2, want2)
	}
	wantIDs2, err := p.tok.Encode(want2)
	if err != nil {
		t.Fatalf("encode full rebuild 2: %v", err)
	}
	if !reflect.DeepEqual(ids2, wantIDs2) {
		t.Fatalf("changed Source layer ids differ from the full tokenization: got %v want %v", ids2, wantIDs2)
	}
}

// TestInKernelIncrementalContextFallsBackWhenRegistryNotPrefix proves a Registry
// layer that is not an exact extension of the held one (a dropped turn) refuses
// the reuse path and still emits the full rebuild.
func TestInKernelIncrementalContextFallsBackWhenRegistryNotPrefix(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}

	sys := Message{Role: RoleSystem, Content: "stable system"}
	user1 := Message{Role: RoleUser, Content: "turn one"}
	user2 := Message{Role: RoleUser, Content: "turn two"}
	if _, _, _, err := p.RenderIncrementalContext([]Message{sys, user1, user2}, nil, sp); err != nil {
		t.Fatalf("request 1 RenderIncrementalContext: %v", err)
	}

	// Drop the last turn: the held registry is no longer a prefix of the request.
	msgs2 := []Message{sys, user1}
	prompt2, ids2, stats2, err := p.RenderIncrementalContext(msgs2, nil, sp)
	if err != nil {
		t.Fatalf("request 2 RenderIncrementalContext: %v", err)
	}
	if !stats2.FellBack {
		t.Fatalf("non-prefix Registry: FellBack = false, want true (stats=%+v)", stats2)
	}
	if len(stats2.Reused) != 0 {
		t.Fatalf("non-prefix Registry: Reused = %v, want none", stats2.Reused)
	}
	want2 := inKernelRenderPrompt(msgs2, nil, p.m.Cfg, sp)
	if prompt2 != want2 {
		t.Fatalf("non-prefix Registry prompt differs from the full rebuild:\n got %q\nwant %q", prompt2, want2)
	}
	wantIDs2, err := p.tok.Encode(want2)
	if err != nil {
		t.Fatalf("encode full rebuild 2: %v", err)
	}
	if !reflect.DeepEqual(ids2, wantIDs2) {
		t.Fatalf("non-prefix Registry ids differ from the full tokenization: got %v want %v", ids2, wantIDs2)
	}
}

// TestInKernelIncrementalContextNilMemoIsInert proves the seam is inert without a
// memo: a nil planner and a zero-value planner error instead of panicking, a
// planner with no memo returns the full rebuild with every layer rendered, and a
// nil *ContextLayerMemo passed straight to PlanContextLayers does the same.
func TestInKernelIncrementalContextNilMemoIsInert(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "system"},
		{Role: RoleUser, Content: "hello"},
	}
	sp := SampleParams{}

	var nilPlanner *InKernelPlanner
	nilPlanner.SetContextLayerMemo(nil) // must not panic
	if _, _, _, err := nilPlanner.RenderIncrementalContext(msgs, nil, sp); err == nil {
		t.Fatal("nil planner: got nil error, want a refusal")
	}

	zero := &InKernelPlanner{}
	if _, _, _, err := zero.RenderIncrementalContext(msgs, nil, sp); err == nil {
		t.Fatal("zero-value planner: got nil error, want a refusal")
	}

	p := newIncrementalContextPlanner(t)
	prompt, ids, stats, err := p.RenderIncrementalContext(msgs, nil, sp)
	if err != nil {
		t.Fatalf("no memo installed: %v", err)
	}
	want := inKernelRenderPrompt(msgs, nil, p.m.Cfg, sp)
	if prompt != want {
		t.Fatalf("no memo installed prompt differs from the full rebuild:\n got %q\nwant %q", prompt, want)
	}
	wantIDs, err := p.tok.Encode(want)
	if err != nil {
		t.Fatalf("encode full rebuild: %v", err)
	}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Fatalf("no memo installed ids differ from the full tokenization: got %v want %v", ids, wantIDs)
	}
	if stats.FellBack {
		t.Fatal("no memo installed must not report FellBack")
	}
	if !reflect.DeepEqual(stats.Rendered, contextLayerAll()) {
		t.Fatalf("no memo installed Rendered = %v, want all three layers", stats.Rendered)
	}

	// A nil memo used directly is inert too.
	full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
	plan, err := (*ContextLayerMemo)(nil).PlanContextLayers(msgs, nil, full, p.tok.Encode)
	if err != nil {
		t.Fatalf("nil memo PlanContextLayers: %v", err)
	}
	if plan.Prompt != want {
		t.Fatalf("nil memo prompt differs from the full rebuild:\n got %q\nwant %q", plan.Prompt, want)
	}
	if plan.Stats.FellBack {
		t.Fatal("nil memo must not report FellBack")
	}
	if !reflect.DeepEqual(plan.Stats.Rendered, contextLayerAll()) {
		t.Fatalf("nil memo Rendered = %v, want all three layers", plan.Stats.Rendered)
	}

	// A nil renderer or encoder is refused, never a panic.
	if _, err := NewContextLayerMemo().PlanContextLayers(msgs, nil, nil, p.tok.Encode); err == nil {
		t.Fatal("nil full renderer: got nil error, want a refusal")
	}
	if _, err := NewContextLayerMemo().PlanContextLayers(msgs, nil, full, nil); err == nil {
		t.Fatal("nil encoder: got nil error, want a refusal")
	}
}

// TestInKernelIncrementalContextThreeConsecutiveTurns proves the reuse holds
// across a run: turns 2 and 3 each reuse the Source and Snapshot layers, re-render
// only the Registry layer, and stay byte-identical to their full rebuilds.
func TestInKernelIncrementalContextThreeConsecutiveTurns(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}
	tools := incrementalContextTools()

	sys := Message{Role: RoleSystem, Content: "stable system"}
	base := []Message{sys, {Role: RoleUser, Content: "turn one"}}

	for turn := 1; turn <= 3; turn++ {
		msgs := append([]Message(nil), base...)
		for i := 2; i <= turn; i++ {
			msgs = append(msgs, Message{Role: RoleUser, Content: "turn " + itoa(i)})
		}
		prompt, ids, stats, err := p.RenderIncrementalContext(msgs, tools, sp)
		if err != nil {
			t.Fatalf("turn %d RenderIncrementalContext: %v", turn, err)
		}
		want := inKernelRenderPrompt(msgs, tools, p.m.Cfg, sp)
		if prompt != want {
			t.Fatalf("turn %d prompt differs from the full rebuild:\n got %q\nwant %q", turn, prompt, want)
		}
		wantIDs, err := p.tok.Encode(want)
		if err != nil {
			t.Fatalf("turn %d encode full rebuild: %v", turn, err)
		}
		if !reflect.DeepEqual(ids, wantIDs) {
			t.Fatalf("turn %d ids differ from the full tokenization: got %v want %v", turn, ids, wantIDs)
		}
		if stats.FellBack {
			t.Fatalf("turn %d: FellBack = true, want false", turn)
		}
		if turn == 1 {
			if !reflect.DeepEqual(stats.Rendered, contextLayerAll()) {
				t.Fatalf("turn 1 Rendered = %v, want all three layers (cold memo)", stats.Rendered)
			}
			continue
		}
		if !reflect.DeepEqual(stats.Rendered, []ContextLayerKind{LayerRegistry}) {
			t.Fatalf("turn %d Rendered = %v, want exactly [registry]", turn, stats.Rendered)
		}
		if !reflect.DeepEqual(stats.Reused, []ContextLayerKind{LayerSource, LayerSnapshot}) {
			t.Fatalf("turn %d Reused = %v, want [source snapshot]", turn, stats.Reused)
		}
	}
}

// TestInKernelIncrementalContextToolResultDeltaGuarded proves the mandatory guard
// covers the toolByID hazard: a Registry holding an assistant tool_call whose id a
// LATER tool result references cannot be re-rendered from the delta alone (the
// delta render has no tool_call to build the id->name map from). The guard must
// either prove the assembly byte-identical to the full rebuild, or fall back with
// FellBack=true. A prompt that differs from the full rebuild is a FAIL.
func TestInKernelIncrementalContextToolResultDeltaGuarded(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}
	tools := incrementalContextTools()

	sys := Message{Role: RoleSystem, Content: "stable system"}
	user1 := Message{Role: RoleUser, Content: "please look it up"}
	assistant := Message{
		Role:    RoleAssistant,
		Content: "",
		ToolCalls: []ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: Func{Name: "lookup", Arguments: `{"key":"x"}`},
		}},
	}
	msgs1 := []Message{sys, user1, assistant}
	if _, _, _, err := p.RenderIncrementalContext(msgs1, tools, sp); err != nil {
		t.Fatalf("request 1 RenderIncrementalContext: %v", err)
	}

	// The tool result carries only the call id: its rendered name comes from the
	// assistant tool_call in the HELD registry, which the delta render cannot see.
	toolResult := Message{Role: RoleTool, ToolCallID: "call_1", Content: "42"}
	msgs2 := []Message{sys, user1, assistant, toolResult}

	prompt2, ids2, stats2, err := p.RenderIncrementalContext(msgs2, tools, sp)
	if err != nil {
		t.Fatalf("request 2 RenderIncrementalContext: %v", err)
	}
	want2 := inKernelRenderPrompt(msgs2, tools, p.m.Cfg, sp)
	if prompt2 != want2 {
		t.Fatalf("tool-result delta prompt differs from the full rebuild:\n got %q\nwant %q", prompt2, want2)
	}
	wantIDs2, err := p.tok.Encode(want2)
	if err != nil {
		t.Fatalf("encode full rebuild 2: %v", err)
	}
	if !reflect.DeepEqual(ids2, wantIDs2) {
		t.Fatalf("tool-result delta ids differ from the full tokenization: got %v want %v", ids2, wantIDs2)
	}
	// The guard ALWAYS falls back for this scenario, deterministically: the delta
	// render has an empty toolByID, so the tool result renders as bare content,
	// while the full render resolves call_1 -> lookup from the HELD assistant
	// tool_call and wraps it in <tool_response>. The two assemblies can never be
	// byte-identical, so the reuse branch is unreachable here.
	if !stats2.FellBack {
		t.Fatalf("tool-result delta: FellBack = false, want true (stats=%+v)", stats2)
	}
	if len(stats2.Reused) != 0 {
		t.Fatalf("FellBack with Reused = %v, want none", stats2.Reused)
	}
}

// ---------------------------------------------------------------------------
// Fresh-context adversarial layer-memo tests (spec-only author; the
// implementation's internal choices were not consulted).
// ---------------------------------------------------------------------------

// incLayerContains reports whether kinds names want.
func incLayerContains(kinds []ContextLayerKind, want ContextLayerKind) bool {
	for _, k := range kinds {
		if k == want {
			return true
		}
	}
	return false
}

// incAssertMatchesFull renders through the planner and asserts invariant A: the
// returned prompt is byte-identical to the full rebuild and the ids are exactly
// the tokenization of that prompt.
func incAssertMatchesFull(t *testing.T, p *InKernelPlanner, msgs []Message, tools []ToolDef, sp SampleParams, label string) ContextLayerStats {
	t.Helper()
	prompt, ids, stats, err := p.RenderIncrementalContext(msgs, tools, sp)
	if err != nil {
		t.Fatalf("%s: RenderIncrementalContext: %v", label, err)
	}
	want := inKernelRenderPrompt(msgs, tools, p.m.Cfg, sp)
	if prompt != want {
		t.Fatalf("%s: prompt differs from the full rebuild:\n got %q\nwant %q", label, prompt, want)
	}
	wantIDs, err := p.tok.Encode(want)
	if err != nil {
		t.Fatalf("%s: encode full rebuild: %v", label, err)
	}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Fatalf("%s: ids differ from the full tokenization:\n got %v\nwant %v", label, ids, wantIDs)
	}
	return stats
}

// TestInKernelIncrementalContextAdversarialByteIdentity is invariant A over a
// spread of message shapes: cold and pure-append renders must both equal the full
// rebuild byte-for-byte (and token-for-token).
func TestInKernelIncrementalContextAdversarialByteIdentity(t *testing.T) {
	sp := SampleParams{}
	tools := incrementalContextTools()
	sys := Message{Role: RoleSystem, Content: "stable system"}

	shapes := []struct {
		name  string
		msgs  []Message
		tools []ToolDef
	}{
		{"system-only", []Message{sys}, nil},
		{"system+1user", []Message{sys, {Role: RoleUser, Content: "u1"}}, nil},
		{"system+2users", []Message{sys, {Role: RoleUser, Content: "u1"}, {Role: RoleUser, Content: "u2"}}, nil},
		{"no-system", []Message{{Role: RoleUser, Content: "u1"}}, nil},
		{"empty-system", []Message{{Role: RoleSystem, Content: ""}, {Role: RoleUser, Content: "u1"}}, nil},
		{"multi-system", []Message{{Role: RoleSystem, Content: "s1"}, {Role: RoleSystem, Content: "s2"}, {Role: RoleUser, Content: "u1"}}, nil},
		{"tools-present", []Message{sys, {Role: RoleUser, Content: "u1"}}, tools},
		{"assistant-toolcall-then-result", []Message{
			sys,
			{Role: RoleUser, Content: "look"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Type: "function", Function: Func{Name: "lookup", Arguments: `{"key":"x"}`}}}},
			{Role: RoleTool, ToolCallID: "c1", Content: "42"},
		}, tools},
		{"assistant-reasoning", []Message{sys, {Role: RoleUser, Content: "u1"}, {Role: RoleAssistant, Content: "answer", ReasoningContent: "step by step"}}, nil},
		{"empty-user-content", []Message{sys, {Role: RoleUser, Content: ""}}, nil},
		{"whitespace", []Message{{Role: RoleSystem, Content: "  padded  "}, {Role: RoleUser, Content: "\n lead and trail \t"}}, nil},
	}

	for _, sh := range shapes {
		sh := sh
		t.Run(sh.name, func(t *testing.T) {
			p := newIncrementalContextPlanner(t)
			p.SetContextLayerMemo(NewContextLayerMemo())
			incAssertMatchesFull(t, p, sh.msgs, sh.tools, sp, sh.name+" cold")
			appended := append(append([]Message(nil), sh.msgs...), Message{Role: RoleUser, Content: "appended turn"})
			incAssertMatchesFull(t, p, appended, sh.tools, sp, sh.name+" append")
		})
	}
}

// TestInKernelIncrementalContextAdversarialReuseAccounting is invariant B: a pure
// append reuses Source and Snapshot, re-renders only Registry, and stays
// byte-identical.
func TestInKernelIncrementalContextAdversarialReuseAccounting(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	p.SetContextLayerMemo(NewContextLayerMemo())
	sp := SampleParams{}
	tools := incrementalContextTools()
	sys := Message{Role: RoleSystem, Content: "stable system"}
	msgs1 := []Message{sys, {Role: RoleUser, Content: "turn one"}}

	stats1 := incAssertMatchesFull(t, p, msgs1, tools, sp, "request 1")
	if !reflect.DeepEqual(stats1.Rendered, contextLayerAll()) {
		t.Fatalf("request 1 Rendered = %v, want all three layers", stats1.Rendered)
	}
	if len(stats1.Reused) != 0 {
		t.Fatalf("request 1 Reused = %v, want none on a cold memo", stats1.Reused)
	}

	msgs2 := append(append([]Message(nil), msgs1...), Message{Role: RoleUser, Content: "turn two"})
	stats2 := incAssertMatchesFull(t, p, msgs2, tools, sp, "request 2")
	if stats2.FellBack {
		t.Fatalf("pure append: FellBack = true, want false (stats=%+v)", stats2)
	}
	if !incLayerContains(stats2.Reused, LayerSource) || !incLayerContains(stats2.Reused, LayerSnapshot) {
		t.Fatalf("pure append Reused = %v, want it to contain source and snapshot", stats2.Reused)
	}
	if !incLayerContains(stats2.Rendered, LayerRegistry) {
		t.Fatalf("pure append Rendered = %v, want it to contain registry", stats2.Rendered)
	}
	if incLayerContains(stats2.Rendered, LayerSource) || incLayerContains(stats2.Rendered, LayerSnapshot) {
		t.Fatalf("pure append Rendered = %v, must NOT contain source or snapshot", stats2.Rendered)
	}
}

// TestInKernelIncrementalContextAdversarialFallback is invariant C: a changed
// Source, or a Registry that is not a pure append (removed / reordered / edited
// turn), must fall back — full rebuild, FellBack=true, nothing reused.
func TestInKernelIncrementalContextAdversarialFallback(t *testing.T) {
	sp := SampleParams{}
	sys := Message{Role: RoleSystem, Content: "stable system"}
	u1 := Message{Role: RoleUser, Content: "turn one"}
	u2 := Message{Role: RoleUser, Content: "turn two"}

	cases := []struct {
		name string
		next []Message
	}{
		{"source-changed", []Message{{Role: RoleSystem, Content: "DIFFERENT system"}, u1, u2}},
		{"turn-removed", []Message{sys, u1}},
		{"turn-reordered", []Message{sys, u2, u1}},
		{"turn-edited", []Message{sys, u1, {Role: RoleUser, Content: "turn two EDITED"}}},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			p := newIncrementalContextPlanner(t)
			p.SetContextLayerMemo(NewContextLayerMemo())
			incAssertMatchesFull(t, p, []Message{sys, u1, u2}, nil, sp, c.name+" request 1")
			stats := incAssertMatchesFull(t, p, c.next, nil, sp, c.name+" request 2")
			if !stats.FellBack {
				t.Fatalf("%s: FellBack = false, want true (stats=%+v)", c.name, stats)
			}
			if len(stats.Reused) != 0 {
				t.Fatalf("%s: Reused = %v, want none", c.name, stats.Reused)
			}
		})
	}
}

// TestInKernelIncrementalContextAdversarialNilSafety is invariant D: nil planner,
// zero-value planner, nil memo, and nil renderer/encoder are all refusals or full
// rebuilds — never a panic.
func TestInKernelIncrementalContextAdversarialNilSafety(t *testing.T) {
	sp := SampleParams{}
	msgs := []Message{{Role: RoleSystem, Content: "system"}, {Role: RoleUser, Content: "hello"}}

	var nilPlanner *InKernelPlanner
	nilPlanner.SetContextLayerMemo(nil) // must not panic
	if _, _, _, err := nilPlanner.RenderIncrementalContext(msgs, nil, sp); err == nil {
		t.Fatal("nil planner: got nil error, want a refusal")
	}

	zero := &InKernelPlanner{}
	if _, _, _, err := zero.RenderIncrementalContext(msgs, nil, sp); err == nil {
		t.Fatal("zero-value planner (nil model): got nil error, want a refusal")
	}

	p := newIncrementalContextPlanner(t)
	stats := incAssertMatchesFull(t, p, msgs, nil, sp, "no memo installed")
	if stats.FellBack {
		t.Fatal("no memo installed: FellBack = true, want false")
	}
	if !reflect.DeepEqual(stats.Rendered, contextLayerAll()) {
		t.Fatalf("no memo installed Rendered = %v, want all three layers", stats.Rendered)
	}
	if len(stats.Reused) != 0 {
		t.Fatalf("no memo installed Reused = %v, want none", stats.Reused)
	}

	full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
	want := full(msgs, nil)
	plan, err := (*ContextLayerMemo)(nil).PlanContextLayers(msgs, nil, full, p.tok.Encode)
	if err != nil {
		t.Fatalf("nil memo PlanContextLayers: %v", err)
	}
	if plan.Prompt != want {
		t.Fatalf("nil memo prompt differs from the full rebuild:\n got %q\nwant %q", plan.Prompt, want)
	}
	if plan.Stats.FellBack {
		t.Fatal("nil memo: FellBack = true, want false")
	}
	if !reflect.DeepEqual(plan.Stats.Rendered, contextLayerAll()) {
		t.Fatalf("nil memo Rendered = %v, want all three layers", plan.Stats.Rendered)
	}

	if _, err := NewContextLayerMemo().PlanContextLayers(msgs, nil, nil, p.tok.Encode); err == nil {
		t.Fatal("nil full renderer: got nil error, want a refusal")
	}
	if _, err := NewContextLayerMemo().PlanContextLayers(msgs, nil, full, nil); err == nil {
		t.Fatal("nil encoder: got nil error, want a refusal")
	}
}

// incMergeEncode is a deterministic, boundary-sensitive encoder: two adjacent
// ASCII letters collapse into ONE id. A naive layer-concatenation of ids will
// therefore disagree with encoding the whole prompt whenever a merge spans a
// layer boundary — exactly the case the memo must detect and fall back on.
func incMergeEncode(s string) ([]int, error) {
	rs := []rune(s)
	ids := make([]int, 0, len(rs))
	isLetter := func(r rune) bool { return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') }
	for i := 0; i < len(rs); {
		if i+1 < len(rs) && isLetter(rs[i]) && isLetter(rs[i+1]) {
			ids = append(ids, 100000+int(rs[i])*256+int(rs[i+1]))
			i += 2
			continue
		}
		ids = append(ids, int(rs[i]))
		i++
	}
	return ids, nil
}

// TestInKernelIncrementalContextAdversarialTokenizer is invariant E: a surprising
// encoder must never yield a prompt that differs from the full rebuild, and a
// non-fallback result must carry ids equal to encoding that prompt.
func TestInKernelIncrementalContextAdversarialTokenizer(t *testing.T) {
	sp := SampleParams{}
	tools := incrementalContextTools()
	sys := Message{Role: RoleSystem, Content: "stable system"}
	msgs1 := []Message{sys, {Role: RoleUser, Content: "turn one"}}
	msgs2 := append(append([]Message(nil), msgs1...), Message{Role: RoleUser, Content: "turn two"})

	t.Run("merge-encode", func(t *testing.T) {
		p := newIncrementalContextPlanner(t)
		memo := NewContextLayerMemo()
		full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
		for i, msgs := range [][]Message{msgs1, msgs2} {
			plan, err := memo.PlanContextLayers(msgs, tools, full, incMergeEncode)
			if err != nil {
				t.Fatalf("call %d: PlanContextLayers: %v", i+1, err)
			}
			want := full(msgs, tools)
			if plan.Prompt != want {
				t.Fatalf("call %d: prompt differs from the full rebuild:\n got %q\nwant %q", i+1, plan.Prompt, want)
			}
			if !plan.Stats.FellBack {
				wantIDs, err := incMergeEncode(want)
				if err != nil {
					t.Fatalf("call %d: merge encode: %v", i+1, err)
				}
				if !reflect.DeepEqual(plan.IDs, wantIDs) {
					t.Fatalf("call %d: non-fallback ids differ from the merge tokenization of the prompt:\n got %v\nwant %v", i+1, plan.IDs, wantIDs)
				}
			} else if len(plan.Stats.Reused) != 0 {
				t.Fatalf("call %d: FellBack with Reused = %v, want none", i+1, plan.Stats.Reused)
			}
		}
	})

	t.Run("stateful-length-changing-encode", func(t *testing.T) {
		p := newIncrementalContextPlanner(t)
		memo := NewContextLayerMemo()
		full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
		calls := 0
		stateful := func(s string) ([]int, error) {
			calls++
			ids := make([]int, 0, len(s)+calls)
			for i := 0; i < len(s); i++ {
				ids = append(ids, int(s[i]))
			}
			for i := 0; i < calls; i++ {
				ids = append(ids, -calls)
			}
			return ids, nil
		}
		for i, msgs := range [][]Message{msgs1, msgs2} {
			plan, err := memo.PlanContextLayers(msgs, tools, full, stateful)
			if err != nil {
				t.Fatalf("call %d: PlanContextLayers: %v", i+1, err)
			}
			want := full(msgs, tools)
			if plan.Prompt != want {
				t.Fatalf("call %d: prompt differs from the full rebuild:\n got %q\nwant %q", i+1, plan.Prompt, want)
			}
			if plan.Stats.FellBack && len(plan.Stats.Reused) != 0 {
				t.Fatalf("call %d: FellBack with Reused = %v, want none", i+1, plan.Stats.Reused)
			}
		}
	})
}

// TestInKernelIncrementalContextAdversarialFullRenderer is invariant F: a `full`
// renderer that appends an open-assistant-turn tail must be reproduced exactly,
// including on the reuse path.
func TestInKernelIncrementalContextAdversarialFullRenderer(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	sp := SampleParams{}
	tools := incrementalContextTools()
	sys := Message{Role: RoleSystem, Content: "stable system"}
	msgs1 := []Message{sys, {Role: RoleUser, Content: "turn one"}}
	msgs2 := append(append([]Message(nil), msgs1...), Message{Role: RoleUser, Content: "turn two"})

	const suffix = "\x00<<open-assistant-turn>>"
	full := func(ms []Message, ts []ToolDef) string {
		return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) + suffix
	}
	memo := NewContextLayerMemo()
	for i, msgs := range [][]Message{msgs1, msgs2} {
		plan, err := memo.PlanContextLayers(msgs, tools, full, p.tok.Encode)
		if err != nil {
			t.Fatalf("call %d: PlanContextLayers: %v", i+1, err)
		}
		want := full(msgs, tools)
		if plan.Prompt != want {
			t.Fatalf("call %d: prompt differs from the adversarial full renderer:\n got %q\nwant %q", i+1, plan.Prompt, want)
		}
		if !strings.HasSuffix(plan.Prompt, suffix) {
			t.Fatalf("call %d: prompt %q lost the open-turn suffix", i+1, plan.Prompt)
		}
	}
}

// TestInKernelIncrementalContextAdversarialConcurrent is invariant G: one memo
// shared by ~8 goroutines must never panic and every returned prompt must equal
// the full rebuild. Each goroutine gets its own tokenizer so the tokenizer itself
// is not the shared mutable state under test.
func TestInKernelIncrementalContextAdversarialConcurrent(t *testing.T) {
	p := newIncrementalContextPlanner(t)
	sp := SampleParams{}
	tools := incrementalContextTools()
	memo := NewContextLayerMemo()

	type encoder interface{ Encode(string) ([]int, error) }
	const n = 8
	toks := make([]encoder, n)
	for i := range toks {
		toks[i] = loadProbeTok(t)
	}

	type result struct {
		label  string
		prompt string
		want   string
		err    error
	}
	ch := make(chan result, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			msgs := []Message{
				{Role: RoleSystem, Content: "stable system"},
				{Role: RoleUser, Content: "turn " + itoa(i)},
			}
			full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
			plan, err := memo.PlanContextLayers(msgs, tools, full, toks[i].Encode)
			ch <- result{label: "goroutine " + itoa(i), prompt: plan.Prompt, want: full(msgs, tools), err: err}
		}(i)
	}
	for i := 0; i < n; i++ {
		r := <-ch
		if r.err != nil {
			t.Fatalf("%s: PlanContextLayers: %v", r.label, r.err)
		}
		if r.prompt != r.want {
			t.Fatalf("%s: prompt differs from the full rebuild:\n got %q\nwant %q", r.label, r.prompt, r.want)
		}
	}
}

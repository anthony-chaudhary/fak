package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

type principalScopeEngine struct{ calls int }

func (*principalScopeEngine) Caps() []abi.Capability { return nil }
func (e *principalScopeEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.calls++
	b := []byte(`{"ok":true}`)
	ref, _ := abi.ActiveResolver().Put(ctx, b)
	return &abi.Result{Status: abi.StatusOK, Payload: ref, Meta: map[string]string{"world_version": "1"}}, nil
}

func TestNativeLoopScopesVdsoPerPrincipal(t *testing.T) {
	vdso.Default.BumpWorld()
	ctx := context.Background()
	eng := &principalScopeEngine{}
	abi.RegisterEngine("principal-scope-5707", eng)
	k := kernel.New("principal-scope-5707")
	k.SetVDSO(true)
	ev := traceEvent{}
	_, _ = execViaKernel(ctx, k, toolSearch, `{"q":"same"}`, "", ev, "tenant-a")
	_, _ = execViaKernel(ctx, k, toolSearch, `{"q":"same"}`, "", ev, "tenant-b")
	if eng.calls != 2 {
		t.Fatalf("cross-principal call was cached: engine calls=%d want 2", eng.calls)
	}
	_, _ = execViaKernel(ctx, k, toolSearch, `{"q":"same"}`, "", ev, "tenant-b")
	if eng.calls != 2 {
		t.Fatalf("same-principal repeat missed cache: calls=%d want 2", eng.calls)
	}
}

func TestNativeLoopEmptyPrincipalPreservesSharedScope(t *testing.T) {
	vdso.Default.BumpWorld()
	ctx := context.Background()
	eng := &principalScopeEngine{}
	abi.RegisterEngine("empty-scope-5707", eng)
	k := kernel.New("empty-scope-5707")
	k.SetVDSO(true)
	ev := traceEvent{}
	_, _ = execViaKernel(ctx, k, toolSearch, `{"q":"empty"}`, "", ev)
	_, _ = execViaKernel(ctx, k, toolSearch, `{"q":"empty"}`, "", ev, "")
	if eng.calls != 1 {
		t.Fatalf("empty principal no longer shares legacy scope: calls=%d", eng.calls)
	}
	_ = vdso.MetaPrincipal // pin the public key used by the runtime lowering.
}

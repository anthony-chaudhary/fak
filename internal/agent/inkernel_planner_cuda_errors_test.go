package agent

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestRecoverDevicePanic_CUDATypedErrorsHandled(t *testing.T) {
	// 1. *compute.CUDAOpError
	opErr := &compute.CUDAOpError{
		Op:    "MatMul",
		Site:  "gemv-forward",
		Msg:   "cuda graph capture/launch failed",
		Code:  1,
		Class: compute.MemoryActivation,
	}
	err, handled := recoverDevicePanic(opErr)
	if !handled {
		t.Fatal("recoverDevicePanic must handle *compute.CUDAOpError")
	}
	var recoveredOp *compute.CUDAOpError
	if !errors.As(err, &recoveredOp) {
		t.Fatalf("want *compute.CUDAOpError, got %T (%v)", err, err)
	}
	if recoveredOp.Op != "MatMul" || recoveredOp.Site != "gemv-forward" || recoveredOp.Code != 1 {
		t.Fatalf("unexpected fields on recovered CUDAOpError: %+v", recoveredOp)
	}

	// 2. *compute.CUDALaunchError
	launchErr := &compute.CUDALaunchError{
		CUDAOpError: compute.CUDAOpError{
			Op:   "GraphEndLaunch",
			Site: "cuda-graph-launch",
			Msg:  "capture launch failed",
			Code: 2,
		},
	}
	err, handled = recoverDevicePanic(launchErr)
	if !handled {
		t.Fatal("recoverDevicePanic must handle *compute.CUDALaunchError")
	}
	var recoveredLaunch *compute.CUDALaunchError
	if !errors.As(err, &recoveredLaunch) {
		t.Fatalf("want *compute.CUDALaunchError, got %T (%v)", err, err)
	}
	if recoveredLaunch.Op != "GraphEndLaunch" || recoveredLaunch.Code != 2 {
		t.Fatalf("unexpected fields on recovered CUDALaunchError: %+v", recoveredLaunch)
	}

	// 3. *compute.DeviceFaultError
	faultErr := &compute.DeviceFaultError{
		Backend: "cuda",
		Site:    "qwen35-gdn-decode",
		Class:   compute.DeviceFaultExecution,
		Code:    31,
	}
	err, handled = recoverDevicePanic(faultErr)
	if !handled {
		t.Fatal("recoverDevicePanic must handle *compute.DeviceFaultError")
	}
	var recoveredFault *compute.DeviceFaultError
	if !errors.As(err, &recoveredFault) {
		t.Fatalf("want *compute.DeviceFaultError, got %T (%v)", err, err)
	}
	if recoveredFault.Site != "qwen35-gdn-decode" || recoveredFault.Code != 31 {
		t.Fatalf("unexpected fields on recovered DeviceFaultError: %+v", recoveredFault)
	}

	// 4. Wrapped CUDAOpError
	wrapped := fmt.Errorf("step failure: %w", opErr)
	err, handled = recoverDevicePanic(wrapped)
	if !handled {
		t.Fatal("recoverDevicePanic must handle wrapped *compute.CUDAOpError")
	}
	if !errors.As(err, &recoveredOp) || recoveredOp.Op != "MatMul" {
		t.Fatalf("wrapped error recovery failed: %v", err)
	}
}

type cudaPanicInjectionBackend struct {
	compute.Backend
	panicOnMatMul bool
	panicPayload  any
	callCount     int
}

func (b *cudaPanicInjectionBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.callCount++
	if b.panicOnMatMul {
		b.panicOnMatMul = false
		panic(b.panicPayload)
	}
	return b.Backend.MatMul(w, x)
}

func TestInKernelPlanner_CUDAPanicRecoveryAndSubsequentHealth(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)

	injected := &compute.CUDAOpError{
		Op:    "MatMul",
		Site:  "forward-gemm",
		Msg:   "injected cuda gemm failure",
		Class: compute.MemoryActivation,
	}

	be := &cudaPanicInjectionBackend{
		Backend:       compute.Default(),
		panicOnMatMul: true,
		panicPayload:  injected,
	}

	p := &InKernelPlanner{m: m, modelID: "cuda-panic-test", backend: be}
	ids := synthIDs(cfg.VocabSize, 4, 10523)

	// 1. First request fails with the injected panic, recovered into typed error cleanly
	_, err := p.generateReusedRecovering(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err == nil {
		t.Fatal("expected request 1 to fail with recovered CUDA error")
	}

	var opErr *compute.CUDAOpError
	if !errors.As(err, &opErr) {
		t.Fatalf("request 1 error = %T (%v), want *compute.CUDAOpError", err, err)
	}
	if opErr.Op != "MatMul" || opErr.Site != "forward-gemm" {
		t.Fatalf("unexpected CUDA error fields: %+v", opErr)
	}

	// 2. Subsequent request on the SAME planner must succeed (remains healthy)
	res, err := p.generateReusedRecovering(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err != nil {
		t.Fatalf("subsequent request failed: %v", err)
	}
	if res.gen != 2 {
		t.Fatalf("subsequent request generated %d tokens, want 2", res.gen)
	}
}

func TestInKernelPlanner_CUDALaunchPanicRecoveryAndSubsequentHealth(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)

	launchErr := &compute.CUDALaunchError{
		CUDAOpError: compute.CUDAOpError{
			Op:   "GraphEndLaunch",
			Site: "cuda-graph-launch",
			Msg:  "cuda graph capture/launch failed",
			Code: 1,
		},
	}

	be := &cudaPanicInjectionBackend{
		Backend:       compute.Default(),
		panicOnMatMul: true,
		panicPayload:  launchErr,
	}

	p := &InKernelPlanner{m: m, modelID: "cuda-launch-panic-test", backend: be}
	ids := synthIDs(cfg.VocabSize, 4, 10524)

	// Request 1 fails with typed launch error
	_, err := p.generateReusedRecovering(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err == nil {
		t.Fatal("expected request 1 to fail with launch error")
	}
	var recoveredLaunch *compute.CUDALaunchError
	if !errors.As(err, &recoveredLaunch) {
		t.Fatalf("want *compute.CUDALaunchError, got %T (%v)", err, err)
	}
	if recoveredLaunch.Code != 1 {
		t.Fatalf("code = %d, want 1", recoveredLaunch.Code)
	}

	// Request 2 succeeds
	res, err := p.generateReusedRecovering(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err != nil {
		t.Fatalf("subsequent request failed: %v", err)
	}
	if res.gen != 2 {
		t.Fatalf("subsequent request generated %d tokens, want 2", res.gen)
	}
}

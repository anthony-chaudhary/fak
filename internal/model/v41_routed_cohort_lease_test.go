package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU protocol witness only: the mutex is a recording domain, not a GPU.
type v41CohortLeaseWitness struct {
	*v41ScalarCohortWitness
	owner   sync.Mutex
	uploads []string
	adds    int
}

func (b *v41CohortLeaseWitness) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41CohortLeaseWitness) checkOwner() {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned routed cohort operation")
	}
}
func (b *v41CohortLeaseWitness) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.checkOwner()
	if class == compute.MemoryActivation {
		b.uploads = append(b.uploads, site)
	}
	return b.Backend.Upload(x, dt)
}
func (b *v41CohortLeaseWitness) MatMul(w, x compute.Tensor) compute.Tensor {
	b.checkOwner()
	return b.v41ScalarCohortWitness.MatMul(w, x)
}
func (b *v41CohortLeaseWitness) AddInPlace(dst, x compute.Tensor) {
	b.checkOwner()
	b.adds++
	b.Backend.AddInPlace(dst, x)
}
func (b *v41CohortLeaseWitness) Free(x compute.Tensor) {
	b.checkOwner()
	b.v41ScalarCohortWitness.Free(x)
}
func (b *v41CohortLeaseWitness) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.checkOwner()
	return b.v41ScalarCohortWitness.V41ReadDeferred(st, rows)
}

// fak-test:runtime medium est=5s lane=default
func TestV41RoutedCohortLeaseProtocol(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedFusedEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	m := v41PinnedSlateFixture(t, false)
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	t.Cleanup(func() { m.v41SetExpertFaultPhase(V41PhaseUnknown) })
	inputs := make([][]float32, 2)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r+i)%9-4) / 16
		}
	}
	picks := [][]routePick{{{expert: 1, weight: .25}, {expert: 0, weight: .5}}, {{expert: 1, weight: .125}, {expert: 0, weight: .75}}}
	newSessions := func() ([]*Session, *v41CohortLeaseWitness) {
		b := &v41CohortLeaseWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}}
		sessions := make([]*Session, 2)
		for i := range sessions {
			var err error
			sessions[i], err = m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(sessions[i].Close)
		}
		return sessions, b
	}
	prepare := func(t *testing.T, sessions []*Session, scope *backendExecutionScope) []*v41RoutedCohortLease {
		t.Helper()
		for _, s := range sessions {
			scope.bind(s)
		}
		groups := [][]v41RoutedCohortRow{
			{{Lane: 2, Session: sessions[0], Input: inputs[0], Weight: .25}, {Lane: 7, Session: sessions[1], Input: inputs[1], Weight: .125}},
			{{Lane: 2, Session: sessions[0], Input: inputs[0], Weight: .5}, {Lane: 7, Session: sessions[1], Input: inputs[1], Weight: .75}},
		}
		leases := make([]*v41RoutedCohortLease, 0, 2)
		for i, rows := range groups {
			lease, outcome, lane, err := prepareV41RoutedExpertCohortOwned(scope, 0, "ffn.experts."+itoa(1-i), rows)
			if err != nil || outcome != v41RoutedExpertHandled || lane != -1 || lease == nil {
				t.Fatalf("prepare %d: outcome=%v lane=%d error=%v", i, outcome, lane, err)
			}
			leases = append(leases, lease)
		}
		return leases
	}
	closeLeases := func(leases []*v41RoutedCohortLease) {
		for i := len(leases) - 1; i >= 0; i-- {
			leases[i].Close()
			leases[i].Close()
		}
	}
	sums := []v41RoutedCohortSum{{Lane: 7, Rows: []int{3, 1}}, {Lane: 2, Rows: []int{2, 0}}}
	controlSessions, control := newSessions()
	want := make([][]float32, 2)
	func() {
		scope := beginBackendExecution(control)
		defer scope.finish()
		for i, s := range controlSessions {
			scope.bind(s)
			var outcome v41RoutedExpertOutcome
			var err error
			want[i], outcome, err = s.v41RoutedLayerFunc()(0, picks[i], inputs[i])
			if outcome != v41RoutedExpertHandled || err != nil {
				t.Fatalf("control lane %d: %v %v", i, outcome, err)
			}
		}
	}()
	sessions, b := newSessions()
	func() {
		scope := beginBackendExecution(b)
		defer scope.finish()
		leases := prepare(t, sessions, scope)
		defer closeLeases(leases)
		if len(b.matmulInputs) != 0 || b.fences != 0 {
			t.Fatal("prepare selected expert arithmetic")
		}
		before := m.V41ExpertFaultAttribution().Decode
		got, lane, err := runV41RoutedCohortLeases(leases, sums)
		if err != nil || lane != -1 || len(got) != 2 || !v41SameBits(got[0], want[1]) || !v41SameBits(got[1], want[0]) {
			t.Fatalf("combined lane sums differ: lane=%d error=%v", lane, err)
		}
		after := m.V41ExpertFaultAttribution().Decode
		if after.IncrementalDeviceGateUpCalls-before.IncrementalDeviceGateUpCalls != 4 || after.IncrementalDeviceDownCalls-before.IncrementalDeviceDownCalls != 4 || after.Contractions-before.Contractions != 4 {
			t.Fatalf("successful dispatch/contraction counters differ: before=%+v after=%+v", before, after)
		}
		if b.fences != 1 || control.fences != 2 || b.adds != 4 || len(b.matmulInputs) != 12 || len(b.pending) != 0 {
			t.Fatalf("candidate/control fences=%d/%d adds=%d matmuls=%d pending=%d", b.fences, control.fences, b.adds, len(b.matmulInputs), len(b.pending))
		}
		if !reflect.DeepEqual(b.uploads, []string{"moe expert gate/up activation", "moe expert gate/up activation", "V4.1 routed-sum accumulator", "V4.1 routed-sum accumulator"}) {
			t.Fatalf("inputs were not reused by lane: %v", b.uploads)
		}
		// Expert-major recording changes only cross-lane order. Each scalar
		// gate/up/down operand and output matches the original lane execution.
		for row, original := range []int{0, 2, 1, 3} {
			for stage := 0; stage < 3; stage++ {
				if !v41SameBits(b.matmulInputs[3*row+stage], control.matmulInputs[3*original+stage]) || !v41SameBits(b.matmulOutputs[3*row+stage], control.matmulOutputs[3*original+stage]) {
					t.Fatal("scalar kernel operand/output changed")
				}
			}
		}
		if _, _, err := runV41RoutedCohortLeases(leases, sums); err == nil || b.fences != 1 {
			t.Fatal("consumed lease replayed")
		}
	}()
	t.Run("earlier_verdict_keeps_partial_attempt_counts_and_retires_all", func(t *testing.T) {
		sessions, b := newSessions()
		scope := beginBackendExecution(b)
		defer scope.finish()
		leases := prepare(t, sessions, scope)
		defer closeLeases(leases)
		b.publishFaultAt = map[int]bool{1: true}
		b.failMatmul = 4
		b.failure = &compute.BackendError{Backend: "test", Site: "later gate", Err: errors.New("selected later recording failure")}
		before := m.V41ExpertFaultAttribution().Decode
		out, lane, err := runV41RoutedCohortLeases(leases, sums)
		var numeric *V41ExpertOperationError
		if out != nil || lane != -1 || !errors.As(err, &numeric) || numeric.Stage != "down" {
			t.Fatalf("earlier verdict lost: lane=%d err=%v", lane, err)
		}
		after := m.V41ExpertFaultAttribution().Decode
		if after.IncrementalDeviceGateUpCalls-before.IncrementalDeviceGateUpCalls != 2 || after.IncrementalDeviceDownCalls-before.IncrementalDeviceDownCalls != 1 || after.Contractions != before.Contractions {
			t.Fatal("partial selected attempts were hidden or fabricated")
		}
		for _, s := range sessions {
			if !s.BackendSessionClosed() {
				t.Fatal("selected cohort participant remains reusable")
			}
		}
		if b.fences != 1 || b.adds != 0 || len(b.pending) != 0 || len(b.matmulInputs) != 4 {
			t.Fatal("failed selected cohort replayed or leaked")
		}
	})
}

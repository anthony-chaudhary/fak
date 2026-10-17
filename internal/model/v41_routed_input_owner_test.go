package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41InputOwnerWitness struct {
	*v41ExpertTileWitness
	mu                     sync.Mutex
	inputBytes             int
	liveInput              map[compute.Buffer]bool
	inputFrees             int
	failUpload, failGather bool
	recycled               bool
}

func (b *v41InputOwnerWitness) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *v41InputOwnerWitness) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 layer routed input" && b.failUpload {
		panic(&compute.BackendError{Backend: "test", Site: "input upload", Err: errors.New("upload failure")})
	}
	if site == "V4.1 layer routed input" || site == "V4.1 same-expert token tile" || site == "moe expert gate/up activation" {
		b.inputBytes += x.Numel() * 4
	}
	y := b.Backend.Upload(x, dt)
	if site == "V4.1 layer routed input" {
		b.liveInput[y.Buf()] = true
		b.ownedTiles[y.Buf()] = true
	}
	return y
}
func (b *v41InputOwnerWitness) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	y, err := b.v41ExpertTileWitness.V41RowLogGather(spans, width)
	if err == nil && b.failGather && b.liveInput[spans[0].Src.Buf()] {
		return y, &compute.BackendError{Backend: "test", Site: "input gather", Err: errors.New("gather failure")}
	}
	return y, err
}
func (b *v41InputOwnerWitness) Free(x compute.Tensor) {
	if b.liveInput[x.Buf()] {
		if b.recycled {
			panic("input released after recycle")
		}
		delete(b.liveInput, x.Buf())
		b.inputFrees++
	}
	b.v41ExpertTileWitness.Free(x)
}
func (b *v41InputOwnerWitness) Recycle() {
	if len(b.liveInput) != 0 {
		panic("live layer input at recycle")
	}
	b.recycled = true
}

// fak-test:runtime medium est=5s lane=default
func TestV41LayerInputOwnerUploadAndFailure(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs := make([][]float32, 2)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((i+r)%7-3) / 16
		}
	}
	for _, mode := range []string{"reuse", "upload", "gather"} {
		t.Run(mode, func(t *testing.T) {
			b := &v41InputOwnerWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, liveInput: map[compute.Buffer]bool{}}
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			scope := s.beginBackendExecution()
			defer scope.finish()
			defer s.closeBackendOwned()
			weights := []float32{.25, .5}
			control, controlOutcome, controlErr := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			if controlOutcome != v41RoutedExpertHandled || controlErr != nil {
				t.Fatalf("control=%v/%v", controlOutcome, controlErr)
			}
			baselineBytes := b.inputBytes
			b.inputBytes, b.panels = 0, nil
			owner := newV41RoutedInputOwner(s, 0, inputs)
			if owner == nil {
				t.Fatal("input owner declined")
			}
			defer owner.release()
			b.failUpload, b.failGather = mode == "upload", mode == "gather"
			rows := []v41ExpertRow{{Token: 0}, {Token: 1}}
			for n := 0; n < 2; n++ {
				selectedInputs, selectedWeights := inputs, weights
				if n == 1 {
					rows = []v41ExpertRow{{Token: 1}, {Token: 0}}
					selectedInputs = [][]float32{inputs[1], inputs[0]}
					selectedWeights = []float32{weights[1], weights[0]}
				}
				got, outcome, cause := s.v41RoutedExpertCohortInputs(0, "ffn.experts.0", selectedInputs, selectedWeights, nil, owner.handoff(rows))
				if mode != "reuse" {
					if outcome != v41RoutedExpertError || cause == nil || !s.halClosed || len(b.liveInput) != 0 || len(b.ownedTiles) != 0 {
						t.Fatalf("failure outcome=%v cause=%v closed=%v input=%d tiles=%d", outcome, cause, s.halClosed, len(b.liveInput), len(b.ownedTiles))
					}
					return
				}
				if outcome != v41RoutedExpertHandled || cause != nil {
					t.Fatalf("outcome=%v cause=%v", outcome, cause)
				}
				for r := range got {
					if !v41SameBits(control[rows[r].Token], got[r]) {
						t.Fatal("resident input changed baseline output")
					}
				}
				b.panels = nil
			}
			if baselineBytes != len(inputs)*m.Cfg.HiddenSize*4 || b.inputBytes != baselineBytes || b.inputFrees != 0 {
				t.Fatalf("bytes=%d frees=%d", b.inputBytes, b.inputFrees)
			}
			owner.release()
			if b.inputFrees != 1 || len(b.liveInput) != 0 {
				t.Fatal("layer input not released exactly once")
			}
		})
	}
}

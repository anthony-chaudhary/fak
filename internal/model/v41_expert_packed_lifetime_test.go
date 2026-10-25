package model

import (
	"bytes"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41PackedLifetimeBackend struct {
	compute.Backend
	during         func()
	weight, scales []byte
}

func (b *v41PackedLifetimeBackend) GemvExpertMXFP4E8M0(o, i int, weight, scales []byte, x []float32) ([]float32, error) {
	if b.during != nil {
		b.during()
	}
	b.weight = append([]byte(nil), weight...)
	b.scales = append([]byte(nil), scales...)
	return make([]float32, o), nil
}

func (b *v41PackedLifetimeBackend) GemvExpertFP8Block32E8M0(o, i int, weight, scales []byte, x []float32) ([]float32, error) {
	panic("MXFP4 fixture selected the FP8 path")
}

// The existing reduced-profile fixture is process-global, so this root and its
// subtests must remain nonparallel; every contender joins before restoration.
// fak-test:runtime fast est=100ms lane=race
func TestV41ExpertPackedModelLifetime(t *testing.T) {
	saved := v41ExpertQuantSpecs
	defer func() { v41ExpertQuantSpecs = saved }()
	const rows, packedCols, scaleCols = 2, 16, 1
	name, weights, scales := v41ExpertDeviceFixture(t, rows, packedCols, scaleCols)
	name += ".weight"
	scaleName := name[:len(name)-len("weight")] + "scale"
	we := stEntry{Dtype: "I8", Shape: []int{rows, packedCols}}
	se := stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}}
	newModel := func() *Model {
		return &Model{Cfg: Config{DeepSeekV41: &DeepSeekV41Config{}}}
	}
	install := func(m *Model) error {
		return installV41ExpertQuantDevice(name, scaleName, we, se, weights, scales, m)
	}
	assertRetired := func(t *testing.T, m *Model) {
		t.Helper()
		if m.V41ExpertDeviceInstalled(name) || m.V41ExpertDevicePackedPreserved() || v41ExpertDeviceStore(m, false) != nil {
			t.Fatal("packed store still visible after final retirement")
		}
		if err := install(m); err == nil {
			t.Fatal("install resurrected closed ownership")
		}
	}

	t.Run("copy_isolation_duplicate_and_shared_view", func(t *testing.T) {
		m, other := newModel(), newModel()
		defer m.CloseWeights()
		defer other.CloseWeights()
		if m.V41ExpertDeviceInstalled(name) || m.V41ExpertDevicePackedPreserved() || m.weightCloser != nil {
			t.Fatal("empty observation created an owner")
		}
		if err := install(m); err != nil {
			t.Fatal(err)
		}
		if other.V41ExpertDeviceInstalled(name) {
			t.Fatal("unrelated model adopted the store")
		}
		view := newModel()
		view.weightCloser = m.weightCloser
		if !view.V41ExpertDeviceInstalled(name) || v41ExpertDeviceStore(view, false) != v41ExpertDeviceStore(m, false) {
			t.Fatal("shared closer view lost its single packed owner")
		}
		beforeW, beforeS := append([]byte(nil), weights...), append([]byte(nil), scales...)
		weights[0] ^= 1
		scales[0]++
		defer func() { copy(weights, beforeW); copy(scales, beforeS) }()
		if err := install(view); err == nil {
			t.Fatal("duplicate install replaced immutable source")
		}
		b := &v41PackedLifetimeBackend{Backend: compute.Default()}
		if _, err := view.V41ExpertDeviceMatmul(name, b, make([]float32, packedCols*2)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b.weight, beforeW) || !bytes.Equal(b.scales, beforeS) {
			t.Fatal("byte-in consumer did not retain the original copied pair")
		}
		if err := view.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		assertRetired(t, m)
		assertRetired(t, view)
	})

	t.Run("last_real_session_close", func(t *testing.T) {
		m := newModel()
		if err := install(m); err != nil {
			t.Fatal(err)
		}
		store := v41ExpertDeviceStore(m, false)
		a, b := m.NewSession(), m.NewSession()
		defer a.Close()
		defer b.Close()
		var active *WeightSessionsActiveError
		if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 2 {
			t.Fatalf("close with sessions: %v", err)
		}
		if err := install(m); err == nil {
			t.Fatal("closing owner admitted an install")
		}
		if m.holdModelWeights() {
			m.releaseWeightSession()
			t.Fatal("closing owner admitted a new session pin")
		}
		a.Close()
		be := &v41PackedLifetimeBackend{Backend: compute.Default()}
		if _, err := m.V41ExpertDeviceMatmul(name, be, make([]float32, packedCols*2)); err != nil {
			t.Fatalf("remaining session lost its source: %v", err)
		}
		b.Close()
		assertRetired(t, m)
		retired := len(store.weights) == 0 && !store.packedPreserved
		if !retired {
			t.Fatal("detached store retained its map")
		}
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("repeat close: %v", err)
		}
	})

	t.Run("first_install_contests_close", func(t *testing.T) {
		for n := 0; n < 32; n++ {
			m := newModel()
			start := make(chan struct{})
			var wg sync.WaitGroup
			var installErr, closeErr error
			wg.Add(2)
			go func() { defer wg.Done(); <-start; installErr = install(m) }()
			go func() { defer wg.Done(); <-start; closeErr = m.CloseWeights() }()
			close(start)
			wg.Wait()
			if closeErr != nil {
				t.Fatal(closeErr)
			}
			// Either lock winner is legal; both must end on the same closed owner.
			if installErr != nil && !errors.Is(installErr, ErrV4ExpertQuantMetadata) {
				t.Fatalf("unexpected install refusal: %v", installErr)
			}
			assertRetired(t, m)
		}
	})

	t.Run("one_concurrent_install_wins", func(t *testing.T) {
		m := newModel()
		defer m.CloseWeights()
		const contenders = 8
		start, outcomes := make(chan struct{}), make(chan error, contenders)
		var wg sync.WaitGroup
		for n := 0; n < contenders; n++ {
			wg.Add(1)
			go func() { defer wg.Done(); <-start; outcomes <- install(m) }()
		}
		close(start)
		wg.Wait()
		wins := 0
		for n := 0; n < contenders; n++ {
			if err := <-outcomes; err == nil {
				wins++
			} else if !errors.Is(err, ErrV4ExpertQuantMetadata) {
				t.Fatal(err)
			}
		}
		if wins != 1 {
			t.Fatalf("successful publications = %d, want one", wins)
		}
	})

	t.Run("concurrent_public_access_and_close", func(t *testing.T) {
		m := newModel()
		if err := install(m); err != nil {
			t.Fatal(err)
		}
		wantW, wantS := append([]byte(nil), weights...), append([]byte(nil), scales...)
		entered, release, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
		ready := make(chan struct{}, 2)
		outcome, closed := make(chan error, 1), make(chan error, 1)
		var releaseOnce, stopOnce sync.Once
		var consumers, observers, closers sync.WaitGroup
		releaseBackend := func() { releaseOnce.Do(func() { close(release) }) }
		stopObservers := func() { stopOnce.Do(func() { close(stop) }) }
		t.Cleanup(func() {
			releaseBackend()
			stopObservers()
			consumers.Wait()
			observers.Wait()
			closers.Wait()
			_ = m.CloseWeights()
		})
		deadline := time.NewTimer(5 * time.Second)
		defer deadline.Stop()
		b := &v41PackedLifetimeBackend{Backend: compute.Default(), during: func() {
			close(entered)
			<-release
		}}
		consumers.Add(1)
		go func() {
			defer consumers.Done()
			_, err := m.V41ExpertDeviceMatmul(name, b, make([]float32, packedCols*2))
			outcome <- err
		}()
		select {
		case <-entered:
		case <-deadline.C:
			t.Fatal("byte-in consumer did not enter backend")
		}
		for i := 0; i < 2; i++ {
			observers.Add(1)
			go func() {
				defer observers.Done()
				m.V41ExpertDeviceInstalled(name)
				m.V41ExpertDevicePackedPreserved()
				ready <- struct{}{}
				for {
					select {
					case <-stop:
						return
					default:
						// Separate observations may straddle retirement.
						m.V41ExpertDeviceInstalled(name)
						m.V41ExpertDevicePackedPreserved()
					}
				}
			}()
		}
		for i := 0; i < 2; i++ {
			select {
			case <-ready:
			case <-deadline.C:
				t.Fatal("public observers did not start")
			}
		}
		closers.Add(1)
		go func() { defer closers.Done(); closed <- m.CloseWeights() }()
		select {
		case err := <-closed:
			if err != nil {
				t.Fatal(err)
			}
		case <-deadline.C:
			t.Fatal("retirement blocked behind an in-flight byte-in consumer")
		}
		stopObservers()
		observers.Wait()
		assertRetired(t, m)
		releaseBackend()
		select {
		case err := <-outcome:
			if err != nil {
				t.Fatal(err)
			}
		case <-deadline.C:
			t.Fatal("byte-in consumer did not finish after release")
		}
		consumers.Wait()
		if !bytes.Equal(b.weight, wantW) || !bytes.Equal(b.scales, wantS) {
			t.Fatal("concurrent retirement invalidated detached immutable bytes")
		}
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		assertRetired(t, m)
	})

	t.Run("byte_in_call_keeps_detached_bytes", func(t *testing.T) {
		m := newModel()
		if err := install(m); err != nil {
			t.Fatal(err)
		}
		b := &v41PackedLifetimeBackend{Backend: compute.Default(), during: func() {
			// Reentrant model teardown proves no owner/store lock crosses the
			// backend call. Only a local immutable pair remains live afterward.
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			assertRetired(t, m)
		}}
		if _, err := m.V41ExpertDeviceMatmul(name, b, make([]float32, packedCols*2)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b.weight, weights) || !bytes.Equal(b.scales, scales) {
			t.Fatal("retirement invalidated an already-started byte-in call")
		}
	})
}

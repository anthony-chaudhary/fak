package framebus

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestConcurrentPublishAndSubscribe(t *testing.T) {
	cfg := BusConfig{
		DefaultBufferSize: 1024,
		DefaultDropPolicy: DropPolicyDropNewest,
		PublishTimeout:    50 * time.Millisecond,
	}
	bus := NewBus(cfg)
	defer bus.Close()

	subWildcard, err := bus.Subscribe("*")
	if err != nil {
		t.Fatalf("wildcard subscribe failed: %v", err)
	}
	defer subWildcard.Close()

	const (
		numPublishers      = 8
		framesPerPublisher = 50
		totalFrames        = numPublishers * framesPerPublisher
	)

	var receivedCount atomic.Uint64
	var consumerWg sync.WaitGroup
	consumerWg.Add(1)

	stopConsumer := make(chan struct{})
	go func() {
		defer consumerWg.Done()
		for {
			select {
			case _, ok := <-subWildcard.Channel():
				if !ok {
					return
				}
				receivedCount.Add(1)
			case <-stopConsumer:
				// Drain any remaining
				for {
					select {
					case <-subWildcard.Channel():
						receivedCount.Add(1)
					default:
						return
					}
				}
			}
		}
	}()

	var pubWg sync.WaitGroup
	for p := 0; p < numPublishers; p++ {
		pubWg.Add(1)
		go func(pubID int) {
			defer pubWg.Done()
			for i := 0; i < framesPerPublisher; i++ {
				topic := fmt.Sprintf("tenant.%d.event", pubID%4)
				f, err := NewFrame(FrameTypeEvent, topic, []byte(fmt.Sprintf("pub-%d-seq-%d", pubID, i)))
				if err != nil {
					t.Errorf("NewFrame error: %v", err)
					return
				}
				if err := bus.Publish(f); err != nil {
					t.Errorf("Publish error: %v", err)
					return
				}
			}
		}(p)
	}

	pubWg.Wait()

	// Wait up to 1 second for consumer to receive all frames
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if receivedCount.Load() == totalFrames {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	close(stopConsumer)
	consumerWg.Wait()

	total := receivedCount.Load() + subWildcard.DroppedCount()
	if total < totalFrames {
		t.Fatalf("expected at least %d processed frames (received+dropped), got received=%d dropped=%d",
			totalFrames, receivedCount.Load(), subWildcard.DroppedCount())
	}
}

func TestConcurrentSubscribeUnsubscribePublish(t *testing.T) {
	bus := NewBus(DefaultConfig())
	defer bus.Close()

	stopCh := make(chan struct{})
	var wg sync.WaitGroup

	// Continuous publisher
	wg.Add(1)
	go func() {
		defer wg.Done()
		seq := 0
		for {
			select {
			case <-stopCh:
				return
			default:
				f, _ := NewFrame(FrameTypeTelemetry, "churn.topic", []byte(fmt.Sprintf("val-%d", seq)))
				seq++
				_ = bus.Publish(f)
				time.Sleep(100 * time.Microsecond)
			}
		}
	}()

	// Churning subscriber workers
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					sub, err := bus.Subscribe("churn.topic", WithBufferSize(64), WithDropPolicy(DropPolicyDropNewest))
					if err != nil {
						return
					}
					// Drain briefly
					timer := time.NewTimer(5 * time.Millisecond)
				drain:
					for {
						select {
						case <-sub.Channel():
						case <-timer.C:
							break drain
						}
					}
					_ = sub.Close()
				}
			}
		}()
	}

	time.Sleep(100 * time.Millisecond)
	close(stopCh)
	wg.Wait()
}

func TestConcurrentCloseDuringPublish(t *testing.T) {
	bus := NewBus(DefaultConfig())
	_, _ = bus.Subscribe("close.test", WithBufferSize(10))

	var wg sync.WaitGroup
	numPublishers := 10

	for i := 0; i < numPublishers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				f, _ := NewFrame(FrameTypeEvent, "close.test", []byte("payload"))
				err := bus.Publish(f)
				if err != nil && !errors.Is(err, ErrBusClosed) && !errors.Is(err, ErrBufferFull) {
					t.Errorf("unexpected error during concurrent close: %v", err)
				}
				time.Sleep(100 * time.Microsecond)
			}
		}()
	}

	time.Sleep(2 * time.Millisecond)
	_ = bus.Close()

	wg.Wait()

	if !bus.IsClosed() {
		t.Fatal("expected bus to be closed")
	}
}

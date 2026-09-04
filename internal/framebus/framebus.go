package framebus

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrBusClosed is returned when an operation is attempted on a closed bus.
	ErrBusClosed = errors.New("framebus: bus closed")

	// ErrBufferFull is returned when a subscriber queue overflows under fail-closed backpressure.
	ErrBufferFull = errors.New("framebus: subscriber buffer full")

	// ErrSubscriberNotFound is returned when an unsubscribe or lookup operation references a nonexistent subscriber.
	ErrSubscriberNotFound = errors.New("framebus: subscriber not found")

	// ErrInvalidFrame is returned when a frame fails structural or semantic validation.
	ErrInvalidFrame = errors.New("framebus: invalid frame")
)

// FrameType represents the classification of a frame traversing the bus.
type FrameType string

const (
	// FrameTypeEvent denotes an asynchronous system or domain event.
	FrameTypeEvent FrameType = "event"

	// FrameTypeTelemetry denotes metric, tracing, or operational telemetry.
	FrameTypeTelemetry FrameType = "telemetry"

	// FrameTypeControl denotes bus control or synchronization frames.
	FrameTypeControl FrameType = "control"

	// FrameTypeMessage denotes point-to-point or general inter-component bus messages.
	FrameTypeMessage FrameType = "message"
)

// DropPolicy specifies the fail-closed or lossy backpressure behavior when a subscriber's buffer is saturated.
type DropPolicy string

const (
	// DropPolicyFailClosed rejects the publish operation and returns ErrBufferFull when buffer is saturated.
	DropPolicyFailClosed DropPolicy = "fail_closed"

	// DropPolicyDropNewest discards the newest incoming frame when buffer is saturated without blocking the publisher.
	DropPolicyDropNewest DropPolicy = "drop_newest"

	// DropPolicyDropOldest evicts the oldest queued frame to make room for the new frame.
	DropPolicyDropOldest DropPolicy = "drop_oldest"

	// DropPolicyBlock blocks the publisher until buffer space becomes available or the context/timeout expires.
	DropPolicyBlock DropPolicy = "block"
)

// Frame represents an atomic payload traversing the framebus.
type Frame struct {
	// ID is the unique identifier for the frame.
	ID string

	// Type categorizes the frame payload.
	Type FrameType

	// Topic is the routing topic for pub/sub matching.
	Topic string

	// Payload carries arbitrary message bytes.
	Payload []byte

	// Metadata carries arbitrary key-value headers.
	Metadata map[string]string

	// Timestamp is when the frame was created or published.
	Timestamp time.Time
}

// Validate verifies the structural and semantic validity of the frame.
// Invariant: Frames must have non-empty ID, Topic, and Type.
func (f *Frame) Validate() error {
	if f == nil {
		return ErrInvalidFrame
	}
	if f.ID == "" || f.Topic == "" || f.Type == "" {
		return ErrInvalidFrame
	}
	return nil
}

var (
	frameIDCounter atomic.Uint64
	subIDCounter   atomic.Uint64
)

// NewFrame constructs a new Frame with a generated timestamp and unique ID.
func NewFrame(frameType FrameType, topic string, payload []byte) (*Frame, error) {
	if frameType == "" || topic == "" {
		return nil, ErrInvalidFrame
	}
	id := fmt.Sprintf("frm-%d-%d", time.Now().UnixNano(), frameIDCounter.Add(1))
	return &Frame{
		ID:        id,
		Type:      frameType,
		Topic:     topic,
		Payload:   payload,
		Metadata:  make(map[string]string),
		Timestamp: time.Now().UTC(),
	}, nil
}

// FilterFunc is a predicate deciding whether a given subscriber receives a frame.
type FilterFunc func(f *Frame) bool

type subscriptionConfig struct {
	bufferSize int
	dropPolicy DropPolicy
	frameTypes []FrameType
	filter     FilterFunc
}

// SubscriptionOption configures optional parameters when registering a subscription.
type SubscriptionOption func(*subscriptionConfig)

// WithBufferSize sets a custom buffer size for the subscription.
func WithBufferSize(size int) SubscriptionOption {
	return func(sc *subscriptionConfig) {
		if size > 0 {
			sc.bufferSize = size
		}
	}
}

// WithDropPolicy sets a custom drop policy for the subscription.
func WithDropPolicy(policy DropPolicy) SubscriptionOption {
	return func(sc *subscriptionConfig) {
		if policy != "" {
			sc.dropPolicy = policy
		}
	}
}

// WithFrameTypeFilter restricts the subscription to specific frame types.
func WithFrameTypeFilter(types ...FrameType) SubscriptionOption {
	return func(sc *subscriptionConfig) {
		sc.frameTypes = append(sc.frameTypes, types...)
	}
}

// WithFilter sets a custom predicate filter for incoming frames.
func WithFilter(fn FilterFunc) SubscriptionOption {
	return func(sc *subscriptionConfig) {
		sc.filter = fn
	}
}

// Subscription represents an active subscriber's handle to consume frames.
type Subscription struct {
	id         string
	topic      string
	frameTypes []FrameType
	filter     FilterFunc
	ch         chan *Frame
	closeCh    chan struct{}
	policy     DropPolicy
	closed     atomic.Bool
	closeOnce  sync.Once
	bus        *Bus
	dropped    atomic.Uint64
	mu         sync.RWMutex
}

// ID returns the unique subscription identifier.
func (s *Subscription) ID() string {
	return s.id
}

// Topic returns the subscription routing topic.
func (s *Subscription) Topic() string {
	return s.topic
}

// Channel returns the receive-only channel carrying delivered frames.
func (s *Subscription) Channel() <-chan *Frame {
	return s.ch
}

// DroppedCount returns the total number of frames dropped due to backpressure.
func (s *Subscription) DroppedCount() uint64 {
	return s.dropped.Load()
}

// Close unsubscribes and closes the subscription's delivery channel.
func (s *Subscription) Close() error {
	if s.closed.Load() {
		return nil
	}
	if s.bus != nil {
		err := s.bus.Unsubscribe(s.id)
		if errors.Is(err, ErrSubscriberNotFound) {
			return nil
		}
		return err
	}
	s.closeChannel()
	return nil
}

func (s *Subscription) closeChannel() {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		close(s.closeCh)
		s.mu.Lock()
		close(s.ch)
		s.mu.Unlock()
	})
}

func (s *Subscription) matches(f *Frame) bool {
	if s.closed.Load() {
		return false
	}
	if !topicMatches(s.topic, f.Topic) {
		return false
	}
	if len(s.frameTypes) > 0 {
		matchedType := false
		for _, ft := range s.frameTypes {
			if ft == f.Type {
				matchedType = true
				break
			}
		}
		if !matchedType {
			return false
		}
	}
	if s.filter != nil && !s.filter(f) {
		return false
	}
	return true
}

func (s *Subscription) deliver(f *Frame, timeout time.Duration, ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed.Load() {
		return nil
	}

	// Guard: fail-closed buffer backpressure rejects or drops frames deterministically.
	switch s.policy {
	case DropPolicyFailClosed:
		select {
		case <-s.closeCh:
			return ErrBusClosed
		case <-ctx.Done():
			s.dropped.Add(1)
			return ctx.Err()
		case s.ch <- f:
			return nil
		default:
			s.dropped.Add(1)
			return ErrBufferFull
		}

	case DropPolicyDropNewest:
		select {
		case <-s.closeCh:
			return ErrBusClosed
		case <-ctx.Done():
			s.dropped.Add(1)
			return ctx.Err()
		case s.ch <- f:
			return nil
		default:
			s.dropped.Add(1)
			return nil
		}

	case DropPolicyDropOldest:
		for {
			select {
			case <-s.closeCh:
				return ErrBusClosed
			case <-ctx.Done():
				s.dropped.Add(1)
				return ctx.Err()
			case s.ch <- f:
				return nil
			default:
				select {
				case <-s.closeCh:
					return ErrBusClosed
				case <-ctx.Done():
					s.dropped.Add(1)
					return ctx.Err()
				case <-s.ch:
					s.dropped.Add(1)
				default:
				}
			}
		}

	case DropPolicyBlock:
		if timeout <= 0 {
			select {
			case s.ch <- f:
				return nil
			case <-s.closeCh:
				return ErrBusClosed
			case <-ctx.Done():
				s.dropped.Add(1)
				return ctx.Err()
			}
		}
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		select {
		case s.ch <- f:
			return nil
		case <-s.closeCh:
			return ErrBusClosed
		case <-timer.C:
			s.dropped.Add(1)
			return ErrBufferFull
		case <-ctx.Done():
			s.dropped.Add(1)
			return ctx.Err()
		}

	default:
		select {
		case <-s.closeCh:
			return ErrBusClosed
		case <-ctx.Done():
			s.dropped.Add(1)
			return ctx.Err()
		case s.ch <- f:
			return nil
		default:
			s.dropped.Add(1)
			return ErrBufferFull
		}
	}
}

// BusConfig defines configuration options for a Bus instance.
type BusConfig struct {
	// DefaultBufferSize sets the channel capacity for new subscriptions.
	DefaultBufferSize int

	// DefaultDropPolicy dictates behavior when subscriber queues are full.
	DefaultDropPolicy DropPolicy

	// PublishTimeout defines maximum block duration for DropPolicyBlock.
	PublishTimeout time.Duration
}

// DefaultConfig returns recommended baseline configuration for a Bus.
func DefaultConfig() BusConfig {
	return BusConfig{
		DefaultBufferSize: 256,
		DefaultDropPolicy: DropPolicyFailClosed,
		PublishTimeout:    100 * time.Millisecond,
	}
}

// Bus coordinates thread-safe frame distribution across subscribers with backpressure handling.
type Bus struct {
	cfg         BusConfig
	mu          sync.RWMutex
	subscribers map[string]*Subscription
	closed      atomic.Bool
	closeCh     chan struct{}
}

// NewBus creates and initializes a new Bus instance with the given configuration.
func NewBus(cfg BusConfig) *Bus {
	if cfg.DefaultBufferSize <= 0 {
		cfg.DefaultBufferSize = 256
	}
	if cfg.DefaultDropPolicy == "" {
		cfg.DefaultDropPolicy = DropPolicyFailClosed
	}
	if cfg.PublishTimeout <= 0 {
		cfg.PublishTimeout = 100 * time.Millisecond
	}
	return &Bus{
		cfg:         cfg,
		subscribers: make(map[string]*Subscription),
		closeCh:     make(chan struct{}),
	}
}

// Subscribe registers a new subscriber for a topic with optional custom options.
func (b *Bus) Subscribe(topic string, opts ...SubscriptionOption) (*Subscription, error) {
	if b.closed.Load() {
		return nil, ErrBusClosed
	}

	sc := subscriptionConfig{
		bufferSize: b.cfg.DefaultBufferSize,
		dropPolicy: b.cfg.DefaultDropPolicy,
	}
	for _, opt := range opts {
		opt(&sc)
	}

	if sc.bufferSize < 1 {
		sc.bufferSize = 1
	}

	subID := fmt.Sprintf("sub-%d-%d", time.Now().UnixNano(), subIDCounter.Add(1))
	sub := &Subscription{
		id:         subID,
		topic:      topic,
		frameTypes: sc.frameTypes,
		filter:     sc.filter,
		ch:         make(chan *Frame, sc.bufferSize),
		closeCh:    make(chan struct{}),
		policy:     sc.dropPolicy,
		bus:        b,
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed.Load() {
		return nil, ErrBusClosed
	}
	b.subscribers[subID] = sub
	return sub, nil
}

// Unsubscribe removes an active subscriber by its ID and closes its channel.
func (b *Bus) Unsubscribe(subID string) error {
	b.mu.Lock()
	sub, ok := b.subscribers[subID]
	if !ok {
		b.mu.Unlock()
		return ErrSubscriberNotFound
	}
	delete(b.subscribers, subID)
	b.mu.Unlock()

	sub.closeChannel()
	return nil
}

// Publish delivers a frame to all matching subscribers according to their drop policy.
// Invariant: Publish is safe for concurrent invocations from multiple goroutines.
func (b *Bus) Publish(f *Frame) error {
	return b.PublishSync(context.Background(), f)
}

// PublishSync delivers a frame synchronously with context cancellation support.
func (b *Bus) PublishSync(ctx context.Context, f *Frame) error {
	if b.closed.Load() {
		return ErrBusClosed
	}
	if f == nil {
		return ErrInvalidFrame
	}
	if err := f.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	b.mu.RLock()
	if b.closed.Load() {
		b.mu.RUnlock()
		return ErrBusClosed
	}
	matched := make([]*Subscription, 0, len(b.subscribers))
	for _, sub := range b.subscribers {
		if sub.matches(f) {
			matched = append(matched, sub)
		}
	}
	b.mu.RUnlock()

	var lastErr error
	for _, sub := range matched {
		if b.closed.Load() {
			return ErrBusClosed
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := sub.deliver(f, b.cfg.PublishTimeout, ctx); err != nil {
			lastErr = err
		}
	}
	if b.closed.Load() {
		return ErrBusClosed
	}
	return lastErr
}

// SubscriberCount returns the total number of registered subscribers.
func (b *Bus) SubscriberCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subscribers)
}

// Close shuts down the bus and closes all active subscriptions.
func (b *Bus) Close() error {
	if !b.closed.CompareAndSwap(false, true) {
		return ErrBusClosed
	}
	close(b.closeCh)

	b.mu.Lock()
	subs := make([]*Subscription, 0, len(b.subscribers))
	for _, sub := range b.subscribers {
		subs = append(subs, sub)
	}
	b.subscribers = make(map[string]*Subscription)
	b.mu.Unlock()

	for _, sub := range subs {
		sub.closeChannel()
	}
	return nil
}

// IsClosed reports whether the bus has been closed.
func (b *Bus) IsClosed() bool {
	return b.closed.Load()
}

func topicMatches(subTopic, frameTopic string) bool {
	if subTopic == "*" || subTopic == frameTopic {
		return true
	}
	if strings.HasSuffix(subTopic, ".*") {
		prefix := strings.TrimSuffix(subTopic, ".*")
		if strings.HasPrefix(frameTopic, prefix+".") || frameTopic == prefix {
			return true
		}
	}
	return false
}

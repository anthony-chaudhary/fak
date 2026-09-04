package l3kv

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/l3server/client"
)

const (
	// DefaultDaemonAddr is the standard local loopback endpoint for the L3 store daemon.
	DefaultDaemonAddr = "127.0.0.1:18000"

	// EnvDaemonAddr is the environment variable for configuring the L3 store daemon address.
	EnvDaemonAddr = "FAK_L3_DAEMON_ADDR"
)

// Deleter is an optional capability interface for stores that support explicit deletion of stored keys.
type Deleter interface {
	Delete(ctx context.Context, key string) error
}

// DaemonStore connects to an external L3 store daemon over TCP speaking the L3
// binary wire protocol (13-byte header: 0xBE 0xEF magic, 1-byte version, 1-byte opcode,
// 1-byte flags, 4-byte reqID, 4-byte bodyLen).
//
// Concurrency: DaemonStore is safe for concurrent access across multiple goroutines.
type DaemonStore struct {
	mu     sync.Mutex
	addr   string
	client *client.Client
	closed bool
}

var (
	_ Store   = (*DaemonStore)(nil)
	_ Deleter = (*DaemonStore)(nil)
)

// NewDaemonStore creates a new DaemonStore connecting to the specified daemon address.
// If addr is empty, it checks FAK_L3_DAEMON_ADDR, FAK_L3_STORE_ADDR, and defaults to 127.0.0.1:18000.
func NewDaemonStore(addr string) (*DaemonStore, error) {
	if addr == "" {
		if env := os.Getenv(EnvDaemonAddr); env != "" {
			addr = env
		} else if env := os.Getenv("FAK_L3_STORE_ADDR"); env != "" {
			addr = env
		} else {
			addr = DefaultDaemonAddr
		}
	}
	c, err := client.New(addr)
	if err != nil {
		return nil, fmt.Errorf("l3kv: connect to daemon %s: %w", addr, err)
	}
	return &DaemonStore{
		addr:   addr,
		client: c,
	}, nil
}

// NewDaemonStoreWithClient creates a DaemonStore using an already-connected client instance.
func NewDaemonStoreWithClient(addr string, c *client.Client) *DaemonStore {
	return &DaemonStore{
		addr:   addr,
		client: c,
	}
}

// Addr returns the configured daemon address.
func (s *DaemonStore) Addr() string {
	return s.addr
}

func (s *DaemonStore) getClientLocked() (*client.Client, error) {
	if s.closed {
		return nil, errors.New("l3kv: daemon store is closed")
	}
	if s.client != nil {
		return s.client, nil
	}
	c, err := client.New(s.addr)
	if err != nil {
		return nil, fmt.Errorf("l3kv: connect to daemon %s: %w", s.addr, err)
	}
	s.client = c
	return s.client, nil
}

func (s *DaemonStore) resetClientLocked() {
	if s.client != nil {
		_ = s.client.Close()
		s.client = nil
	}
}

// Put stores payload under key in the remote L3 daemon.
func (s *DaemonStore) Put(ctx context.Context, key string, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if key == "" {
		return fmt.Errorf("l3kv: empty key")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	c, err := s.getClientLocked()
	if err != nil {
		return err
	}
	if err := c.Set([]byte(key), payload, 0); err != nil {
		s.resetClientLocked()
		return fmt.Errorf("l3kv: daemon put %s: %w", key, err)
	}
	return nil
}

// Get retrieves the payload staged under key from the remote L3 daemon.
// found=false is a clean miss.
func (s *DaemonStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if key == "" {
		return nil, false, fmt.Errorf("l3kv: empty key")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	c, err := s.getClientLocked()
	if err != nil {
		return nil, false, err
	}
	val, err := c.Get([]byte(key))
	if err != nil {
		s.resetClientLocked()
		return nil, false, fmt.Errorf("l3kv: daemon get %s: %w", key, err)
	}
	if val == nil {
		return nil, false, nil
	}
	return val, true, nil
}

// Delete removes the entry staged under key from the remote L3 daemon.
func (s *DaemonStore) Delete(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if key == "" {
		return fmt.Errorf("l3kv: empty key")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	c, err := s.getClientLocked()
	if err != nil {
		return err
	}
	if err := c.Delete([]byte(key)); err != nil {
		s.resetClientLocked()
		return fmt.Errorf("l3kv: daemon delete %s: %w", key, err)
	}
	return nil
}

// Exists checks if key is resident in the remote L3 daemon.
func (s *DaemonStore) Exists(ctx context.Context, key string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if key == "" {
		return false, fmt.Errorf("l3kv: empty key")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	c, err := s.getClientLocked()
	if err != nil {
		return false, err
	}
	exists, err := c.Exists([]byte(key))
	if err != nil {
		s.resetClientLocked()
		return false, fmt.Errorf("l3kv: daemon exists %s: %w", key, err)
	}
	return exists, nil
}

// Close closes the underlying daemon client connection.
func (s *DaemonStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.client != nil {
		err := s.client.Close()
		s.client = nil
		return err
	}
	return nil
}

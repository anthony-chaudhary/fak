package client

import (
	"encoding/json"
	"fmt"
	"net"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// Client is an L3 TCP client.
type Client struct {
	codec *protocol.Codec
	reqID atomic.Uint32
}

// New creates a new client connected to the given address.
func New(addr string) (*Client, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("connect to %s: %w", addr, err)
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetReadBuffer(65536)
		tc.SetWriteBuffer(65536)
	}
	return &Client{
		codec: protocol.NewCodec(conn),
	}, nil
}

// Close closes the connection.
func (c *Client) Close() error {
	return c.codec.Close()
}

func (c *Client) nextReqID() uint32 {
	return c.reqID.Add(1)
}

// Get retrieves a value by key. Returns (nil, nil) if not found.
func (c *Client) Get(key []byte) ([]byte, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpGet,
			RequestID: reqID,
		},
		Body: protocol.EncodeKeyBody(key),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	value, found, err := protocol.DecodeValueResponse(resp.Body)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, nil
	}
	return value, nil
}

// Set stores a key-value pair with optional TTL (0 = no expiry).
func (c *Client) Set(key, value []byte, ttlMs int64) error {
	reqID := c.nextReqID()
	flags := uint8(protocol.FlagNone)
	if ttlMs > 0 {
		flags = protocol.FlagWithTTL
	}
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpSet,
			Flags:     flags,
			RequestID: reqID,
		},
		Body: protocol.EncodeKVBody(key, value, ttlMs, flags),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Delete removes a key.
func (c *Client) Delete(key []byte) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpDelete,
			RequestID: reqID,
		},
		Body: protocol.EncodeKeyBody(key),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Exists checks if a key exists.
func (c *Client) Exists(key []byte) (bool, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpTest,
			RequestID: reqID,
		},
		Body: protocol.EncodeKeyBody(key),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return false, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return false, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return false, fmt.Errorf("server error: %s", string(resp.Body))
	}
	if len(resp.Body) > 0 && resp.Body[0] == 1 {
		return true, nil
	}
	return false, nil
}

// Lease grants a lease protecting a key from eviction.
func (c *Client) Lease(key []byte, durationMs int64) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpLease,
			RequestID: reqID,
		},
		Body: protocol.EncodeLeaseBody(key, durationMs),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// MGet retrieves multiple values.
func (c *Client) MGet(keys [][]byte) ([][]byte, []bool, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpMGet,
			RequestID: reqID,
		},
		Body: protocol.EncodeMGetBody(keys),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	return protocol.DecodeMultiValueResponse(resp.Body)
}

// MSet stores multiple key-value pairs. Returns per-key status (0=ok, nonzero=error)
// on partial failure, or nil on full success (backward compat with RespOK).
func (c *Client) MSet(keys, values [][]byte) ([]byte, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpMSet,
			RequestID: reqID,
		},
		Body: protocol.EncodeMSetBody(keys, values),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}
	if resp.Header.OpCode == protocol.RespMSetResult {
		statuses, perr := protocol.DecodeMSetResultResponse(resp.Body)
		return statuses, perr
	}
	return nil, nil // RespOK â€” all succeeded
}

// MExists checks existence of multiple keys.
func (c *Client) MExists(keys [][]byte) ([]bool, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpMTest,
			RequestID: reqID,
		},
		Body: protocol.EncodeMGetBody(keys),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	// MTest response uses MultiValue format â€” extract founds only
	_, founds, err := protocol.DecodeMultiValueResponse(resp.Body)
	return founds, err
}

// MDelete removes multiple keys.
func (c *Client) MDelete(keys [][]byte) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpMDel,
			RequestID: reqID,
		},
		Body: protocol.EncodeMGetBody(keys),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Keys returns all keys matching a regex pattern.
func (c *Client) Keys(pattern string) ([][]byte, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpKeys,
			RequestID: reqID,
		},
		Body: protocol.EncodeKeysBody([]byte(pattern)),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	return protocol.DecodeKeysResponse(resp.Body)
}

// Pin permanently protects a key from eviction.
func (c *Client) Pin(key []byte) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpPin, RequestID: reqID},
		Body:   protocol.EncodeKeyBody(key),
	}
	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}
	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}
	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Unpin removes eviction protection.
func (c *Client) Unpin(key []byte) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{OpCode: protocol.OpUnpin, RequestID: reqID},
		Body:   protocol.EncodeKeyBody(key),
	}
	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}
	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}
	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// SetReplicated stores a key-value pair with FlagReplicated set.
// Used by the replicator to prevent replication loops.
func (c *Client) SetReplicated(key, value []byte, ttlMs int64) error {
	reqID := c.nextReqID()
	flags := uint8(protocol.FlagReplicated)
	if ttlMs > 0 {
		flags |= protocol.FlagWithTTL
	}
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpSet,
			Flags:     flags,
			RequestID: reqID,
		},
		Body: protocol.EncodeKVBody(key, value, ttlMs, flags),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// DeleteReplicated removes a key with FlagReplicated set.
// Used by the replicator to prevent replication loops.
func (c *Client) DeleteReplicated(key []byte) error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpDelete,
			Flags:     protocol.FlagReplicated,
			RequestID: reqID,
		},
		Body: protocol.EncodeKeyBody(key),
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Flush clears all data from the server.
func (c *Client) Flush() error {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpFlush,
			RequestID: reqID,
		},
		Body: []byte{},
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}

	if resp.Header.OpCode == protocol.RespError {
		return fmt.Errorf("server error: %s", string(resp.Body))
	}
	return nil
}

// Info returns server metadata.
func (c *Client) Info() (map[string]interface{}, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpInfo,
			RequestID: reqID,
		},
		Body: []byte{},
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse info response: %w", err)
	}
	return result, nil
}

// Stats returns bandwidth and connection statistics from the server.
func (c *Client) Stats() (map[string]interface{}, error) {
	reqID := c.nextReqID()
	msg := protocol.Message{
		Header: protocol.Header{
			OpCode:    protocol.OpStats,
			RequestID: reqID,
		},
		Body: []byte{},
	}

	if err := c.codec.WriteMessage(msg); err != nil {
		return nil, err
	}

	resp, err := c.codec.ReadMessage()
	if err != nil {
		return nil, err
	}

	if resp.Header.OpCode == protocol.RespError {
		return nil, fmt.Errorf("server error: %s", string(resp.Body))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse stats response: %w", err)
	}
	return result, nil
}

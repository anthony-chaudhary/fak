package client

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
	"github.com/anthony-chaudhary/fak/internal/l3server/version"
)

// ErrHandshake identifies a peer that did not complete the native L3 handshake.
var ErrHandshake = errors.New("L3 handshake failed")

// Client is an L3 TCP client.
type Client struct {
	codec *protocol.Codec
	reqID atomic.Uint32
}

// New creates a client after validating the peer's native L3 handshake.
func New(addr string) (*Client, error) {
	return newWithTimeout(addr, 5*time.Second)
}

// ADAPT: ggml-org/llama.cpp ggml/src/ggml-rpc/ggml-rpc.cpp, start/get_dispatcher
// at bc180757161f9f7d56e0efac44257e9029951a3b: admit only after successful HELLO.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
func newWithTimeout(addr string, timeout time.Duration) (*Client, error) {
	deadline := time.Now().Add(timeout)
	conn, err := (&net.Dialer{Deadline: deadline}).Dial("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("connect to %s: %w", addr, err)
	}
	admitted := false
	defer func() {
		if !admitted {
			_ = conn.Close()
		}
	}()
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetReadBuffer(65536)
		tc.SetWriteBuffer(65536)
	}
	c := &Client{
		codec: protocol.NewCodec(conn),
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrHandshake, err)
	}
	if err := c.handshake(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrHandshake, err)
	}
	if err := conn.SetDeadline(time.Time{}); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrHandshake, err)
	}
	admitted = true
	return c, nil
}

func (c *Client) handshake() error {
	body, err := json.Marshal(struct {
		ClientVersion string `json:"client_version"`
	}{ClientVersion: version.ServerVersion})
	if err != nil {
		return err
	}
	id := c.nextReqID()
	if err := c.codec.WriteMessage(protocol.Message{Header: protocol.Header{OpCode: protocol.OpHandshake, RequestID: id}, Body: body}); err != nil {
		return err
	}
	response, err := c.codec.ReadMessage()
	if err != nil {
		return err
	}
	defer protocol.PutBodyBuf(response.Body, response.BodyPoolIdx)
	if response.Header.OpCode != protocol.RespOK || response.Header.RequestID != id {
		return errors.New("unexpected handshake response")
	}
	var hello struct {
		Status          string `json:"status"`
		APIVersion      int    `json:"api_version"`
		ProtocolVersion int    `json:"protocol_version"`
	}
	if err := json.Unmarshal(response.Body, &hello); err != nil {
		return err
	}
	if hello.Status != "ok" || hello.APIVersion != version.APIVersion || hello.ProtocolVersion != version.ProtocolVersion {
		return errors.New("incompatible handshake response")
	}
	return nil
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

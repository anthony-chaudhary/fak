package protocol

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
)

const (
	MagicByte0 = 0xBE
	MagicByte1 = 0xEF
	Version1   = 0x01
	HeaderSize = 13 // 2 magic + 1 version + 1 opcode + 1 flags + 4 reqID + 4 bodyLen

	OpGet    = 0x01
	OpSet    = 0x02
	OpDelete = 0x03
	OpTest   = 0x04
	OpLease  = 0x05
	OpPin    = 0x06
	OpUnpin  = 0x07

	OpMGet  = 0x10
	OpMSet  = 0x11
	OpMTest = 0x12
	OpMDel  = 0x13

	OpInfo        = 0x20
	OpCluster     = 0x21
	OpKeys        = 0x22
	OpHandshake   = 0x23
	OpFlush       = 0x24
	OpStats       = 0x25
	OpReportStats = 0x26
	OpMaintenance = 0x27

	OpRegMem        = 0x30
	OpDeregMem      = 0x31
	OpRDMAReadReady = 0x32 // serverâ†’client: RDMA Read metadata (rkey, addr, len)
	OpReadAck       = 0x33 // clientâ†’server: RDMA Read completion ack (1-byte WC status)
	OpMGetRDMA      = 0x34 // clientâ†’server: batch GET requesting RDMA metadata
	OpMGetReadReady = 0x35 // serverâ†’client: batch RDMA coordinates
	OpBatchReadAck  = 0x36 // clientâ†’server: batch WC statuses

	OpCXLRegionMap   = 0x37 // clientâ†’server: request CXL region map
	OpCXLReadReady   = 0x38 // serverâ†’client: load from this CXL device offset
	RespCXLRegionMap = 0xF5 // serverâ†’client: CXL region entries

	RespOK         = 0xF0
	RespError      = 0xF1
	RespValue      = 0xF2
	RespMultiValue = 0xF3
	RespMSetResult = 0xF4 // per-key MSET status bytes (partial failure)
	RespOOM        = 0xF6 // memory pressure â€” SET rejected with diagnostics
	RespNotReady   = 0xF7 // server still starting â€” shard not allocated yet

	OpSnapshot = 0x28 // trigger snapshot to configured dir
	OpRestore  = 0x29 // restore from configured dir

	FlagNone       = 0x00
	FlagWithTTL    = 0x01
	FlagReplicated = 0x02 // writes forwarded by replication â€” do not re-replicate

	MaxBodySize = 64 * 1024 * 1024 // 64 MB max body
)

var (
	ErrBadMagic     = errors.New("invalid magic bytes")
	ErrBadVersion   = errors.New("unsupported protocol version")
	ErrBodyTooLarge = errors.New("body exceeds maximum size")
)

// M3: Body buffer pools â€” bucketed by size class to reduce GC pressure.
// Only pool bodies >= 16KB (GC handles small ones well).
const bodyPoolMinSize = 16 * 1024

var bodyPools = [...]sync.Pool{
	{New: func() any { b := make([]byte, 16*1024); return &b }},      // 16KB
	{New: func() any { b := make([]byte, 256*1024); return &b }},     // 256KB
	{New: func() any { b := make([]byte, 1024*1024); return &b }},    // 1MB
	{New: func() any { b := make([]byte, 64*1024*1024); return &b }}, // 64MB
}

var bodyPoolSizes = [...]int{16 * 1024, 256 * 1024, 1024 * 1024, 64 * 1024 * 1024}

// getBodyBuf returns a buffer of at least size bytes from a pool, or allocates
// directly for very small buffers. Returns the buffer slice and the pool index
// (or -1 if not pooled). Caller must call PutBodyBuf when done.
func getBodyBuf(size int) ([]byte, int) {
	if size < bodyPoolMinSize {
		return make([]byte, size), -1
	}
	for i, ps := range bodyPoolSizes {
		if size <= ps {
			bp := bodyPools[i].Get().(*[]byte)
			return (*bp)[:size], i
		}
	}
	return make([]byte, size), -1
}

// PutBodyBuf returns a buffer to its pool. poolIdx is the value returned by
// getBodyBuf. Safe to call with poolIdx=-1 (no-op) or with nil buf.
func PutBodyBuf(buf []byte, poolIdx int) {
	if poolIdx < 0 || poolIdx >= len(bodyPools) || buf == nil {
		return
	}
	b := buf[:cap(buf)]
	bodyPools[poolIdx].Put(&b)
}

// Header represents a wire protocol header.
type Header struct {
	OpCode    uint8
	Flags     uint8
	RequestID uint32
	BodyLen   uint32
}

// Message is a complete protocol message (header + body).
type Message struct {
	Header      Header
	Body        []byte
	BodyPoolIdx int // M3: pool index for body buffer (-1 = not pooled). Call PutBodyBuf to return.
}

// ReadMessage reads a complete message from a reader.
// For bodies >= 16KB, the body buffer comes from a pool. Callers processing
// the message should call PutBodyBuf(msg.Body, msg.BodyPoolIdx) when done
// to return the buffer. The BodyPoolIdx field on Message tracks which pool
// owns the buffer (-1 = not pooled).
func ReadMessage(r io.Reader) (Message, error) {
	var hdr [HeaderSize]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return Message{}, err
	}

	if hdr[0] != MagicByte0 || hdr[1] != MagicByte1 {
		return Message{}, ErrBadMagic
	}
	if hdr[2] != Version1 {
		return Message{}, ErrBadVersion
	}

	h := Header{
		OpCode:    hdr[3],
		Flags:     hdr[4],
		RequestID: binary.LittleEndian.Uint32(hdr[5:9]),
		BodyLen:   binary.LittleEndian.Uint32(hdr[9:13]),
	}

	if h.BodyLen > MaxBodySize {
		return Message{}, ErrBodyTooLarge
	}

	body, poolIdx := getBodyBuf(int(h.BodyLen))
	if h.BodyLen > 0 {
		if _, err := io.ReadFull(r, body); err != nil {
			PutBodyBuf(body, poolIdx)
			return Message{}, err
		}
	}

	return Message{Header: h, Body: body, BodyPoolIdx: poolIdx}, nil
}

// WriteMessage writes a complete message to a writer.
func WriteMessage(w io.Writer, msg Message) error {
	var hdr [HeaderSize]byte
	hdr[0] = MagicByte0
	hdr[1] = MagicByte1
	hdr[2] = Version1
	hdr[3] = msg.Header.OpCode
	hdr[4] = msg.Header.Flags
	binary.LittleEndian.PutUint32(hdr[5:9], msg.Header.RequestID)
	binary.LittleEndian.PutUint32(hdr[9:13], uint32(len(msg.Body)))

	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(msg.Body) > 0 {
		if _, err := w.Write(msg.Body); err != nil {
			return err
		}
	}
	return nil
}

// --- Body encoding helpers ---

// EncodeKeyBody encodes a key for GET/DELETE/TEST/LEASE/PIN/UNPIN.
// Format: keyLen(2) + key
func EncodeKeyBody(key []byte) []byte {
	body := make([]byte, 2+len(key))
	binary.LittleEndian.PutUint16(body[0:2], uint16(len(key)))
	copy(body[2:], key)
	return body
}

// DecodeKeyBody decodes a key from the body.
func DecodeKeyBody(body []byte) ([]byte, error) {
	if len(body) < 2 {
		return nil, fmt.Errorf("key body too short")
	}
	keyLen := binary.LittleEndian.Uint16(body[0:2])
	if int(keyLen)+2 > len(body) {
		return nil, fmt.Errorf("key body truncated")
	}
	return body[2 : 2+keyLen], nil
}

// EncodeKVBody encodes a key-value pair for SET.
// Format: keyLen(2) + key + valueLen(4) + value [+ ttlMs(8) if FlagWithTTL]
func EncodeKVBody(key, value []byte, ttlMs int64, flags uint8) []byte {
	size := 2 + len(key) + 4 + len(value)
	if flags&FlagWithTTL != 0 {
		size += 8
	}
	body := make([]byte, size)
	off := 0
	binary.LittleEndian.PutUint16(body[off:], uint16(len(key)))
	off += 2
	copy(body[off:], key)
	off += len(key)
	binary.LittleEndian.PutUint32(body[off:], uint32(len(value)))
	off += 4
	copy(body[off:], value)
	off += len(value)
	if flags&FlagWithTTL != 0 {
		binary.LittleEndian.PutUint64(body[off:], uint64(ttlMs))
	}
	return body
}

// DecodeKVBody decodes a key-value pair from the body.
func DecodeKVBody(body []byte, flags uint8) (key, value []byte, ttlMs int64, err error) {
	if len(body) < 2 {
		return nil, nil, 0, fmt.Errorf("kv body too short")
	}
	off := 0
	keyLen := binary.LittleEndian.Uint16(body[off:])
	off += 2
	if off+int(keyLen) > len(body) {
		return nil, nil, 0, fmt.Errorf("kv body key truncated")
	}
	key = body[off : off+int(keyLen)]
	off += int(keyLen)
	if off+4 > len(body) {
		return nil, nil, 0, fmt.Errorf("kv body value length truncated")
	}
	valueLen := binary.LittleEndian.Uint32(body[off:])
	off += 4
	if off+int(valueLen) > len(body) {
		return nil, nil, 0, fmt.Errorf("kv body value truncated")
	}
	value = body[off : off+int(valueLen)]
	off += int(valueLen)
	if flags&FlagWithTTL != 0 && off+8 <= len(body) {
		ttlMs = int64(binary.LittleEndian.Uint64(body[off:]))
	}
	return key, value, ttlMs, nil
}

// EncodeValueResponse encodes a value for RespValue.
// Format: found(1) + valueLen(4) + value
func EncodeValueResponse(value []byte, found bool) []byte {
	if !found {
		return []byte{0}
	}
	body := make([]byte, 1+4+len(value))
	body[0] = 1
	binary.LittleEndian.PutUint32(body[1:5], uint32(len(value)))
	copy(body[5:], value)
	return body
}

// DecodeValueResponse decodes a value response.
func DecodeValueResponse(body []byte) (value []byte, found bool, err error) {
	if len(body) < 1 {
		return nil, false, fmt.Errorf("value response too short")
	}
	if body[0] == 0 {
		return nil, false, nil
	}
	if len(body) < 5 {
		return nil, false, fmt.Errorf("value response truncated")
	}
	valueLen := binary.LittleEndian.Uint32(body[1:5])
	if 5+int(valueLen) > len(body) {
		return nil, false, fmt.Errorf("value response value truncated")
	}
	return body[5 : 5+valueLen], true, nil
}

// EncodeErrorResponse creates an error response body.
func EncodeErrorResponse(errMsg string) []byte {
	return []byte(errMsg)
}

// EncodeOKResponse creates an OK response body.
func EncodeOKResponse() []byte {
	return []byte{1}
}

// EncodeLeaseBody encodes key + lease duration.
// Format: keyLen(2) + key + durationMs(8)
func EncodeLeaseBody(key []byte, durationMs int64) []byte {
	body := make([]byte, 2+len(key)+8)
	binary.LittleEndian.PutUint16(body[0:2], uint16(len(key)))
	copy(body[2:], key)
	binary.LittleEndian.PutUint64(body[2+len(key):], uint64(durationMs))
	return body
}

// DecodeLeaseBody decodes a lease request body.
func DecodeLeaseBody(body []byte) (key []byte, durationMs int64, err error) {
	if len(body) < 2 {
		return nil, 0, fmt.Errorf("lease body too short")
	}
	keyLen := binary.LittleEndian.Uint16(body[0:2])
	if 2+int(keyLen)+8 > len(body) {
		return nil, 0, fmt.Errorf("lease body truncated")
	}
	key = body[2 : 2+keyLen]
	durationMs = int64(binary.LittleEndian.Uint64(body[2+keyLen:]))
	return key, durationMs, nil
}

// --- RDMA Read Ready helpers ---

// RDMAReadReadySize is the body size for an RDMA Read Ready response.
// Format: found(1) + rkey(4) + remoteAddr(8) + length(4) = 17 bytes
const RDMAReadReadySize = 17

// EncodeRDMAReadReady encodes MR metadata for the client to perform RDMA Read.
func EncodeRDMAReadReady(rkey uint32, remoteAddr uint64, length uint32) []byte {
	body := make([]byte, RDMAReadReadySize)
	body[0] = 1 // found
	binary.LittleEndian.PutUint32(body[1:5], rkey)
	binary.LittleEndian.PutUint64(body[5:13], remoteAddr)
	binary.LittleEndian.PutUint32(body[13:17], length)
	return body
}

// DecodeRDMAReadReady decodes MR metadata from an RDMA Read Ready response.
func DecodeRDMAReadReady(body []byte) (rkey uint32, remoteAddr uint64, length uint32, err error) {
	if len(body) < RDMAReadReadySize {
		return 0, 0, 0, fmt.Errorf("rdma read ready body too short: %d < %d", len(body), RDMAReadReadySize)
	}
	if body[0] == 0 {
		return 0, 0, 0, nil // not found
	}
	rkey = binary.LittleEndian.Uint32(body[1:5])
	remoteAddr = binary.LittleEndian.Uint64(body[5:13])
	length = binary.LittleEndian.Uint32(body[13:17])
	return rkey, remoteAddr, length, nil
}

// EncodeReadAck encodes an OpReadAck body (1-byte WC status).
func EncodeReadAck(wcStatus uint8) []byte {
	return []byte{wcStatus}
}

// DecodeReadAck decodes an OpReadAck body, returning the WC status.
func DecodeReadAck(body []byte) (uint8, error) {
	if len(body) < 1 {
		return 0, fmt.Errorf("read ack body too short")
	}
	return body[0], nil
}

// --- Batch RDMA Read Ready helpers ---

// MGetReadReadyEntry is a single entry in a batch RDMA Read Ready response.
type MGetReadReadyEntry struct {
	Found      bool
	RKey       uint32
	RemoteAddr uint64
	Length     uint32
}

// EncodeMGetReadReady encodes batch RDMA metadata for OpMGetReadReady.
// Format: count(4) + [found(1) + rkey(4) + remote_addr(8) + length(4)] Ã— count
func EncodeMGetReadReady(entries []MGetReadReadyEntry) []byte {
	n := len(entries)
	body := make([]byte, 4+n*17)
	binary.LittleEndian.PutUint32(body[0:4], uint32(n))
	off := 4
	for _, e := range entries {
		if e.Found {
			body[off] = 1
		} else {
			body[off] = 0
		}
		off++
		binary.LittleEndian.PutUint32(body[off:off+4], e.RKey)
		off += 4
		binary.LittleEndian.PutUint64(body[off:off+8], e.RemoteAddr)
		off += 8
		binary.LittleEndian.PutUint32(body[off:off+4], e.Length)
		off += 4
	}
	return body
}

// DecodeMGetReadReady decodes batch RDMA metadata from an OpMGetReadReady body.
func DecodeMGetReadReady(body []byte) ([]MGetReadReadyEntry, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("mget read ready body too short: %d", len(body))
	}
	n := int(binary.LittleEndian.Uint32(body[0:4]))
	if len(body) < 4+n*17 {
		return nil, fmt.Errorf("mget read ready body truncated: need %d, got %d", 4+n*17, len(body))
	}
	entries := make([]MGetReadReadyEntry, n)
	off := 4
	for i := 0; i < n; i++ {
		entries[i].Found = body[off] != 0
		off++
		entries[i].RKey = binary.LittleEndian.Uint32(body[off : off+4])
		off += 4
		entries[i].RemoteAddr = binary.LittleEndian.Uint64(body[off : off+8])
		off += 8
		entries[i].Length = binary.LittleEndian.Uint32(body[off : off+4])
		off += 4
	}
	return entries, nil
}

// EncodeBatchReadAck encodes batch WC statuses for OpBatchReadAck.
// Format: count(4) + [wcStatus(1)] Ã— count
func EncodeBatchReadAck(statuses []uint8) []byte {
	n := len(statuses)
	body := make([]byte, 4+n)
	binary.LittleEndian.PutUint32(body[0:4], uint32(n))
	copy(body[4:], statuses)
	return body
}

// DecodeBatchReadAck decodes batch WC statuses from an OpBatchReadAck body.
func DecodeBatchReadAck(body []byte) ([]uint8, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("batch read ack body too short: %d", len(body))
	}
	n := int(binary.LittleEndian.Uint32(body[0:4]))
	if len(body) < 4+n {
		return nil, fmt.Errorf("batch read ack body truncated: need %d, got %d", 4+n, len(body))
	}
	statuses := make([]uint8, n)
	copy(statuses, body[4:4+n])
	return statuses, nil
}

// SerializeMessage serializes a message into dst, returning the number of bytes written.
// This is used by the RDMA transport for buffer-based serialization (avoids bufio.Writer).
// dst must be large enough: HeaderSize + len(msg.Body).
func SerializeMessage(dst []byte, msg Message) int {
	dst[0] = MagicByte0
	dst[1] = MagicByte1
	dst[2] = Version1
	dst[3] = msg.Header.OpCode
	dst[4] = msg.Header.Flags
	binary.LittleEndian.PutUint32(dst[5:9], msg.Header.RequestID)
	binary.LittleEndian.PutUint32(dst[9:13], uint32(len(msg.Body)))
	copy(dst[HeaderSize:], msg.Body)
	return HeaderSize + len(msg.Body)
}

// DeserializeMessage parses a message from a buffer.
// Returns the message and number of bytes consumed.
func DeserializeMessage(src []byte) (Message, int, error) {
	if len(src) < HeaderSize {
		return Message{}, 0, fmt.Errorf("buffer too short for header: %d < %d", len(src), HeaderSize)
	}
	if src[0] != MagicByte0 || src[1] != MagicByte1 {
		return Message{}, 0, ErrBadMagic
	}
	if src[2] != Version1 {
		return Message{}, 0, ErrBadVersion
	}
	h := Header{
		OpCode:    src[3],
		Flags:     src[4],
		RequestID: binary.LittleEndian.Uint32(src[5:9]),
		BodyLen:   binary.LittleEndian.Uint32(src[9:13]),
	}
	if h.BodyLen > MaxBodySize {
		return Message{}, 0, ErrBodyTooLarge
	}
	total := HeaderSize + int(h.BodyLen)
	if len(src) < total {
		return Message{}, 0, fmt.Errorf("buffer too short for body: %d < %d", len(src), total)
	}
	body := make([]byte, h.BodyLen)
	copy(body, src[HeaderSize:total])
	return Message{Header: h, Body: body}, total, nil
}

// --- CXL Read Ready helpers ---

// CXLReadReadySize is the body size for a CXL Read Ready response.
// Format: found(1) + deviceOffset(8) + valueSize(4) = 13 bytes
const CXLReadReadySize = 13

// EncodeCXLReadReady encodes device offset and size for CXL direct load.
func EncodeCXLReadReady(deviceOffset uint64, valueSize uint32) []byte {
	body := make([]byte, CXLReadReadySize)
	body[0] = 1 // found
	binary.LittleEndian.PutUint64(body[1:9], deviceOffset)
	binary.LittleEndian.PutUint32(body[9:13], valueSize)
	return body
}

// DecodeCXLReadReady decodes CXL device offset and size from a CXL Read Ready body.
func DecodeCXLReadReady(body []byte) (deviceOffset uint64, valueSize uint32, found bool, err error) {
	if len(body) < CXLReadReadySize {
		return 0, 0, false, fmt.Errorf("cxl read ready body too short: %d < %d", len(body), CXLReadReadySize)
	}
	if body[0] == 0 {
		return 0, 0, false, nil
	}
	deviceOffset = binary.LittleEndian.Uint64(body[1:9])
	valueSize = binary.LittleEndian.Uint32(body[9:13])
	return deviceOffset, valueSize, true, nil
}

// CXLRegionEntry describes a single devdax region for the CXL region map.
type CXLRegionEntry struct {
	ShardID  uint16
	ClassIdx uint16
	Offset   uint64
	Size     uint64
}

// EncodeCXLRegionMap encodes a CXL region map for RespCXLRegionMap.
// Format: pathLen(2) + path + numEntries(4) + [shardID(2) + classIdx(2) + offset(8) + size(8)] x N
func EncodeCXLRegionMap(devdaxPath string, entries []CXLRegionEntry) []byte {
	pathBytes := []byte(devdaxPath)
	n := len(entries)
	size := 2 + len(pathBytes) + 4 + n*20
	body := make([]byte, size)
	off := 0
	binary.LittleEndian.PutUint16(body[off:], uint16(len(pathBytes)))
	off += 2
	copy(body[off:], pathBytes)
	off += len(pathBytes)
	binary.LittleEndian.PutUint32(body[off:], uint32(n))
	off += 4
	for _, e := range entries {
		binary.LittleEndian.PutUint16(body[off:], e.ShardID)
		off += 2
		binary.LittleEndian.PutUint16(body[off:], e.ClassIdx)
		off += 2
		binary.LittleEndian.PutUint64(body[off:], e.Offset)
		off += 8
		binary.LittleEndian.PutUint64(body[off:], e.Size)
		off += 8
	}
	return body
}

// DecodeCXLRegionMap decodes a CXL region map from a RespCXLRegionMap body.
func DecodeCXLRegionMap(body []byte) (devdaxPath string, entries []CXLRegionEntry, err error) {
	if len(body) < 6 {
		return "", nil, fmt.Errorf("cxl region map body too short: %d", len(body))
	}
	off := 0
	pathLen := int(binary.LittleEndian.Uint16(body[off:]))
	off += 2
	if off+pathLen > len(body) {
		return "", nil, fmt.Errorf("cxl region map path truncated")
	}
	devdaxPath = string(body[off : off+pathLen])
	off += pathLen
	if off+4 > len(body) {
		return "", nil, fmt.Errorf("cxl region map entry count truncated")
	}
	n := int(binary.LittleEndian.Uint32(body[off:]))
	off += 4
	if n > MaxBatchCount {
		return "", nil, fmt.Errorf("cxl region map entry count %d exceeds max %d", n, MaxBatchCount)
	}
	if off+n*20 > len(body) {
		return "", nil, fmt.Errorf("cxl region map entries truncated: need %d, got %d", off+n*20, len(body))
	}
	entries = make([]CXLRegionEntry, n)
	for i := 0; i < n; i++ {
		entries[i].ShardID = binary.LittleEndian.Uint16(body[off:])
		off += 2
		entries[i].ClassIdx = binary.LittleEndian.Uint16(body[off:])
		off += 2
		entries[i].Offset = binary.LittleEndian.Uint64(body[off:])
		off += 8
		entries[i].Size = binary.LittleEndian.Uint64(body[off:])
		off += 8
	}
	return devdaxPath, entries, nil
}

// --- OOM (Out of Memory) response helpers ---

// EncodeOOMResponse encodes a RespOOM body with diagnostic information.
// Format: utilPct(1) + allocatedBytes(8) + totalBytes(8) + msgLen(2) + msg
func EncodeOOMResponse(utilPct uint8, allocatedBytes, totalBytes uint64, msg string) []byte {
	msgBytes := []byte(msg)
	if len(msgBytes) > 1024 {
		msgBytes = msgBytes[:1024]
	}
	body := make([]byte, 1+8+8+2+len(msgBytes))
	body[0] = utilPct
	binary.LittleEndian.PutUint64(body[1:9], allocatedBytes)
	binary.LittleEndian.PutUint64(body[9:17], totalBytes)
	binary.LittleEndian.PutUint16(body[17:19], uint16(len(msgBytes)))
	copy(body[19:], msgBytes)
	return body
}

// DecodeOOMResponse decodes a RespOOM body.
func DecodeOOMResponse(body []byte) (utilPct uint8, allocatedBytes, totalBytes uint64, msg string, err error) {
	if len(body) < 19 {
		return 0, 0, 0, "", fmt.Errorf("oom response too short: %d", len(body))
	}
	utilPct = body[0]
	allocatedBytes = binary.LittleEndian.Uint64(body[1:9])
	totalBytes = binary.LittleEndian.Uint64(body[9:17])
	msgLen := binary.LittleEndian.Uint16(body[17:19])
	if 19+int(msgLen) > len(body) {
		return 0, 0, 0, "", fmt.Errorf("oom response msg truncated")
	}
	msg = string(body[19 : 19+msgLen])
	return utilPct, allocatedBytes, totalBytes, msg, nil
}

// --- Not Ready response helpers ---

// EncodeNotReadyResponse encodes a RespNotReady body with startup progress.
// Format: shardsReady(4 LE) + shardsTotal(4 LE) + msgLen(2 LE) + msg
func EncodeNotReadyResponse(shardsReady, shardsTotal uint32, msg string) []byte {
	msgBytes := []byte(msg)
	if len(msgBytes) > 1024 {
		msgBytes = msgBytes[:1024]
	}
	body := make([]byte, 4+4+2+len(msgBytes))
	binary.LittleEndian.PutUint32(body[0:4], shardsReady)
	binary.LittleEndian.PutUint32(body[4:8], shardsTotal)
	binary.LittleEndian.PutUint16(body[8:10], uint16(len(msgBytes)))
	copy(body[10:], msgBytes)
	return body
}

// DecodeNotReadyResponse decodes a RespNotReady body.
func DecodeNotReadyResponse(body []byte) (shardsReady, shardsTotal uint32, msg string, err error) {
	if len(body) < 10 {
		return 0, 0, "", fmt.Errorf("not ready response too short: %d", len(body))
	}
	shardsReady = binary.LittleEndian.Uint32(body[0:4])
	shardsTotal = binary.LittleEndian.Uint32(body[4:8])
	msgLen := binary.LittleEndian.Uint16(body[8:10])
	if 10+int(msgLen) > len(body) {
		return 0, 0, "", fmt.Errorf("not ready response msg truncated")
	}
	msg = string(body[10 : 10+msgLen])
	return shardsReady, shardsTotal, msg, nil
}

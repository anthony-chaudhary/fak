package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// PagedKVTransferSerializerID is the stable serializer axis for the native paged
// K/Kraw/V wire frame. The frame is host f32 and paged-block granular; high-perf
// GPUDirect/NVLink/UCX bindings are transport follow-ons behind KVTransport.
const PagedKVTransferSerializerID = "fak-paged-kv-f32-v1"

const pagedKVTransferMagic = "FAKPKVT1\n"

// KVTransport is the native P/D KV data-plane seam: it moves a paged KV span plus
// its cachemeta.KVTransfer descriptor and returns the receiver's reconstructed span.
type KVTransport interface {
	Send(seq *PagedKV, transfer cachemeta.KVTransfer, from, n int) (PagedKVTransferReceipt, error)
}

// PagedKVTransferReceipt is the receive-side reconstruction of one moved span.
type PagedKVTransferReceipt struct {
	KV        *PagedKV
	Positions []int
	Transfer  cachemeta.KVTransfer
	Entry     cachemeta.Entry
}

// Verdict replays the transfer entry through the shared cachemeta verdict fold.
func (r PagedKVTransferReceipt) Verdict() cachemeta.LookupVerdict {
	return cachemeta.KVTransferVerdict(r.Entry)
}

type pagedKVTransferMeta struct {
	Serializer  string               `json:"serializer"`
	SpanStart   int                  `json:"span_start"`
	Positions   []int                `json:"positions"`
	BlockTokens int                  `json:"block_tokens"`
	Stride      int                  `json:"stride"`
	Layers      int                  `json:"layers"`
	Planes      int                  `json:"planes"`
	Tokens      int                  `json:"tokens"`
	Blocks      int                  `json:"blocks"`
	Transfer    cachemeta.KVTransfer `json:"transfer"`
}

// MarshalPagedKVTransfer serializes the logical span [from, from+n) of a PagedKV
// sequence into a self-describing wire frame. The source must be a Kraw-capable
// paged pool, which is the structural gate that keeps this serializer on the
// paged/block allocator and preserves exact-span eviction after transfer.
func MarshalPagedKVTransfer(seq *PagedKV, transfer cachemeta.KVTransfer, from, n int) ([]byte, error) {
	blockBytes, positions, nBlocks, err := marshalPagedKVBlockBytes(seq, from, n)
	if err != nil {
		return nil, err
	}
	if transfer.Tokens == 0 {
		transfer.Tokens = int64(n)
	}
	if transfer.Tokens != int64(n) {
		return nil, fmt.Errorf("model: KVTransfer tokens=%d does not match serialized span tokens=%d", transfer.Tokens, n)
	}
	if transfer.SerializerID == "" {
		transfer.SerializerID = PagedKVTransferSerializerID
	}
	if transfer.BytesMoved == 0 {
		transfer.BytesMoved = int64(len(blockBytes))
	}
	if transfer.Outcome == "" {
		transfer.Outcome = cachemeta.KVTransferOK
	}
	if transfer.SpanDigest == "" {
		transfer.SpanDigest = digestPagedKVSpan(positions, blockBytes)
	}

	p := seq.pool
	meta := pagedKVTransferMeta{
		Serializer:  PagedKVTransferSerializerID,
		SpanStart:   from,
		Positions:   positions,
		BlockTokens: p.blockTokens,
		Stride:      p.stride,
		Layers:      p.nLayers,
		Planes:      p.planes,
		Tokens:      n,
		Blocks:      nBlocks,
		Transfer:    transfer,
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return nil, fmt.Errorf("model: marshal paged KV transfer descriptor: %w", err)
	}
	if len(metaJSON) > int(^uint32(0)) {
		return nil, errors.New("model: paged KV transfer descriptor too large")
	}
	frame := make([]byte, 0, len(pagedKVTransferMagic)+4+len(metaJSON)+len(blockBytes))
	frame = append(frame, pagedKVTransferMagic...)
	var hdr [4]byte
	binary.LittleEndian.PutUint32(hdr[:], uint32(len(metaJSON)))
	frame = append(frame, hdr[:]...)
	frame = append(frame, metaJSON...)
	frame = append(frame, blockBytes...)
	return frame, nil
}

// UnmarshalPagedKVTransfer restores a wire frame into fresh owned pages in pool
// and returns the transfer descriptor that crossed the wire with the bytes.
func UnmarshalPagedKVTransfer(pool *PagedKVPool, frame []byte) (PagedKVTransferReceipt, error) {
	if pool == nil {
		return PagedKVTransferReceipt{}, errors.New("model: cannot receive paged KV into nil pool")
	}
	if len(frame) < len(pagedKVTransferMagic)+4 {
		return PagedKVTransferReceipt{}, errors.New("model: short paged KV transfer frame")
	}
	if string(frame[:len(pagedKVTransferMagic)]) != pagedKVTransferMagic {
		return PagedKVTransferReceipt{}, errors.New("model: invalid paged KV transfer frame")
	}
	off := len(pagedKVTransferMagic)
	metaLen := int(binary.LittleEndian.Uint32(frame[off : off+4]))
	off += 4
	if metaLen < 0 || metaLen > len(frame)-off {
		return PagedKVTransferReceipt{}, errors.New("model: paged KV transfer descriptor length exceeds frame")
	}
	var meta pagedKVTransferMeta
	if err := json.Unmarshal(frame[off:off+metaLen], &meta); err != nil {
		return PagedKVTransferReceipt{}, fmt.Errorf("model: unmarshal paged KV transfer descriptor: %w", err)
	}
	off += metaLen
	if err := validatePagedKVTransferMeta(pool, meta); err != nil {
		return PagedKVTransferReceipt{}, err
	}
	blockBytes := frame[off:]
	wantBytes := meta.Blocks * pool.blockFloats() * 4
	if len(blockBytes) != wantBytes {
		return PagedKVTransferReceipt{}, fmt.Errorf("model: paged KV transfer block bytes=%d, want %d", len(blockBytes), wantBytes)
	}

	seq := &PagedKV{pool: pool, table: make([]int, meta.Blocks), nTokens: meta.Tokens}
	fail := func(err error) (PagedKVTransferReceipt, error) {
		seq.Free()
		return PagedKVTransferReceipt{}, err
	}
	blockFloats := pool.blockFloats()
	for bi := 0; bi < meta.Blocks; bi++ {
		id := pool.alloc()
		seq.table[bi] = id
		blk := pool.blocks[id]
		start := bi * blockFloats * 4
		for j := range blk {
			blk[j] = math.Float32frombits(binary.LittleEndian.Uint32(blockBytes[start+j*4 : start+(j+1)*4]))
		}
	}
	positions := append([]int(nil), meta.Positions...)
	transfer := meta.Transfer
	entry := cachemeta.FromKVTransfer(transfer)
	if meta.Tokens == 0 && meta.Blocks != 0 {
		return fail(errors.New("model: empty paged KV transfer carried blocks"))
	}
	return PagedKVTransferReceipt{KV: seq, Positions: positions, Transfer: transfer, Entry: entry}, nil
}

func validatePagedKVTransferMeta(pool *PagedKVPool, meta pagedKVTransferMeta) error {
	if meta.Serializer != PagedKVTransferSerializerID {
		return fmt.Errorf("model: paged KV transfer serializer=%q, want %q", meta.Serializer, PagedKVTransferSerializerID)
	}
	if !pool.SupportsRaw() {
		return errors.New("model: paged KV transfer requires a Kraw-capable destination pool")
	}
	if meta.BlockTokens != pool.blockTokens || meta.Stride != pool.stride || meta.Layers != pool.nLayers || meta.Planes != pool.planes {
		return fmt.Errorf("model: paged KV transfer geometry mismatch: blob block=%d stride=%d layers=%d planes=%d, pool block=%d stride=%d layers=%d planes=%d",
			meta.BlockTokens, meta.Stride, meta.Layers, meta.Planes, pool.blockTokens, pool.stride, pool.nLayers, pool.planes)
	}
	if meta.Tokens < 0 || meta.Blocks < 0 {
		return errors.New("model: paged KV transfer has negative token/block count")
	}
	if len(meta.Positions) != meta.Tokens {
		return fmt.Errorf("model: paged KV transfer positions=%d, want tokens=%d", len(meta.Positions), meta.Tokens)
	}
	wantBlocks := 0
	if meta.Tokens > 0 {
		wantBlocks = (meta.Tokens + pool.blockTokens - 1) / pool.blockTokens
	}
	if meta.Blocks != wantBlocks {
		return fmt.Errorf("model: paged KV transfer blocks=%d, want %d for %d tokens", meta.Blocks, wantBlocks, meta.Tokens)
	}
	if meta.Transfer.Tokens != int64(meta.Tokens) {
		return fmt.Errorf("model: paged KV transfer descriptor tokens=%d, want %d", meta.Transfer.Tokens, meta.Tokens)
	}
	return nil
}

func marshalPagedKVBlockBytes(seq *PagedKV, from, n int) ([]byte, []int, int, error) {
	if seq == nil || seq.pool == nil {
		return nil, nil, 0, errors.New("model: cannot serialize nil PagedKV")
	}
	p := seq.pool
	if !p.SupportsRaw() {
		return nil, nil, 0, errors.New("model: paged KV transfer requires a Kraw-capable source pool")
	}
	if from < 0 || n < 0 {
		return nil, nil, 0, fmt.Errorf("model: invalid paged KV transfer span from=%d n=%d", from, n)
	}
	if from > seq.nTokens || from+n > seq.nTokens {
		return nil, nil, 0, fmt.Errorf("model: paged KV transfer span [%d,%d) exceeds Len=%d", from, from+n, seq.nTokens)
	}
	if p.blockTokens <= 0 {
		return nil, nil, 0, errors.New("model: paged KV transfer source has invalid block size")
	}
	nBlocks := 0
	if n > 0 {
		nBlocks = (n + p.blockTokens - 1) / p.blockTokens
	}
	blockFloats := p.blockFloats()
	maxInt := int(^uint(0) >> 1)
	if blockFloats > 0 && nBlocks > maxInt/(blockFloats*4) {
		return nil, nil, 0, errors.New("model: paged KV transfer span too large")
	}
	blockBytes := make([]byte, nBlocks*blockFloats*4)
	positions := make([]int, n)
	for i := 0; i < n; i++ {
		srcPos := from + i
		positions[i] = srcPos
		srcBlockIdx := srcPos / p.blockTokens
		if srcBlockIdx >= len(seq.table) {
			return nil, nil, 0, fmt.Errorf("model: paged KV transfer source table missing logical block %d", srcBlockIdx)
		}
		srcID := seq.table[srcBlockIdx]
		if srcID < 0 || srcID >= len(p.blocks) || p.ref[srcID] == 0 {
			return nil, nil, 0, fmt.Errorf("model: paged KV transfer source has invalid block id %d", srcID)
		}
		srcBlock := p.blocks[srcID]
		srcOff := srcPos % p.blockTokens
		dstBlockIdx := i / p.blockTokens
		dstOff := i % p.blockTokens
		for l := 0; l < p.nLayers; l++ {
			for plane := 0; plane < p.planes; plane++ {
				src := p.slot(l, plane, srcOff)
				dstFloat := dstBlockIdx*blockFloats + p.slot(l, plane, dstOff)
				for j := 0; j < p.stride; j++ {
					binary.LittleEndian.PutUint32(blockBytes[(dstFloat+j)*4:(dstFloat+j+1)*4], math.Float32bits(srcBlock[src+j]))
				}
			}
		}
	}
	return blockBytes, positions, nBlocks, nil
}

func digestPagedKVSpan(positions []int, blockBytes []byte) string {
	h := sha256.New()
	var buf [8]byte
	for _, pos := range positions {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(pos)))
		_, _ = h.Write(buf[:])
	}
	_, _ = h.Write(blockBytes)
	return hex.EncodeToString(h.Sum(nil))
}

// LocalKVTransport moves through the same serializer/deserializer without a socket.
// It is useful as the behavior oracle for a real framed transport.
type LocalKVTransport struct {
	Pool *PagedKVPool
}

func (t LocalKVTransport) Send(seq *PagedKV, transfer cachemeta.KVTransfer, from, n int) (PagedKVTransferReceipt, error) {
	frame, err := MarshalPagedKVTransfer(seq, transfer, from, n)
	if err != nil {
		return PagedKVTransferReceipt{}, err
	}
	return UnmarshalPagedKVTransfer(t.Pool, frame)
}

// TCPKVTransport is the loopback-proven wire adapter for native KV transfer. It
// moves the exact same paged-KV frame over a real stream connection; a UCX/RDMA
// backend is correct only if it reproduces this byte contract.
type TCPKVTransport struct {
	conn net.Conn
	pool *PagedKVPool
}

func NewTCPKVTransport(conn net.Conn, pool *PagedKVPool) *TCPKVTransport {
	return &TCPKVTransport{conn: conn, pool: pool}
}

func (t *TCPKVTransport) Send(seq *PagedKV, transfer cachemeta.KVTransfer, from, n int) (PagedKVTransferReceipt, error) {
	if t == nil || t.conn == nil {
		return PagedKVTransferReceipt{}, errors.New("model: TCPKVTransport has no connection")
	}
	frame, err := MarshalPagedKVTransfer(seq, transfer, from, n)
	if err != nil {
		return PagedKVTransferReceipt{}, err
	}
	if err := writeFrame(t.conn, frame); err != nil {
		return PagedKVTransferReceipt{}, fmt.Errorf("model: TCPKVTransport send: %w", err)
	}
	reply, err := readFrame(t.conn)
	if err != nil {
		return PagedKVTransferReceipt{}, fmt.Errorf("model: TCPKVTransport recv: %w", err)
	}
	return UnmarshalPagedKVTransfer(t.pool, reply)
}

// EchoKVTransferFrames is the loopback peer for TCPKVTransport: it proves the
// transport preserves the transfer frame byte-for-byte before a band-running
// native prefill/decode worker consumes it.
func EchoKVTransferFrames(conn net.Conn) error {
	for {
		frame, err := readFrame(conn)
		if err != nil {
			if err == io.EOF || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		if err := writeFrame(conn, frame); err != nil {
			return err
		}
	}
}

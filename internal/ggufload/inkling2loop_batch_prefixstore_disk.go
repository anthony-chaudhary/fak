package ggufload

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/kvcache"
)

const (
	ink2DiskMagic      = "INK2CKPT"
	ink2DiskVersion    = uint32(1)
	ink2DiskHeaderSize = 8 + 4 + 4 + sha256.Size // magic + version + flags + sha256
)

var (
	errInk2DiskCorrupt   = errors.New("inkling2 prefix disk: corrupt checkpoint file")
	errInk2DiskLoopSlots = errors.New("inkling2 prefix disk: loopSlots mismatch")
)

type ink2DiskEntry struct {
	tokens    []int
	file      string
	fileBytes int64
	used      uint64
	loopSlots int
}

// ink2PrefixDisk is an opt-in NVMe/disk spillover tier for inkling2PrefixStore.
// Evicted host-RAM checkpoints are written atomically with a SHA-256 header and
// reloaded on a later prefix hit. Any I/O, checksum, or shape error fails closed
// by deleting the suspect file and returning a miss.
type ink2PrefixDisk struct {
	dir       string
	capBytes  int64
	heldBytes int64
	entries   []*ink2DiskEntry
}

func newInkling2PrefixDisk(dir string, capBytes int64) *ink2PrefixDisk {
	return &ink2PrefixDisk{
		dir:      dir,
		capBytes: capBytes,
	}
}

func ink2DiskFilename(tokens []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(t)))
		_, _ = h.Write(buf[:])
	}
	sum := h.Sum(nil)
	return "ckpt-" + hex.EncodeToString(sum[:16]) + ".ink2"
}

func encodeInk2DiskCkpt(tokens []int, loopSlots int, ck *ink2RowCkpt) ([]byte, error) {
	if ck == nil || len(tokens) != ck.pos || ck.pos < 0 || ck.posFloats < 0 || loopSlots < 0 {
		return nil, errInk2DiskCorrupt
	}
	payloadLen := 4 + 4 + 8*len(tokens) + 4 + 4 + 4 + 4
	for _, sl := range ck.slots {
		if sl.lo < 0 {
			return nil, errInk2DiskCorrupt
		}
		payloadLen += 16 + 4*(len(sl.k)+len(sl.v)+len(sl.hist))
	}
	buf := make([]byte, ink2DiskHeaderSize+payloadLen)
	off := ink2DiskHeaderSize

	binary.LittleEndian.PutUint32(buf[off:], uint32(loopSlots))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(len(tokens)))
	off += 4
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(buf[off:], uint64(int64(t)))
		off += 8
	}
	binary.LittleEndian.PutUint32(buf[off:], uint32(ck.pos))
	off += 4
	if ck.half {
		buf[off] = 1
	}
	off += 4 // half (1 byte) + 3 reserved zero bytes
	binary.LittleEndian.PutUint32(buf[off:], uint32(ck.posFloats))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(len(ck.slots)))
	off += 4

	for _, sl := range ck.slots {
		binary.LittleEndian.PutUint32(buf[off:], uint32(sl.lo))
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], uint32(len(sl.k)))
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], uint32(len(sl.v)))
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], uint32(len(sl.hist)))
		off += 4
		for _, f := range sl.k {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
		for _, f := range sl.v {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
		for _, f := range sl.hist {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
	}

	copy(buf[0:8], ink2DiskMagic)
	binary.LittleEndian.PutUint32(buf[8:12], ink2DiskVersion)
	binary.LittleEndian.PutUint32(buf[12:16], 0)
	sum := sha256.Sum256(buf[ink2DiskHeaderSize:])
	copy(buf[16:48], sum[:])
	return buf, nil
}

// encodeInk2SnapshotBytes serializes an Inkling-2 row checkpoint into the shared
// versioned kvcache.Snapshot wire format.
func encodeInk2SnapshotBytes(tokens []int, loopSlots, kvHeads, headDim int, ck *ink2RowCkpt) ([]byte, error) {
	if ck == nil || len(tokens) != ck.pos || ck.pos < 0 || loopSlots <= 0 {
		return nil, errInk2DiskCorrupt
	}
	prec := kvcache.PrecisionFP32
	if ck.half {
		prec = kvcache.PrecisionFP16
	}
	layers := make([]kvcache.SnapshotLayer, len(ck.slots))
	for i, sl := range ck.slots {
		layers[i] = kvcache.SnapshotLayer{
			Index:  sl.lo,
			KCodes: kvcache.Float32sToBytes(sl.k),
			VCodes: kvcache.Float32sToBytes(sl.v),
			Extra:  kvcache.Float32sToBytes(sl.hist),
		}
	}
	return kvcache.EncodeSnapshot(&kvcache.Snapshot{
		Layout: kvcache.LayoutDesc{
			Arch:      "inkling2",
			Precision: prec,
			Layers:    len(ck.slots),
			KVHeads:   kvHeads,
			HeadDim:   headDim,
			Capacity:  ck.posFloats,
			Layout:    kvcache.LayoutContiguous,
		},
		Tokens: append([]int(nil), tokens...),
		Layers: layers,
	})
}

// decodeInk2SnapshotBytes restores an Inkling-2 row checkpoint from the shared
// kvcache.Snapshot wire format.
func decodeInk2SnapshotBytes(data []byte, expectedLoopSlots int) ([]int, *ink2RowCkpt, error) {
	var exp *kvcache.LayoutDesc
	if expectedLoopSlots > 0 {
		exp = &kvcache.LayoutDesc{Arch: "inkling2", Layers: expectedLoopSlots}
	}
	snap, err := kvcache.DecodeSnapshot(data, exp)
	if err != nil {
		if errors.Is(err, kvcache.ErrSnapshotIncompatible) {
			return nil, nil, errInk2DiskLoopSlots
		}
		return nil, nil, errInk2DiskCorrupt
	}
	slots := make([]ink2SlotCkpt, len(snap.Layers))
	for i, l := range snap.Layers {
		k, err := kvcache.BytesToFloat32s(l.KCodes)
		if err != nil {
			return nil, nil, errInk2DiskCorrupt
		}
		v, err := kvcache.BytesToFloat32s(l.VCodes)
		if err != nil {
			return nil, nil, errInk2DiskCorrupt
		}
		hist, err := kvcache.BytesToFloat32s(l.Extra)
		if err != nil {
			return nil, nil, errInk2DiskCorrupt
		}
		slots[i] = ink2SlotCkpt{lo: l.Index, k: k, v: v, hist: hist}
	}
	return snap.Tokens, &ink2RowCkpt{
		pos:       len(snap.Tokens),
		half:      snap.Layout.Precision == kvcache.PrecisionFP16,
		posFloats: snap.Layout.Capacity,
		slots:     slots,
	}, nil
}

func decodeInk2DiskCkpt(data []byte, expectedLoopSlots int) ([]int, *ink2RowCkpt, error) {
	if len(data) < ink2DiskHeaderSize {
		return nil, nil, errInk2DiskCorrupt
	}
	if string(data[0:8]) != ink2DiskMagic {
		return nil, nil, errInk2DiskCorrupt
	}
	if binary.LittleEndian.Uint32(data[8:12]) != ink2DiskVersion {
		return nil, nil, errInk2DiskCorrupt
	}
	if binary.LittleEndian.Uint32(data[12:16]) != 0 {
		return nil, nil, errInk2DiskCorrupt
	}
	payload := data[ink2DiskHeaderSize:]
	sum := sha256.Sum256(payload)
	var want [sha256.Size]byte
	copy(want[:], data[16:48])
	if sum != want {
		return nil, nil, errInk2DiskCorrupt
	}

	off := 0
	readU32 := func() (uint32, error) {
		if len(payload)-off < 4 {
			return 0, errInk2DiskCorrupt
		}
		v := binary.LittleEndian.Uint32(payload[off:])
		off += 4
		return v, nil
	}
	readF32Slice := func(n uint32) ([]float32, error) {
		bytesNeeded := int64(n) * 4
		if bytesNeeded < 0 || int64(len(payload)-off) < bytesNeeded {
			return nil, errInk2DiskCorrupt
		}
		if n == 0 {
			return nil, nil
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off:]))
			off += 4
		}
		return out, nil
	}

	fileLoopSlotsU32, err := readU32()
	if err != nil {
		return nil, nil, err
	}
	fileLoopSlots := int(fileLoopSlotsU32)
	if expectedLoopSlots != 0 && fileLoopSlots != expectedLoopSlots {
		return nil, nil, errInk2DiskLoopSlots
	}

	numTokensU32, err := readU32()
	if err != nil {
		return nil, nil, err
	}
	tokBytes := int64(numTokensU32) * 8
	if tokBytes < 0 || int64(len(payload)-off) < tokBytes {
		return nil, nil, errInk2DiskCorrupt
	}
	tokens := make([]int, numTokensU32)
	for i := range tokens {
		tokens[i] = int(int64(binary.LittleEndian.Uint64(payload[off:])))
		off += 8
	}

	posU32, err := readU32()
	if err != nil {
		return nil, nil, err
	}
	if int(posU32) != len(tokens) {
		return nil, nil, errInk2DiskCorrupt
	}
	if len(payload)-off < 4 {
		return nil, nil, errInk2DiskCorrupt
	}
	if payload[off] > 1 || payload[off+1] != 0 || payload[off+2] != 0 || payload[off+3] != 0 {
		return nil, nil, errInk2DiskCorrupt
	}
	half := payload[off] == 1
	off += 4

	posFloatsU32, err := readU32()
	if err != nil {
		return nil, nil, err
	}
	numSlotsU32, err := readU32()
	if err != nil {
		return nil, nil, err
	}
	numSlots := int(numSlotsU32)
	if expectedLoopSlots != 0 && numSlots != expectedLoopSlots {
		return nil, nil, errInk2DiskLoopSlots
	}
	if fileLoopSlots != 0 && numSlots != fileLoopSlots {
		return nil, nil, errInk2DiskLoopSlots
	}

	slots := make([]ink2SlotCkpt, numSlots)
	for i := range slots {
		lo, err := readU32()
		if err != nil {
			return nil, nil, err
		}
		lenK, err := readU32()
		if err != nil {
			return nil, nil, err
		}
		lenV, err := readU32()
		if err != nil {
			return nil, nil, err
		}
		lenHist, err := readU32()
		if err != nil {
			return nil, nil, err
		}
		k, err := readF32Slice(lenK)
		if err != nil {
			return nil, nil, err
		}
		v, err := readF32Slice(lenV)
		if err != nil {
			return nil, nil, err
		}
		hist, err := readF32Slice(lenHist)
		if err != nil {
			return nil, nil, err
		}
		slots[i] = ink2SlotCkpt{lo: int(lo), k: k, v: v, hist: hist}
	}
	if off != len(payload) {
		return nil, nil, errInk2DiskCorrupt
	}

	ck := &ink2RowCkpt{
		pos:       int(posU32),
		half:      half,
		posFloats: int(posFloatsU32),
		slots:     slots,
	}
	return tokens, ck, nil
}

func (d *ink2PrefixDisk) removeExact(tokens []int) {
	for i, e := range d.entries {
		if len(e.tokens) == len(tokens) && inkling2HasPrefix(tokens, e.tokens) {
			_ = os.Remove(e.file)
			d.heldBytes -= e.fileBytes
			d.entries = append(d.entries[:i], d.entries[i+1:]...)
			return
		}
	}
}

func (d *ink2PrefixDisk) removeAt(idx int) {
	e := d.entries[idx]
	_ = os.Remove(e.file)
	d.heldBytes -= e.fileBytes
	d.entries = append(d.entries[:idx], d.entries[idx+1:]...)
}

func writeFileAtomic(dir, finalPath string, data []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".ink2ckpt-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tmpPath)
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return fmt.Errorf("rename %s -> %s: %w", tmpPath, finalPath, err)
	}
	cleanup = false
	return nil
}

func (st *inkling2PrefixStore) spillToDisk(victim *ink2PrefixEntry) {
	d := st.disk
	if d == nil || victim == nil || d.capBytes <= 0 {
		return
	}
	data, err := encodeInk2DiskCkpt(victim.tokens, st.loopSlots, victim.ck)
	if err != nil {
		st.stats.DiskErrors++
		return
	}
	fileBytes := int64(len(data))
	if fileBytes > d.capBytes {
		return
	}
	d.removeExact(victim.tokens)
	finalPath := filepath.Join(d.dir, ink2DiskFilename(victim.tokens))
	if err := writeFileAtomic(d.dir, finalPath, data); err != nil {
		st.stats.DiskErrors++
		return
	}
	st.tick++
	d.entries = append(d.entries, &ink2DiskEntry{
		tokens:    append([]int(nil), victim.tokens...),
		file:      finalPath,
		fileBytes: fileBytes,
		used:      st.tick,
		loopSlots: st.loopSlots,
	})
	d.heldBytes += fileBytes
	st.stats.DiskSpills++

	for d.heldBytes > d.capBytes && len(d.entries) > 0 {
		lru := 0
		for i, e := range d.entries {
			if e.used < d.entries[lru].used {
				lru = i
			}
		}
		d.removeAt(lru)
		st.stats.DiskEvictions++
	}
}

func (st *inkling2PrefixStore) lookupDiskLongerThan(prompt []int, minLen int) *ink2PrefixEntry {
	d := st.disk
	if d == nil {
		return nil
	}
	for {
		bestIdx := -1
		for i, e := range d.entries {
			if len(e.tokens) < len(prompt) && len(e.tokens) > minLen &&
				(bestIdx < 0 || len(e.tokens) > len(d.entries[bestIdx].tokens)) &&
				inkling2HasPrefix(prompt, e.tokens) {
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			return nil
		}
		cand := d.entries[bestIdx]
		data, err := os.ReadFile(cand.file)
		if err != nil {
			d.removeAt(bestIdx)
			st.stats.DiskErrors++
			continue
		}
		tokens, ck, err := decodeInk2DiskCkpt(data, st.loopSlots)
		if err != nil || len(tokens) != len(cand.tokens) || !inkling2HasPrefix(tokens, cand.tokens) {
			d.removeAt(bestIdx)
			st.stats.DiskErrors++
			continue
		}
		d.removeAt(bestIdx)

		ramBytes := ck.Bytes() + int64(8*len(tokens))
		st.tick++
		promoted := &ink2PrefixEntry{
			loopSlots: st.loopSlots,
			tokens:    tokens,
			ck:        ck,
			bytes:     ramBytes,
			used:      st.tick,
		}
		if ramBytes <= st.capBytes {
			st.entries = append(st.entries, promoted)
			st.held += ramBytes
			st.evictRAMToCap()
		}
		st.stats.Hits++
		st.stats.DiskHits++
		st.stats.HitTokens += uint64(len(promoted.tokens))
		return promoted
	}
}

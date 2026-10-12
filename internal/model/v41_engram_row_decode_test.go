package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41EngramRowDecodeRef transcribes the device row decode contract
// (shaders/v41_engram_row_decode.comp) independently of the host decode: the
// packed product is built from integer fields, the Q2_K table from explicitly
// rounded binary32 steps. It returns the F32 bits of the first dim elements and
// the row status word.
func v41EngramRowDecodeRef(row []byte, encoding compute.V41EngramRowEncoding, dim int) ([]uint32, uint32) {
	out := make([]uint32, dim)
	fault := uint32(math.MaxUint32)
	note := func(j int, kind uint32) {
		if key := uint32(2*j) + kind; key < fault {
			fault = key
		}
	}
	finite := func(bits uint32) bool { return bits&0x7f800000 != 0x7f800000 }
	for j := 0; j < dim; j++ {
		var bits uint32
		switch encoding {
		case compute.V41EngramRowPacked:
			code, scale := uint32(row[j]), uint32(row[256+j/32])
			if code&0x7f == 0x7f || scale == 0xff {
				note(j, 0)
				continue
			}
			sign := (code & 0x80) << 24
			e := (code >> 3) & 0xf
			mant, q := code&7, -9
			if e != 0 {
				mant, q = 8|code&7, int(e)-10
			}
			bits = sign
			if mant != 0 {
				k := q + int(scale) - 127
				msb := 31 - int(bitsLeadingZeros32(mant))
				switch x := msb + k; {
				case x > 127:
					bits = sign | 0x7f800000
				case x >= -126:
					bits = sign | uint32(x+127)<<23 | (mant<<uint(23-msb))&0x7fffff
				default:
					bits = sign | mant<<uint(k+149)
				}
			}
			if finite(bits) {
				bits += 0x7fff + (bits>>16)&1
			}
			bits &= 0xffff0000
		case compute.V41EngramRowQ2K:
			half, r := j>>7, j&127
			sub, h, l := r>>5, (r>>4)&1, r&15
			sc := row[half*8+sub*2+h]
			code := (row[16+half*32+h*16+l] >> (2 * sub)) & 3
			d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(row[80:])))
			m := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(row[82:])))
			dl := float32(d * float32(sc&15))
			ml := float32(m * float32(sc>>4))
			switch code {
			case 0:
				if ml == 0 {
					bits = 0
				} else {
					bits = math.Float32bits(ml) ^ 0x80000000
				}
			case 1:
				bits = math.Float32bits(float32(dl - ml))
			default:
				bits = math.Float32bits(float32(float32(dl*float32(code)) - ml))
			}
		default:
			bits = binary.LittleEndian.Uint32(row[4*j:])
		}
		if !finite(bits) {
			note(j, 1)
		}
		out[j] = bits
	}
	if fault == math.MaxUint32 {
		return out, 0
	}
	return out, fault + 1
}

func bitsLeadingZeros32(x uint32) uint {
	n := uint(0)
	for x&0x80000000 == 0 && n < 32 {
		x <<= 1
		n++
	}
	return n
}

func (b *v41MHCTestBackend) SupportsV41EngramRowDecode() bool { return b.rowDecode }

func (b *v41MHCTestBackend) V41EngramRowDecode(encoded []byte, encoding compute.V41EngramRowEncoding, rows, dim int) (compute.Tensor, compute.Tensor, error) {
	stride := compute.V41EngramRowStride(encoding)
	if stride == 0 || rows <= 0 || dim <= 0 || dim > compute.V41EngramRowDecodeMaxDim || len(encoded) != rows*stride {
		panic(&compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassInvalidGeometry, Err: compute.ErrVulkanInvalidGeometry, Message: "invalid Engram row decode operand"})
	}
	values := make([]float32, rows*dim)
	words := make([]float32, rows)
	for r := 0; r < rows; r++ {
		bits, word := v41EngramRowDecodeRef(encoded[r*stride:(r+1)*stride], encoding, dim)
		for j, v := range bits {
			values[r*dim+j] = math.Float32frombits(v)
		}
		words[r] = math.Float32frombits(word)
	}
	panel, err := b.publish("row-decode", []int{rows, dim}, values)
	if err != nil {
		return compute.Tensor{}, compute.Tensor{}, err
	}
	status := compute.NewF32(b, []int{rows}, words)
	b.outputs[status.Buf()] = "row-decode-status"
	return panel, status, nil
}

func (b *v41MHCTestBackend) V41EngramRowDecodePoison(target, status compute.Tensor, rows, index int) error {
	words := b.operand(status, rows)
	b.calls["row-poison"]++
	for _, w := range words {
		if math.Float32bits(w) != 0 {
			b.Backend.Read(target)[index] = float32(math.NaN())
			break
		}
	}
	return nil
}

// The device contract's integer packed product and explicitly rounded Q2_K
// table equal the host decode bit for bit: every E4M3 code against every E8M0
// scale, random Q2_K blocks (with zero, subnormal, Inf and NaN f16 scales) and
// F32 rows, with the same first refusal.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramRowDecodeReferenceMatchesHostDecode(t *testing.T) {
	const dim = 256
	host := func(row []byte, rowBytes int) ([]uint32, error) {
		values, err := decodeV41EngramRow(row, dim, rowBytes)
		if err != nil {
			return nil, err
		}
		bits := make([]uint32, len(values))
		for i, v := range values {
			bits[i] = math.Float32bits(v)
		}
		return bits, nil
	}
	check := func(label string, row []byte, encoding compute.V41EngramRowEncoding, rowBytes int) {
		t.Helper()
		got, word := v41EngramRowDecodeRef(row, encoding, dim)
		want, err := host(row, rowBytes)
		if err != nil {
			j, nanByte, ok := compute.V41EngramRowFault(word)
			wantMsg := map[bool]string{true: "NaN code/scale byte at ", false: " value "}[nanByte]
			if !ok || !strings.Contains(err.Error(), wantMsg+itoa(j)) {
				t.Fatalf("%s: host refusal %v, device status %#x", label, err, word)
			}
			return
		}
		if word != 0 {
			t.Fatalf("%s: device status %#x on a row the host admits", label, word)
		}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("%s element %d: device %08x host %08x", label, j, got[j], want[j])
			}
		}
	}
	lanes := 0
	for scale := 0; scale < 256; scale++ {
		row := make([]byte, V41EngramPackedRowBytes)
		for j := 0; j < 256; j++ {
			row[j] = byte(j)
		}
		for s := range 8 {
			row[256+s] = byte(scale)
		}
		check("packed", row, compute.V41EngramRowPacked, V41EngramPackedRowBytes)
		// Every code in its own row, so one fault cannot hide another.
		for code := 0; code < 256; code++ {
			single := make([]byte, V41EngramPackedRowBytes)
			single[0] = byte(code)
			for s := range 8 {
				single[256+s] = byte(scale)
			}
			got, word := v41EngramRowDecodeRef(single, compute.V41EngramRowPacked, 1)
			want, err := decodeV41EngramRow(single, 1, V41EngramPackedRowBytes)
			switch {
			case (err != nil) != (word != 0):
				t.Fatalf("code %#x scale %d: host err=%v device status %#x", code, scale, err, word)
			case err == nil && got[0] != math.Float32bits(want[0]):
				t.Fatalf("code %#x scale %d: device %08x host %08x", code, scale, got[0], math.Float32bits(want[0]))
			}
			lanes++
		}
	}
	rng := rand.New(rand.NewSource(13668))
	f16s := []uint16{0, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00}
	for i := 0; i < 4000; i++ {
		row := make([]byte, v41EngramQ2KEncodedRowBytes)
		rng.Read(row)
		if i%3 == 0 {
			binary.LittleEndian.PutUint16(row[80:], f16s[rng.Intn(len(f16s))])
			binary.LittleEndian.PutUint16(row[82:], f16s[rng.Intn(len(f16s))])
		} else {
			binary.LittleEndian.PutUint16(row[80:], uint16(rng.Intn(0x7c00)))
			binary.LittleEndian.PutUint16(row[82:], uint16(rng.Intn(0x7c00))|uint16(rng.Intn(2))<<15)
		}
		if runtime.GOARCH == "amd64" {
			// Elsewhere Go may fuse the host table's dl*3 - ml; the device keeps
			// the two roundings amd64 performs.
			check("q2_k", row, compute.V41EngramRowQ2K, v41EngramQ2KEncodedRowBytes)
		}
		f32 := make([]byte, V41EngramF32RowBytes)
		for j := 0; j < 256; j++ {
			binary.LittleEndian.PutUint32(f32[4*j:], rng.Uint32())
		}
		check("f32", f32, compute.V41EngramRowF32, V41EngramF32RowBytes)
		lanes += 2 * dim
	}
	t.Logf("V41_ENGRAM_ROW_DECODE_REF lanes=%d", lanes)
}

// fak-test:runtime medium est=5s lane=default
func TestV41EngramRowDecodeRunsOnDeviceWithNoHostDecode(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		m, _ := v41IncrementalEngramFixture(t, compressed)
		oracle, _ := v41IncrementalEngramFixture(t, compressed)
		control, _ := v41IncrementalEngramFixture(t, compressed)
		b := v41EngramPoolBackend(true, true)
		hostDecode := v41EngramPoolBackend(true, true)
		hostDecode.rowDecode = false
		device, hostArm := v41DenseTestSession(t, m, b), v41DenseTestSession(t, control, hostDecode)
		prefix, next := v41IncrementalPrefix(m.Cfg, 4)
		// The cpu-ref fakes round BF16 on the host themselves, so host rounds are
		// witnessed on hardware (TestV41EngramRowDecodeVulkan), not here.
		run := func(s *Session) (logits [][]float32, decodes int64) {
			stop := enableV41BF16Witness()
			v41HostEngramRowDecodes.Store(0)
			logits = append(logits, s.Prefill(prefix))
			logits = append(logits, s.Step(next))
			logits = append(logits, s.Prefill([]int{2, 5}))
			decodes = v41HostEngramRowDecodes.Load()
			stop()
			return logits, decodes
		}
		got, decodes := run(device)
		want, hostDecodes := run(hostArm)
		if decodes != 0 {
			t.Fatalf("compressed=%t device decode session made %d host row decodes", compressed, decodes)
		}
		if hostDecodes == 0 || b.calls["row-decode"] != hostDecode.calls["engram"] || b.calls["row-poison"] != b.calls["engram"] {
			t.Fatalf("compressed=%t decodes=%d host=%d gate=%d poison=%d control gate=%d", compressed, b.calls["row-decode"], hostDecodes, b.calls["engram"], b.calls["row-poison"], hostDecode.calls["engram"])
		}
		for i := range want {
			for j := range want[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
					t.Fatalf("compressed=%t row %d logit %d: device decode %08x host decode %08x", compressed, i, j, math.Float32bits(got[i][j]), math.Float32bits(want[i][j]))
				}
			}
		}
		host := oracle.NewSession()
		assertV41RowsClose(t, "Engram prefill", got[0], host.Prefill(prefix), 1e-6)
		host.Close()
		if len(b.outputs) != 0 {
			t.Fatalf("compressed=%t device decode left %d live outputs", compressed, len(b.outputs))
		}
	}
}

// A malformed row refuses with the host decode's typed error, closes the
// session, invalidates the layer cache and never publishes the panel.
// fak-test:runtime fast est=2s lane=default
func TestV41EngramRowDecodeFaultMatchesHostRefusal(t *testing.T) {
	refusal := func(rowDecode bool) (error, *v41MHCTestBackend, *Session) {
		m, sources := v41IncrementalEngramFixture(t, false)
		b := v41EngramPoolBackend(true, false)
		b.rowDecode = rowDecode
		s := v41DenseTestSession(t, m, b)
		prefix, next := v41IncrementalPrefix(m.Cfg, 4)
		s.Prefill(prefix)
		for _, layer := range m.v41EngramStageFor().caches {
			layer.invalidateRows()
		}
		for _, src := range sources {
			src.malformed = true
		}
		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err, _ = r.(error)
				}
			}()
			_, _, err = m.forwardV41Step(next, s.v41State(), &v41ProjScratch{})
		}()
		return err, b, s
	}
	deviceErr, b, s := refusal(true)
	hostErr, _, _ := refusal(false)
	if hostErr == nil || deviceErr == nil || !errors.Is(deviceErr, ErrV41NativeUnsupported) {
		t.Fatalf("device refusal %v, host refusal %v", deviceErr, hostErr)
	}
	const want = "Engram row has a NaN code/scale byte at 0"
	if !strings.Contains(deviceErr.Error(), want) || !strings.Contains(hostErr.Error(), want) {
		t.Fatalf("device refusal %q and host refusal %q must both name %q", deviceErr, hostErr, want)
	}
	if !s.BackendSessionClosed() || b.calls["row-decode"] == 0 || b.streamUploads != 0 {
		t.Fatalf("closed=%t decodes=%d republications=%d", s.BackendSessionClosed(), b.calls["row-decode"], b.streamUploads)
	}
	for _, cache := range s.M.v41EngramStageFor().caches {
		if len(cache.inner.entries) != 0 {
			t.Fatal("device decode fault left the faulting rows cached")
		}
	}
}

// Under device-only a gate without the row decode refuses typed at the Engram
// layer before decoding anything on the host (admission names it first:
// TestV41DeviceOnlyNamesHostEngramRowDecode).
// fak-test:runtime fast est=2s lane=default
func TestV41DeviceOnlyRefusesHostEngramRowDecode(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, false)
	b := v41EngramPoolBackend(true, false)
	b.rowDecode = false
	s := v41DenseTestSession(t, m, b)
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	stop := enableV41BF16Witness()
	v41HostEngramRowDecodes.Store(0)
	_, err := s.v41State().gatherOps.engramGateFunc()(0, nil, nil, 1, v41EngramGeometry{}, 1e-6)
	decodes := v41HostEngramRowDecodes.Load()
	stop()
	var closed *BackendForwardOperationError
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Stage != "engram-row-decode declined" {
		t.Fatalf("device-only Engram gate without the row decode = %v, want the typed engram-row-decode refusal", err)
	}
	if decodes != 0 || !s.BackendSessionClosed() || b.calls["engram"] != 0 {
		t.Fatalf("refusal decoded %d rows on the host, closed=%t, gate dispatches=%d", decodes, s.BackendSessionClosed(), b.calls["engram"])
	}
}

// v41EngramQ2KFixtureSource returns a verified Q2_K source over random blocks:
// its served rows are their host dequantization, and its binding is the tree
// over the blocks, so the stage caches the blocks themselves.
func v41EngramQ2KFixtureSource(t *testing.T, rows int, seed int64) (V41EngramRowSource, []byte) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	blocks := make([]byte, rows*v41EngramQ2KEncodedRowBytes)
	rng.Read(blocks)
	decoded := make([]byte, rows*V41EngramF32RowBytes)
	for r := 0; r < rows; r++ {
		blk := blocks[r*v41EngramQ2KEncodedRowBytes : (r+1)*v41EngramQ2KEncodedRowBytes]
		binary.LittleEndian.PutUint16(blk[80:], 0x2000+uint16(rng.Intn(0x800)))
		binary.LittleEndian.PutUint16(blk[82:], 0x1c00+uint16(rng.Intn(0x800)))
		var row [qkK]float32
		q2kDequantSuperBlockScalar(row[:], blk)
		for j, v := range row {
			binary.LittleEndian.PutUint32(decoded[r*V41EngramF32RowBytes+4*j:], math.Float32bits(v))
		}
	}
	reader := v41EngramEncodedFake{ReaderAt: bytes.NewReader(decoded), encoded: blocks, encoding: V41EngramEncodingQ2K}
	shard := "engram-q2k-" + itoa(int(seed))
	src, err := NewV41AcquiredEngramRowSource(reader, V41EngramShard{ID: shard, ExpectedID: shard, Size: int64(len(decoded)), Rows: rows},
		V41EngramArtifactBinding{Shard: shard, Size: int64(len(decoded)), Rows: rows, RowBytes: V41EngramF32RowBytes})
	if err != nil {
		t.Fatal(err)
	}
	return src, decoded
}

// A verified Q2_K table is cached as its blocks and decoded where consumed: the
// host arm's logits equal those of the same table served as dequantized rows,
// and a device decode session matches them bit for bit with no host decode.
// fak-test:runtime medium est=5s lane=default
func TestV41EngramQ2KBlocksDecodeWhereConsumed(t *testing.T) {
	wire := func(encoded bool) *Model {
		m, _ := v41IncrementalEngramFixture(t, false)
		layout := m.v41EngramStageFor().layout
		v41EngramStages.Delete(m)
		srcs := make([]V41EngramRowSource, len(layout.Rows))
		for i := range srcs {
			src, decoded := v41EngramQ2KFixtureSource(t, int(layout.Rows[i]), int64(i+1))
			if !encoded {
				shard := "engram-f32-" + itoa(i)
				binding := V41EngramArtifactBinding{Shard: shard, Size: int64(len(decoded)), Rows: int(layout.Rows[i]), RowBytes: V41EngramF32RowBytes, Digest: v41EngramTestDigest(decoded)}
				var err error
				src, err = NewV41VerifiedEngramRowSource(bytes.NewReader(decoded), V41EngramShard{ID: shard, ExpectedID: shard, Size: binding.Size, Rows: binding.Rows}, binding)
				if err != nil {
					t.Fatal(err)
				}
			}
			srcs[i] = src
		}
		if err := m.wireV41Engram(layout, srcs, int64(V41EngramF32RowBytes)*4); err != nil {
			t.Fatal(err)
		}
		return m
	}
	blocks, rows, device := wire(true), wire(false), wire(true)
	if got := blocks.v41EngramStageFor().rowBytes; got != v41EngramQ2KEncodedRowBytes {
		t.Fatalf("verified Q2_K stage caches %d-byte rows, want the %d-byte blocks", got, v41EngramQ2KEncodedRowBytes)
	}
	if got := rows.v41EngramStageFor().rowBytes; got != V41EngramF32RowBytes {
		t.Fatalf("flat-digest f32 stage caches %d-byte rows", got)
	}
	ids := []int{1, 3, 5, 2}
	want := rows.Forward(ids)
	got := blocks.Forward(ids)
	b := v41EngramPoolBackend(true, false)
	s := v41DenseTestSession(t, device, b)
	stop := enableV41BF16Witness()
	v41HostEngramRowDecodes.Store(0)
	dev := s.Prefill(ids)
	decodes := v41HostEngramRowDecodes.Load()
	stop()
	if decodes != 0 || b.calls["row-decode"] == 0 {
		t.Fatalf("device session made %d host decodes and %d device decodes", decodes, b.calls["row-decode"])
	}
	for pos := range ids {
		for i := range want.Logits[pos] {
			if math.Float32bits(got.Logits[pos][i]) != math.Float32bits(want.Logits[pos][i]) {
				t.Fatalf("position %d logit %d: q2_k blocks %08x f32 rows %08x", pos, i, math.Float32bits(got.Logits[pos][i]), math.Float32bits(want.Logits[pos][i]))
			}
		}
	}
	assertV41RowsClose(t, "Q2_K device decode", dev, want.Logits[len(ids)-1], 1e-6)
}

// v41EngramRoleFixture is the device-only-admitted Q2_K/Q3_K routed-expert role
// fixture (compressed KV and the indexer) with layer 0 declared an Engram layer
// whose table is served in the given dialect: packed 264-byte rows, or a
// verified Q2_K table the stage caches as its blocks.
func v41EngramRoleFixture(t *testing.T, encoding compute.V41EngramRowEncoding) *Model {
	t.Helper()
	m := v41IncrementalExpertFixture(t, true, false)
	c := &m.Cfg
	H := c.HiddenSize
	d := c.DeepSeekV41
	d.EngramLayerIDs, d.EngramNumEmbeddings = []int{0}, []int{408}
	d.EngramMaxNgramSize, d.EngramNHeads, d.EngramHeadDim = 4, 8, H
	const cols = 24
	man, raw := synthBuildRaw([]synthTensor{
		{layerName(0, "engram_kv.weight"), []int{cols * H, 5 * H}},
		{layerName(0, "engram_q_norm.weight"), []int{4 * H}},
		{layerName(0, "engram_k_norm.weight"), []int{4 * H}},
	}, func(name string, next func() float32) float32 { return synthMatmulFill(name, next) })
	for k, v := range man {
		v.Offset += len(m.raw)
		m.manifest[k] = v
	}
	m.raw = append(m.raw, raw...)
	layout := V41EngramLayout{
		TokenMap: make([]uint32, c.VocabSize), CompressedVocab: uint32(c.VocabSize),
		Rows: []uint32{408}, Multipliers: [][]uint64{{17, 13, 19, 11}}, Primes: [][]uint32{make([]uint32, cols)},
		MaxNgramSize: 4, HeadsPerNgram: 8,
	}
	for i := range layout.TokenMap {
		layout.TokenMap[i] = uint32(i)
	}
	for i := range layout.Primes[0] {
		layout.Primes[0][i] = 17
	}
	var src V41EngramRowSource = &v41EngramMemorySource{packed: v41EngramTestPackedRows(408), rows: 408}
	if encoding == compute.V41EngramRowQ2K {
		src, _ = v41EngramQ2KFixtureSource(t, 408, 13668)
	}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, 64*1024); err != nil {
		t.Fatal(err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("Engram role fixture admission: %v", err)
	}
	return m
}

// On the role fixture the device decode session's logits equal the host
// decode session's bit for bit in both dialects, with no host row decode.
// fak-test:runtime medium est=10s lane=default
func TestV41EngramRoleFixtureDecodeParity(t *testing.T) {
	for _, encoding := range []compute.V41EngramRowEncoding{compute.V41EngramRowPacked, compute.V41EngramRowQ2K} {
		device, hostArm := v41EngramRoleFixture(t, encoding), v41EngramRoleFixture(t, encoding)
		want := map[compute.V41EngramRowEncoding]int{compute.V41EngramRowPacked: V41EngramPackedRowBytes, compute.V41EngramRowQ2K: v41EngramQ2KEncodedRowBytes}[encoding]
		if got := device.v41EngramStageFor().rowBytes; got != want {
			t.Fatalf("encoding %d stage caches %d-byte rows, want %d", encoding, got, want)
		}
		b, hb := v41EngramPoolBackend(true, true), v41EngramPoolBackend(true, true)
		hb.rowDecode = false
		ds, hs := v41DenseTestSession(t, device, b), v41DenseTestSession(t, hostArm, hb)
		stop := enableV41BF16Witness()
		v41HostEngramRowDecodes.Store(0)
		got := [][]float32{ds.Prefill([]int{1, 2, 3}), ds.Step(4), ds.Step(5)}
		decodes := v41HostEngramRowDecodes.Load()
		stop()
		ref := [][]float32{hs.Prefill([]int{1, 2, 3}), hs.Step(4), hs.Step(5)}
		if decodes != 0 || b.calls["row-decode"] == 0 || hb.calls["row-decode"] != 0 {
			t.Fatalf("encoding %d: host decodes=%d device decodes=%d control decodes=%d", encoding, decodes, b.calls["row-decode"], hb.calls["row-decode"])
		}
		for i := range ref {
			for j := range ref[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(ref[i][j]) {
					t.Fatalf("encoding %d step %d logit %d: device decode %08x host decode %08x", encoding, i, j, math.Float32bits(got[i][j]), math.Float32bits(ref[i][j]))
				}
			}
		}
	}
}

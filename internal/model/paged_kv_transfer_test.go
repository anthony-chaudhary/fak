package model

import (
	"math"
	"net"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func copyPagedSpan(t *testing.T, src *PagedKV, from, n int) *PagedKV {
	t.Helper()
	pool := NewPagedKVPoolWithRaw(src.poolConfig(), src.pool.BlockTokens())
	dst := pool.NewSequence()
	for pos := from; pos < from+n; pos++ {
		k := make([][]float32, src.pool.nLayers)
		kraw := make([][]float32, src.pool.nLayers)
		v := make([][]float32, src.pool.nLayers)
		for l := 0; l < src.pool.nLayers; l++ {
			k[l] = append([]float32(nil), src.GatherK(l)[pos*src.pool.stride:(pos+1)*src.pool.stride]...)
			kraw[l] = append([]float32(nil), src.GatherKraw(l)[pos*src.pool.stride:(pos+1)*src.pool.stride]...)
			v[l] = append([]float32(nil), src.GatherV(l)[pos*src.pool.stride:(pos+1)*src.pool.stride]...)
		}
		dst.AppendRaw(k, kraw, v)
	}
	return dst
}

func (s *PagedKV) poolConfig() Config {
	return Config{NumLayers: s.pool.nLayers, NumKVHeads: 1, HeadDim: s.pool.stride}
}

func assertPagedBitsEqual(t *testing.T, label string, want, got *PagedKV) {
	t.Helper()
	if got.Len() != want.Len() || got.Blocks() != want.Blocks() {
		t.Fatalf("%s Len/Blocks = %d/%d, want %d/%d", label, got.Len(), got.Blocks(), want.Len(), want.Blocks())
	}
	for l := 0; l < want.pool.nLayers; l++ {
		assertFloat32BitsEqual(t, label+" K l"+itoa(l), want.GatherK(l), got.GatherK(l))
		assertFloat32BitsEqual(t, label+" Kraw l"+itoa(l), want.GatherKraw(l), got.GatherKraw(l))
		assertFloat32BitsEqual(t, label+" V l"+itoa(l), want.GatherV(l), got.GatherV(l))
	}
}

func TestPagedKVTransferRoundTripCarriesDescriptor(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	ref := m.NewSession()
	ref.Prefill([]int{3, 17, 5, 23, 41, 2, 19})

	srcPool := NewPagedKVPoolWithRaw(cfg, 4)
	src := snapshotCacheToPaged(srcPool, ref.Cache)
	const from, n = 1, 5
	want := copyPagedSpan(t, src, from, n)
	transfer := cachemeta.KVTransfer{
		Direction:        cachemeta.KVMigrate,
		SpanDigest:       "span-issue-29",
		Tokens:           n,
		ModelID:          "synthetic",
		TokenizerID:      "unit",
		PositionMode:     cachemeta.PositionRelocatable,
		FromTier:         cachemeta.TierHBM,
		ToTier:           cachemeta.TierRemote,
		Owner:            "prefill-worker",
		Lease:            "lease-29",
		SecuritySet:      true,
		Taint:            abi.TaintQuarantined,
		Scope:            abi.ScopeAgent,
		AdmissionVerdict: cachemeta.AdmissionQuarantine,
		AdmittedBy:       "admission-gate",
		Outcome:          cachemeta.KVTransferOK,
	}

	frame, err := MarshalPagedKVTransfer(src, transfer, from, n)
	if err != nil {
		t.Fatalf("MarshalPagedKVTransfer: %v", err)
	}
	got, err := UnmarshalPagedKVTransfer(NewPagedKVPoolWithRaw(cfg, 4), frame)
	if err != nil {
		t.Fatalf("UnmarshalPagedKVTransfer: %v", err)
	}
	assertPagedBitsEqual(t, "round-trip", want, got.KV)
	for i, pos := range got.Positions {
		if pos != from+i {
			t.Fatalf("position[%d]=%d, want %d", i, pos, from+i)
		}
	}
	if got.Transfer.Lease != "lease-29" || got.Transfer.Outcome != cachemeta.KVTransferOK {
		t.Fatalf("transfer descriptor did not round-trip: %+v", got.Transfer)
	}
	if got.Entry.Security.Taint != abi.TaintQuarantined ||
		got.Entry.Security.Scope != abi.ScopeAgent ||
		got.Entry.Security.AdmissionVerdict != cachemeta.AdmissionQuarantine ||
		got.Entry.Security.AdmittedBy != "admission-gate" {
		t.Fatalf("trust descriptor not reconstructable: %+v", got.Entry.Security)
	}
	if got.Entry.Derivation.SerializerID != PagedKVTransferSerializerID {
		t.Fatalf("serializer id = %q, want %q", got.Entry.Derivation.SerializerID, PagedKVTransferSerializerID)
	}
	if got.Verdict().Kind != cachemeta.LookupHit {
		t.Fatalf("ok transfer verdict = %s, want hit", got.Verdict().Kind)
	}

	// Kraw must survive the move: after receipt, a middle-span eviction should be
	// byte-identical to applying the same eviction before transfer.
	want.Evict(1, 2, cfg)
	got.KV.Evict(1, 2, cfg)
	assertPagedBitsEqual(t, "post-transfer evict", want, got.KV)
}

func TestTCPKVTransportMatchesLocal(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	ref := m.NewSession()
	ref.Prefill([]int{11, 7, 29, 31, 43, 47})

	srcPool := NewPagedKVPoolWithRaw(cfg, 3)
	src := snapshotCacheToPaged(srcPool, ref.Cache)
	transfer := cachemeta.KVTransfer{
		Direction:    cachemeta.KVMigrate,
		SpanDigest:   "span-tcp-kv",
		Tokens:       int64(src.Len()),
		ModelID:      "synthetic",
		PositionMode: cachemeta.PositionRelocatable,
		FromTier:     cachemeta.TierHBM,
		ToTier:       cachemeta.TierRemote,
		Owner:        "prefill-worker",
		Lease:        "tcp-loopback",
		Outcome:      cachemeta.KVTransferOK,
	}
	local, err := (LocalKVTransport{Pool: NewPagedKVPoolWithRaw(cfg, 3)}).Send(src, transfer, 0, src.Len())
	if err != nil {
		t.Fatalf("LocalKVTransport: %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		peer, aerr := ln.Accept()
		if aerr != nil {
			return
		}
		defer peer.Close()
		_ = EchoKVTransferFrames(peer)
	}()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	tcp, err := NewTCPKVTransport(conn, NewPagedKVPoolWithRaw(cfg, 3)).Send(src, transfer, 0, src.Len())
	if err != nil {
		t.Fatalf("TCPKVTransport: %v", err)
	}
	conn.Close()
	wg.Wait()

	assertPagedBitsEqual(t, "tcp-vs-local", local.KV, tcp.KV)
	if tcp.Transfer.BytesMoved <= 0 {
		t.Fatalf("TCP transfer did not report moved bytes: %+v", tcp.Transfer)
	}
	if tcp.Entry.Residency.Lease != "tcp-loopback" || tcp.Entry.Residency.Tier != cachemeta.TierRemote {
		t.Fatalf("residency descriptor missing after TCP transfer: %+v", tcp.Entry.Residency)
	}
}

func TestPagedKVTransferRequiresRawPagedKV(t *testing.T) {
	cfg := pagedEvictCfg()
	pool := NewPagedKVPool(cfg, 2)
	seq := pool.NewSequence()
	k := make([][]float32, cfg.NumLayers)
	v := make([][]float32, cfg.NumLayers)
	for l := range k {
		k[l] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
		v[l] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
		for i := range k[l] {
			k[l][i] = float32(l*10 + i)
			v[l][i] = float32(math.Copysign(float64(i+1), -1))
		}
	}
	seq.Append(k, v)
	if _, err := MarshalPagedKVTransfer(seq, cachemeta.KVTransfer{Direction: cachemeta.KVMigrate}, 0, seq.Len()); err == nil {
		t.Fatal("2-plane paged KV transfer should fail closed because Kraw is absent")
	}
}

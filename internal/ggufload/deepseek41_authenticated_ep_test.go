package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// The packed authenticated fixture uses one Q2_K/Q3_K block per hidden row.
const ds41EPHiddenWidth = 256

// The fixture interleaves two selected experts per rank. Return a fresh list so
// neither model construction nor expected-call derivation can mutate the source.
func ds41EPSelectedExperts() []int { return []int{258, 130, 2, 257, 129, 1} }

func ds41EPOwnedProjectionCounts(band ExpertShard, rows int) map[string]int {
	expected := make(map[string]int)
	for _, expert := range ds41EPSelectedExperts() {
		if expert < band.Lo || expert >= band.Hi {
			continue
		}
		for _, projection := range []string{"gate_proj", "up_proj", "down_proj"} {
			expected[batchedExpertCanonicalName("deepseek41", 0, expert, projection)] += rows
		}
	}
	return expected
}

type ds41EPSource struct {
	io.ReaderAt
	io.Closer
	closed atomic.Int32
}

func (r *ds41EPSource) Close() error {
	r.closed.Add(1)
	return r.Closer.Close()
}

// The recorder executes only CPU reference operations. Advertising DeviceMemory
// exercises the production owner-device admission and dispatch seams; it never
// establishes physical device execution, residency, DMA or performance.
type ds41EPRecorder struct {
	compute.Backend
	allowed   map[[32]byte]string
	names     map[compute.Buffer]string
	live      map[compute.Buffer]bool
	frees     map[compute.Buffer]int
	order     []string
	routed    int
	sumAcc    map[compute.Buffer]int
	sumRows   map[compute.Buffer]bool
	sums      [][][32]byte
	finalRows [][32]byte
}

func newDS41EPRecorder(m *model.Model) *ds41EPRecorder {
	b := &ds41EPRecorder{Backend: compute.Default(), allowed: map[[32]byte]string{},
		names: map[compute.Buffer]string{}, live: map[compute.Buffer]bool{}, frees: map[compute.Buffer]int{},
		sumAcc: map[compute.Buffer]int{}, sumRows: map[compute.Buffer]bool{}}
	for expert := range m.Cfg.NumExperts {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := batchedExpertCanonicalName("deepseek41", 0, expert, proj)
			if raw, ok := m.KQuantRaw(name); ok {
				digest := sha256.Sum256(raw)
				if _, duplicate := b.allowed[digest]; duplicate {
					panic("fixture has ambiguous expert projection bytes")
				}
				b.allowed[digest] = name
			}
		}
	}
	return b
}

func (b *ds41EPRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}

func (b *ds41EPRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || dt == compute.Q8_0 || dt == compute.Q2_K || dt == compute.Q3_K
}

func (b *ds41EPRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	name := ""
	if dt == compute.Q2_K || dt == compute.Q3_K {
		host, ok := x.Buf().(compute.HostBuffer)
		if !ok || x.Dtype != dt || !slices.Equal(x.Shape, []int{ds41EPHiddenWidth, ds41EPHiddenWidth}) {
			panic("expert upload lost packed dtype or geometry")
		}
		raw := make([]byte, len(host.I8()))
		for i, value := range host.I8() {
			raw[i] = byte(value)
		}
		name = b.allowed[sha256.Sum256(raw)]
		if name == "" {
			panic("nonowned or altered expert upload")
		}
		if (dt == compute.Q3_K) != strings.HasSuffix(name, ".w2.weight") {
			panic("owned expert projection has the wrong packed dtype")
		}
	}
	out := b.Backend.Upload(x, dt)
	b.live[out.Buf()] = true
	if name != "" {
		b.names[out.Buf()] = name
	}
	return out
}

func (b *ds41EPRecorder) Free(x compute.Tensor) {
	if _, uploaded := b.live[x.Buf()]; uploaded {
		b.live[x.Buf()] = false
		b.frees[x.Buf()]++
	}
	b.Backend.Free(x)
}

// Observe the production upload's exact site identity. Shared/dense additions
// cannot enter this trace just because they happen to have the same shape.
func (b *ds41EPRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	out := b.Upload(x, dt)
	switch site {
	case "V4.1 shared-add accumulator":
		// Observe the exact routed F32 row before adding the shared expert.
		// Every rank consumes this site, including followers with no sum.
		if class != compute.MemoryActivation || dt != compute.F32 || !slices.Equal(out.Shape, []int{ds41EPHiddenWidth}) {
			panic("final routed row lost its F32 geometry or class")
		}
		values := b.Backend.Read(out)
		if len(values) != ds41EPHiddenWidth {
			panic("final routed row changed width")
		}
		raw := make([]byte, 4*len(values))
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				panic("final routed row is nonfinite")
			}
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(value))
		}
		b.finalRows = append(b.finalRows, sha256.Sum256(raw))
	case "V4.1 routed-sum accumulator", "V4.1 routed-sum expert output":
		if class != compute.MemoryActivation || dt != compute.F32 || !slices.Equal(out.Shape, []int{ds41EPHiddenWidth}) {
			panic("routed-sum upload lost site geometry or class")
		}
		if site == "V4.1 routed-sum accumulator" {
			for _, value := range b.Backend.Read(out) {
				if math.Float32bits(value) != 0 {
					panic("routed sum did not start from positive zero")
				}
			}
			b.sumAcc[out.Buf()] = len(b.sums)
			b.sums = append(b.sums, nil)
		} else {
			b.sumRows[out.Buf()] = true
		}
	}
	return out
}

func (b *ds41EPRecorder) AddInPlace(dst, src compute.Tensor) {
	round, routed := b.sumAcc[dst.Buf()]
	if routed {
		if !b.sumRows[src.Buf()] || !b.live[dst.Buf()] || !b.live[src.Buf()] {
			panic("routed sum did not consume its live tagged expert row")
		}
		values := b.Backend.Read(src)
		if len(values) != ds41EPHiddenWidth {
			panic("routed sum contribution changed width")
		}
		raw := make([]byte, 4*len(values))
		for i, value := range values {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(value))
		}
		b.sums[round] = append(b.sums[round], sha256.Sum256(raw))
	} else if b.sumRows[src.Buf()] {
		panic("routed expert output escaped its tagged accumulator")
	}
	b.Backend.AddInPlace(dst, src)
}

func (b *ds41EPRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	if w.Dtype == compute.Q2_K || w.Dtype == compute.Q3_K {
		name := b.names[w.Buf()]
		if name == "" || !b.live[w.Buf()] {
			panic("expert MatMul has no live owner upload")
		}
		b.order = append(b.order, name)
	}
	return b.Backend.MatMul(w, x)
}

func (*ds41EPRecorder) V41RoutedSwiGLUAvailable() bool { return true }

func (b *ds41EPRecorder) V41RoutedSwiGLU(gate, up compute.Tensor, limit, weight float32) (compute.Tensor, error) {
	b.routed++
	round := func(x float32) float32 {
		bits := math.Float32bits(x)
		return math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
	}
	g, u := b.Backend.Read(gate), b.Backend.Read(up)
	out := make([]float32, len(g))
	for i := range out {
		a, c := round(g[i]), round(u[i])
		if limit > 0 {
			a, c = min(a, limit), max(-limit, min(c, limit))
		}
		out[i] = round((a / (1 + float32(math.Exp(float64(-a))))) * c * weight)
	}
	return b.Upload(compute.NewF32(b.Backend, []int{len(out)}, out), compute.F32), nil
}

func (b *ds41EPRecorder) retired() error {
	for buffer, live := range b.live {
		if live || b.frees[buffer] != 1 {
			return fmt.Errorf("uploaded buffer live=%t frees=%d", live, b.frees[buffer])
		}
	}
	return nil
}

// This is an actual GGUF/authentication/owner-range/native-Session/TCP software
// composition. Authentication scans whole files; ONLY subsequent materialization
// is owner-local. It does not authorize the production V4.1 EP serve path.
// fak-test:runtime medium est=5s lane=default
func TestDeepSeek41AuthenticatedExpertParallelSession(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	shards, digest := ds41AuthenticatedEPShards(t)
	if bad, err := OpenWeightsAuthenticated(t.Context(), shards, fmt.Sprintf("%064x", 0)); err == nil {
		_ = bad.Close()
		t.Fatal("untrusted manifest entered the native path")
	}
	const ranks = 3
	weights := make([]*WeightSource, ranks+1)
	sources := make([][]*ds41EPSource, ranks+1)
	for i := range weights {
		ws, err := OpenWeightsAuthenticated(t.Context(), shards, digest)
		if err != nil {
			t.Fatal(err)
		}
		weights[i] = ws
		t.Cleanup(func() { _ = ws.Close() })
		retained := map[io.ReaderAt]*ds41EPSource{}
		for j, closer := range ws.closers {
			reader, ok := closer.(io.ReaderAt)
			if !ok {
				t.Fatal("portable authenticated source did not retain its file")
			}
			tracked := &ds41EPSource{ReaderAt: reader, Closer: closer}
			retained[reader], ws.closers[j] = tracked, tracked
			sources[i] = append(sources[i], tracked)
		}
		for j := range ws.readerFor {
			tracked := retained[ws.readerFor[j]]
			if tracked == nil {
				t.Fatal("tensor escaped its authenticated reader")
			}
			ws.readerFor[j] = tracked
		}
	}
	if runtime.GOOS != "windows" {
		// All consumers opened A. Replacing the paths with B cannot change
		// their later payload reads; the original open files stay immutable.
		for _, shard := range shards {
			if err := os.Rename(shard.Path, shard.Path+".retained"); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(shard.Path, []byte("replacement must not be reopened"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if bad, err := OpenWeightsAuthenticated(t.Context(), shards, digest); err == nil {
			_ = bad.Close()
			t.Fatal("replacement shard paths retained the old authenticated identity")
		}
	}
	models := make([]*model.Model, ranks+1)
	bands := make([]ExpertShard, ranks)
	for i, ws := range weights {
		var opts []Q4KLoadOption
		var reads []*ownerRangeReader
		if i < ranks {
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatal(err)
			}
			bands[i], err = ExpertShardForRank(cfg.NumExperts, ranks, i)
			if err != nil {
				t.Fatal(err)
			}
			opts = append(opts, WithExpertShard(bands[i].Lo, bands[i].Hi))
			for j, info := range ws.File.Tensors {
				if _, _, ok := glmMoeDsaBatchedExpert(info.Name); ok {
					r := &ownerRangeReader{ReaderAt: ws.readerFor[j], bands: []ownerRangeReadBand{ownerRangeBand(info, bands[i])}}
					ws.readerFor[j] = r
					reads = append(reads, r)
				}
			}
		}
		m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, opts...)
		if err != nil {
			t.Fatalf("owner %d load: %v", i, err)
		}
		models[i] = m
		m.SetWeightCloser(ws)
		t.Cleanup(func() { _ = m.CloseWeights() })
		if m.Cfg.HeadDim != 512 || m.Cfg.HiddenSize != 256 || m.Cfg.NumLayers != 1 {
			t.Fatal("fixture failed to derive its full-path compact geometry from GGUF")
		}
		if i < ranks {
			if len(reads) != 3 {
				t.Fatal("fixture did not split all three expert projections")
			}
			var selectedBytes int64
			for _, r := range reads {
				want := r.bands[0].hi - r.bands[0].lo
				if r.rejected != 0 || r.reads != bands[i].Hi-bands[i].Lo || r.readBytes != want {
					t.Fatalf("rank %d selected payload reads=%d bytes=%d rejected=%d", i, r.reads, r.readBytes, r.rejected)
				}
				selectedBytes += want
			}
			if got := m.ResidentReport(); got.KQuantBytes != selectedBytes || got.KQuantTensors != 3*(bands[i].Hi-bands[i].Lo) {
				t.Fatalf("rank %d retained a nonowned payload: %+v", i, got)
			}
		}
	}
	oracle := models[ranks]
	for rank, band := range bands {
		got, want := models[rank].ResidentReport(), oracle.ResidentReport()
		if got.Q8Tensors != want.Q8Tensors || got.Q8Bytes != want.Q8Bytes || got.F32Tensors != want.F32Tensors || got.F32Bytes != want.F32Bytes || got.Q4KTensors != want.Q4KTensors || got.Q4KBytes != want.Q4KBytes {
			t.Fatalf("rank %d changed replicated stores or expanded nonowned experts", rank)
		}
		for expert := range oracle.Cfg.NumExperts {
			for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
				name := batchedExpertCanonicalName("deepseek41", 0, expert, proj)
				got, present := models[rank].KQuantRaw(name)
				want, _ := oracle.KQuantRaw(name)
				owned := expert >= band.Lo && expert < band.Hi
				if present != owned || (owned && (!bytes.Equal(got, want) || len(got) != cap(got))) {
					t.Fatalf("rank %d projection %s violated exact ownership", rank, name)
				}
			}
		}
	}
	checkSources := func(rank int, want int32) error {
		for _, r := range sources[rank] {
			if got := r.closed.Load(); got != want {
				return fmt.Errorf("rank %d authenticated source closes=%d want=%d", rank, got, want)
			}
		}
		return nil
	}
	refBackend := newDS41EPRecorder(oracle)
	reference, err := oracle.NewBackendSessionChecked(refBackend)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(reference.Close)
	tokens := []int{1, 2, 3}
	want := [][]float32{reference.Prefill(tokens[:2]), reference.Step(tokens[2])}
	reference.Close()
	if err := oracle.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if err := refBackend.retired(); err != nil {
		t.Fatal(err)
	}
	if err := checkSources(ranks, 1); err != nil {
		t.Fatal(err)
	}
	if len(refBackend.order) != 3*model.V41RouterTopK*3 || refBackend.routed != 3*model.V41RouterTopK {
		t.Fatal("full reference did not execute the expected mixed expert graph")
	}
	if len(refBackend.sums) != 3 || len(refBackend.finalRows) != 3 {
		t.Fatal("reference did not record one routed sum and final F32 row per position")
	}
	for _, round := range refBackend.sums {
		unique := map[[32]byte]bool{}
		for _, contribution := range round {
			unique[contribution] = true
		}
		if len(round) != model.V41RouterTopK || len(unique) != len(round) {
			t.Fatal("reference contributions cannot distinguish missing, duplicate or reordered routed rows")
		}
	}
	// Synthetic source agreement is a protocol binding, not build provenance.
	source := fmt.Sprintf("%x", sha256.Sum256([]byte("authenticated EP software fixture source")))
	errs := ds41EPGroup(t, ranks, func(g *model.DistComm) error {
		rank := g.Rank()
		m := models[rank]
		collective, err := model.NewV41ExpertParallelCollective(g, model.V41ExpertParallelBinding{ArtifactSHA256: digest, SourceSHA256: source})
		if err != nil {
			return err
		}
		m.SetExpertParallelRanks(g.Size())
		m.SetExpertParallelRank(rank)
		m.SetExpertParallelCollective(collective)
		backend := newDS41EPRecorder(m)
		var session *model.Session
		defer func() {
			if session != nil {
				session.Close()
			}
			_ = m.CloseWeights()
		}()
		newSession := func() *model.Session {
			var err error
			session, err = m.NewBackendSessionChecked(backend)
			if err != nil {
				panic(err)
			}
			// Model closure must defer retained-reader retirement until its
			// live native session finishes, including subsequent Step calls.
			var active *model.WeightSessionsActiveError
			if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 1 {
				panic(fmt.Errorf("owner close did not retain exactly one native session: %v", err))
			}
			if err := checkSources(rank, 0); err != nil {
				panic(err)
			}
			return session
		}
		if rank == 0 {
			coord, err := model.NewEPDecodeCoordinator(g)
			if err != nil {
				return err
			}
			m.SetEPDecodeCoordinator(coord)
			s := newSession()
			got := [][]float32{s.Prefill(tokens[:2]), s.Step(tokens[2])}
			for step := range got {
				if len(got[step]) != len(want[step]) || len(got[step]) == 0 {
					return fmt.Errorf("operation %d logit shape changed", step)
				}
				for i, value := range got[step] {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value) != math.Float32bits(want[step][i]) {
						return fmt.Errorf("operation %d logit %d differs: %g vs %g", step, i, value, want[step][i])
					}
				}
			}
			s.Close()
			if err := coord.Shutdown(); err != nil {
				return err
			}
		} else if err := model.RunEPFollower(g, newSession); err != nil {
			return err
		}
		// The ordinary multi-token prefill is expert-major while EP is
		// token-major. Their projection schedules may differ; exact owned
		// identities/counts and the actual arithmetic order must not.
		expected, actual := map[string]int{}, map[string]int{}
		for _, name := range refBackend.order {
			for _, own := range backend.allowed {
				if own == name {
					expected[name]++
					break
				}
			}
		}
		for _, name := range backend.order {
			actual[name]++
		}
		declared := ds41EPOwnedProjectionCounts(bands[rank], len(tokens))
		wantCalls := 0
		for _, count := range declared {
			wantCalls += count
		}
		if !maps.Equal(expected, declared) || len(backend.order) != wantCalls || !maps.Equal(actual, expected) || backend.routed != 6 {
			return fmt.Errorf("rank %d owned projection identities/counts differ: actual=%v expected=%v routed=%d", rank, actual, expected, backend.routed)
		}
		for name, count := range expected {
			if count != 3 {
				return fmt.Errorf("rank %d reference projection %s executions=%d want3", rank, name, count)
			}
		}
		if rank == 0 {
			if len(backend.sums) != len(refBackend.sums) {
				return fmt.Errorf("root routed sum count=%d want%d", len(backend.sums), len(refBackend.sums))
			}
			for i, round := range backend.sums {
				if !slices.Equal(round, refBackend.sums[i]) {
					return fmt.Errorf("root position %d routed contribution bytes/order differ", i)
				}
			}
		} else if len(backend.sums) != 0 {
			// Keep shutdown joined even in the baseline witness; returning
			// here would cancel peers still receiving the shutdown frame.
			t.Errorf("follower rank %d executed %d routed sums", rank, len(backend.sums))
		}
		if !slices.Equal(backend.finalRows, refBackend.finalRows) {
			return fmt.Errorf("rank %d final routed F32 row bytes differ", rank)
		}
		if err := backend.retired(); err != nil {
			return err
		}
		return checkSources(rank, 1)
	})
	for rank, err := range errs {
		if err != nil {
			t.Fatalf("authenticated native rank %d: %v", rank, err)
		}
	}
}

// Join every worker even on refusal. Closing socket handles, rather than groups
// from another goroutine, avoids racing DistComm's mutable connection tables.
func ds41EPGroup(t *testing.T, ranks int, run func(*model.DistComm) error) []error {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var sockets []net.Conn
	closed := false
	remember := func(conn net.Conn) {
		mu.Lock()
		defer mu.Unlock()
		if closed {
			_ = conn.Close()
			return
		}
		sockets = append(sockets, conn)
	}
	closeSockets := func() {
		_ = listener.Close()
		mu.Lock()
		defer mu.Unlock()
		closed = true
		for _, conn := range sockets {
			_ = conn.Close()
		}
	}
	defer closeSockets()
	errs := make([]error, ranks)
	var workers sync.WaitGroup
	for rank := range ranks {
		workers.Add(1)
		go func(rank int) {
			defer workers.Done()
			defer func() {
				if cause := recover(); cause != nil {
					errs[rank] = fmt.Errorf("native rank panic: %v", cause)
				}
				if errs[rank] != nil {
					closeSockets()
				}
			}()
			var group *model.DistComm
			if rank == 0 {
				group, errs[rank] = model.Coordinate(ds41EPListener{listener, remember}, ranks)
			} else {
				var conn net.Conn
				conn, errs[rank] = net.DialTimeout("tcp", listener.Addr().String(), time.Second)
				if errs[rank] != nil {
					return
				}
				remember(conn)
				group, errs[rank] = model.Join(conn, rank, ranks)
			}
			if errs[rank] != nil {
				return
			}
			defer group.Close()
			errs[rank] = run(group)
		}(rank)
	}
	done := make(chan struct{})
	go func() { workers.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		closeSockets()
		// Never unwind into model/source cleanup while workers can still
		// use them. The enclosing test and process deadlines bound a stuck
		// join; only those process-level guards may terminate live workers.
		<-done
		t.Fatal("authenticated EP exceeded its bounded software deadline")
	}
	return errs
}

type ds41EPListener struct {
	net.Listener
	remember func(net.Conn)
}

func (l ds41EPListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err == nil {
		l.remember(conn)
	}
	return conn, err
}

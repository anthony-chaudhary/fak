package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"net"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func v41EPFixtureBinding(m *Model) V41ExpertParallelBinding {
	artifact := sha256.Sum256(m.raw)
	source := sha256.Sum256([]byte("synthetic V4.1 TCP regression fixture, not production provenance"))
	return V41ExpertParallelBinding{ArtifactSHA256: hex.EncodeToString(artifact[:]), SourceSHA256: hex.EncodeToString(source[:])}
}

func v41EPFixtureShard(full *Model, band TPShard) *Model {
	m := &Model{Cfg: full.Cfg, manifest: map[string]tensorMeta{}}
	names := make([]string, 0, len(full.manifest))
	for name := range full.manifest {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if i := strings.Index(name, ".ffn.experts."); i >= 0 {
			id := strings.Split(name[i+len(".ffn.experts."):], ".")[0]
			expert, _ := strconv.Atoi(id)
			if expert < band.Lo || expert >= band.Hi {
				continue
			}
		}
		meta := full.manifest[name]
		bytes := full.raw[meta.Offset : meta.Offset+meta.Nbytes]
		meta.Offset = len(m.raw)
		m.raw = append(m.raw, bytes...)
		m.manifest[name] = meta
	}
	return m
}

// A real loopback group with joined teardown, including the timeout failure arm.
// Unlike a Fatal-only watchdog, no socket or follower goroutine survives a test.
func v41EPTestGroup(t *testing.T, fn func(*DistComm) error) []error {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	var mu sync.Mutex
	var sockets []net.Conn
	remember := func(conn net.Conn) { mu.Lock(); sockets = append(sockets, conn); mu.Unlock() }
	groups := make([]*DistComm, 3)
	formed := make(chan error, 1)
	go func() { var err error; groups[0], err = Coordinate(ln, 3); formed <- err }()
	for rank := 1; rank < 3; rank++ {
		conn, err := net.DialTimeout("tcp", ln.Addr().String(), time.Second)
		if err != nil {
			t.Fatal(err)
		}
		remember(conn)
		groups[rank], err = Join(conn, rank, 3)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := <-formed; err != nil {
		t.Fatal(err)
	}
	for _, conn := range groups[0].workers {
		if conn != nil {
			remember(conn)
		}
	}
	closeSockets := func() {
		mu.Lock()
		defer mu.Unlock()
		for _, conn := range sockets {
			_ = conn.Close()
		}
	}
	defer closeSockets()
	var wg sync.WaitGroup
	errs := make([]error, 3)
	for rank, g := range groups {
		wg.Add(1)
		go func(rank int, g *DistComm) {
			defer wg.Done()
			defer g.Close()
			err := panicAsError(func() { errs[rank] = fn(g) })
			if err != nil {
				errs[rank] = err
			}
		}(rank, g)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		closeSockets()
		// Never unwind into model/session cleanup while a worker still uses
		// it. Test/process deadlines bound this join after socket cancellation.
		<-done
		t.Fatal("EP operation exceeded bounded regression deadline; joined workers after socket cancellation")
	}
	return errs
}

func v41EPConfigure(m *Model, g *DistComm, binding V41ExpertParallelBinding) error {
	coll, err := NewV41ExpertParallelCollective(g, binding)
	if err != nil {
		return err
	}
	m.SetExpertParallelRanks(g.Size())
	m.SetExpertParallelRank(g.Rank())
	m.SetExpertParallelCollective(coll)
	return nil
}

func v41EPCheckBits(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("logit widths differ: %d vs %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return fmt.Errorf("logit %d differs: %g vs %g", i, got[i], want[i])
		}
	}
	return nil
}

// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelSessionTCPExact(t *testing.T) {
	for _, routed := range [][]int{{1, 2, 3, 129, 257, 259}, {257, 258, 259, 260, 261, 262}} {
		t.Run(fmt.Sprint(routed), func(t *testing.T) {
			full := v41FullStepModel(t)
			bias := make([]float32, full.Cfg.NumExperts)
			for i, expert := range routed {
				bias[expert] = 100 + float32(i)
			}
			v41WriteTensorF32(t, full, layerName(0, "ffn.gate.e_score_correction_bias"), bias)
			binding := v41EPFixtureBinding(full)
			owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
			models := make([]*Model, 3)
			for rank, band := range owners.Shards {
				models[rank] = v41EPFixtureShard(full, band)
			}
			reference := full.NewSession()
			want := [][]float32{reference.Prefill([]int{1, 2, 3}), reference.Step(4), reference.Prefill([]int{5, 6}), reference.Step(7)}
			wantState := []*v41ForwardSnapshot{captureV41ForwardSnapshot(reference.v41Forward)}
			reference.Close()
			reference = full.NewSession()
			reference.PrefillNoLogits([]int{2, 1})
			want = append(want, reference.Step(3))
			wantState = append(wantState, captureV41ForwardSnapshot(reference.v41Forward))
			reference.Close()
			counts := make([]map[int]int, 3)
			mirrors := make([][]*Session, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank, m := g.Rank(), models[g.Rank()]
				if err := v41EPConfigure(m, g, binding); err != nil {
					return err
				}
				counts[rank] = map[int]int{}
				newSession := func() *Session {
					s := m.NewSession()
					st := s.v41State()
					st.routedExpert = func(_ int, stem string, _ []float32, _ float32) ([]float32, v41RoutedExpertOutcome, error) {
						expert, err := strconv.Atoi(strings.TrimPrefix(stem, "ffn.experts."))
						band := owners.Shards[rank]
						if err != nil || expert < band.Lo || expert >= band.Hi {
							return nil, v41RoutedExpertError, fmt.Errorf("rank %d computed peer expert %s", rank, stem)
						}
						counts[rank][expert]++
						return nil, v41RoutedExpertDeclined, nil // real host triple still computes the pick
					}
					mirrors[rank] = append(mirrors[rank], s)
					return s
				}
				if rank != 0 {
					return RunEPFollower(g, newSession)
				}
				coord, err := NewEPDecodeCoordinator(g)
				if err != nil {
					return err
				}
				m.SetEPDecodeCoordinator(coord)
				s := newSession()
				got := [][]float32{s.Prefill([]int{1, 2, 3}), s.Step(4), s.Prefill([]int{5, 6}), s.Step(7)}
				if s.epSeqLen() != 7 {
					return fmt.Errorf("V4.1 position=%d want 7", s.epSeqLen())
				}
				s.Close()
				s = newSession()
				s.PrefillNoLogits([]int{2, 1})
				got = append(got, s.Step(3))
				s.Close()
				for i := range got {
					if err := v41EPCheckBits(got[i], want[i]); err != nil {
						return fmt.Errorf("operation %d: %w", i, err)
					}
				}
				return coord.Shutdown()
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			for rank, band := range owners.Shards {
				stats := models[rank].V41ExpertParallelTransportStats()
				if stats.CompletedGathers != 10 || stats.SentFrameBytes == 0 || stats.ReceivedFrameBytes == 0 {
					t.Fatalf("rank %d transport stats=%+v", rank, stats)
				}
				for _, expert := range routed {
					wantCount := 0
					if expert >= band.Lo && expert < band.Hi {
						wantCount = 10
					}
					if got := counts[rank][expert]; got != wantCount {
						t.Fatalf("rank %d expert %d calls=%d want=%d", rank, expert, got, wantCount)
					}
				}
				if len(mirrors[rank]) != 2 {
					t.Fatalf("rank %d did not replace the mirror", rank)
				}
				for i, sess := range mirrors[rank] {
					if !reflect.DeepEqual(captureV41ForwardSnapshot(sess.v41Forward), wantState[i]) {
						t.Fatalf("rank %d mirror %d retained continuation differs", rank, i)
					}
					if sess.modelWeightsHeld {
						t.Fatalf("rank %d mirror %d retained model ownership after close", rank, i)
					}
					wantLen := 7
					if i == 1 {
						wantLen = 3
					}
					if sess.epSeqLen() != wantLen {
						t.Fatalf("rank %d mirror %d continuation=%d want=%d", rank, i, sess.epSeqLen(), wantLen)
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelCompactSlots(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	picks := []routePick{{expert: 8, weight: .5}, {expert: 0, weight: .25}, {expert: 0, weight: .75}}
	plan, err := v41EPGatherPlan(picks, owners, 1)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Dim != 3*v41EPHeaderWords+3 || plan.Shards[1].Width() != v41EPHeaderWords {
		t.Fatal("empty owner was padded with expert payload")
	}
	digest := sha256.Sum256([]byte("slots"))
	gathered := make([]float32, plan.Dim)
	for _, shard := range plan.Shards {
		v41EPHeader(gathered[shard.Lo:shard.Hi], digest, false)
	}
	gathered[plan.Shards[0].Lo+v41EPHeaderWords] = 1 << 24
	gathered[plan.Shards[0].Lo+v41EPHeaderWords+1] = -(1 << 24)
	gathered[plan.Shards[2].Lo+v41EPHeaderWords] = 1
	rows, err := v41EPUnpack(gathered, plan, owners, digest, picks, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(rows, [][]float32{{1}, {1 << 24}, {-(1 << 24)}}) {
		t.Fatalf("original slots changed: %v", rows)
	}
	sum, err := v41FullRoutedSumWith(0, picks, rows, 1, nil)
	if err != nil || sum[0] != 1 || !reflect.DeepEqual(v41RoutedSumOrder(picks), []int{1, 2, 0}) {
		t.Fatalf("stable exact sum=%v err=%v", sum, err)
	}
	for _, mutate := range []func([]float32){
		func(v []float32) { v[plan.Shards[1].Lo+1] = math.Float32frombits(1) },
		func(v []float32) {
			v[plan.Shards[2].Lo+2] = math.Float32frombits(math.Float32bits(v[plan.Shards[2].Lo+2]) ^ 1)
		},
		func(v []float32) { v[len(v)-1] = float32(math.NaN()) },
	} {
		bad := append([]float32(nil), gathered...)
		mutate(bad)
		if _, err := v41EPUnpack(bad, plan, owners, digest, picks, 1); !errors.Is(err, ErrV41ExpertParallel) {
			t.Fatalf("malformed row admitted: %v", err)
		}
	}
	if _, err := v41EPUnpack(gathered[:len(gathered)-1], plan, owners, digest, picks, 1); err == nil {
		t.Fatal("short payload admitted")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelSessionTCPFailures(t *testing.T) {
	for _, mode := range []string{"missing-owner", "binding-mismatch", "follower-group", "selected-refusal", "disconnect"} {
		t.Run(mode, func(t *testing.T) {
			full := v41FullStepModel(t)
			bias := make([]float32, full.Cfg.NumExperts)
			for _, expert := range []int{1, 2, 129, 130, 257, 258} {
				bias[expert] = 100
			}
			v41WriteTensorF32(t, full, layerName(0, "ffn.gate.e_score_correction_bias"), bias)
			owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
			models := make([]*Model, 3)
			for rank, band := range owners.Shards {
				models[rank] = v41EPFixtureShard(full, band)
			}
			if mode == "missing-owner" {
				delete(models[2].manifest, layerName(0, "ffn.experts.257.w2.weight"))
			}
			binding := v41EPFixtureBinding(full)
			sessions := make([]*Session, 3)
			selected := errors.New("selected owner device refusal")
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank, m := g.Rank(), models[g.Rank()]
				localBinding := binding
				if mode == "binding-mismatch" && rank == 2 {
					localBinding.SourceSHA256 = strings.Repeat("0", 64)
				}
				bindingGroup := g
				if mode == "follower-group" && rank == 2 {
					bindingGroup = &DistComm{rank: rank, size: g.Size()}
				}
				if err := v41EPConfigure(m, bindingGroup, localBinding); err != nil {
					return err
				}
				newSession := func() *Session {
					s := m.NewSession()
					sessions[rank] = s
					st := s.v41State()
					if mode == "selected-refusal" && rank == 2 {
						calls := 0
						st.routedExpert = func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
							calls++
							if calls > 6 {
								return nil, v41RoutedExpertError, selected
							}
							return nil, v41RoutedExpertDeclined, nil
						}
					}
					return s
				}
				if rank != 0 {
					if mode == "disconnect" && rank == 2 {
						payload, err := g.BroadcastFromRoot(nil)
						if err != nil {
							return err
						}
						_, _, ids, err := decodeEPDecodeFrame(payload)
						if err != nil {
							return err
						}
						s := newSession()
						defer s.Close()
						s.Prefill(ids)
						g.abortV41EP()
						return errors.New("injected peer disconnect")
					}
					return RunEPFollower(g, newSession)
				}
				coord, err := NewEPDecodeCoordinator(g)
				if err != nil {
					return err
				}
				m.SetEPDecodeCoordinator(coord)
				s := newSession()
				defer s.Close()
				return panicAsError(func() {
					s.Prefill([]int{1, 2, 3})
					if mode == "selected-refusal" || mode == "disconnect" {
						s.Step(4)
					}
				})
			})
			for rank, err := range errs {
				if err == nil {
					t.Fatalf("rank %d accepted %s", rank, mode)
				}
			}
			if mode == "selected-refusal" && !errors.Is(errs[2], selected) {
				t.Fatalf("selected owner cause lost: %v", errs[2])
			}
			for rank, s := range sessions {
				if s == nil || s.modelWeightsHeld {
					t.Fatalf("rank %d did not retire session", rank)
				}
				wantLen := 0
				if mode == "selected-refusal" || mode == "disconnect" {
					wantLen = 3
				}
				if got := s.epSeqLen(); got != wantLen {
					t.Fatalf("rank %d committed failed operation: history=%d want=%d", rank, got, wantLen)
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelAdmissionRefusesGenericTransport(t *testing.T) {
	m := v41FullStepModel(t)
	m.SetExpertParallelRanks(3)
	m.SetExpertParallelRank(0)
	m.SetExpertParallelCollective(LocalCollective{})
	if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("generic reduction admitted: %v", err)
	}
	if _, err := NewV41ExpertParallelCollective(nil, V41ExpertParallelBinding{}); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("nil group admitted: %v", err)
	}
	g := &DistComm{size: 3}
	if _, err := NewV41ExpertParallelCollective(g, V41ExpertParallelBinding{}); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("absent identity admitted: %v", err)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestV41ExpertParallelAnnouncementBoundsStalledPeer(t *testing.T) {
	full := v41FullStepModel(t)
	binding := v41EPFixtureBinding(full)
	owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
	models := make([]*Model, 3)
	for rank, band := range owners.Shards {
		models[rank] = v41EPFixtureShard(full, band)
	}
	errs := v41EPTestGroup(t, func(g *DistComm) error {
		m := models[g.Rank()]
		if err := v41EPConfigure(m, g, binding); err != nil {
			return err
		}
		if g.Rank() == 2 {
			// Keep TCP connected but do not enter the announcement protocol.
			var one [1]byte
			_, err := g.coord.Read(one[:])
			return err
		}
		if g.Rank() == 1 {
			return RunEPFollower(g, m.NewSession)
		}
		coord, err := NewEPDecodeCoordinator(g)
		if err != nil {
			return err
		}
		m.SetEPDecodeCoordinator(coord)
		coord.v41Timeout = 50 * time.Millisecond
		s := m.NewSession()
		defer s.Close()
		return panicAsError(func() { s.Prefill([]int{1, 2, 3}) })
	})
	for rank, err := range errs {
		if err == nil {
			t.Fatalf("rank %d accepted stalled announcement", rank)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelRejectsBatchAndPrefix(t *testing.T) {
	m := v41FullStepModel(t)
	s := m.NewSession()
	defer s.Close()
	s.Prefill([]int{1, 2})
	prefix, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer prefix.Close()
	other := m.NewSession()
	defer other.Close()
	m.SetExpertParallelRanks(3)
	if _, err := s.PrefixSnapshot(); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("snapshot admitted: %v", err)
	}
	if err := prefix.Restore(other); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("restore admitted: %v", err)
	}
	if other.epSeqLen() != 0 || s.epSeqLen() != 2 {
		t.Fatal("refused prefix operation mutated a continuation")
	}
	if err := m.v41StepCohortAdmit([]v41StepCohortLane{{session: s, id: 3}, {session: other, id: 3}}); !errors.Is(err, errV41StepCohortDeclined) {
		t.Fatalf("cohort admitted: %v", err)
	}
	m.SetExpertParallelRanks(1)
	if err := prefix.Restore(other); err != nil {
		t.Fatalf("ordinary single-rank restore changed: %v", err)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41ExpertParallelRejectsInterleavedSessions(t *testing.T) {
	full := v41FullStepModel(t)
	binding := v41EPFixtureBinding(full)
	owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
	models := make([]*Model, 3)
	for rank, band := range owners.Shards {
		models[rank] = v41EPFixtureShard(full, band)
	}
	created := make([]int, 3)
	errs := v41EPTestGroup(t, func(g *DistComm) error {
		m := models[g.Rank()]
		if err := v41EPConfigure(m, g, binding); err != nil {
			return err
		}
		newSession := func() *Session { created[g.Rank()]++; return m.NewSession() }
		if g.Rank() != 0 {
			return RunEPFollower(g, newSession)
		}
		coord, err := NewEPDecodeCoordinator(g)
		if err != nil {
			return err
		}
		m.SetEPDecodeCoordinator(coord)
		first := newSession()
		defer first.Close()
		first.Prefill([]int{1, 2, 3})
		second := newSession()
		defer second.Close()
		return panicAsError(func() { second.Prefill([]int{4, 5, 6}) })
	})
	if !errors.Is(errs[0], ErrV41ExpertParallel) {
		t.Fatalf("root did not refuse session replacement: %v", errs[0])
	}
	for rank := 1; rank < 3; rank++ {
		if errs[rank] == nil || created[rank] != 1 {
			t.Fatalf("rank %d replaced a live mirror: created=%d err=%v", rank, created[rank], errs[rank])
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41ExpertParallelDuplicatePicksTCP(t *testing.T) {
	full := v41FullStepModel(t)
	binding := v41EPFixtureBinding(full)
	owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
	models := make([]*Model, 3)
	for rank, band := range owners.Shards {
		models[rank] = v41EPFixtureShard(full, band)
	}
	picks := []routePick{{expert: 260, weight: .501953125}, {expert: 2, weight: .25}, {expert: 2, weight: .75}}
	input := make([]float32, full.Cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%5-2) / 8
	}
	rows := make([][]float32, len(picks))
	for slot, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		scratch := &v41ProjScratch{}
		var err error
		rows[slot], err = full.v41FullRoutedExpertPublished(0, stem, input, pick.weight, full.Cfg, nil, nil, nil,
			func() ([]float32, []float32, []float32, error) {
				return full.v41ExpertTripleInto(0, stem, scratch, false)
			},
			func() ([]float32, error) { return full.hostExpertDown(0, stem, scratch) }, false, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	want, err := v41FullRoutedSum(0, picks, rows, full.Cfg.HiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	errs := v41EPTestGroup(t, func(g *DistComm) error {
		m := models[g.Rank()]
		if err := v41EPConfigure(m, g, binding); err != nil {
			return err
		}
		s := m.NewSession()
		defer s.Close()
		got, outcome, err := s.v41State().routedLayer(0, picks, input)
		if err != nil {
			return err
		}
		if outcome != v41RoutedExpertHandled {
			return fmt.Errorf("duplicate picks declined")
		}
		return v41EPCheckBits(got, want)
	})
	for rank, err := range errs {
		if err != nil {
			t.Fatalf("rank %d: %v", rank, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41ExpertParallelForbidsRankPresum(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	picks := []routePick{{expert: 8, weight: 1}, {expert: 0, weight: 1}, {expert: 8, weight: 1}}
	plan, err := v41EPGatherPlan(picks, owners, 1)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte("rank reassociation witness"))
	gathered := make([]float32, plan.Dim)
	for _, shard := range plan.Shards {
		v41EPHeader(gathered[shard.Lo:shard.Hi], digest, false)
	}
	gathered[plan.Shards[0].Lo+v41EPHeaderWords] = 1
	gathered[plan.Shards[2].Lo+v41EPHeaderWords] = 1 << 24
	gathered[plan.Shards[2].Lo+v41EPHeaderWords+1] = -(1 << 24)
	rows, err := v41EPUnpack(gathered, plan, owners, digest, picks, 1)
	if err != nil {
		t.Fatal(err)
	}
	exact, err := v41FullRoutedSum(0, picks, rows, 1)
	presummed := float32(rows[1][0] + float32(rows[0][0]+rows[2][0]))
	if err != nil || exact[0] != 0 || presummed != 1 {
		t.Fatalf("exact=%v rank-presummed=%g err=%v", exact, presummed, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelSessionNeedsCoordinator(t *testing.T) {
	m := v41FullStepModel(t)
	g := &DistComm{rank: 0, size: 3}
	if err := v41EPConfigure(m, g, v41EPFixtureBinding(m)); err != nil {
		t.Fatal(err)
	}
	s := m.NewSession()
	defer s.Close()
	if err := panicAsError(func() { s.Prefill([]int{1}) }); !errors.Is(err, ErrV41ExpertParallel) {
		t.Fatalf("uncoordinated rank-0 Session admitted: %v", err)
	}
	if s.epSeqLen() != 0 {
		t.Fatal("unannounced request changed continuation")
	}
}

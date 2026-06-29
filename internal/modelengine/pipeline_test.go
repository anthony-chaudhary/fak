package modelengine

import (
	"context"
	"net"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPipelineEngineAdmitRunsServeBandWorker is the EngineDriver reachability gate
// for the network PP worker (#30): the request enters through abi.AdmitOrShim, the
// remote stage runs only inside model.ServeBand over a real TCPTransport peer, and the
// streamed tokens match monolithic greedy generation.
func TestPipelineEngineAdmitRunsServeBandWorker(t *testing.T) {
	ctx := context.Background()
	cfg := SyntheticConfig()
	cfg.NumLayers = 3

	mono := model.NewSynthetic(cfg)
	stageA := model.NewSynthetic(cfg)
	stageB := model.NewSynthetic(cfg)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	var wg sync.WaitGroup
	workerErr := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		peer, aerr := ln.Accept()
		if aerr != nil {
			workerErr <- aerr
			return
		}
		defer peer.Close()
		workerErr <- model.ServeBand(peer, model.PipelineStage{
			Spec:  model.StageSpec{Lo: 1, Hi: 3, Last: true},
			Model: stageB,
		}, nil)
	}()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial worker: %v", err)
	}
	defer conn.Close()

	eng := NewPipelineEngine(model.PipelineStage{
		Spec:  model.StageSpec{Lo: 0, Hi: 1, First: true},
		Model: stageA,
	}, model.NewTCPTransport(conn))
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatal("PipelineEngine must implement the lifecycle seam")
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatal("PipelineEngine must advertise lifecycle support")
	}

	call := inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`)
	prompt := tokenize(call.Tool, refBytes(ctx, call.Args), cfg.VocabSize)
	want := mono.NewSession().Generate(prompt, genTokens)

	req, err := abi.AdmitOrShim(ctx, eng, call)
	if err != nil {
		t.Fatalf("AdmitOrShim: %v", err)
	}
	var streamed []int
	for tok := range req.Tokens() {
		streamed = append(streamed, tok.ID)
	}
	if !equalInts(streamed, want) {
		t.Fatalf("pipeline engine streamed tokens != monolithic:\n got=%v\nwant=%v", streamed, want)
	}

	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["engine"] != PipelineEngineID {
		t.Fatalf("meta engine = %q, want %q", res.Meta["engine"], PipelineEngineID)
	}
	g := decodeGen(t, ctx, res)
	if g.Engine != PipelineEngineID {
		t.Fatalf("payload engine = %q, want %q", g.Engine, PipelineEngineID)
	}
	if !equalInts(g.Tokens, streamed) {
		t.Fatalf("result tokens != streamed tokens:\n result=%v\n stream=%v", g.Tokens, streamed)
	}

	conn.Close()
	wg.Wait()
	close(workerErr)
	for e := range workerErr {
		if e != nil {
			t.Fatalf("ServeBand worker error: %v", e)
		}
	}
}

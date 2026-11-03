package modelengine

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// fak-test:runtime medium est=3s lane=default
func TestNativeOutputIsolation(t *testing.T) {
	for _, imported := range []bool{false, true} {
		name := "ordinary"
		if imported {
			name = "imported"
		}
		t.Run(name, func(t *testing.T) {
			m := model.NewSynthetic(SyntheticConfig())
			e := New()
			e.Preload(m)
			s := e.nativeScheduler()
			s.SetMaxRunning(1)
			s.SetKVPreemptionPolicy(NativePreemptionPolicy{})
			s.SetInBatchPrefixDedup(false)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			s.executor.Lock()
			var unlock sync.Once
			release := func() { unlock.Do(s.executor.Unlock) }
			var closeMu sync.Mutex
			closes := make(map[*model.Session]int)
			s.closeSession = func(session *model.Session) {
				closeMu.Lock()
				closes[session]++
				closeMu.Unlock()
				session.Close()
			}
			peerDone := make(chan struct{})
			peerStarted := false
			t.Cleanup(func() {
				cancel()
				s.Close()
				release()
				wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
				defer stop()
				if err := s.CloseAndWait(wait); err != nil {
					t.Errorf("scheduler did not stop: %v", err)
				}
				if peerStarted {
					select {
					case <-peerDone:
					case <-wait.Done():
						t.Error("Complete caller did not join")
					}
				}
			})

			var req abi.EngineRequest
			var err error
			if imported {
				prompt := []int{3, 7, 11}
				session := m.NewSession()
				logits := session.Prefill(prompt)
				req, err = s.AdmitImported(ctx, inlineCall("output-slow", "{}"), ImportedSequence{
					Session: session, Logits: logits, Prompt: prompt, PromptLen: len(prompt), Tool: "output-slow",
				})
				if err != nil {
					session.Close() // rejected import retains caller ownership
				}
			} else {
				req, err = e.Admit(ctx, inlineCall("output-slow", "{}"))
			}
			if err != nil {
				t.Fatal(err)
			}
			ln := req.(*schedLane)
			session := ln.sess
			wantCapacity := 1
			if nativeOutputIsolationEnabled {
				wantCapacity = genTokens
			}
			if got := cap(ln.tokens); got != wantCapacity {
				t.Fatalf("actual admitted outbox capacity=%d, want %d", got, wantCapacity)
			}
			prompt := append([]int(nil), ln.prompt...)
			if imported {
				prompt = []int{3, 7, 11}
			}
			want := nativeOutputOracle(m, prompt)
			if len(want) != genTokens {
				t.Fatal("fixture needs a full request budget, not early EOS")
			}
			firstPublished := make(chan struct{})
			var first sync.Once
			s.observeNativeEvent = func(event nativeSchedulerEvent) {
				if event.Kind == nativeSchedulerEventDecode && event.Lane == ln {
					first.Do(func() { close(firstPublished) })
				}
			}
			peerPrepared := make(chan struct{})
			prepare := s.prepare
			s.prepare = func(ctx context.Context, call *abi.ToolCall, m *model.Model) schedPrepare {
				prep := prepare(ctx, call, m)
				if call.Tool == "output-peer" {
					close(peerPrepared)
				}
				return prep
			}
			release()
			select {
			case <-firstPublished:
			case <-ctx.Done():
				t.Fatal("slow stream never entered actual token publication")
			}
			type outcome struct {
				res *abi.Result
				err error
			}
			peer := make(chan outcome, 1)
			peerStarted = true
			go func() {
				defer close(peerDone)
				res, err := e.Complete(ctx, inlineCall("output-peer", "{}"))
				peer <- outcome{res, err}
			}()
			select {
			case <-peerPrepared:
			case <-ctx.Done():
				t.Fatal("peer did not reach Engine.Complete -> Admit")
			}

			if !nativeOutputIsolationEnabled {
				// The real first lane occupies the only running slot. Its one
				// output slot is full and no reader exists; the next publication
				// cannot advance the shared executor. This is the old-route
				// control, not a wall-time or scheduler-throughput measurement.
				s.mu.Lock()
				terminal := ln.terminal
				running := len(s.lanes) == 1 && s.lanes[0] == ln
				s.mu.Unlock()
				if terminal || !running || len(ln.tokens) != 1 {
					t.Fatal("control did not retain the live undrained owner")
				}
				select {
				case <-peer:
					t.Fatal("control peer completed behind an undrained running owner")
				default:
				}
				req.Cancel()
			}
			var out outcome
			select {
			case out = <-peer:
			case <-ctx.Done():
				t.Fatal("peer Complete stalled on the undrained slow stream")
			}
			if out.err != nil || out.res == nil || out.res.Status != abi.StatusOK {
				t.Fatalf("peer result=%v err=%v", out.res, out.err)
			}
			result, resultErr := req.Result() // deliberately BEFORE Tokens drain
			if nativeOutputIsolationEnabled {
				if resultErr != nil || result == nil || result.Status != abi.StatusOK || len(ln.tokens) != genTokens {
					t.Fatalf("selected terminal result=%v err=%v buffered=%d", result, resultErr, len(ln.tokens))
				}
				var payload struct {
					Tokens []int `json:"generated_tokens"`
				}
				if err := json.Unmarshal(refutil.Bytes(ctx, result.Payload), &payload); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(payload.Tokens, want) ||
					result.Meta["input_tokens"] != strconv.Itoa(len(prompt)) ||
					result.Meta["output_tokens"] != strconv.Itoa(len(want)) {
					t.Fatalf("terminal payload/usage changed: payload=%v meta=%v", payload, result.Meta)
				}
			} else if result != nil || !errors.Is(resultErr, context.Canceled) {
				t.Fatalf("control cancellation result=%v err=%v", result, resultErr)
			}
			var got []int
			for token := range req.Tokens() {
				if token.Text != "" {
					t.Fatal("native ID-only outbox unexpectedly owns text payload")
				}
				got = append(got, token.ID)
			}
			if nativeOutputIsolationEnabled {
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("delayed stream changed: got %v want %v", got, want)
				}
			} else if len(got) != 1 || got[0] != want[0] {
				t.Fatalf("cancellation lost already-published output: %v", got)
			}
			again, againErr := req.Result()
			req.Cancel()
			req.Cancel()
			if again != result || againErr != resultErr || !ln.Reclaimed() || ln.sess != nil || ln.inflightSess != nil {
				t.Fatal("terminal output retained or re-created a session owner")
			}
			closeMu.Lock()
			closed := closes[session]
			closeMu.Unlock()
			if closed != 1 {
				t.Fatalf("slow real session closed %d times, want once", closed)
			}
		})
	}

	// All terminal causes use the same close owner, independently of delivery.
	// Hold execution rather than racing a reader against model speed.
	for _, cause := range []string{"cancel", "fault", "shutdown"} {
		t.Run(cause, func(t *testing.T) {
			s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			s.executor.Lock()
			var unlock sync.Once
			release := func() { unlock.Do(s.executor.Unlock) }
			t.Cleanup(func() {
				cancel()
				s.Close()
				release()
				wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
				defer stop()
				if err := s.CloseAndWait(wait); err != nil {
					t.Errorf("terminal fixture did not join: %v", err)
				}
			})
			req, err := s.Admit(ctx, inlineCall("output-terminal", "{}"))
			if err != nil {
				t.Fatal(err)
			}
			ln := req.(*schedLane)
			session := ln.sess
			closes := 0
			s.closeSession = func(actual *model.Session) {
				if actual == session {
					closes++
				}
				actual.Close()
			}
			wantErr := error(context.Canceled)
			switch cause {
			case "cancel":
				req.Cancel()
			case "fault":
				ln.logits = nil // actual decode-readiness refusal, not a fake result
				wantErr = errNativeSchedulerLaneNotDecodeReady
			case "shutdown":
				s.Close()
			}
			release()
			select {
			case <-ln.done:
			case <-ctx.Done():
				t.Fatal("terminal owner waited on output delivery")
			}
			result, err := req.Result()
			if result != nil || !errors.Is(err, wantErr) || !ln.Reclaimed() || closes != 1 {
				t.Fatalf("terminal result=%v err=%v closes=%d", result, err, closes)
			}
			if _, open := <-req.Tokens(); open {
				t.Fatal("terminal-before-execution unexpectedly published a token")
			}
			req.Cancel()
			_, _ = req.Result()
			if closes != 1 {
				t.Fatal("repeat terminal access closed the owner twice")
			}
		})
	}
}

func nativeOutputOracle(m *model.Model, prompt []int) []int {
	session := m.NewSession()
	defer session.Close()
	logits := session.Prefill(prompt)
	tokens := make([]int, 0, genTokens)
	for len(tokens) < genTokens {
		next := argmax(logits)
		tokens = append(tokens, next)
		if m.Cfg.IsEOS(next) {
			break
		}
		logits = session.Step(next)
	}
	return tokens
}

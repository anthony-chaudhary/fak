package engine

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestRequestFinishCompleteIsSingleAssignment(t *testing.T) {
	const contenders = 3
	for round := 0; round < 100; round++ {
		var finish requestFinish
		tokens := make(chan abi.EngineToken)
		done := make(chan struct{})
		results := []*abi.Result{
			{Status: abi.StatusOK, Meta: map[string]string{"winner": "success"}},
			nil,
			nil,
		}
		errs := []error{nil, errors.New("transport failure"), errors.New("request canceled")}

		start := make(chan struct{})
		var calls sync.WaitGroup
		calls.Add(contenders)
		for i := range contenders {
			go func() {
				defer calls.Done()
				<-start
				finish.complete(tokens, done, results[i], errs[i])
			}()
		}
		close(start)
		<-done
		calls.Wait()

		winner := -1
		for i := range contenders {
			if finish.res == results[i] && finish.err == errs[i] {
				winner = i
				break
			}
		}
		if winner < 0 {
			t.Fatalf("round %d: terminal fields do not match any complete call: res=%p err=%v", round, finish.res, finish.err)
		}
		if finish.res != results[winner] || finish.err != errs[winner] {
			t.Fatalf("round %d: observed partially written terminal result", round)
		}
		if _, ok := <-tokens; ok {
			t.Fatalf("round %d: tokens channel remains open", round)
		}
		select {
		case <-done:
		default:
			t.Fatalf("round %d: done channel remains open", round)
		}
	}
}

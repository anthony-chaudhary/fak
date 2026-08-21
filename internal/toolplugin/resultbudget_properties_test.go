package toolplugin

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"sync"
	"testing"
)

func TestResolveBudgetClampProperties(t *testing.T) {
	contract := []BudgetContract{{Tool: "search", Pointers: []string{"/limit"}, Ceiling: 100}}
	rng := rand.New(rand.NewSource(8041))
	for i := 0; i < 2000; i++ {
		requested := rng.Int63n(1 << 40)
		args, _ := json.Marshal(map[string]int64{"limit": requested})
		first, err := ResolveBudget("search", args, contract)
		if err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		if first.Effective > requested || first.Effective > contract[0].Ceiling {
			t.Fatalf("clamp raised value: requested=%d effective=%d", requested, first.Effective)
		}
		second, err := ResolveBudget("search", args, contract)
		if err != nil || !bytes.Equal(first.Arguments, second.Arguments) || first.Decision != second.Decision {
			t.Fatalf("nondeterministic resolution: first=%+v second=%+v err=%v", first, second, err)
		}
		replay, err := ResolveBudget("search", first.Arguments, contract)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(replay.Arguments, first.Arguments) {
			t.Fatalf("replay changed effective arguments: %s -> %s", first.Arguments, replay.Arguments)
		}
	}
}

func TestResolveBudgetFailSafeAndDenyPreservation(t *testing.T) {
	contract := []BudgetContract{{Tool: "search", Pointers: []string{"/limit"}, Ceiling: 10}}
	for _, args := range []string{`null`, `[]`, `{"limit":true}`, `{"limit":1e3}`, `{"limit":{}}`, `{"limit":[]}`, `{"limit":18446744073709551615}`} {
		got, err := ResolveBudget("search", []byte(args), contract)
		if err == nil && args != `null` && args != `[]` {
			t.Fatalf("hostile value %s unexpectedly admitted: %+v", args, got)
		}
		if string(got.Arguments) != args {
			t.Fatalf("hostile value mutated: %s -> %s", args, got.Arguments)
		}
	}
	// Tool admission is deliberately outside this argument rewriter: unknown/denied tools remain unchanged.
	denied := []byte(`{"limit":999}`)
	got, err := ResolveBudget("denied_tool", denied, contract)
	if err != nil || got.Decision != "pass" || !bytes.Equal(got.Arguments, denied) {
		t.Fatalf("deny preservation failed: %+v err=%v", got, err)
	}
}

func TestResolveBudgetConcurrentDeterminism(t *testing.T) {
	contract := []BudgetContract{{Tool: "search", Pointers: []string{"/limit"}, Ceiling: 7}}
	const workers = 32
	results := make(chan BudgetResolution, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := ResolveBudget("search", []byte(`{"limit":99,"continuation":"cursor-1"}`), contract)
			if err != nil {
				t.Error(err)
				return
			}
			results <- got
		}()
	}
	wg.Wait()
	close(results)
	var first BudgetResolution
	for got := range results {
		if first.Decision == "" {
			first = got
			continue
		}
		if got.Decision != first.Decision || !bytes.Equal(got.Arguments, first.Arguments) {
			t.Fatalf("concurrent result differs: %+v vs %+v", first, got)
		}
	}
	if !bytes.Contains(first.Arguments, []byte(`"continuation":"cursor-1"`)) {
		t.Fatalf("clamp hid continuation state: %s", first.Arguments)
	}
}

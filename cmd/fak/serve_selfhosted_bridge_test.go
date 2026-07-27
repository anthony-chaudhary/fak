package main

import (
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// The bridge is where an attribution goes to die.
//
// The gateway now resolves which side served each turn, and the durable row now has
// somewhere to put it, but the two are joined by one hand-written struct literal in
// gatewayUsageCounters. Add a seventh split field to Counters, forget the line here,
// and nothing complains: the field is omitempty, so it silently never reaches the
// wire and every consumer reads NOT INSTRUMENTED forever. That is exactly the
// failure this whole track exists to fix -- a number computed and discarded a line
// apart -- so the bridge gets a witness rather than a promise.
//
// Scoped to the who-served-it family on purpose. The rest of the literal renames
// fields as it copies them (CacheTTLUpgradesUpgraded <- adj.CacheTTLUpgraded), so a
// whole-struct name match would be a false alarm generator, and a test that cries
// wolf gets deleted.
func TestEverySelfHostedSplitFieldIsBridgedToTheDurableRow(t *testing.T) {
	src, err := os.ReadFile("serve.go")
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	body, ok := funcBodyText(string(src), "func gatewayUsageCounters(")
	if !ok {
		t.Fatal("gatewayUsageCounters not found in serve.go — if it moved, move this witness with it")
	}

	rt := reflect.TypeOf(gatewayusageledger.Counters{})
	var checked int
	for i := 0; i < rt.NumField(); i++ {
		name := rt.Field(i).Name
		if !strings.HasPrefix(name, "SelfHosted") && !strings.HasPrefix(name, "Vendor") {
			continue
		}
		checked++
		if !strings.Contains(body, name+":") {
			t.Errorf("Counters.%s is never assigned in gatewayUsageCounters — the gateway measures it and the row drops it", name)
		}
	}
	if checked == 0 {
		t.Fatal("no self-hosted split fields found on Counters; this witness has stopped witnessing anything")
	}
}

// funcBodyText returns the text from the start of the named func to the first
// line-initial closing brace, which is where a top-level Go func ends under gofmt.
func funcBodyText(src, decl string) (string, bool) {
	i := strings.Index(src, decl)
	if i < 0 {
		return "", false
	}
	rest := src[i:]
	if end := strings.Index(rest, "\n}\n"); end >= 0 {
		return rest[:end], true
	}
	return rest, true
}

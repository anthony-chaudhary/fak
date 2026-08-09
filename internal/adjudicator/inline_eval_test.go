package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestPolicyInlineEvalExtendsLiveSelfModifyFloor(t *testing.T) {
	base := DefaultPolicy()
	a := New(base)
	call := inlineCall("Bash", `{"command":"perl -MJSON -e 'open(F,\"w\",\"internal/abi/x.go\")'"}`)
	if got := a.Adjudicate(context.Background(), call); got.Reason == abi.ReasonSelfModify {
		t.Fatalf("omitted inline_eval changed baseline: %+v", got)
	}

	base.InlineEval = []InlineEvalSpec{{Interp: "perl", Flags: []string{"-e", "-E"}}}
	a.SetPolicy(base)
	for _, command := range []string{
		`perl -MJSON -e 'open(F,"w","internal/abi/x.go")'`,
		`perl -MJSON -e='open(F,"w","internal/abi/x.go")'`,
	} {
		got := a.Adjudicate(context.Background(), inlineCall("Bash", `{"command":`+quoteJSON(command)+`}`))
		if got.Reason != abi.ReasonSelfModify {
			t.Fatalf("%q: got %+v, want SELF_MODIFY", command, got)
		}
	}
}

func quoteJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

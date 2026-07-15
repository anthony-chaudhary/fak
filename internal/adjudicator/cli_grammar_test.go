package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func cliGrammarVerdict(t *testing.T, command string) (abi.Verdict, map[string]any) {
	t.Helper()
	a := New(Policy{Allow: map[string]bool{"Bash": true}, ArgPredicates: []ArgPredicate{{Tool: "Bash", Arg: "command", Kind: ArgCLIReadOnly, Reason: abi.ReasonPolicyBlock}}})
	v := a.Adjudicate(context.Background(), inlineCall("Bash", `{"command":`+mustJSON(command)+`}`))
	if v.Kind != abi.VerdictTransform {
		return v, nil
	}
	tp := v.Payload.(abi.TransformPayload)
	b, err := abi.ActiveResolver().Resolve(context.Background(), tp.NewArgs)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return v, out
}
func mustJSON(s string) string { b, _ := json.Marshal(s); return string(b) }

func TestCLIGrammarReadOnlyAllowlistAndAttenuation(t *testing.T) {
	if got, changed, err := attenuateCLIGrammar("gh search issues bug repo:other/x org:other user:alice --limit 5"); err != nil || !changed || got != "gh search issues bug --limit 5" {
		t.Fatalf("direct attenuation got=%q changed=%v err=%v", got, changed, err)
	}
	for _, command := range []string{"gh issue list --limit 5", "gh pr view 12", "git status --short", "git log -3 --oneline"} {
		v, _ := cliGrammarVerdict(t, command)
		if v.Kind != abi.VerdictAllow {
			t.Errorf("%q action=%v reason=%v", command, v.Kind, v.Reason)
		}
	}
	for _, command := range []string{"gh pr create --title x", "gh issue close 12", "git push origin main", "gh issue list; gh pr create"} {
		v, _ := cliGrammarVerdict(t, command)
		if v.Kind != abi.VerdictDeny {
			t.Errorf("%q action=%v", command, v.Kind)
		}
	}
	v, args := cliGrammarVerdict(t, "gh search issues bug repo:other/x org:other user:alice --limit 5")
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("action=%v reason=%v", v.Kind, v.Reason)
	}
	if got := args["command"]; got != "gh search issues bug --limit 5" {
		t.Fatalf("rewritten command=%q", got)
	}
}

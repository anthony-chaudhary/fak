package adjudicator

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func buildCacheCleanAdj(tool, arg string) *Adjudicator {
	return New(Policy{
		Allow: map[string]bool{tool: true},
		ArgPredicates: []ArgPredicate{{
			Tool: tool, Arg: arg, Kind: ArgDenyRegex,
			Re: regexp.MustCompile(defaultBuildCacheCleanDenyRegex), Reason: abi.ReasonPolicyBlock,
		}},
	})
}

func TestBuildCacheCleanPolicyDeciderDistinguishesUseFromMention(t *testing.T) {
	for _, tool := range []struct{ name, arg string }{
		{"Bash", "command"},
		{"PowerShell", "command"},
		{"shell_command", "command"},
		{"functions.shell_command", "command"},
		{"exec_command", "cmd"},
	} {
		t.Run(tool.name, func(t *testing.T) {
			a := buildCacheCleanAdj(tool.name, tool.arg)
			for _, command := range []string{
				"go clean -cache",
				"go clean -testcache -cache",
				"env GOENV=off go clean --cache=true",
				"cd /tmp && go clean -cache -testcache",
			} {
				v := a.Adjudicate(context.Background(), inlineCall(tool.name, `{"`+tool.arg+`":"`+command+`"}`))
				if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Errorf("%q = %v/%s, want Deny/POLICY_BLOCK", command, v.Kind, abi.ReasonName(v.Reason))
				}
			}
			for _, command := range []string{
				"go clean -testcache",
				"go test ./...",
				"echo 'go clean -cache'",
				`rg -n 'go clean -cache' docs`,
				`git commit -m 'docs: avoid go clean -cache'`,
			} {
				v := a.Adjudicate(context.Background(), inlineCall(tool.name, `{"`+tool.arg+`":"`+command+`"}`))
				if v.Kind != abi.VerdictAllow {
					t.Errorf("near miss %q = %v/%s, want Allow", command, v.Kind, abi.ReasonName(v.Reason))
				}
			}
		})
	}
}

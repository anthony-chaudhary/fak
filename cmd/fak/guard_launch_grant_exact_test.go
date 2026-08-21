package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestGuardDenialRowGrantStaysExactOnly(t *testing.T) {
	rejected := []string{"*", "deploy*", "deploy?", "[deploy]", "deploy|shell", "../deploy", "deploy..preview", "deploy preview", "ｄｅｐｌｏｙ＊"}
	for _, name := range rejected {
		t.Run("reject/"+name, func(t *testing.T) {
			var flag launchToolFlag
			if err := flag.Set(name); err == nil {
				t.Fatalf("Set(%q) accepted widening syntax", name)
			}
		})
	}
	accepted := []string{"deploy_preview", "opencode.bash", "mcp:search/read", "Deploy-Preview"}
	for _, name := range accepted {
		t.Run("accept/"+name, func(t *testing.T) {
			var flag launchToolFlag
			if err := flag.Set(name); err != nil {
				t.Fatalf("Set(%q): %v", name, err)
			}
			setLaunchToolGrant(flag)
			grant := launchToolGrant()
			if len(grant.Allow) != 1 || grant.Allow[0] != name || len(grant.AllowPrefix) != 0 {
				t.Fatalf("grant widened: %+v", grant)
			}
			var rt policy.Runtime
			guardApplyAllowOverlay(&rt, grant)
			if len(rt.Policy().AllowPrefix) != 0 {
				t.Fatalf("runtime gained prefix authority: %+v", rt.Policy())
			}
		})
	}
}

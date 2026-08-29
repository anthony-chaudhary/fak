package repoguard

import (
	"strings"
	"testing"
)

func TestClassifyBuildCacheCleanDeniesExecutedCacheClears(t *testing.T) {
	cases := []string{
		"go clean -cache",
		"go clean -cache -testcache",
		"go clean -testcache -cache",
		"GOFLAGS=-x go clean --cache",
		"env GOENV=off /usr/local/bin/go clean -cache=true",
		"cd /tmp && go clean -cache",
		"go -C ./internal clean -cache",
		"command go clean -cache",
		"sh -c 'go clean -cache'",
	}
	for _, command := range cases {
		t.Run(command, func(t *testing.T) {
			got := ClassifyBuildCacheClean(command)
			if len(got) != 1 || got[0].Reason != ReasonBuildCacheCleanRace {
				t.Fatalf("ClassifyBuildCacheClean(%q) = %+v, want one %s", command, got, ReasonBuildCacheCleanRace)
			}
		})
	}
}

func TestClassifyBuildCacheCleanAllowsNearMissesAndMentions(t *testing.T) {
	cases := []string{
		"go clean -testcache",
		"go clean -cache=false",
		"go test ./...",
		"go vet ./...",
		"echo 'go clean -cache'",
		`printf '%s\\n' "go clean -cache"`,
		`rg -n 'go clean -cache' docs`,
		`git commit -m "docs: never run go clean -cache"`,
		"cat <<'EOF'\ngo clean -cache\nEOF",
		"go env GOCACHE",
		"go clean ./... -cache",
		"go clean -- -cache",
	}
	for _, command := range cases {
		t.Run(command, func(t *testing.T) {
			if got := ClassifyBuildCacheClean(command); len(got) != 0 {
				t.Fatalf("ClassifyBuildCacheClean(%q) = %+v, want allow", command, got)
			}
		})
	}
}

func TestBuildCacheCleanDoesNotTrustInlineGOCACHEOwnershipClaim(t *testing.T) {
	got := ClassifyBuildCacheClean(`GOCACHE="$PWD/.gocache" go clean -cache`)
	if len(got) != 1 {
		t.Fatalf("inline GOCACHE claim yielded %+v, want deny until ownership is externally proven", got)
	}
	for _, want := range []string{"fak-dev buildcheck --vet", "fak validate --mine", "explicitly allocated OS-temp directory", "platform Trash/Recycle Bin", "FAK_REPO_GUARD_SEVERITY=BUILD_CACHE_CLEAN_RACE=off"} {
		if !strings.Contains(got[0].Fix, want) {
			t.Fatalf("fix = %q, want %q", got[0].Fix, want)
		}
	}
}

func TestBuildCacheCleanDefaultsToDeny(t *testing.T) {
	if got := DefaultSeverity(ReasonBuildCacheCleanRace); got != SeverityDeny {
		t.Fatalf("DefaultSeverity(%s) = %v, want deny", ReasonBuildCacheCleanRace, got)
	}
}

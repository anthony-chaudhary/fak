package workerworktree

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This entry returns without assertions in the parent. Child invocation is
// explicit and uses the already compiled test executable, never a shell script.
func TestPreservingAttributeFilterHelperProcess(t *testing.T) {
	args := flag.Args()
	if len(args) != 2 || args[0] != "preserving-attribute-filter-fixture" {
		return
	}
	if !filepath.IsAbs(args[1]) {
		t.Fatal("absolute fixture marker required")
	}
	file, err := os.OpenFile(args[1], os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("native filter helper executed\n"); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	os.Exit(86)
}

func TestPreservingNativeFixtureParentAndChildAttributeIdentity(t *testing.T) {
	f := newQualifiedPreservingFixture(t)
	home := preservingFixtureIdentity(t)
	attributes := filepath.Join(home, "git", "attributes")
	if err := os.MkdirAll(filepath.Dir(attributes), 0700); err != nil {
		t.Fatal(err)
	}
	data := []byte("target.txt filter=private-parent-child\n")
	if err := os.WriteFile(attributes, data, 0600); err != nil {
		t.Fatal(err)
	}
	preservingFixtureGit(t, f.repo, "config", "filter.private-parent-child.smudge", "must-not-execute")
	// The child's actual effective attribute lookup must see the file inspected
	// by the parent policy. There is no rewriting of native Git output.
	got := preservingFixtureGit(t, f.repo, "check-attr", "--source="+f.base, "-z", "filter", "--", "target.txt")
	if got != "target.txt\x00filter\x00private-parent-child\x00" {
		t.Fatalf("native child did not use parent XDG identity: %q", got)
	}
	var calls []string
	if _, err := preservingCheckoutPolicy(f.repo, f.base, preservingTestRunner(t, &calls)); err == nil || !strings.Contains(err.Error(), "global attributes unqualified") {
		t.Fatalf("parent policy ignored its matching private XDG file: %v", err)
	}
	if after, err := os.ReadFile(attributes); err != nil || string(after) != string(data) {
		t.Fatalf("fixture attributes changed: %v", err)
	}
}

func TestPreparePreservingNativePinnedAttributesRejectDirtyHiding(t *testing.T) {
	for _, kind := range []string{"root", "nested", "macro"} {
		t.Run(kind, func(t *testing.T) {
			f := newQualifiedPreservingFixture(t)
			attributePath, sourcePath := ".gitattributes", "target.txt"
			committed := "target.txt filter=preserving-native\n"
			switch kind {
			case "nested":
				attributePath, sourcePath = "nested/.gitattributes", "nested/target.txt"
				if err := os.MkdirAll(filepath.Join(f.repo, "nested"), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(f.repo, sourcePath), []byte("nested pinned source\n"), 0600); err != nil {
					t.Fatal(err)
				}
			case "macro":
				committed = "[attr]preserving-active filter=preserving-native\ntarget.txt preserving-active\n"
			}
			if err := os.WriteFile(filepath.Join(f.repo, attributePath), []byte(committed), 0600); err != nil {
				t.Fatal(err)
			}
			// Commit fixture attributes before defining any executable filter. The
			// fresh fixture has no inherited system/global filter configuration.
			preservingFixtureGit(t, f.repo, "add", "--", attributePath, sourcePath)
			preservingFixtureGit(t, f.repo, "commit", "-q", "-m", "pinned active attribute fixture")
			base := strings.TrimSpace(preservingFixtureGit(t, f.repo, "rev-parse", "HEAD"))
			dirty := []byte("# dirty checkout hides the committed active attribute\n")
			if err := os.WriteFile(filepath.Join(f.repo, attributePath), dirty, 0600); err != nil {
				t.Fatal(err)
			}
			marker := filepath.Join(t.TempDir(), "filter-executed")
			quote := func(value string) string { return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'" }
			command := quote(os.Args[0]) + " -test.run=^TestPreservingAttributeFilterHelperProcess$ -- preserving-attribute-filter-fixture " + quote(marker)
			for _, key := range []string{"clean", "smudge", "process"} {
				preservingFixtureGit(t, f.repo, "config", "filter.preserving-native."+key, command)
			}
			preservingFixtureGit(t, f.repo, "config", "filter.preserving-native.required", "true")
			pinned := preservingFixtureGit(t, f.repo, "check-attr", "--source="+base, "-z", "filter", "--", sourcePath)
			working := preservingFixtureGit(t, f.repo, "check-attr", "-z", "filter", "--", sourcePath)
			if pinned != sourcePath+"\x00filter\x00preserving-native\x00" || working != sourcePath+"\x00filter\x00unspecified\x00" {
				t.Fatalf("fixture failed to establish pinned/dirty disagreement: pinned=%q dirty=%q", pinned, working)
			}
			before := map[string]string{}
			for _, path := range []string{attributePath, sourcePath, ".git/config", ".git/index", ".git/HEAD"} {
				data, err := os.ReadFile(filepath.Join(f.repo, path))
				if err != nil {
					t.Fatal(err)
				}
				before[path] = fmt.Sprintf("%x", sha256.Sum256(data))
			}
			workers := t.TempDir()
			var calls []string
			res := preparePreserving(context.Background(), f.repo, "cmd", "pinned-attribute-negative", base, workers,
				OwnerStamp{PID: os.Getpid(), LeaseID: "fixture-admission"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
			if res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" || !strings.Contains(res.Reason, "active, reordered or unknown pinned filter attribute") {
				t.Fatalf("native pinned attribute proof did not cause refusal: %+v", res)
			}
			for _, call := range calls {
				if strings.Contains(call, "worktree add") || strings.HasPrefix(call, "status ") || strings.HasPrefix(call, "diff ") {
					t.Fatalf("checkout/helper-capable command after negative proof: %s", call)
				}
			}
			if _, err := os.Lstat(marker); !os.IsNotExist(err) {
				t.Fatalf("filter helper ran during proof/refusal: %v", err)
			}
			if entries, err := os.ReadDir(workers); err != nil || len(entries) != 0 {
				t.Fatalf("new target or sidecars created before refusal: %v %v", entries, err)
			}
			for path, digest := range before {
				data, err := os.ReadFile(filepath.Join(f.repo, path))
				if err != nil || fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
					t.Fatalf("fixture evidence changed: %s %v", path, err)
				}
			}
			evidence, err := json.Marshal(struct {
				Kind, Base, Pinned, Dirty, Code string
				Preserved                       map[string]string
				Queries                         []string
			}{kind, base, pinned, working, res.Code, before, calls})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("native pinned attribute refusal evidence: %s", evidence)
			// Positive control follows every no-helper assertion. It executes the
			// native helper directly; Git never executes a filter in this fixture.
			gitPath, err := exec.LookPath("git")
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPreservingAttributeFilterHelperProcess$", "--", "preserving-attribute-filter-fixture", marker)
			child.WaitDelay = 250 * time.Millisecond
			child.Env = preservingLibraryFixtureEnv(os.Environ(), gitPath, preservingFixtureIdentity(t))
			output, err := child.CombinedOutput()
			exit, ok := err.(*exec.ExitError)
			if ctx.Err() != nil || !ok || exit.ExitCode() != 86 {
				t.Fatalf("native helper positive control failed: %v %q", err, output)
			}
			if data, err := os.ReadFile(marker); err != nil || string(data) != "native filter helper executed\n" {
				t.Fatalf("native helper marker positive control failed: %v", err)
			}
		})
	}
}

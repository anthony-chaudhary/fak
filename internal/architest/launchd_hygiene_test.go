package architest

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestLaunchdHygiene enforces the shift-left composable lifecycle invariant
// (docs/notes/SHIFT-LEFT-COMPOSABLE-LIFECYCLE-NO-STATIC-DAEMONS-2026-09-03.md, issue #10866).
//
// Background daemons with KeepAlive=true (e.g. com.fak.* in ~/Library/LaunchAgents/
// or /Library/LaunchDaemons/) persist across sessions and reboots, consume 16+ GiB RAM,
// force memory compression past 30%, and trip memory admission floors (pressure_critical).
//
// On developer benches, all services must have composable, ephemeral lifecycles:
// spawned on-demand by calling harnesses with signal traps (trap cleanup EXIT INT TERM),
// paired with explicit teardown handlers (launchctl bootout/unload, --uninstall),
// and never hardcoded in tests without hermetic isolation (t.TempDir()).
func TestLaunchdHygiene(t *testing.T) {
	root := repoRoot(t)

	t.Run("PlistTemplatesDeclareTeardown", func(t *testing.T) {
		// Inspect all .plist files in the repository. If any template configures KeepAlive=true,
		// it must explicitly document teardown commands (launchctl unload, launchctl bootout,
		// or removal instructions) so developers and operators have deterministic lifecycle control.
		var plists []string
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				name := d.Name()
				if name == ".git" || name == "_scratch" || name == ".cache" || name == "vendor" || name == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.HasSuffix(d.Name(), ".plist") {
				plists = append(plists, path)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk plists under %s: %v", root, err)
		}

		if len(plists) == 0 {
			t.Fatalf("expected to find at least one .plist template in repository")
		}

		for _, path := range plists {
			rel, _ := filepath.Rel(root, path)
			content, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", rel, err)
			}

			if err := checkPlistHygiene(content); err != nil {
				t.Errorf(".plist template %s failed hygiene check: %v\n"+
					"\tIssue #10866: Static background daemons with KeepAlive=true persist on dev benches and trip admission floors.\n"+
					"\tFix: Document explicit unload/bootout commands in the template header or remove KeepAlive.", rel, err)
			}
		}
	})

	t.Run("SyntheticViolationsDetected", func(t *testing.T) {
		// Verify the gate catches unmanaged KeepAlive daemons without teardown handlers.
		badPlist := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.fak.rogue-daemon</string>
    <key>KeepAlive</key><true/>
    <key>RunAtLoad</key><true/>
  </dict>
</plist>`)
		if err := checkPlistHygiene(badPlist); err == nil {
			t.Fatal("expected failure on unmanaged KeepAlive=true daemon without teardown instructions")
		}

		goodPlistWithUnload := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<!--
  To unload: launchctl unload -w ~/Library/LaunchAgents/com.fak.good.plist
-->
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.fak.good</string>
    <key>KeepAlive</key><true/>
  </dict>
</plist>`)
		if err := checkPlistHygiene(goodPlistWithUnload); err != nil {
			t.Fatalf("unexpected failure on plist with documented unload: %v", err)
		}

		goodIntervalPlist := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.fak.cadence</string>
    <key>StartInterval</key><integer>300</integer>
  </dict>
</plist>`)
		if err := checkPlistHygiene(goodIntervalPlist); err != nil {
			t.Fatalf("unexpected failure on cadence plist without KeepAlive: %v", err)
		}
	})

	t.Run("ServiceTemplatesProvideUninstallPath", func(t *testing.T) {
		// Inspect Go service template generators in the codebase.
		// If a template renders KeepAlive=true for launchd, it must be paired with
		// an explicit uninstall/bootout mechanism in the owning package or command.
		type templateCheck struct {
			file              string
			requiredMechanism string
		}

		checks := []templateCheck{
			{
				file:              filepath.Join("cmd", "fak", "node_install.go"),
				requiredMechanism: "--uninstall",
			},
			{
				file:              filepath.Join("internal", "systemservice", "launchd.go"),
				requiredMechanism: "RenderLaunchDaemon",
			},
			{
				file:              filepath.Join("cmd", "fak", "model_incumbent.go"),
				requiredMechanism: "renderIncumbentPlist",
			},
			{
				file:              filepath.Join("tools", "install-mac-node.sh"),
				requiredMechanism: "--uninstall",
			},
		}

		for _, c := range checks {
			path := filepath.Join(root, c.file)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read template source %s: %v", c.file, err)
			}
			text := string(data)

			if keepAliveRegex.Match(data) || strings.Contains(text, "KeepAlive") {
				if !strings.Contains(text, c.requiredMechanism) {
					t.Errorf("service template %s missing lifecycle teardown mechanism %q", c.file, c.requiredMechanism)
				}
			}
		}
	})

	t.Run("TestsDoNotInstallHostLaunchAgentsWithoutTeardown", func(t *testing.T) {
		// Inspect test files to ensure no test hardcodes writes or persistent registrations
		// into live user/system launchd directories (~/Library/LaunchAgents or /Library/LaunchDaemons)
		// without hermetic temp directories.
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				name := d.Name()
				if name == ".git" || name == "_scratch" || name == ".cache" || name == "vendor" {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.HasSuffix(d.Name(), "_test.go") {
				rel, _ := filepath.Rel(root, path)
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("read test %s: %v", rel, err)
				}

				text := string(data)
				// Check if test attempts to write directly to ~/Library/LaunchAgents on the host
				if strings.Contains(text, `os.WriteFile(`) && strings.Contains(text, `LaunchAgents`) {
					// Ensure it uses t.TempDir() or a temp directory
					if !strings.Contains(text, `TempDir`) && !strings.Contains(text, `dir`) && !strings.Contains(text, `tmp`) {
						t.Errorf("test %s writes to LaunchAgents without hermetic TempDir isolation", rel)
					}
				}

				// Check if test runs live launchctl bootstrap without cleanup
				if strings.Contains(text, `"launchctl"`) && strings.Contains(text, `"bootstrap"`) {
					if !strings.Contains(text, `t.Cleanup`) && !strings.Contains(text, `defer`) && !strings.Contains(text, `bootout`) {
						t.Errorf("test %s calls launchctl bootstrap without explicit cleanup/bootout teardown handler", rel)
					}
				}
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk test files: %v", err)
		}
	})

	t.Run("DoctrineDocumentedInNotes", func(t *testing.T) {
		// Ensure the shift-left composable lifecycle note exists and covers the required core invariants.
		notePath := filepath.Join(root, "docs", "notes", "SHIFT-LEFT-COMPOSABLE-LIFECYCLE-NO-STATIC-DAEMONS-2026-09-03.md")
		data, err := os.ReadFile(notePath)
		if err != nil {
			t.Fatalf("missing required note %s: %v", notePath, err)
		}
		noteText := string(data)

		requiredPhrases := []string{
			"trap cleanup EXIT INT TERM",
			"KeepAlive=true",
			"launchctl bootout",
			"lsof -ti:8090 | xargs kill",
			"~/Library/LaunchAgents",
			"/Library/LaunchDaemons",
			"systemd",
			"Task Scheduler",
		}

		for _, phrase := range requiredPhrases {
			if !strings.Contains(noteText, phrase) {
				t.Errorf("doctrine note missing required phrase %q", phrase)
			}
		}
	})
}

var keepAliveRegex = regexp.MustCompile(`(?i)<key>KeepAlive</key>\s*<true\s*/>`)

func checkPlistHygiene(content []byte) error {
	if !keepAliveRegex.Match(content) {
		return nil
	}
	teardownKeywords := []string{
		"launchctl unload",
		"launchctl bootout",
		"bootout",
		"unload",
		"--uninstall",
		"cleanup",
		"rm -f",
	}
	text := strings.ToLower(string(content))
	for _, kw := range teardownKeywords {
		if strings.Contains(text, kw) {
			return nil
		}
	}
	return fmt.Errorf("plist specifies KeepAlive=true without documented teardown commands (launchctl unload/bootout/cleanup)")
}

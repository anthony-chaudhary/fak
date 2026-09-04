package adjudicator

import (
	"testing"
)

func TestShellEditNudge(t *testing.T) {
	const wantAdvisory = "Prefer structured tool 'Edit' or 'Write' over shell mutation for atomic, diff-witnessed changes."

	t.Run("sed -i variations", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "basic sed -i",
				cmd:        "sed -i 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "sed -i with backup suffix",
				cmd:        "sed -i.bak 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "sed -i empty quotes macOS style",
				cmd:        "sed -i '' 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "sed -i with spaces in path",
				cmd:        `sed -i 's/foo/bar/g' "path with spaces/file.go"`,
				wantTool:   "sed",
				wantTarget: "path with spaces/file.go",
			},
			{
				name:       "sed -i multiple -e flags",
				cmd:        "sed -i -e 's/a/b/' -e 's/c/d/' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, false)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
				if res.Blocked {
					t.Errorf("expected Blocked false in non-strict mode")
				}
				if res.Suggestion != wantAdvisory {
					t.Errorf("Suggestion = %q, want %q", res.Suggestion, wantAdvisory)
				}
			})
		}
	})

	t.Run("sed --in-place variations", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "sed --in-place basic",
				cmd:        "sed --in-place 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "sed --in-place with suffix",
				cmd:        "sed --in-place=.bak 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, false)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
			})
		}
	})

	t.Run("awk and gawk -i inplace", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "awk -i inplace",
				cmd:        "awk -i inplace '{print $1}' data.csv",
				wantTool:   "awk",
				wantTarget: "data.csv",
			},
			{
				name:       "gawk -i inplace",
				cmd:        "gawk -i inplace '{print $1}' data.csv",
				wantTool:   "gawk",
				wantTarget: "data.csv",
			},
			{
				name:       "gawk --include inplace",
				cmd:        "gawk --include inplace '{print $1}' data.csv",
				wantTool:   "gawk",
				wantTarget: "data.csv",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, false)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
			})
		}
	})

	t.Run("perl -pi -e and perl -i variations", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "perl -pi -e basic",
				cmd:        "perl -pi -e 's/foo/bar/g' file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
			{
				name:       "perl -i -pe basic",
				cmd:        "perl -i -pe 's/foo/bar/g' file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
			{
				name:       "perl -i with script arg",
				cmd:        "perl -i 's/foo/bar/g' file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
			{
				name:       "perl -pie clustered",
				cmd:        "perl -pie 's/foo/bar/g' file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
			{
				name:       "perl -i.bak with -ne",
				cmd:        "perl -i.bak -ne 'print if /foo/' file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
			{
				name:       "perl -pi -e with spaces in path",
				cmd:        `perl -pi -e 's/foo/bar/g' "path with spaces/file.go"`,
				wantTool:   "perl",
				wantTarget: "path with spaces/file.go",
			},
			{
				name:       "perl -i file only",
				cmd:        "perl -i file.go",
				wantTool:   "perl",
				wantTarget: "file.go",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, false)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
			})
		}
	})

	t.Run("python -c in-place write variations", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "python -c open write",
				cmd:        `python -c "open('file.go','w').write('hello')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python -c open write with spaces",
				cmd:        `python -c "open('file.go', 'w').write('hello')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python3 -c open write",
				cmd:        `python3 -c "open('file.go', 'w').write('hello')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python -c with double quotes inside",
				cmd:        `python -c 'open("file.go", "w").write("hello")'`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python -c with open context manager",
				cmd:        `python -c 'with open("file.go", "w") as f: f.write("hello")'`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python -c append mode",
				cmd:        `python -c "open('file.go', 'a').write('more')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python3 -c pathlib Path write_text",
				cmd:        `python3 -c "from pathlib import Path; Path('file.go').write_text('hello')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python3 -c pathlib.Path write_bytes",
				cmd:        `python3 -c "import pathlib; pathlib.Path('file.go').write_bytes(b'hello')"`,
				wantTool:   "python",
				wantTarget: "file.go",
			},
			{
				name:       "python -c open with spaces in filename",
				cmd:        `python -c "open('dir with spaces/file.go', 'w').write('hello')"`,
				wantTool:   "python",
				wantTarget: "dir with spaces/file.go",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, false)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
			})
		}
	})

	t.Run("strictMode policy enforcement", func(t *testing.T) {
		cmd := "sed -i 's/foo/bar/g' file.go"

		nonStrictRes := CheckShellEditNudge(cmd, false)
		if !nonStrictRes.IsShellEdit {
			t.Fatal("expected IsShellEdit true")
		}
		if nonStrictRes.Blocked {
			t.Errorf("nonStrictRes.Blocked = %v, want false", nonStrictRes.Blocked)
		}
		if nonStrictRes.Suggestion != wantAdvisory {
			t.Errorf("nonStrictRes.Suggestion = %q, want %q", nonStrictRes.Suggestion, wantAdvisory)
		}

		strictRes := CheckShellEditNudge(cmd, true)
		if !strictRes.IsShellEdit {
			t.Fatal("expected IsShellEdit true")
		}
		if !strictRes.Blocked {
			t.Errorf("strictRes.Blocked = %v, want true", strictRes.Blocked)
		}
		if strictRes.Suggestion != wantAdvisory {
			t.Errorf("strictRes.Suggestion = %q, want %q", strictRes.Suggestion, wantAdvisory)
		}
	})

	t.Run("SuggestStructuredEditTool helper", func(t *testing.T) {
		cmd := "sed -i 's/foo/bar/g' file.go"

		isEdit, suggestion, blocked := SuggestStructuredEditTool(cmd, false)
		if !isEdit || blocked || suggestion != wantAdvisory {
			t.Errorf("SuggestStructuredEditTool non-strict got (%v, %q, %v), want (true, %q, false)",
				isEdit, suggestion, blocked, wantAdvisory)
		}

		isEdit, suggestion, blocked = SuggestStructuredEditTool(cmd, true)
		if !isEdit || !blocked || suggestion != wantAdvisory {
			t.Errorf("SuggestStructuredEditTool strict got (%v, %q, %v), want (true, %q, true)",
				isEdit, suggestion, blocked, wantAdvisory)
		}

		// Non-edit command
		isEdit, suggestion, blocked = SuggestStructuredEditTool("cat file.go", true)
		if isEdit || blocked || suggestion != "" {
			t.Errorf("SuggestStructuredEditTool for cat got (%v, %q, %v), want (false, \"\", false)",
				isEdit, suggestion, blocked)
		}
	})

	t.Run("negative non-mutating commands", func(t *testing.T) {
		commands := []string{
			"sed 's/foo/bar/' file.go",
			"sed -n 'p' file.go",
			"awk '{print $1}' data.csv",
			"gawk '{print $1}' data.csv",
			"perl -e 'print \"hello\"'",
			"perl script.pl file.go",
			"perl -ne 'print if /foo/' file.go",
			"python -c 'print(1+1)'",
			"python -c \"open('file.go').read()\"",
			"python -c \"open('file.go', 'r').read()\"",
			"python -c \"open('file.go', 'rb').read()\"",
			"python script.py",
			"cat file.go",
			"git status",
			"echo hello",
			"ls -la",
		}

		for _, cmd := range commands {
			t.Run(cmd, func(t *testing.T) {
				res := CheckShellEditNudge(cmd, true)
				if res.IsShellEdit {
					t.Errorf("expected IsShellEdit false for %q, got tool %q target %q",
						cmd, res.DetectedTool, res.TargetPath)
				}
				if res.Blocked {
					t.Errorf("expected Blocked false for %q", cmd)
				}
				if res.Suggestion != "" {
					t.Errorf("expected empty Suggestion for %q, got %q", cmd, res.Suggestion)
				}
			})
		}
	})

	t.Run("compound commands and subshells", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantTool   string
			wantTarget string
		}{
			{
				name:       "compound with && prefix",
				cmd:        "echo start && sed -i 's/a/b/' file.go && echo done",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "compound with ; prefix",
				cmd:        "git checkout main; python -c \"open('v.txt', 'w').write('1')\"",
				wantTool:   "python",
				wantTarget: "v.txt",
			},
			{
				name:       "nested bash -c",
				cmd:        `bash -c "sed -i 's/foo/bar/g' file.go"`,
				wantTool:   "sed",
				wantTarget: "file.go",
			},
			{
				name:       "sudo prefix",
				cmd:        "sudo sed -i 's/foo/bar/g' file.go",
				wantTool:   "sed",
				wantTarget: "file.go",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				res := CheckShellEditNudge(tc.cmd, true)
				if !res.IsShellEdit {
					t.Fatalf("expected IsShellEdit true for %q", tc.cmd)
				}
				if res.DetectedTool != tc.wantTool {
					t.Errorf("DetectedTool = %q, want %q", res.DetectedTool, tc.wantTool)
				}
				if res.TargetPath != tc.wantTarget {
					t.Errorf("TargetPath = %q, want %q", res.TargetPath, tc.wantTarget)
				}
				if !res.Blocked {
					t.Errorf("expected Blocked true in strict mode for %q", tc.cmd)
				}
			})
		}
	})
}

package adjudicator

import (
	"testing"
)

func TestHeredocTarget(t *testing.T) {
	t.Run("basic forms", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantPath   string
			wantAppend bool
			wantDelim  string
			wantKnown  bool
		}{
			{
				name:       "standard heredoc with single quote delim",
				cmd:        "cat << 'EOF' > file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "heredoc append with double quote delim",
				cmd:        `cat << "EOF" >> file.txt`,
				wantPath:   "file.txt",
				wantAppend: true,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "unquoted delim adjacent to operator",
				cmd:        "cat <<EOF > file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "redirect before heredoc",
				cmd:        "cat > file.txt << 'EOF'",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "append redirect before unquoted heredoc",
				cmd:        "cat >> file.txt <<EOF",
				wantPath:   "file.txt",
				wantAppend: true,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "double quoted path with spaces",
				cmd:        `cat << 'EOF' > "path with spaces/file.go"`,
				wantPath:   "path with spaces/file.go",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "single quoted path with spaces",
				cmd:        "cat << 'EOF' > 'path with spaces/file.go'",
				wantPath:   "path with spaces/file.go",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "redirect first with quoted path with spaces",
				cmd:        `cat > "path with spaces/file.go" << 'EOF'`,
				wantPath:   "path with spaces/file.go",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "append redirect first with quoted path with spaces",
				cmd:        `cat >> "path with spaces/file.go" << 'EOF'`,
				wantPath:   "path with spaces/file.go",
				wantAppend: true,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "backslash escaped spaces in path",
				cmd:        `cat << 'EOF' > path\ with\ spaces/file.go`,
				wantPath:   "path with spaces/file.go",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "no space between redirect and path",
				cmd:        "cat << 'EOF' >file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "no space between append redirect and path",
				cmd:        "cat << 'EOF' >>file.txt",
				wantPath:   "file.txt",
				wantAppend: true,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "explicit fd 1 redirect",
				cmd:        "cat 1> file.txt << 'EOF'",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "explicit fd 1 append redirect",
				cmd:        "cat 1>> file.txt << 'EOF'",
				wantPath:   "file.txt",
				wantAppend: true,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "clobber redirect",
				cmd:        "cat << 'EOF' >| file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "custom delimiter",
				cmd:        "cat << 'CUSTOM_DELIM_123' > output.log",
				wantPath:   "output.log",
				wantAppend: false,
				wantDelim:  "CUSTOM_DELIM_123",
				wantKnown:  true,
			},
			{
				name:       "backslash escaped delimiter",
				cmd:        `cat << \EOF > file.txt`,
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "tab stripping operator with single quote delim",
				cmd:        "cat <<- 'EOF' > file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
			{
				name:       "tab stripping operator with unquoted delim",
				cmd:        "cat <<-EOF > file.txt",
				wantPath:   "file.txt",
				wantAppend: false,
				wantDelim:  "EOF",
				wantKnown:  true,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				targets := ExtractHeredocTargets(tc.cmd)
				if len(targets) != 1 {
					t.Fatalf("ExtractHeredocTargets(%q) returned %d targets, want 1", tc.cmd, len(targets))
				}
				target := targets[0]
				if target.Path != tc.wantPath {
					t.Errorf("Path = %q, want %q", target.Path, tc.wantPath)
				}
				if target.Append != tc.wantAppend {
					t.Errorf("Append = %v, want %v", target.Append, tc.wantAppend)
				}
				if target.Delimiter != tc.wantDelim {
					t.Errorf("Delimiter = %q, want %q", target.Delimiter, tc.wantDelim)
				}
				if target.TreeKnown != tc.wantKnown {
					t.Errorf("TreeKnown = %v, want %v", target.TreeKnown, tc.wantKnown)
				}

				targetPath, appendMode, ok := ExtractHeredocWriteTarget(tc.cmd)
				if !ok {
					t.Fatalf("ExtractHeredocWriteTarget(%q) ok = false, want true", tc.cmd)
				}
				if targetPath != tc.wantPath {
					t.Errorf("ExtractHeredocWriteTarget Path = %q, want %q", targetPath, tc.wantPath)
				}
				if appendMode != tc.wantAppend {
					t.Errorf("ExtractHeredocWriteTarget Append = %v, want %v", appendMode, tc.wantAppend)
				}
			})
		}
	})

	t.Run("multi-line heredocs", func(t *testing.T) {
		cmd := "cat << 'EOF' > file.txt\n" +
			"This is line 1 of payload\n" +
			"This is line 2 with rm -rf / and echo > fake.txt\n" +
			"EOF\n"

		targets := ExtractHeredocTargets(cmd)
		if len(targets) != 1 {
			t.Fatalf("ExtractHeredocTargets returned %d targets, want 1", len(targets))
		}
		if targets[0].Path != "file.txt" {
			t.Errorf("Path = %q, want file.txt", targets[0].Path)
		}
		if targets[0].Append != false {
			t.Errorf("Append = %v, want false", targets[0].Append)
		}
		if targets[0].Delimiter != "EOF" {
			t.Errorf("Delimiter = %q, want EOF", targets[0].Delimiter)
		}
		if !targets[0].TreeKnown {
			t.Errorf("TreeKnown = %v, want true", targets[0].TreeKnown)
		}

		targetPath, appendMode, ok := ExtractHeredocWriteTarget(cmd)
		if !ok || targetPath != "file.txt" || appendMode != false {
			t.Errorf("ExtractHeredocWriteTarget = (%q, %v, %v), want (file.txt, false, true)", targetPath, appendMode, ok)
		}
	})

	t.Run("nested looking syntax inside payload stays inert", func(t *testing.T) {
		cmd := "cat << 'EOF' > real.txt\n" +
			"cat << 'INNER' > fake_inner.txt\n" +
			"INNER\n" +
			"EOF\n"

		targets := ExtractHeredocTargets(cmd)
		if len(targets) != 1 {
			t.Fatalf("ExtractHeredocTargets returned %d targets, want 1 (payload must be inert)", len(targets))
		}
		if targets[0].Path != "real.txt" {
			t.Errorf("Path = %q, want real.txt", targets[0].Path)
		}

		targetPath, _, ok := ExtractHeredocWriteTarget(cmd)
		if !ok || targetPath != "real.txt" {
			t.Errorf("ExtractHeredocWriteTarget = (%q, %v), want (real.txt, true)", targetPath, ok)
		}
	})

	t.Run("sequential heredocs in multi-line script", func(t *testing.T) {
		cmd := "cat << 'EOF' > first.txt\n" +
			"payload 1\n" +
			"EOF\n" +
			"cat >> second.txt << 'EOF'\n" +
			"payload 2\n" +
			"EOF\n"

		targets := ExtractHeredocTargets(cmd)
		if len(targets) != 2 {
			t.Fatalf("ExtractHeredocTargets returned %d targets, want 2", len(targets))
		}
		if targets[0].Path != "first.txt" || targets[0].Append != false {
			t.Errorf("target 0 = (%q, %v), want (first.txt, false)", targets[0].Path, targets[0].Append)
		}
		if targets[1].Path != "second.txt" || targets[1].Append != true {
			t.Errorf("target 1 = (%q, %v), want (second.txt, true)", targets[1].Path, targets[1].Append)
		}

		targetPath, appendMode, ok := ExtractHeredocWriteTarget(cmd)
		if !ok || targetPath != "first.txt" || appendMode != false {
			t.Errorf("ExtractHeredocWriteTarget = (%q, %v, %v), want (first.txt, false, true)", targetPath, appendMode, ok)
		}
	})

	t.Run("compound commands on line", func(t *testing.T) {
		tests := []struct {
			name       string
			cmd        string
			wantPath   string
			wantAppend bool
		}{
			{
				name: "preceding mkdir with &&",
				cmd: "mkdir -p dir && cat << 'EOF' > dir/file.txt\n" +
					"data\n" +
					"EOF\n",
				wantPath:   "dir/file.txt",
				wantAppend: false,
			},
			{
				name: "succeeding chmod with &&",
				cmd: "cat << 'EOF' > script.sh && chmod +x script.sh\n" +
					"#!/bin/bash\n" +
					"echo ok\n" +
					"EOF\n",
				wantPath:   "script.sh",
				wantAppend: false,
			},
			{
				name: "semicolon sequencing",
				cmd: "cd /tmp; cat > output.txt << 'EOF'\n" +
					"hello\n" +
					"EOF\n",
				wantPath:   "output.txt",
				wantAppend: false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				targets := ExtractHeredocTargets(tc.cmd)
				if len(targets) != 1 {
					t.Fatalf("ExtractHeredocTargets returned %d targets, want 1", len(targets))
				}
				if targets[0].Path != tc.wantPath || targets[0].Append != tc.wantAppend {
					t.Errorf("target = (%q, %v), want (%q, %v)", targets[0].Path, targets[0].Append, tc.wantPath, tc.wantAppend)
				}
			})
		}
	})

	t.Run("tab stripping heredoc with leading tabs", func(t *testing.T) {
		cmd := "cat <<- 'EOF' > tabbed.txt\n" +
			"\tline 1\n" +
			"\tEOF\n"

		targets := ExtractHeredocTargets(cmd)
		if len(targets) != 1 {
			t.Fatalf("ExtractHeredocTargets returned %d targets, want 1", len(targets))
		}
		if targets[0].Path != "tabbed.txt" || targets[0].Delimiter != "EOF" {
			t.Errorf("target = (%q, %q), want (tabbed.txt, EOF)", targets[0].Path, targets[0].Delimiter)
		}
	})

	t.Run("stderr redirect alongside stdout redirect", func(t *testing.T) {
		tests := []string{
			"cat << 'EOF' > out.txt 2>/dev/null",
			"cat 2>/dev/null << 'EOF' > out.txt",
			"cat << 'EOF' > out.txt 2>&1",
		}
		for _, cmd := range tests {
			targets := ExtractHeredocTargets(cmd)
			if len(targets) != 1 || targets[0].Path != "out.txt" {
				t.Errorf("ExtractHeredocTargets(%q) = %v, want out.txt", cmd, targets)
			}
		}
	})

	t.Run("cat wrappers and paths", func(t *testing.T) {
		tests := []string{
			"/bin/cat << 'EOF' > wrapped.txt",
			"/usr/bin/cat << 'EOF' > wrapped.txt",
			"sudo cat << 'EOF' > wrapped.txt",
			"sudo -u root cat << 'EOF' > wrapped.txt",
			"env FOO=bar cat << 'EOF' > wrapped.txt",
		}
		for _, cmd := range tests {
			targetPath, _, ok := ExtractHeredocWriteTarget(cmd)
			if !ok || targetPath != "wrapped.txt" {
				t.Errorf("ExtractHeredocWriteTarget(%q) = (%q, %v), want (wrapped.txt, true)", cmd, targetPath, ok)
			}
		}
	})

	t.Run("tree known evaluation", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantPath  string
			wantKnown bool
			wantOK    bool
		}{
			{
				name:      "concrete relative path",
				cmd:       "cat << 'EOF' > src/main.go",
				wantPath:  "src/main.go",
				wantKnown: true,
				wantOK:    true,
			},
			{
				name:      "quoted path with spaces",
				cmd:       `cat << 'EOF' > "path with spaces/file.go"`,
				wantPath:  "path with spaces/file.go",
				wantKnown: true,
				wantOK:    true,
			},
			{
				name:      "single quoted path with literal dollar",
				cmd:       "cat << 'EOF' > 'dir/$not_var/file.txt'",
				wantPath:  "dir/$not_var/file.txt",
				wantKnown: true,
				wantOK:    true,
			},
			{
				name:      "dynamic variable",
				cmd:       "cat << 'EOF' > $DEST",
				wantPath:  "$DEST",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "double quoted variable",
				cmd:       `cat << 'EOF' > "$DEST/file.txt"`,
				wantPath:  "$DEST/file.txt",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "braced variable",
				cmd:       "cat << 'EOF' > ${DEST}",
				wantPath:  "${DEST}",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "command substitution subshell",
				cmd:       "cat << 'EOF' > $(mktemp)",
				wantPath:  "$(mktemp)",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "command substitution backticks",
				cmd:       "cat << 'EOF' > `mktemp`",
				wantPath:  "`mktemp`",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "wildcard in path",
				cmd:       "cat << 'EOF' > dir/*.txt",
				wantPath:  "dir/*.txt",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "null sink",
				cmd:       "cat << 'EOF' > /dev/null",
				wantPath:  "/dev/null",
				wantKnown: false,
				wantOK:    false,
			},
			{
				name:      "empty quoted path",
				cmd:       `cat << 'EOF' > ""`,
				wantPath:  "",
				wantKnown: false,
				wantOK:    false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				targets := ExtractHeredocTargets(tc.cmd)
				if len(targets) != 1 {
					t.Fatalf("ExtractHeredocTargets(%q) returned %d targets, want 1", tc.cmd, len(targets))
				}
				if targets[0].Path != tc.wantPath {
					t.Errorf("Path = %q, want %q", targets[0].Path, tc.wantPath)
				}
				if targets[0].TreeKnown != tc.wantKnown {
					t.Errorf("TreeKnown = %v, want %v", targets[0].TreeKnown, tc.wantKnown)
				}

				targetPath, _, ok := ExtractHeredocWriteTarget(tc.cmd)
				if ok != tc.wantOK {
					t.Errorf("ExtractHeredocWriteTarget ok = %v, want %v", ok, tc.wantOK)
				}
				if tc.wantOK && targetPath != tc.wantPath {
					t.Errorf("ExtractHeredocWriteTarget targetPath = %q, want %q", targetPath, tc.wantPath)
				}
			})
		}
	})

	t.Run("negative cases non-heredoc or non-cat", func(t *testing.T) {
		negativeCmds := []string{
			"ls -la",
			"cat file.txt",
			"echo 'hello' > file.txt",
			"cat << 'EOF'",
			"sh << 'EOF' > file.txt",
			"python3 << 'EOF' > file.txt",
			`echo "cat << 'EOF' > file.txt"`,
			"# cat << 'EOF' > file.txt",
			"cat << 'EOF' | grep foo",
			"cat << 'EOF' | sh",
		}

		for _, cmd := range negativeCmds {
			targets := ExtractHeredocTargets(cmd)
			if len(targets) != 0 {
				t.Errorf("ExtractHeredocTargets(%q) = %v, want empty", cmd, targets)
			}
			targetPath, _, ok := ExtractHeredocWriteTarget(cmd)
			if ok {
				t.Errorf("ExtractHeredocWriteTarget(%q) ok = true (target %q), want false", cmd, targetPath)
			}
		}
	})

	t.Run("comments and whitespace", func(t *testing.T) {
		cmd := "# A preparatory comment\n" +
			"\n" +
			"   cat << 'EOF' > indented.txt\n" +
			"hello world\n" +
			"EOF\n"

		targets := ExtractHeredocTargets(cmd)
		if len(targets) != 1 || targets[0].Path != "indented.txt" {
			t.Fatalf("ExtractHeredocTargets failed on indented command: %v", targets)
		}
	})
}

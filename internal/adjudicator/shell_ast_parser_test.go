package adjudicator

import (
	"testing"
)

func TestShellParse(t *testing.T) {
	t.Run("Redirections", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "simple stdout overwrite",
				cmd:  "echo hello > out.txt",
				wantTargs: []BashWriteTarget{
					{Path: "out.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "simple stdout append",
				cmd:  "echo hello >> append.log",
				wantTargs: []BashWriteTarget{
					{Path: "append.log", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "explicit fd 1 overwrite",
				cmd:  "cmd 1> out1.txt",
				wantTargs: []BashWriteTarget{
					{Path: "out1.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "explicit fd 1 append",
				cmd:  "cmd 1>> append1.log",
				wantTargs: []BashWriteTarget{
					{Path: "append1.log", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "attached redirect without space",
				cmd:  "echo foo>out.txt",
				wantTargs: []BashWriteTarget{
					{Path: "out.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "attached fd 1 append without space",
				cmd:  "echo foo 1>>append.log",
				wantTargs: []BashWriteTarget{
					{Path: "append.log", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "clobber redirect >|",
				cmd:  "echo foo >| clobber.txt",
				wantTargs: []BashWriteTarget{
					{Path: "clobber.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "quoted target with spaces",
				cmd:  `echo "hello" > "path with spaces/file.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "path with spaces/file.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "single quoted target with spaces",
				cmd:  "echo 'hello' > 'dir/another file.txt'",
				wantTargs: []BashWriteTarget{
					{Path: "dir/another file.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "multiple redirects in single command",
				cmd:  "echo foo > file1.txt >> file2.txt",
				wantTargs: []BashWriteTarget{
					{Path: "file1.txt", Op: "write", TreeKnown: true},
					{Path: "file2.txt", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "dynamic target variable",
				cmd:  "echo foo > $DYNAMIC_PATH",
				wantTargs: []BashWriteTarget{
					{Path: "$DYNAMIC_PATH", Op: "write", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name:      "null device sink excluded",
				cmd:       "echo foo > /dev/null",
				wantTargs: []BashWriteTarget{},
				wantKnown: true,
			},
			{
				name:      "null device sink with fd dup 2>&1",
				cmd:       "cmd > /dev/null 2>&1",
				wantTargs: []BashWriteTarget{},
				wantKnown: true,
			},
			{
				name: "bare stdout redirect without leading command",
				cmd:  "> empty.txt",
				wantTargs: []BashWriteTarget{
					{Path: "empty.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("PipelinesWithTee", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "tee overwrite",
				cmd:  "cat in.txt | tee out.txt",
				wantTargs: []BashWriteTarget{
					{Path: "out.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "tee append flag -a",
				cmd:  "echo 'log line' | tee -a log.txt",
				wantTargs: []BashWriteTarget{
					{Path: "log.txt", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "tee long append flag --append",
				cmd:  "echo 'log line' | tee --append log.txt",
				wantTargs: []BashWriteTarget{
					{Path: "log.txt", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "tee bundled flag -ia",
				cmd:  "echo 'data' | tee -ia log.txt",
				wantTargs: []BashWriteTarget{
					{Path: "log.txt", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "tee multiple files",
				cmd:  "echo test | tee file1.txt file2.txt",
				wantTargs: []BashWriteTarget{
					{Path: "file1.txt", Op: "write", TreeKnown: true},
					{Path: "file2.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "tee with dynamic destination",
				cmd:  "cat file | tee -a $OUTPUT_FILE",
				wantTargs: []BashWriteTarget{
					{Path: "$OUTPUT_FILE", Op: "append", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name:      "tee to /dev/null excluded",
				cmd:       "cat file | tee /dev/null",
				wantTargs: []BashWriteTarget{},
				wantKnown: true,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("CoreutilsWrites", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "cp src dest",
				cmd:  "cp a.txt b.txt",
				wantTargs: []BashWriteTarget{
					{Path: "b.txt", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cp recursive with flags",
				cmd:  "cp -r src/ dest_dir/",
				wantTargs: []BashWriteTarget{
					{Path: "dest_dir/", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cp multiple sources to directory",
				cmd:  "cp -f a.txt b.txt target/",
				wantTargs: []BashWriteTarget{
					{Path: "target/", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cp with target directory flag -t",
				cmd:  "cp -t dest/ a.txt b.txt",
				wantTargs: []BashWriteTarget{
					{Path: "dest/", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cp with long target directory flag",
				cmd:  "cp --target-directory=dest a.txt",
				wantTargs: []BashWriteTarget{
					{Path: "dest", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cp with dynamic destination",
				cmd:  "cp file.txt $DEST",
				wantTargs: []BashWriteTarget{
					{Path: "$DEST", Op: "copy", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name: "mv src dest",
				cmd:  "mv old.txt new.txt",
				wantTargs: []BashWriteTarget{
					{Path: "new.txt", Op: "move", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "mv with flags",
				cmd:  "mv -f old.txt new.txt",
				wantTargs: []BashWriteTarget{
					{Path: "new.txt", Op: "move", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "mv with target directory flag -t",
				cmd:  "mv -t backup/ a.txt b.txt",
				wantTargs: []BashWriteTarget{
					{Path: "backup/", Op: "move", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "mv with dynamic destination",
				cmd:  "mv old.txt $NEW_NAME",
				wantTargs: []BashWriteTarget{
					{Path: "$NEW_NAME", Op: "move", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name: "rm single file",
				cmd:  "rm file.txt",
				wantTargs: []BashWriteTarget{
					{Path: "file.txt", Op: "remove", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "rm -rf directory",
				cmd:  "rm -rf /tmp/data",
				wantTargs: []BashWriteTarget{
					{Path: "/tmp/data", Op: "remove", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "rm multiple files",
				cmd:  "rm -f a.txt b.txt c.txt",
				wantTargs: []BashWriteTarget{
					{Path: "a.txt", Op: "remove", TreeKnown: true},
					{Path: "b.txt", Op: "remove", TreeKnown: true},
					{Path: "c.txt", Op: "remove", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "rm with wildcard pattern",
				cmd:  "rm -rf /tmp/dir/*",
				wantTargs: []BashWriteTarget{
					{Path: "/tmp/dir/*", Op: "remove", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name: "rm with dynamic variable",
				cmd:  "rm -rf $SCRATCH_DIR",
				wantTargs: []BashWriteTarget{
					{Path: "$SCRATCH_DIR", Op: "remove", TreeKnown: false},
				},
				wantKnown: false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("StreamEditors", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "sed -i in-place replacement",
				cmd:  "sed -i 's/foo/bar/g' config.json",
				wantTargs: []BashWriteTarget{
					{Path: "config.json", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed -i with backup extension",
				cmd:  "sed -i.bak 's/foo/bar/g' config.json",
				wantTargs: []BashWriteTarget{
					{Path: "config.json", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed -i with empty backup extension macOS style",
				cmd:  "sed -i '' 's/foo/bar/g' config.json",
				wantTargs: []BashWriteTarget{
					{Path: "config.json", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed --in-place long flag",
				cmd:  "sed --in-place 's/foo/bar/g' config.json",
				wantTargs: []BashWriteTarget{
					{Path: "config.json", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed -i with multiple -e expressions and multiple files",
				cmd:  "sed -i -e 's/a/b/' -e 's/c/d/' file1.txt file2.txt",
				wantTargs: []BashWriteTarget{
					{Path: "file1.txt", Op: "stream_edit", TreeKnown: true},
					{Path: "file2.txt", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed -i with script file -f",
				cmd:  "sed -i -f script.sed file.txt",
				wantTargs: []BashWriteTarget{
					{Path: "file.txt", Op: "stream_edit", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name:      "sed without -i does not write in-place",
				cmd:       "sed 's/foo/bar/g' file.txt",
				wantTargs: []BashWriteTarget{},
				wantKnown: true,
			},
			{
				name: "sed without -i redirected to file",
				cmd:  "sed 's/foo/bar/g' in.txt > out.txt",
				wantTargs: []BashWriteTarget{
					{Path: "out.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sed -i with dynamic target",
				cmd:  "sed -i 's/foo/bar/g' $TARGET",
				wantTargs: []BashWriteTarget{
					{Path: "$TARGET", Op: "stream_edit", TreeKnown: false},
				},
				wantKnown: false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("HeredocRedirects", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "cat <<EOF > path overwrite",
				cmd: "cat <<EOF > data.txt\n" +
					"hello world\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "data.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cat <<EOF >> path append",
				cmd: "cat <<EOF >> audit.log\n" +
					"entry 1\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "audit.log", Op: "append", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cat <<'EOF' > path with single quoted delimiter",
				cmd: "cat <<'EOF' > script.sh\n" +
					"echo $TEST\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "script.sh", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cat <<-EOF > path with tab stripping",
				cmd: "cat <<-EOF > config.yaml\n" +
					"\tkey: value\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "config.yaml", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cat > path <<EOF redirect before delimiter",
				cmd: "cat > output.txt <<EOF\n" +
					"some text\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "output.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "cat <<EOF > path with deceptive payload body",
				cmd: "cat <<EOF > notes.txt\n" +
					"rm -rf /root\n" +
					"echo test > malicious.txt\n" +
					"cp confidential /tmp\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "notes.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "heredoc followed by coreutils command",
				cmd: "cat <<EOF > source.txt\n" +
					"first line\n" +
					"EOF\n" +
					"cp source.txt backup.txt",
				wantTargs: []BashWriteTarget{
					{Path: "source.txt", Op: "write", TreeKnown: true},
					{Path: "backup.txt", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "heredoc with dynamic target",
				cmd: "cat <<EOF > $TARGET_FILE\n" +
					"data\n" +
					"EOF",
				wantTargs: []BashWriteTarget{
					{Path: "$TARGET_FILE", Op: "write", TreeKnown: false},
				},
				wantKnown: false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("SubshellUnwrapping", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "bash -c redirect write",
				cmd:  `bash -c "echo foo > file.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "file.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "sh -c coreutils remove",
				cmd:  `sh -c "rm -rf /tmp/scratch"`,
				wantTargs: []BashWriteTarget{
					{Path: "/tmp/scratch", Op: "remove", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "/bin/bash -c cp",
				cmd:  `/bin/bash -c "cp a.txt b.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "b.txt", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "bash -lc login shell flag with multiple statements",
				cmd:  `bash -lc "echo 1 > a.txt && cp a.txt b.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "a.txt", Op: "write", TreeKnown: true},
					{Path: "b.txt", Op: "copy", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "nested subshells bash -c and sh -c",
				cmd:  `bash -c "sh -c 'tee output.txt' < in.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "output.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "compound statement with bash -c",
				cmd:  `mkdir -p dir && bash -c "echo data > dir/file.txt"`,
				wantTargs: []BashWriteTarget{
					{Path: "dir/file.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "subshell with outer and inner redirects",
				cmd:  `bash -c "echo inner > inner.txt" > outer.txt`,
				wantTargs: []BashWriteTarget{
					{Path: "inner.txt", Op: "write", TreeKnown: true},
					{Path: "outer.txt", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "bash -c with dynamic argument",
				cmd:  `bash -c "$CMD_VAR"`,
				wantTargs: []BashWriteTarget{
					{Path: "$CMD_VAR", Op: "write", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name: "bash -c with dynamic target inside",
				cmd:  `bash -c "echo test > $TARGET_PATH"`,
				wantTargs: []BashWriteTarget{
					{Path: "$TARGET_PATH", Op: "write", TreeKnown: false},
				},
				wantKnown: false,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("CompoundPipelinesAndChaining", func(t *testing.T) {
		tests := []struct {
			name      string
			cmd       string
			wantTargs []BashWriteTarget
			wantKnown bool
		}{
			{
				name: "chain of write, copy, and remove",
				cmd:  "echo foo > a.txt && cp a.txt b.txt && rm a.txt",
				wantTargs: []BashWriteTarget{
					{Path: "a.txt", Op: "write", TreeKnown: true},
					{Path: "b.txt", Op: "copy", TreeKnown: true},
					{Path: "a.txt", Op: "remove", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "parenthesized subshell with redirect",
				cmd:  "( echo foo > a.txt ; cp a.txt b.txt ) > wrapper.log",
				wantTargs: []BashWriteTarget{
					{Path: "a.txt", Op: "write", TreeKnown: true},
					{Path: "b.txt", Op: "copy", TreeKnown: true},
					{Path: "wrapper.log", Op: "write", TreeKnown: true},
				},
				wantKnown: true,
			},
			{
				name: "mixed known and unknown targets in pipeline",
				cmd:  "echo foo > a.txt; rm -rf $DYNAMIC_DIR",
				wantTargs: []BashWriteTarget{
					{Path: "a.txt", Op: "write", TreeKnown: true},
					{Path: "$DYNAMIC_DIR", Op: "remove", TreeKnown: false},
				},
				wantKnown: false,
			},
			{
				name:      "read only pipeline",
				cmd:       "cat in.txt | grep pattern | wc -l",
				wantTargs: []BashWriteTarget{},
				wantKnown: true,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				gotTargs, gotKnown, err := ParseBashWriteTargets(tc.cmd)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if gotKnown != tc.wantKnown {
					t.Errorf("treeKnown = %v, want %v", gotKnown, tc.wantKnown)
				}
				assertTargetsMatch(t, gotTargs, tc.wantTargs)
			})
		}
	})

	t.Run("ErrorsAndEdgeCases", func(t *testing.T) {
		// Empty string
		targs, known, err := ParseBashWriteTargets("")
		if err != nil || !known || len(targs) != 0 {
			t.Errorf("expected empty result, got %v, known=%v, err=%v", targs, known, err)
		}

		// Whitespace only
		targs, known, err = ParseBashWriteTargets("   \t\n  ")
		if err != nil || !known || len(targs) != 0 {
			t.Errorf("expected empty result, got %v, known=%v, err=%v", targs, known, err)
		}

		// Unterminated single quote
		_, _, err = ParseBashWriteTargets("echo 'unterminated")
		if err == nil {
			t.Errorf("expected error for unterminated quote, got nil")
		}

		// Unterminated double quote
		_, _, err = ParseBashWriteTargets(`echo "unterminated`)
		if err == nil {
			t.Errorf("expected error for unterminated quote, got nil")
		}

		// Missing -c argument
		_, _, err = ParseBashWriteTargets("bash -c")
		if err == nil {
			t.Errorf("expected error for missing -c argument, got nil")
		}
	})
}

func assertTargetsMatch(t *testing.T, got, want []BashWriteTarget) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d targets %v, want %d %v", len(got), got, len(want), want)
	}
	for i := range got {
		if got[i].Path != want[i].Path || got[i].Op != want[i].Op || got[i].TreeKnown != want[i].TreeKnown {
			t.Errorf("[%d] got %+v, want %+v", i, got[i], want[i])
		}
	}
}

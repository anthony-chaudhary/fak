package adjudicator

import (
	"testing"
)

func TestSedMutation(t *testing.T) {
	t.Run("basic sed in-place", func(t *testing.T) {
		cmd := "sed -i 's/foo/bar/g' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" {
			t.Errorf("got Tool %q, want 'sed'", m.Tool)
		}
		if !m.InPlace {
			t.Errorf("got InPlace %v, want true", m.InPlace)
		}
		if m.TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", m.TargetPath)
		}
		if m.Script != "s/foo/bar/g" {
			t.Errorf("got Script %q, want 's/foo/bar/g'", m.Script)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("sed in-place with attached backup suffix", func(t *testing.T) {
		cmd := "sed -i.bak 's/foo/bar/g' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" || !m.InPlace || m.TargetPath != "file.txt" || m.Script != "s/foo/bar/g" || !m.TreeKnown {
			t.Errorf("unexpected target: %+v", m)
		}
	})

	t.Run("sed in-place macOS BSD style empty suffix", func(t *testing.T) {
		cmd := "sed -i '' 's/foo/bar/g' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" || !m.InPlace || m.TargetPath != "file.txt" || m.Script != "s/foo/bar/g" || !m.TreeKnown {
			t.Errorf("unexpected target: %+v", m)
		}
	})

	t.Run("sed long flag --in-place", func(t *testing.T) {
		cmd := "sed --in-place 's/foo/bar/g' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" || !m.InPlace || m.TargetPath != "file.txt" || m.Script != "s/foo/bar/g" || !m.TreeKnown {
			t.Errorf("unexpected target: %+v", m)
		}
	})

	t.Run("sed multiple -e expressions", func(t *testing.T) {
		cmd := "sed -i -e 's/a/b/' -e 's/c/d/' path/to/file.go"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" {
			t.Errorf("got Tool %q, want 'sed'", m.Tool)
		}
		if !m.InPlace {
			t.Errorf("got InPlace %v, want true", m.InPlace)
		}
		if m.TargetPath != "path/to/file.go" {
			t.Errorf("got TargetPath %q, want 'path/to/file.go'", m.TargetPath)
		}
		if m.Script != "s/a/b/; s/c/d/" {
			t.Errorf("got Script %q, want 's/a/b/; s/c/d/'", m.Script)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("gawk in-place", func(t *testing.T) {
		cmd := "gawk -i inplace '{print $1}' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "gawk" {
			t.Errorf("got Tool %q, want 'gawk'", m.Tool)
		}
		if !m.InPlace {
			t.Errorf("got InPlace %v, want true", m.InPlace)
		}
		if m.TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", m.TargetPath)
		}
		if m.Script != "{print $1}" {
			t.Errorf("got Script %q, want '{print $1}'", m.Script)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("awk in-place", func(t *testing.T) {
		cmd := "awk -i inplace 'BEGIN{...}' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "awk" {
			t.Errorf("got Tool %q, want 'awk'", m.Tool)
		}
		if !m.InPlace {
			t.Errorf("got InPlace %v, want true", m.InPlace)
		}
		if m.TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", m.TargetPath)
		}
		if m.Script != "BEGIN{...}" {
			t.Errorf("got Script %q, want 'BEGIN{...}'", m.Script)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("quoted target path with spaces", func(t *testing.T) {
		cmd := `sed -i 's/foo/bar/g' "path with spaces/file.go"`
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.TargetPath != "path with spaces/file.go" {
			t.Errorf("got TargetPath %q, want 'path with spaces/file.go'", m.TargetPath)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("multiple target files", func(t *testing.T) {
		cmd := "sed -i 's/a/b/' f1.txt f2.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 2 {
			t.Fatalf("expected 2 targets, got %d", len(muts))
		}
		if muts[0].TargetPath != "f1.txt" || !muts[0].InPlace || muts[0].Script != "s/a/b/" {
			t.Errorf("unexpected target 0: %+v", muts[0])
		}
		if muts[1].TargetPath != "f2.txt" || !muts[1].InPlace || muts[1].Script != "s/a/b/" {
			t.Errorf("unexpected target 1: %+v", muts[1])
		}
	})

	t.Run("distinguishes non-in-place read-only sed", func(t *testing.T) {
		cmd := "sed 's/a/b/' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "sed" {
			t.Errorf("got Tool %q, want 'sed'", m.Tool)
		}
		if m.InPlace {
			t.Errorf("got InPlace true, want false (read-only)")
		}
		if m.TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", m.TargetPath)
		}
		if m.Script != "s/a/b/" {
			t.Errorf("got Script %q, want 's/a/b/'", m.Script)
		}
		if !m.TreeKnown {
			t.Errorf("got TreeKnown %v, want true", m.TreeKnown)
		}
	})

	t.Run("distinguishes non-in-place read-only awk", func(t *testing.T) {
		cmd := "awk '{print $1}' file.txt"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		m := muts[0]
		if m.Tool != "awk" {
			t.Errorf("got Tool %q, want 'awk'", m.Tool)
		}
		if m.InPlace {
			t.Errorf("got InPlace true, want false")
		}
		if m.TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", m.TargetPath)
		}
	})

	t.Run("dynamic expressions evaluate TreeKnown false", func(t *testing.T) {
		dynamicCases := []struct {
			cmd        string
			targetPath string
		}{
			{"sed -i 's/a/b/' $FILE", "$FILE"},
			{`sed -i 's/a/b/' "$TARGET"`, "$TARGET"},
			{"sed -i 's/a/b/' ${DEST_FILE}", "${DEST_FILE}"},
			{"sed -i 's/a/b/' *.go", "*.go"},
			{"sed -i 's/a/b/' src/*.txt", "src/*.txt"},
			{"sed -i 's/a/b/' file?.txt", "file?.txt"},
			{"sed -i 's/a/b/' file{1,2}.txt", "file{1,2}.txt"},
			{"sed -i 's/a/b/' $(find . -name '*.go')", "$(find . -name '*.go')"},
			{"gawk -i inplace '{print}' `cat list.txt`", "`cat list.txt`"},
		}

		for _, tc := range dynamicCases {
			muts, err := ExtractSedAwkMutations(tc.cmd)
			if err != nil {
				t.Fatalf("%q failed: %v", tc.cmd, err)
			}
			if len(muts) != 1 {
				t.Fatalf("%q expected 1 target, got %d", tc.cmd, len(muts))
			}
			if muts[0].TreeKnown {
				t.Errorf("%q got TreeKnown true, want false for dynamic path %q", tc.cmd, muts[0].TargetPath)
			}
			if muts[0].TargetPath != tc.targetPath {
				t.Errorf("%q got TargetPath %q, want %q", tc.cmd, muts[0].TargetPath, tc.targetPath)
			}
		}
	})

	t.Run("lane tree classification", func(t *testing.T) {
		globs := []string{"internal/abi/", "internal/gateway/", "cmd/**"}
		cmd := "sed -i 's/a/b/' internal/abi/kernel.go && awk -i inplace '{print}' cmd/fak/main.go"
		muts, err := ExtractSedAwkMutationsWithTrees(cmd, globs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 2 {
			t.Fatalf("expected 2 targets, got %d", len(muts))
		}
		if muts[0].DeclaredTree != "internal/abi/" {
			t.Errorf("got DeclaredTree %q, want 'internal/abi/'", muts[0].DeclaredTree)
		}
		if muts[1].DeclaredTree != "cmd/**" {
			t.Errorf("got DeclaredTree %q, want 'cmd/**'", muts[1].DeclaredTree)
		}

		// Dynamic target has empty DeclaredTree even if matching substring
		dynCmd := "sed -i 's/a/b/' internal/abi/$FILE"
		dynMuts, err := ExtractSedAwkMutationsWithTrees(dynCmd, globs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(dynMuts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(dynMuts))
		}
		if dynMuts[0].DeclaredTree != "" {
			t.Errorf("expected empty DeclaredTree for dynamic expression, got %q", dynMuts[0].DeclaredTree)
		}
	})

	t.Run("redirection filtering ignores sinks", func(t *testing.T) {
		cmd := "sed -i 's/a/b/' file.txt > /dev/null 2>&1"
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		if muts[0].TargetPath != "file.txt" {
			t.Errorf("got TargetPath %q, want 'file.txt'", muts[0].TargetPath)
		}
	})

	t.Run("nested subshell execution", func(t *testing.T) {
		cmd := `bash -c "sed -i 's/a/b/' file.go"`
		muts, err := ExtractSedAwkMutations(cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(muts) != 1 {
			t.Fatalf("expected 1 target, got %d", len(muts))
		}
		if muts[0].TargetPath != "file.go" || !muts[0].InPlace {
			t.Errorf("unexpected target: %+v", muts[0])
		}
	})

	t.Run("unterminated quote error", func(t *testing.T) {
		cmd := "sed -i 's/foo/bar file.txt"
		_, err := ExtractSedAwkMutations(cmd)
		if err == nil {
			t.Error("expected error for unterminated quote, got nil")
		}
	})
}

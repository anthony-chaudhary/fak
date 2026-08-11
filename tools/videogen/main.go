// Command video is the single entry point for the repository's shared
// explainer-video renderer.
//
// The root module deliberately has zero external dependencies. The terminal
// renderer needs x/image for its bitmap and antialiased font paths, so it lives
// in the nested tools/videogen/terminal module. This small stdlib-only runner
// preserves the repo-root `go run ./tools/videogen ...` interface and hides
// that module boundary.
package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

var projectName = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "video:", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	root, err := repoRoot()
	if err != nil {
		return err
	}
	if len(args) == 0 || (len(args) == 1 && (args[0] == "-h" || args[0] == "--help")) {
		fmt.Fprint(stdout, usage)
		return nil
	}
	if len(args) == 2 && args[0] == "-new" {
		dst, err := newProject(root, args[1])
		if err != nil {
			return err
		}
		fmt.Fprintf(stdout, "created %s\n", dst)
		fmt.Fprintf(stdout, "next: go run ./tools/videogen -config %s/render.json -verify\n", dst)
		return nil
	}
	if len(args) > 0 && args[0] == "-new" {
		return errors.New("-new needs exactly one lowercase project name")
	}

	callerDir, err := os.Getwd()
	if err != nil {
		return err
	}
	args, ffmpeg, err := extractFFmpegFlag(args, callerDir)
	if err != nil {
		return err
	}
	args, err = absolutePathFlags(args, callerDir)
	if err != nil {
		return err
	}
	moduleDir, err := rendererModule(root)
	if err != nil {
		return err
	}
	cmd := exec.Command("go", append([]string{"run", "."}, args...)...)
	cmd.Dir = moduleDir
	configureDispatchSpawn(cmd)
	if ffmpeg != "" {
		cmd.Env = append(os.Environ(), "VIDEOGEN_FFMPEG="+ffmpeg)
	}
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("terminal renderer: %w", err)
	}
	return nil
}

const usage = `video renders paced terminal captures into reusable explainers.

Usage:
  go run ./tools/videogen -new NAME
  go run ./tools/videogen -config FILE -verify
  go run ./tools/videogen -config FILE -all [-ffmpeg PATH]
  command 2>&1 | go run ./tools/videogen \
    -record-typescript FILE.typescript -record-timing FILE.timing

Modes:
  -new NAME              copy the checked terminal template to projects/NAME
  -verify                plan and verify pacing; rasterise nothing
  -all                   write GIF, hi-res frames, chapters, and MP4; verify all
  -record-typescript P   tee stdin to a script(1)-compatible capture
  -record-timing P       timing companion; required with -record-typescript

Render inputs and outputs:
  -config FILE           JSON story, capture paths, pacing rules, and gates
  -timeline FILE         frame-by-frame audit log (default timeline.json)
  -frames DIR            hi-res PNG sequence and ffconcat playlist
  -png DIR               optional bitmap debug frames
  -cell-w N              hi-res terminal cell width
  -cell-h N              hi-res terminal cell height

Paths supplied on the command line are relative to the caller. Paths inside a
render config are relative to that config, so a project is portable.
`

func rendererModule(root string) (string, error) {
	shared := filepath.Join(root, "tools", "videogen", "terminal")
	if st, statErr := os.Stat(filepath.Join(shared, "go.mod")); statErr == nil && !st.IsDir() {
		return shared, nil
	}
	return "", fmt.Errorf("shared terminal renderer is missing from %s", shared)
}

func repoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		raw, readErr := os.ReadFile(filepath.Join(dir, "go.mod"))
		if readErr == nil && strings.Contains(string(raw), "module github.com/anthony-chaudhary/fak\n") {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("run from inside the fak checkout")
		}
		dir = parent
	}
}

func extractFFmpegFlag(args []string, callerDir string) ([]string, string, error) {
	out := make([]string, 0, len(args))
	var ffmpeg string
	for i := 0; i < len(args); i++ {
		value := ""
		switch {
		case args[i] == "-ffmpeg":
			if i+1 >= len(args) {
				return nil, "", errors.New("-ffmpeg needs a path")
			}
			i++
			value = args[i]
		case strings.HasPrefix(args[i], "-ffmpeg="):
			value = strings.TrimPrefix(args[i], "-ffmpeg=")
		default:
			out = append(out, args[i])
			continue
		}
		if value == "" {
			return nil, "", errors.New("-ffmpeg needs a path")
		}
		if !filepath.IsAbs(value) {
			value = filepath.Join(callerDir, value)
		}
		ffmpeg = value
	}
	return out, ffmpeg, nil
}

var pathFlags = map[string]bool{
	"-config":            true,
	"-frames":            true,
	"-png":               true,
	"-record-timing":     true,
	"-record-typescript": true,
	"-timeline":          true,
}

// absolutePathFlags preserves the caller's path semantics when the runner
// changes cwd into the renderer's nested module. Both "-flag value" and
// "-flag=value" spellings are supported.
func absolutePathFlags(args []string, callerDir string) ([]string, error) {
	out := append([]string(nil), args...)
	abs := func(value string) string {
		if value == "" || filepath.IsAbs(value) {
			return value
		}
		return filepath.Join(callerDir, value)
	}
	for i := 0; i < len(out); i++ {
		if pathFlags[out[i]] {
			if i+1 >= len(out) {
				return nil, fmt.Errorf("%s needs a path", out[i])
			}
			out[i+1] = abs(out[i+1])
			i++
			continue
		}
		for flag := range pathFlags {
			prefix := flag + "="
			if strings.HasPrefix(out[i], prefix) {
				out[i] = prefix + abs(strings.TrimPrefix(out[i], prefix))
				break
			}
		}
	}
	return out, nil
}

func newProject(root, name string) (string, error) {
	if !projectName.MatchString(name) {
		return "", fmt.Errorf("invalid project name %q: use lowercase letters, digits, and hyphens", name)
	}
	src := filepath.Join(root, "tools", "videogen", "templates", "terminal")
	dst := filepath.Join(root, "tools", "videogen", "projects", name)
	if _, err := os.Stat(dst); err == nil {
		return "", fmt.Errorf("%s already exists", dst)
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	projects := filepath.Dir(dst)
	if err := os.MkdirAll(projects, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.MkdirTemp(projects, "."+name+"-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)
	if err := copyTree(src, tmp); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, dst); err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, dst)
	if err != nil {
		return "", err
	}
	return filepath.ToSlash(rel), nil
}

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, raw, 0o644)
	})
}

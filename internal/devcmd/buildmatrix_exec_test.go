package devcmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
)

// fak-test:runtime fast est=1s
func TestBuildMatrixCompileProcess(t *testing.T) {
	type observation struct {
		Program   string
		Args      []string
		Dir       string
		Inherited string
		Override  string
	}
	const stderrLine = "build-matrix child stderr\n"
	if os.Getenv("FAK_BUILDMATRIX_TEST_PROCESS") == "1" {
		dir, err := os.Getwd()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		args := os.Args[1:]
		for i, arg := range args {
			if arg == "--" {
				args = args[i+1:]
				break
			}
		}
		if err := json.NewEncoder(os.Stdout).Encode(observation{
			Program: os.Args[0], Args: args, Dir: dir,
			Inherited: os.Getenv("FAK_BUILDMATRIX_TEST_INHERITED"),
			Override:  os.Getenv("FAK_BUILDMATRIX_TEST_OVERRIDE"),
		}); err != nil {
			os.Exit(1)
		}
		fmt.Fprint(os.Stderr, stderrLine)
		code, err := strconv.Atoi(os.Getenv("FAK_BUILDMATRIX_TEST_EXIT"))
		if err != nil {
			os.Exit(1)
		}
		os.Exit(code)
	}

	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_BUILDMATRIX_TEST_INHERITED", "parent value")
	t.Setenv("FAK_BUILDMATRIX_TEST_OVERRIDE", "parent value")
	for _, tc := range []struct {
		name string
		code int
	}{
		{name: "success", code: 0},
		{name: "exit_error", code: 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			wantArgs := []string{"alpha", "two words"}
			plan := buildwitness.MatrixCompilePlan{
				Command: append([]string{"-test.run=^TestBuildMatrixCompileProcess$", "--"}, wantArgs...),
				Env: []string{
					"FAK_BUILDMATRIX_TEST_PROCESS=1",
					"FAK_BUILDMATRIX_TEST_OVERRIDE=plan value",
					"FAK_BUILDMATRIX_TEST_EXIT=" + strconv.Itoa(tc.code),
				},
			}
			out, err := runBuildMatrixCompileCheck(executable, root, plan)
			if tc.code == 0 {
				if err != nil {
					t.Fatalf("selected executable: %v\n%s", err, out)
				}
			} else {
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != tc.code {
					t.Fatalf("exit error=%v, want code %d; output=%q", err, tc.code, out)
				}
			}
			if !strings.HasSuffix(out, stderrLine) {
				t.Fatalf("combined output lost stderr: %q", out)
			}
			var got observation
			if err := json.Unmarshal([]byte(strings.TrimSuffix(out, stderrLine)), &got); err != nil {
				t.Fatalf("child observation: %v; output=%q", err, out)
			}
			if got.Program != executable || !reflect.DeepEqual(got.Args, wantArgs) {
				t.Fatalf("selected program/argv: %+v, want %q %q", got, executable, wantArgs)
			}
			wantDir, err := os.Stat(root)
			if err != nil {
				t.Fatal(err)
			}
			gotDir, err := os.Stat(got.Dir)
			if err != nil || !os.SameFile(wantDir, gotDir) {
				t.Fatalf("child directory=%q, want %q: %v", got.Dir, root, err)
			}
			if got.Inherited != "parent value" || got.Override != "plan value" {
				t.Fatalf("inherited/overridden environment: %+v", got)
			}
		})
	}
	t.Run("missing_path_tool", func(t *testing.T) {
		t.Chdir(t.TempDir())
		t.Setenv("PATH", "")
		out, err := runBuildMatrixCompileCheck("", ".", buildwitness.MatrixCompilePlan{})
		if out != "" || !errors.Is(err, exec.ErrNotFound) || !strings.HasPrefix(err.Error(), "go toolchain not on PATH: ") {
			t.Fatalf("missing tool output=%q error=%v", out, err)
		}
	})
}

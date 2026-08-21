package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const sessionNewUsage = `usage: fak session new [flags] <prompt>

Launch a new interactive agent terminal through fak guard.

Prompt source (choose one):
  <prompt>        one positional prompt argument
  --stdin         read the prompt from stdin
  --clipboard     read the prompt from the platform clipboard

Flags:
  --agent NAME       child agent executable (default: claude)
  --terminal NAME    auto, windows-terminal, x-terminal, or gnome-terminal
  --terminal-command PATH  override the adapter executable
  --cwd DIR          child working directory (default: current directory)
  --dry-run          print the launch receipt without starting a terminal
  --json             print a JSON receipt

Examples:
  fak session new "inspect this function"
  Get-Clipboard | fak session new --stdin
  fak session new --clipboard
`

type sessionNewReceipt struct {
	Schema       string   `json:"schema"`
	Status       string   `json:"status"`
	Source       string   `json:"source"`
	PromptSHA256 string   `json:"prompt_sha256"`
	PromptBytes  int      `json:"prompt_bytes"`
	Agent        string   `json:"agent"`
	Terminal     string   `json:"terminal"`
	WorkingDir   string   `json:"working_directory"`
	Executable   string   `json:"executable"`
	Arguments    []string `json:"arguments"`
}

type sessionNewDeps struct {
	goos       string
	stdin      io.Reader
	lookPath   func(string) (string, error)
	executable func() (string, error)
	getwd      func() (string, error)
	readClip   func(string) (string, error)
	start      func(string, []string, string) error
}

func defaultSessionNewDeps() sessionNewDeps {
	return sessionNewDeps{
		goos: runtime.GOOS, stdin: os.Stdin, lookPath: exec.LookPath,
		executable: os.Executable, getwd: os.Getwd, readClip: readSessionClipboard,
		start: func(name string, args []string, cwd string) error {
			cmd := exec.Command(name, args...)
			cmd.Dir = cwd
			if err := cmd.Start(); err != nil {
				return err
			}
			return cmd.Process.Release()
		},
	}
}

func runSessionNew(stdout, stderr io.Writer, argv []string) int {
	return runSessionNewWith(stdout, stderr, argv, defaultSessionNewDeps())
}

func runSessionNewWith(stdout, stderr io.Writer, argv []string, deps sessionNewDeps) int {
	fs := flag.NewFlagSet("session new", flag.ContinueOnError)
	fs.SetOutput(stderr)
	useStdin := fs.Bool("stdin", false, "read prompt from stdin")
	useClipboard := fs.Bool("clipboard", false, "read prompt from clipboard")
	agent := fs.String("agent", "claude", "agent executable")
	terminal := fs.String("terminal", "auto", "terminal adapter")
	terminalCommand := fs.String("terminal-command", "", "terminal executable override")
	cwd := fs.String("cwd", "", "child working directory")
	dryRun := fs.Bool("dry-run", false, "print without launching")
	jsonOut := fs.Bool("json", false, "emit JSON receipt")
	fs.Usage = func() { fmt.Fprint(stderr, sessionNewUsage) }
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	sourceCount := 0
	if *useStdin {
		sourceCount++
	}
	if *useClipboard {
		sourceCount++
	}
	if fs.NArg() > 0 {
		sourceCount++
	}
	if sourceCount != 1 || fs.NArg() > 1 {
		fmt.Fprintln(stderr, "session new: choose exactly one prompt source: one positional argument, --stdin, or --clipboard")
		return 2
	}
	if strings.TrimSpace(*agent) == "" {
		fmt.Fprintln(stderr, "session new: --agent must not be empty")
		return 2
	}

	source, prompt, err := sessionNewPrompt(fs.Args(), *useStdin, *useClipboard, deps)
	if err != nil {
		fmt.Fprintf(stderr, "session new: %v\n", err)
		return 1
	}
	// A pipeline normally contributes one record-separator newline; it is not prompt content.
	if source == "stdin" {
		prompt = strings.TrimSuffix(prompt, "\n")
		prompt = strings.TrimSuffix(prompt, "\r")
	}
	if strings.TrimSpace(prompt) == "" {
		fmt.Fprintln(stderr, "session new: prompt is empty")
		return 2
	}

	workdir := *cwd
	if workdir == "" {
		workdir, err = deps.getwd()
		if err != nil {
			fmt.Fprintf(stderr, "session new: determine working directory: %v\n", err)
			return 1
		}
	}
	workdir, err = filepath.Abs(workdir)
	if err != nil {
		fmt.Fprintf(stderr, "session new: resolve working directory: %v\n", err)
		return 1
	}
	fakBin, err := deps.executable()
	if err != nil {
		fmt.Fprintf(stderr, "session new: locate fak executable: %v\n", err)
		return 1
	}
	terminalName, terminalBin, terminalArgs, err := buildSessionTerminal(*terminal, *terminalCommand, deps.goos, deps.lookPath, workdir, fakBin, *agent, prompt)
	if err != nil {
		fmt.Fprintf(stderr, "session new: %v\n", err)
		return 1
	}

	sum := sha256.Sum256([]byte(prompt))
	receipt := sessionNewReceipt{Schema: "fak-session-new/1", Status: "planned", Source: source,
		PromptSHA256: hex.EncodeToString(sum[:]), PromptBytes: len([]byte(prompt)), Agent: *agent,
		Terminal: terminalName, WorkingDir: workdir, Executable: terminalBin, Arguments: redactSessionNewArgs(terminalArgs, prompt, receiptPromptMarker(sum))}
	if !*dryRun {
		if err := deps.start(terminalBin, terminalArgs, workdir); err != nil {
			fmt.Fprintf(stderr, "session new: start %s: %v\n", terminalName, err)
			return 1
		}
		receipt.Status = "launched"
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "session new: encode receipt: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "%s: new guarded %s session via %s (prompt sha256:%s)\n", strings.ToUpper(receipt.Status), receipt.Agent, receipt.Terminal, receipt.PromptSHA256[:12])
	}
	return 0
}

func receiptPromptMarker(sum [sha256.Size]byte) string {
	return "<prompt:sha256:" + hex.EncodeToString(sum[:6]) + ">"
}

func redactSessionNewArgs(args []string, prompt, marker string) []string {
	out := append([]string(nil), args...)
	for i := range out {
		if out[i] == prompt {
			out[i] = marker
		}
	}
	return out
}

func sessionNewPrompt(args []string, stdin, clipboard bool, deps sessionNewDeps) (string, string, error) {
	switch {
	case stdin:
		b, err := io.ReadAll(deps.stdin)
		if err != nil {
			return "", "", fmt.Errorf("read stdin: %w", err)
		}
		return "stdin", string(b), nil
	case clipboard:
		s, err := deps.readClip(deps.goos)
		if err != nil {
			return "", "", fmt.Errorf("read clipboard: %w; copy text again or provide prompt text directly", err)
		}
		return "clipboard", s, nil
	default:
		return "argument", args[0], nil
	}
}

func buildSessionTerminal(requested, commandOverride, goos string, lookPath func(string) (string, error), cwd, fakBin, agent, prompt string) (string, string, []string, error) {
	candidates := []string{requested}
	if requested == "auto" {
		switch goos {
		case "windows":
			candidates = []string{"windows-terminal"}
		case "linux":
			candidates = []string{"x-terminal", "gnome-terminal"}
		default:
			return "", "", nil, fmt.Errorf("no automatic terminal adapter for %s; use a supported --terminal", goos)
		}
	}
	var misses []string
	for _, adapter := range candidates {
		var command string
		var args []string
		switch adapter {
		case "windows-terminal":
			command = "wt.exe"
			args = []string{"new-tab", "--startingDirectory", cwd, fakBin, "guard", "--", agent, prompt}
		case "x-terminal":
			command = "x-terminal-emulator"
			args = []string{"-e", fakBin, "guard", "--", agent, prompt}
		case "gnome-terminal":
			command = "gnome-terminal"
			args = []string{"--working-directory", cwd, "--", fakBin, "guard", "--", agent, prompt}
		default:
			return "", "", nil, fmt.Errorf("unsupported terminal adapter %q", adapter)
		}
		if commandOverride != "" {
			return adapter, commandOverride, args, nil
		}
		bin, err := lookPath(command)
		if err == nil {
			return adapter, bin, args, nil
		}
		misses = append(misses, command)
	}
	return "", "", nil, fmt.Errorf("no supported terminal found (%s)", strings.Join(misses, ", "))
}

func readSessionClipboard(goos string) (string, error) {
	var cmd *exec.Cmd
	switch goos {
	case "windows":
		cmd = exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw")
	case "darwin":
		cmd = exec.Command("pbpaste")
	case "linux":
		if p, err := exec.LookPath("wl-paste"); err == nil {
			cmd = exec.Command(p, "--no-newline")
		} else if p, err := exec.LookPath("xclip"); err == nil {
			cmd = exec.Command(p, "-selection", "clipboard", "-out")
		} else {
			return "", errors.New("clipboard unavailable: install wl-paste or xclip")
		}
	default:
		return "", fmt.Errorf("clipboard unsupported on %s", goos)
	}
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

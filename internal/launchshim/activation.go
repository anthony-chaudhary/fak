package launchshim

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	activationBegin = "# >>> fak launch PATH >>>"
	activationEnd   = "# <<< fak launch PATH <<<"
)

type ShellProfile struct {
	Shell string
	Path  string
}

// Profiles returns supported startup files beneath home. Existing files are selected;
// when none exist, the platform's default profile is returned for creation.
func Profiles(home, goos string) []ShellProfile {
	if goos == "windows" {
		return []ShellProfile{{Shell: "powershell", Path: filepath.Join(home, "Documents", "PowerShell", "Microsoft.PowerShell_profile.ps1")}}
	}
	candidates := []ShellProfile{{"bash", filepath.Join(home, ".bashrc")}, {"zsh", filepath.Join(home, ".zshrc")}, {"fish", filepath.Join(home, ".config", "fish", "config.fish")}}
	var existing []ShellProfile
	for _, p := range candidates {
		if _, err := os.Stat(p.Path); err == nil {
			existing = append(existing, p)
		}
	}
	if len(existing) != 0 {
		return existing
	}
	return candidates[:1]
}

func ActivationBlock(shell, dir string) string {
	quoted := shellQuote(dir)
	line := `export PATH=` + quoted + `:"$PATH"`
	if shell == "powershell" {
		line = `$env:PATH = ` + psQuote(dir+string(os.PathListSeparator)) + ` + $env:PATH`
	}
	if shell == "fish" {
		line = `fish_add_path --prepend ` + quoted
	}
	return activationBegin + "\n" + line + "\n" + activationEnd + "\n"
}

func Activate(profile ShellProfile, dir string) (bool, error) {
	raw, err := os.ReadFile(profile.Path)
	if err != nil && !os.IsNotExist(err) {
		return false, err
	}
	cleaned := removeActivation(raw)
	block := []byte(ActivationBlock(profile.Shell, dir))
	if len(cleaned) > 0 && cleaned[len(cleaned)-1] != '\n' {
		cleaned = append(cleaned, '\n')
	}
	want := append(cleaned, block...)
	if bytes.Equal(raw, want) {
		return false, nil
	}
	if err := os.MkdirAll(filepath.Dir(profile.Path), 0o755); err != nil {
		return false, err
	}
	if err := os.WriteFile(profile.Path, want, 0o600); err != nil {
		return false, err
	}
	return true, nil
}

func Deactivate(profile ShellProfile) (bool, error) {
	raw, err := os.ReadFile(profile.Path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	want := removeActivation(raw)
	if bytes.Equal(raw, want) {
		return false, nil
	}
	return true, os.WriteFile(profile.Path, want, 0o600)
}

func CurrentShellCommand(shell, dir string) string {
	if shell == "powershell" {
		return `$env:PATH = ` + psQuote(dir+string(os.PathListSeparator)) + ` + $env:PATH`
	}
	if shell == "fish" {
		return `fish_add_path --prepend ` + shellQuote(dir)
	}
	return `export PATH=` + shellQuote(dir) + `:"$PATH"`
}

func removeActivation(raw []byte) []byte {
	text := string(raw)
	for {
		start := strings.Index(text, activationBegin)
		if start < 0 {
			break
		}
		endRel := strings.Index(text[start:], activationEnd)
		if endRel < 0 {
			break
		}
		end := start + endRel + len(activationEnd)
		if end < len(text) && text[end] == '\r' {
			end++
		}
		if end < len(text) && text[end] == '\n' {
			end++
		}
		text = text[:start] + text[end:]
	}
	return []byte(text)
}

func shellQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'" }
func psQuote(s string) string    { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }

func ProfileSummary(p ShellProfile) string { return fmt.Sprintf("%s:%s", p.Shell, p.Path) }

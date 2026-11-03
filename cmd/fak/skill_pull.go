package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/skilldiscovery"
)

type skillPuller func(context.Context, skilldiscovery.Options) (skilldiscovery.Receipt, error)

// runSkillPull is an explicit operator acquisition, not agent startup discovery.
// The returned root must be deliberately selected with existing --skills-dir.
func runSkillPull(out, errw io.Writer, args []string) int {
	return runSkillPullWith(out, errw, args, skilldiscovery.Pull)
}
func runSkillPullWith(out, errw io.Writer, args []string, pull skillPuller) int {
	fs := flag.NewFlagSet("fak skill pull", flag.ContinueOnError)
	// Flag parse errors can echo caller-supplied values; keep diagnostics redacted.
	fs.SetOutput(io.Discard)
	source := fs.String("source", "", "explicit HTTPS catalog directory URL")
	pin := fs.String("index-sha256", "", "independently trusted SHA256 of exact index.json bytes")
	cache := fs.String("cache-root", "", "existing canonical directory for private immutable cache generations")
	asJSON := fs.Bool("json", false, "emit activation-neutral receipt")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintln(errw, "fak skill pull: invalid arguments")
		return 2
	}
	if fs.NArg() != 0 || *source == "" || *pin == "" || *cache == "" {
		fmt.Fprintln(errw, "fak skill pull: --source, --index-sha256 and --cache-root are required; no positional arguments")
		return 2
	}
	result, err := pull(context.Background(), skilldiscovery.Options{
		Source: *source, IndexSHA256: *pin, CacheRoot: *cache, ValidateSkill: validatePulledSkill,
	})
	if err != nil {
		// Only the package's redacted reason is exposed. Validators do not leak
		// remote frontmatter or credentials through their parser error messages.
		if typed, ok := err.(*skilldiscovery.Error); ok {
			fmt.Fprintln(errw, typed.Error())
		} else {
			fmt.Fprintln(errw, "fak skill pull: acquisition failed; no activation requested")
		}
		return 1
	}
	if *asJSON {
		if err := json.NewEncoder(out).Encode(result); err != nil {
			fmt.Fprintln(errw, "fak skill pull: receipt encoding failed")
			return 1
		}
		return 0
	}
	fmt.Fprintf(out, "skill data cached: %s\nindex SHA256: %s\nactivated: false\n", result.SkillsRoot, result.IndexSHA256)
	fmt.Fprintln(out, "After reviewing this pinned content, explicitly select the root with fak agent/chat --skills-dir. Existing policy and tool permissions still apply.")
	return 0
}

func validatePulledSkill(name string, body []byte) error {
	// Empty path makes the actual native parser incapable of following canonical
	// files. Remote bundles cannot import another local file at later activation.
	s, err := agent.ParseSkill(body, "")
	if err != nil || s.Name != name || strings.TrimSpace(s.Metadata["canonical"]) != "" {
		return &skilldiscovery.Error{Code: "ENTRYPOINT_INVALID"}
	}
	return nil
}

package main

import "github.com/anthony-chaudhary/fak/internal/harnessprofile"

// guard_repoint.go — the single seam that decides HOW a wrapped child is pointed at the
// gateway, driven by the resolved harness's HarnessProfile.Repoint (C3, #1954) instead of
// the per-harness `if guardIsCodex` / `case "claude"` the installers used to hardcode.
//
// The three repoint mechanisms already exist, one installer each; C3 does NOT rewrite them,
// it only makes the SELECTION data:
//
//   - env          -> guardInjectedEnv          (ANTHROPIC_BASE_URL / OPENAI_BASE_URL+API_BASE)
//   - cli-config   -> installGuardCodexConfig    (Codex `-c model_providers.fak.*` overrides)
//   - settings-file -> the --settings hooks + --mcp-config install (Claude)
//
// env is always-on (guardInjectedEnv fires for every provider), so the interesting gates are
// cli-config (only Codex declares it) and settings-file (only Claude declares it). guardIsCodex
// and guardPreCompactIsClaudeCommand now delegate here, so adding a harness that wants
// settings-file or env is a registry entry (C6: a config entry) — no new `if guardIsX`.

// guardRepointWants reports whether a resolved profile declares repoint mechanism m. It is
// the pure core the guard repoint gates read: an unrecognized agent (ok=false) gets ONLY the
// always-on env repoint (guardInjectedEnv fires regardless) and never cli-config /
// settings-file — matching guard's historical behavior for an agent no profile names.
func guardRepointWants(profile harnessprofile.HarnessProfile, ok bool, m harnessprofile.RepointMechanism) bool {
	if !ok {
		return m == harnessprofile.RepointEnv
	}
	return profile.HasRepoint(m)
}

// guardProfileHasRepoint resolves the wrapped-agent command to its profile and reports
// whether that profile declares mechanism m — the command-resolving wrapper over
// guardRepointWants. It is the one place repoint selection consults the registry, so every
// gate (guardIsCodex, guardPreCompactIsClaudeCommand) agrees on the same normalization
// (harnessprofile.Lookup ports guardAgentBaseName) and the same closed mechanism set.
func guardProfileHasRepoint(command string, m harnessprofile.RepointMechanism) bool {
	profile, ok := harnessprofile.Lookup(command)
	return guardRepointWants(profile, ok, m)
}

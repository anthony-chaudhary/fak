package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

const (
	guardRotateAuto = "auto"
	guardRotateOff  = "off"
)

type guardRotation struct{ Seat, Dir, Reason, EnvKey string }

type guardRotationEvidence struct {
	Kind   string
	Detail string
}

func (e guardRotationEvidence) reason() (string, bool) {
	switch e.Kind {
	case "stale_credential", "provider_auth", "provider_rate_limited":
		if strings.TrimSpace(e.Detail) != "" {
			return e.Kind + ":" + strings.TrimSpace(e.Detail), true
		}
		return e.Kind, true
	default:
		return "", false
	}
}

type guardRotationRuntime struct {
	Launcher    guardChildLauncher
	Mode        string
	CurrentSeat string
	Registry    accounts.Registry
	EnvKey      string
	Headroom    accounts.RotationHeadroom
}

func guardRotationRuntimeFor(command []string, mode string) guardRotationRuntime {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return guardRotationRuntimeForProfile(profile, mode)
}

func guardRotationRuntimeForProfile(profile harnessprofile.HarnessProfile, mode string) guardRotationRuntime {
	r := guardRotationRuntime{Mode: mode}
	if mode == guardRotateOff || !profile.Recognized() || strings.TrimSpace(profile.ConfigHomeGlob) == "" {
		return r
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return r
	}
	homes, err := accounts.DiscoverProfile(home, profile)
	if err != nil {
		return r
	}
	r.Registry = accounts.Registry{Homes: homes}
	r.Headroom = rotationHeadroom(home)
	switch profile.Identity {
	case harnessprofile.IdentityClaude:
		r.EnvKey = "CLAUDE_CONFIG_DIR"
	case harnessprofile.IdentityCodex:
		r.EnvKey = "CODEX_HOME"
	}
	for _, h := range homes {
		if current := os.Getenv(r.EnvKey); current != "" && strings.EqualFold(current, h.Dir) {
			r.CurrentSeat = h.Name
			break
		}
	}
	return r
}

func (rt *guardRotationRuntime) launcher() guardChildLauncher {
	if rt == nil {
		return nil
	}
	return rt.Launcher
}

func (rt *guardRotationRuntime) rotate(command []string, injected [][2]string, reason string, audit *journal.Journal, traceID string, stderr *os.File) ([]string, [][2]string, bool) {
	if rt == nil || rt.Mode == guardRotateOff {
		return command, injected, false
	}
	r, ok := guardNextRotationWithHeadroom(rt.Registry, rt.CurrentSeat, rt.Mode, reason, rt.EnvKey, rt.Headroom)
	if !ok {
		return command, injected, false
	}
	command, injected = guardApplyRotation(command, injected, r)
	rt.CurrentSeat = r.Seat
	if stderr != nil {
		fmt.Fprint(stderr, guardRotationBanner(r))
	}
	if audit != nil {
		audit.AppendAgentEvent("ACCOUNT_ROTATION", traceID, r.Seat+":"+r.Reason)
	}
	return command, injected, true
}

// rotateAfterExit is the child-supervisor boundary: a successful child already completed
// the operator's request and must never be relaunched on another account. Keeping this guard
// beside the rotation primitive makes both supervised and unsupervised launch paths share the
// same no-amplification rule.
func (rt *guardRotationRuntime) rotateAfterExit(runErr error, evidence guardRotationEvidence, command []string, injected [][2]string, audit *journal.Journal, traceID string, stderr *os.File) ([]string, [][2]string, bool) {
	if runErr == nil {
		return command, injected, false
	}
	reason, ok := evidence.reason()
	if !ok {
		return command, injected, false
	}
	return rt.rotate(command, injected, reason, audit, traceID, stderr)
}

func guardRotationEvidenceSince(before, after map[string]uint64) guardRotationEvidence {
	if after["auth"] > before["auth"] {
		return guardRotationEvidence{Kind: "provider_auth", Detail: fmt.Sprintf("upstream_auth_delta=%d", after["auth"]-before["auth"])}
	}
	if after["rate_limited"] > before["rate_limited"] {
		return guardRotationEvidence{Kind: "provider_rate_limited", Detail: fmt.Sprintf("upstream_rate_limited_delta=%d", after["rate_limited"]-before["rate_limited"])}
	}
	return guardRotationEvidence{}
}

func normalizeGuardRotateMode(raw string, explicitlySet, interactive bool) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		if !explicitlySet && interactive {
			return guardRotateOff, nil
		}
		return guardRotateAuto, nil
	}
	if raw == guardRotateAuto || raw == guardRotateOff {
		return raw, nil
	}
	if strings.ContainsAny(raw, "/\\") {
		return "", fmt.Errorf("guard --rotate: seat must be a name, not a path")
	}
	return raw, nil
}

func guardNextRotation(reg accounts.Registry, currentSeat, requested, reason, envKey string) (guardRotation, bool) {
	return guardNextRotationWithHeadroom(reg, currentSeat, requested, reason, envKey, nil)
}

func guardNextRotationWithHeadroom(reg accounts.Registry, currentSeat, requested, reason, envKey string, headroom accounts.RotationHeadroom) (guardRotation, bool) {
	if requested == guardRotateOff {
		return guardRotation{}, false
	}
	var next accounts.RotationSeat
	var ok bool
	if requested != guardRotateAuto {
		for _, s := range reg.RotationPlan().Pool {
			if s.Name == requested {
				next = s
				ok = s.Status == accounts.RotationIncluded
				break
			}
		}
	} else {
		next, ok = reg.NextInRotationWithHeadroom(currentSeat, headroom)
	}
	if !ok || next.Name == currentSeat {
		return guardRotation{}, false
	}
	return guardRotation{Seat: next.Name, Dir: next.Dir, Reason: reason, EnvKey: envKey}, true
}

func guardApplyRotation(command []string, injected [][2]string, r guardRotation) ([]string, [][2]string) {
	if strings.TrimSpace(r.EnvKey) == "" || strings.TrimSpace(r.Dir) == "" {
		return command, injected
	}
	out := append([][2]string(nil), injected...)
	for i := range out {
		if strings.EqualFold(out[i][0], r.EnvKey) {
			out[i][1] = r.Dir
			return command, out
		}
	}
	return command, append(out, [2]string{r.EnvKey, r.Dir})
}

func guardRotationBanner(r guardRotation) string {
	return fmt.Sprintf("fak guard: rotated to account %s (%s)\n", r.Seat, r.Reason)
}

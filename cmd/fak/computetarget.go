package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"text/tabwriter"
	"time"
)

// computetarget — the THIRD console registry axis. The four compute backends a
// user can chat against (`fak c`) exist today only as scattered, unnamed wrappers
// (the Mac Metal gateway in claude_mac_fak.go, the glm-gcp shell preset, a manual
// local `fak serve`, the real Anthropic default in tui.go). This file gives them
// ONE typed list the console can enumerate and resolve.
//
// It must NOT be conflated with the two registries fak already has
// (docs/fak/concept-glossary.md):
//   - internal/modelroute Roster — binds a model-id -> provider credential
//     (per-model, credential axis).
//   - internal/accounts — Claude config-home seats (subscription-login axis).
//
// This registry names WHOLE compute/gateway targets for the interactive console
// (gateway URL + model + locality + a live /healthz), a third axis. It stores only
// env-var NAMES, never a secret — mirroring modelroute's credential discipline.

type targetKind string

const (
	targetGatewayURL    targetKind = "gateway-url"    // an existing fak/OpenAI-compatible gateway, reached over HTTP
	targetLocalSpawn    targetKind = "local-spawn"    // an in-kernel `fak serve` the user starts on this box
	targetProviderProxy targetKind = "provider-proxy" // a real upstream provider (Anthropic) fronted by guard
)

type targetLocality string

const (
	localityLocal  targetLocality = "local"
	localityRemote targetLocality = "remote"
)

// computeTarget names ONE whole compute/gateway backend the interactive console
// can chat against. The credential is held as CredEnv — the NAME of the env var
// holding the bearer/token — never the secret itself, so a registry dump is safe
// to print and diff.
type computeTarget struct {
	Name        string         `json:"name"`
	Kind        targetKind     `json:"kind"`
	GatewayURL  string         `json:"gateway_url,omitempty"` // gateway-url / provider-proxy
	SpawnSpec   string         `json:"spawn_spec,omitempty"`  // local-spawn: the command to start it
	Model       string         `json:"model,omitempty"`
	Locality    targetLocality `json:"locality"`
	HealthzPath string         `json:"healthz_path,omitempty"` // empty => no live probe (lists n/a)
	CredEnv     string         `json:"cred_env,omitempty"`     // env-var NAME only, never the secret
	CostNote    string         `json:"cost_note,omitempty"`
}

// computeTargetEnvNameRE mirrors internal/modelroute's env-var-name discipline: a
// cred reference must LOOK like a variable name, so a pasted secret ("sk-ant-…")
// fails loud at validate time instead of leaking into the manifest.
var computeTargetEnvNameRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func (t computeTarget) validate() error {
	name := strings.TrimSpace(t.Name)
	if name == "" {
		return fmt.Errorf("computetarget: a target name must not be empty")
	}
	switch t.Kind {
	case targetGatewayURL, targetProviderProxy:
		if strings.TrimSpace(t.GatewayURL) == "" {
			return fmt.Errorf("computetarget: target %q (kind %s) requires a gateway_url", name, t.Kind)
		}
		if err := validateGatewayURL(t.GatewayURL); err != nil {
			return fmt.Errorf("computetarget: target %q: %w", name, err)
		}
	case targetLocalSpawn:
		if strings.TrimSpace(t.SpawnSpec) == "" {
			return fmt.Errorf("computetarget: target %q (kind local-spawn) requires a spawn_spec", name)
		}
	default:
		return fmt.Errorf("computetarget: target %q has unknown kind %q (want gateway-url|local-spawn|provider-proxy)", name, t.Kind)
	}
	switch t.Locality {
	case localityLocal, localityRemote:
	default:
		return fmt.Errorf("computetarget: target %q has unknown locality %q (want local|remote)", name, t.Locality)
	}
	if t.CredEnv != "" && !computeTargetEnvNameRE.MatchString(t.CredEnv) {
		return fmt.Errorf("computetarget: target %q cred_env %q is not an env-var NAME "+
			"(it must NAME the variable holding the credential, e.g. FAK_GATEWAY_KEY — never the secret itself)", name, t.CredEnv)
	}
	return nil
}

func validateGatewayURL(raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return fmt.Errorf("gateway_url %q is not parseable: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("gateway_url %q must be http(s)", raw)
	}
	if u.Host == "" {
		return fmt.Errorf("gateway_url %q has no host", raw)
	}
	return nil
}

// builtinComputeTargets returns the four built-in targets, sourced from the
// scattered defaults the issue catalogs (no new secrets; env-var NAMES only):
//   - mac        claude_mac_fak.go defaultClaudeMacGateway/Model (FAK_MAC_GATEWAY/FAK_MAC_MODEL)
//   - gcp        scripts/dogfood-claude.sh glm-gcp preset (FAK_GLM_GCP_BASE_URL, glm-5.2)
//   - local      an in-kernel `fak serve … --backend cuda` on this box (loopback gateway)
//   - anthropic  the real Anthropic API (provider hardcoded in tui.go's guard path)
func builtinComputeTargets() []computeTarget {
	return []computeTarget{
		{
			Name:        "mac",
			Kind:        targetGatewayURL,
			GatewayURL:  envOrDefault("FAK_MAC_GATEWAY", defaultClaudeMacGateway),
			Model:       envOrDefault("FAK_MAC_MODEL", defaultClaudeMacModel),
			Locality:    localityRemote,
			HealthzPath: "/healthz",
			CredEnv:     "FAK_GATEWAY_KEY",
			CostNote:    "your own Mac Metal box — no per-token cost",
		},
		{
			Name:        "gcp",
			Kind:        targetGatewayURL,
			GatewayURL:  envOrDefault("FAK_GLM_GCP_BASE_URL", "http://127.0.0.1:8200/v1"),
			Model:       envOrDefault("FAK_GLM_GCP_MODEL", "glm-5.2"),
			Locality:    localityRemote,
			HealthzPath: "/health", // SGLang/vLLM liveness probe (root, not the /v1 base)
			CredEnv:     "FAK_GATEWAY_KEY",
			CostNote:    "GCP GLM-5.2 serving node (paid GPU compute)",
		},
		{
			Name:        "local",
			Kind:        targetLocalSpawn,
			GatewayURL:  envOrDefault("FAK_LOCAL_GATEWAY", "http://127.0.0.1:8080"),
			SpawnSpec:   "fak serve --gguf <model.gguf> --backend cuda",
			Locality:    localityLocal,
			HealthzPath: "/healthz",
			CostNote:    "in-kernel fak serve on this box — no per-token cost",
		},
		{
			Name:       "anthropic",
			Kind:       targetProviderProxy,
			GatewayURL: envOrDefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
			Model:      defaultLaunchModel,
			Locality:   localityRemote,
			// The real Anthropic API exposes no /healthz; the console proves it live
			// only via a real request, so it lists n/a rather than a phantom "up".
			HealthzPath: "",
			CredEnv:     "ANTHROPIC_API_KEY",
			CostNote:    "real Anthropic API (subscription OAuth or ANTHROPIC_API_KEY; metered per token)",
		},
	}
}

// targetRegistry is an ordered, name-unique set of compute targets.
type targetRegistry struct {
	targets []computeTarget
}

func (r *targetRegistry) add(t computeTarget) error {
	if err := t.validate(); err != nil {
		return err
	}
	name := strings.TrimSpace(t.Name)
	for _, e := range r.targets {
		if strings.EqualFold(e.Name, name) {
			return fmt.Errorf("computetarget: duplicate target %q (names must be unique across the built-ins and the user file)", name)
		}
	}
	t.Name = name
	r.targets = append(r.targets, t)
	return nil
}

func (r *targetRegistry) resolve(name string) (computeTarget, bool) {
	name = strings.TrimSpace(name)
	for _, t := range r.targets {
		if strings.EqualFold(t.Name, name) {
			return t, true
		}
	}
	return computeTarget{}, false
}

func (r *targetRegistry) all() []computeTarget {
	out := make([]computeTarget, len(r.targets))
	copy(out, r.targets)
	return out
}

// nearest returns the registered target name closest to s (case-insensitive) within a
// small edit distance, or "" when nothing is close — the seed for a "did you mean" hint
// on an unknown `fak c <token>` (#938). A prefix relation counts as close (so "an" and
// "anthropic-x" both suggest "anthropic"); otherwise it is the best Levenshtein match
// within distance 2. It is a hint surface only, never a hot path.
func (r *targetRegistry) nearest(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return ""
	}
	best := ""
	bestDist := 3 // suggest only within edit distance <= 2
	for _, t := range r.targets {
		name := strings.ToLower(t.Name)
		if name == s || strings.HasPrefix(name, s) || strings.HasPrefix(s, name) {
			return t.Name
		}
		if d := levenshtein(s, name); d < bestDist {
			bestDist = d
			best = t.Name
		}
	}
	return best
}

// levenshtein is the classic edit-distance DP, used only for the small "did you mean"
// target hint above.
func levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			del, ins, sub := prev[j]+1, cur[j-1]+1, prev[j-1]+cost
			m := del
			if ins < m {
				m = ins
			}
			if sub < m {
				m = sub
			}
			cur[j] = m
		}
		prev = cur
	}
	return prev[len(rb)]
}

// defaultComputeTargetsFile is the optional user override file, additive over the
// built-ins. FAK_TARGETS_FILE wins; otherwise ~/.fak/targets.json.
func defaultComputeTargetsFile() string {
	return envOrHomePath("FAK_TARGETS_FILE", ".fak", "targets.json")
}

// envOrHomePath returns a trimmed env override (envVar) when set, else the
// home-relative path home/<rel...>. It returns "" when no override is set and the
// user home dir can't be determined.
func envOrHomePath(envVar string, rel ...string) string {
	if v := strings.TrimSpace(os.Getenv(envVar)); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(append([]string{home}, rel...)...)
}

// loadComputeTargets builds a registry of the built-ins plus any user targets from
// path (additive over the built-ins). A missing path is fine (built-ins only).
// Malformed JSON, an invalid target, or a duplicate name (a collision with a
// built-in or within the file) fails LOUD.
func loadComputeTargets(path string) (*targetRegistry, error) {
	r := &targetRegistry{}
	for _, t := range builtinComputeTargets() {
		if err := r.add(t); err != nil {
			return nil, fmt.Errorf("computetarget: built-in %q invalid: %w", t.Name, err)
		}
	}
	path = strings.TrimSpace(path)
	if path == "" {
		return r, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return r, nil
		}
		return nil, fmt.Errorf("computetarget: read %s: %w", path, err)
	}
	var user []computeTarget
	if err := json.Unmarshal(data, &user); err != nil {
		return nil, fmt.Errorf("computetarget: parse %s: %w", path, err)
	}
	for _, t := range user {
		if err := r.add(t); err != nil {
			return nil, fmt.Errorf("computetarget: %s: %w", path, err)
		}
	}
	return r, nil
}

// targetHealth is the live liveness verdict for one target, read from a REAL
// /healthz response — never asserted.
type targetHealth struct {
	State  string `json:"state"`            // "up" | "down" | "n/a"
	Detail string `json:"detail,omitempty"` // why it is down / n/a
}

// healthzURL joins the gateway URL's ORIGIN (scheme+host) with the target's
// healthz path — never the gateway's full path, so an OpenAI /v1 base still probes
// /healthz at the root. Returns ok=false when the target declares no probe (the
// real Anthropic API), which lists as n/a, not a phantom up.
func (t computeTarget) healthzURL() (string, bool) {
	if strings.TrimSpace(t.HealthzPath) == "" || strings.TrimSpace(t.GatewayURL) == "" {
		return "", false
	}
	u, err := url.Parse(strings.TrimSpace(t.GatewayURL))
	if err != nil || u.Host == "" {
		return "", false
	}
	return u.Scheme + "://" + u.Host + t.HealthzPath, true
}

// probe GETs the target's /healthz and reports up/down from the REAL response.
// A target with no healthz path is n/a. The credential is read from CredEnv at
// probe time and sent as a bearer; it is never stored on the value or printed.
// This mirrors the one-shot bearer-GET shape in claude_mac_fak.go.
func (t computeTarget) probe(ctx context.Context, hc *http.Client) targetHealth {
	hurl, ok := t.healthzURL()
	if !ok {
		return targetHealth{State: "n/a", Detail: "no healthz endpoint for this target"}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, hurl, nil)
	if err != nil {
		return targetHealth{State: "down", Detail: err.Error()}
	}
	if t.CredEnv != "" {
		if key := strings.TrimSpace(os.Getenv(t.CredEnv)); key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return targetHealth{State: "down", Detail: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return targetHealth{State: "down", Detail: fmt.Sprintf("status %d", resp.StatusCode)}
	}
	return targetHealth{State: "up"}
}

const computeTargetListSchema = "fak.computetarget.list.v1"

type targetListing struct {
	Name       string         `json:"name"`
	Kind       targetKind     `json:"kind"`
	GatewayURL string         `json:"gateway_url,omitempty"`
	SpawnSpec  string         `json:"spawn_spec,omitempty"`
	Model      string         `json:"model,omitempty"`
	Locality   targetLocality `json:"locality"`
	CredEnv    string         `json:"cred_env,omitempty"`
	CostNote   string         `json:"cost_note,omitempty"`
	Health     targetHealth   `json:"health"`
}

// targetListReport is the stable --json shape for `fak c --list-targets --json`.
type targetListReport struct {
	Schema  string          `json:"schema"`
	Targets []targetListing `json:"targets"`
}

// listing probes every target's /healthz (each with its own timeout) and returns
// the stable report shape shared by the table and the --json output.
func (r *targetRegistry) listing(parent context.Context, hc *http.Client, perProbe time.Duration) targetListReport {
	rep := targetListReport{Schema: computeTargetListSchema, Targets: []targetListing{}}
	for _, t := range r.targets {
		ctx, cancel := context.WithTimeout(parent, perProbe)
		health := t.probe(ctx, hc)
		cancel()
		rep.Targets = append(rep.Targets, targetListing{
			Name:       t.Name,
			Kind:       t.Kind,
			GatewayURL: t.GatewayURL,
			SpawnSpec:  t.SpawnSpec,
			Model:      t.Model,
			Locality:   t.Locality,
			CredEnv:    t.CredEnv,
			CostNote:   t.CostNote,
			Health:     health,
		})
	}
	return rep
}

// renderComputeTargetTable writes the human table: name, kind, gateway, locality,
// live health, model. The HEALTH column shows the STATE only; a down/n-a target's
// reason is relocated to the "health notes" block below the table so an
// unreachable gateway is never silent, yet a single long dial error cannot wreck
// the layout.
func renderComputeTargetTable(w io.Writer, rep targetListReport) {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tKIND\tGATEWAY\tLOCALITY\tHEALTH\tMODEL")
	// Keep only the state in the aligned HEALTH cell and collect any detail for the
	// notes block. Inlining the detail here made tabwriter pad every row to the
	// widest cell, so one verbose "connectex: … actively refused it" on a down
	// target shoved `up`/`n/a` and the MODEL column hundreds of columns right and
	// off-screen. State-only stays aligned; the reason is printed once per target
	// underneath and the untruncated detail still rides on --json.
	var notes []string
	for _, t := range rep.Targets {
		gateway := t.GatewayURL
		if gateway == "" {
			gateway = t.SpawnSpec
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n",
			t.Name, t.Kind, blankDash(gateway), t.Locality, t.Health.State, blankDash(t.Model))
		if t.Health.State != "up" && strings.TrimSpace(t.Health.Detail) != "" {
			notes = append(notes, fmt.Sprintf("  %s (%s): %s", t.Name, t.Health.State, strings.TrimSpace(t.Health.Detail)))
		}
	}
	tw.Flush()
	if len(notes) > 0 {
		fmt.Fprintln(w, "\nhealth notes:")
		for _, n := range notes {
			fmt.Fprintln(w, n)
		}
	}
	fmt.Fprintln(w, "\ncredential: env-var NAME only — fak never stores or prints a secret.")
	fmt.Fprintln(w, "override built-ins additively via ~/.fak/targets.json (or FAK_TARGETS_FILE).")
}

// runListComputeTargets backs `fak c --list-targets [--json]`: it loads the
// built-in + user-file registry, probes each target's live /healthz, and renders
// the table (or the stable --json shape). An unreachable target shows down, never
// a phantom up.
func runListComputeTargets(stdout, stderr io.Writer, asJSON bool) int {
	reg, err := loadComputeTargets(defaultComputeTargetsFile())
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 1
	}
	hc := &http.Client{Timeout: 3 * time.Second}
	rep := reg.listing(context.Background(), hc, 3*time.Second)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak console agent")
	}
	renderComputeTargetTable(stdout, rep)
	return 0
}

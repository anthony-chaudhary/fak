package routerlaunch

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// EnvRouterGatewayKey is the operator-set bearer env var for the supervised
// local router; discovery checks it before the on-disk gateway key file.
const EnvRouterGatewayKey = "FAK_ROUTER_GATEWAY_KEY"

// UserGatewayKeyFile is the supervised router's gateway key location relative
// to the user home dir. Forward slashes keep it a constant expression and are
// accepted by filepath.Join on every platform.
const UserGatewayKeyFile = ".fak/router/gateway.key"

// userRoutingTableFile is the supervised router's routing table location
// relative to the user home dir.
const userRoutingTableFile = ".fak/router/routing.json"

// userHomeFunc is the hermetic seam over os.UserHomeDir: every home-relative
// read in this package goes through it so tests can point at a temp dir.
var userHomeFunc = os.UserHomeDir

// advertisedFetch is the hermetic seam over AdvertisedModels:
// PreferredDefaultBackendModel calls it so tests can stub the /v1/models
// fetch without dialing a router.
var advertisedFetch = func(origin, bearer string, timeout time.Duration) []string {
	return AdvertisedModels(origin, bearer, timeout)
}

// normalizeOrigin trims a trailing "/" and a trailing "/v1" so /healthz and
// /v1/* share one origin, matching cmd/fak's router-origin handling.
func normalizeOrigin(origin string) string {
	v := strings.TrimRight(strings.TrimSpace(origin), "/")
	return strings.TrimSuffix(v, "/v1")
}

// ResolveGatewayCredential returns the first non-empty value among the named
// env vars (checked in order), then the trimmed contents of
// <home>/UserGatewayKeyFile. "" when nothing is found. Never logs values.
func ResolveGatewayCredential(envVarNames ...string) string {
	for _, name := range envVarNames {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	home, err := userHomeFunc()
	if err != nil || home == "" {
		return ""
	}
	raw, err := os.ReadFile(filepath.Join(home, UserGatewayKeyFile))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(raw))
}

// AdvertisedModels lists the router's /v1/models data[].id entries (in
// response order) using Bearer authentication when bearer != "". Returns nil
// on any failure (unreachable, non-200, decode error).
func AdvertisedModels(origin, bearer string, timeout time.Duration) []string {
	base := normalizeOrigin(origin)
	if base == "" {
		return nil
	}
	client := &http.Client{
		Timeout: timeout,
		// No redirects: a /v1/models hop to another host is not the router's
		// answer, and following it would leak the bearer off-host.
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	req, err := http.NewRequest(http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return nil
	}
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil
	}
	var doc struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal(body, &doc) != nil {
		return nil
	}
	ids := make([]string, 0, len(doc.Data))
	for _, m := range doc.Data {
		if id := strings.TrimSpace(m.ID); id != "" {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	return ids
}

// UserRoutingTable reads <home>/.fak/router/routing.json best-effort.
func UserRoutingTable() ([]byte, error) {
	home, err := userHomeFunc()
	if err != nil {
		return nil, err
	}
	return os.ReadFile(filepath.Join(home, userRoutingTableFile))
}

// routingTable is the subset of the user routing table this package consumes;
// unknown fields are tolerated.
type routingTable struct {
	DefaultBackend string `json:"default_backend"`
	Routes         []struct {
		Model   string   `json:"model"`
		Aliases []string `json:"aliases"`
		Backend string   `json:"backend"`
	} `json:"routes"`
}

// DefaultBackendModel returns the first advertised id whose route backend
// equals the routing table's default_backend. Route match is by exact model
// id or alias. ok=false when the table does not parse, declares no route on
// the default backend, or no advertised id matches such a route.
func DefaultBackendModel(routingJSON []byte, advertised []string) (string, bool) {
	var table routingTable
	if json.Unmarshal(routingJSON, &table) != nil {
		return "", false
	}
	def := strings.TrimSpace(table.DefaultBackend)
	if def == "" {
		return "", false
	}
	onDefault := make(map[string]bool)
	for _, r := range table.Routes {
		if strings.TrimSpace(r.Backend) != def {
			continue
		}
		if r.Model != "" {
			onDefault[r.Model] = true
		}
		for _, a := range r.Aliases {
			if a != "" {
				onDefault[a] = true
			}
		}
	}
	for _, id := range advertised {
		if onDefault[id] {
			return id, true
		}
	}
	return "", false
}

// PreferredDefaultBackendModel composes the launch-time preference: resolve
// the credential (FAK_CLAUDE_ROUTER_KEY, then EnvRouterGatewayKey, then the
// user key file), fetch advertised ids, read the user routing table, and apply
// DefaultBackendModel. Returns "" whenever any step yields nothing usable;
// callers fall back to their own default.
func PreferredDefaultBackendModel(origin string, timeout time.Duration) string {
	bearer := ResolveGatewayCredential("FAK_CLAUDE_ROUTER_KEY", EnvRouterGatewayKey)
	advertised := advertisedFetch(origin, bearer, timeout)
	if len(advertised) == 0 {
		return ""
	}
	table, err := UserRoutingTable()
	if err != nil || len(table) == 0 {
		return ""
	}
	model, ok := DefaultBackendModel(table, advertised)
	if !ok {
		return ""
	}
	return model
}

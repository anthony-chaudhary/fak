---
loop: goal
goal_slug: server-dashboard-remote-access-link-audit
witness: "go test -v ./internal/gateway/... ./cmd/fak-strix/..."
budget: { max_iters: 20 }
lane: gateway
---
# Objective
Audit, design, and implement context-aware dashboard link creation and navigation across model serving runtimes (`fak serve`, `cmd/fak-server`, and `cmd/fak-strix`). Eliminate the category error where remote access to the server dashboard (e.g. `http://192.168.1.208:8080/?key=...`) issues hardcoded redirects to `http://localhost:3000/d/fak-gateway-observability`, failing with `ERR_CONNECTION_REFUSED` on client workstations. Establish dynamic authority derivation, optional single-port gateway reverse-proxying (`/grafana/*`), dashboard catalog adaptation (`fak-strix-*` on appliances vs generic gateway dashboards in dev), and seamless authentication handover.

---

# Root Cause Audit: Tracing the Broken Link Creation Pipeline

When an operator launches an appliance or remote server and accesses:
```text
http://192.168.1.208:8080/?key=15cc2f12f82370b253c11b214415addf3fc20cea8aa0a202ad9a4069722a1cbe
```
the following execution sequence currently occurs:

1. **Appliance CLI URL Generation (`cmd/fak-strix/main.go`, `tools/strix.py`):**
   - The CLI probes the appliance's network interface via `ip route get 1.1.1.1` and reads `/etc/fak/gateway.env` to discover `IP=192.168.1.208` and `FAK_GATEWAY_KEY`.
   - It outputs the web dashboard URL: `http://192.168.1.208:8080/?key=<KEY>`.
   - It separately outputs the Grafana index URL: `http://192.168.1.208:3000/d/fak-strix-index`.

2. **Gateway Homepage Rendering (`internal/gateway/home.go`):**
   - The user opens `http://192.168.1.208:8080/?key=...` in their browser on a client workstation (e.g. laptop).
   - `handleHome` renders `homePageTemplate`.
   - JavaScript extracts `?key=` from `window.location.search`, preserves it in `sessionStorage`, and dynamically appends `&key=...` to all dashboard cards.
   - Cards are rendered with links such as:
     ```html
     <a class="card rich-dashboard" data-dashboard-uid="fak-gateway-observability" href="/?dashboard=rich&uid=fak-gateway-observability&key=15cc2f...">
     ```

3. **Rich Dashboard Resolution & Redirect (`internal/gateway/rich_dashboard.go`):**
   - The user clicks "Gateway observability" (`fak-gateway-observability`).
   - The browser sends `GET /?dashboard=rich&uid=fak-gateway-observability&key=...` to `http://192.168.1.208:8080`.
   - `handleRichDashboard(w, r)` calls `s.richDashboards.ensure()`.
   - `richDashboardManager.activate()` runs on the appliance. Because `FAK_GRAFANA_URL` is unset, it defaults to:
     ```go
     base = bundledGrafanaURL // "http://localhost:3000"
     ```
   - On the appliance, Grafana is running and bound to `0.0.0.0:3000`. Probing `http://localhost:3000/api/health` from the appliance succeeds!
   - `snap.State` becomes `"ready"` and `snap.URL` is set to `"http://localhost:3000"`.
   - `richDashboardDestination(snap.URL, uid)` computes:
     ```go
     destination := "http://localhost:3000/d/fak-gateway-observability"
     ```
   - `handleRichDashboard` issues:
     ```http
     HTTP/1.1 303 See Other
     Location: http://localhost:3000/d/fak-gateway-observability
     ```

4. **The Client Failure Mode:**
   - The client browser (running on the operator's laptop at `192.168.1.50`) receives the `303 See Other` response and follows `Location: http://localhost:3000/d/fak-gateway-observability`.
   - The client browser attempts to connect to `localhost:3000` **on the client's laptop**, NOT on the remote appliance `192.168.1.208`.
   - The connection immediately fails with:
     ```text
     This site can’t be reached.
     localhost refused to connect.
     ERR_CONNECTION_REFUSED
     ```
   - If the developer happens to run a local dev service (e.g. Next.js, Rails, local Grafana) on port 3000 on their laptop, it connects to that unrelated service instead!

5. **Secondary Defect — Dashboard Catalog Mismatch:**
   - Even if the destination hostname were corrected to `http://192.168.1.208:3000/d/fak-gateway-observability`, the AMD Strix Halo appliance does **not** provision `fak-gateway-observability.json`.
   - The appliance observability suite (`platform/installer/observability/assets/dashboards/`) provisions:
     `fak-strix-index`, `fak-strix-appliance`, `fak-strix-serving`, `fak-strix-cache`, `fak-strix-cluster`, `fak-strix-agents`, `fak-strix-orchestration`.
   - Navigating to `fak-gateway-observability` on the appliance results in a Grafana 404 "Dashboard not found".

6. **Tertiary Defect — Authentication Wall:**
   - The user authenticated to the gateway on port 8080 with `?key=<KEY>`.
   - The appliance Grafana instance on port 3000 enforces `[auth.anonymous] enabled = false` with a separate 48-hex random password (`/etc/fak-observability/grafana-admin.env`).
   - The user is confronted with a Grafana login barrier without credentials.

---

# Exhaustive Analysis of Access Contexts

Link creation must be context-aware across all 6 production operating contexts:

| Access Context | Client Dialed Authority (`r.Host`) | Target Server & Grafana Topology | Defective Current Behavior | Required Correct Behavior |
|:---|:---|:---|:---|:---|
| **1. Local Workstation (Loopback)** | `localhost:8080` or `127.0.0.1:8080` | Gateway & Grafana running on local laptop (`127.0.0.1:3000`) | Redirects to `http://localhost:3000/d/...` (works by coincidence) | Continue resolving to `http://localhost:3000/d/...` |
| **2. Remote LAN IP** | `192.168.1.208:8080` | Gateway & Grafana running on Strix Halo APU on LAN | Redirects to `http://localhost:3000/d/...` (**fails**: connects to client laptop) | Dynamically rewrite host to client authority: `http://192.168.1.208:3000/d/...` |
| **3. Local DNS / mDNS** | `strix-halo-fak.local:8080` | Gateway & Grafana addressed via Zeroconf / Avahi | Redirects to `http://localhost:3000/d/...` (**fails**) | Dynamically rewrite host: `http://strix-halo-fak.local:3000/d/...` |
| **4. VPN / Overlay Mesh (Tailscale)** | `100.64.0.10:8080` or `strix1.tailnet.ts.net:8080` | Appliance accessed over Tailscale / WireGuard | Redirects to `http://localhost:3000/d/...` (**fails**) | Dynamically rewrite host: `http://100.64.0.10:3000/d/...` or `http://strix1.tailnet.ts.net:3000/d/...` |
| **5. Reverse Proxy / Public FQDN** | `ai.example.com` (port 443, `X-Forwarded-Host`) | Ingress exposes port 8080; port 3000 is firewalled | Redirects to `http://localhost:3000/d/...` (**fails**) or `:3000` (**times out**) | Reverse-proxy Grafana via `/grafana/*` on gateway port 8080, or use configured public Grafana URL |
| **6. SSH Port Forwarding** | `localhost:8080` via `ssh -L 8080:localhost:8080` | Gateway tunneled to laptop; port 3000 **not tunneled** | Redirects to `http://localhost:3000/d/...` (**fails**: port 3000 not open on laptop) | Gateway-internal proxy on `/grafana/*` allows single-port SSH tunneling |

---

# Architectural Invariants

1. **Never Emit Foreign Loopback to Remote Clients:**
   A gateway receiving a request from an external network (`r.Host` is not loopback) must **never** redirect the client to `localhost` or `127.0.0.1` unless the operator explicitly configured a loopback override intended for client-side evaluation.
2. **Authority-Preserving Port Remapping:**
   When Grafana is co-located with the gateway (the default bundled or appliance topology) and the target base URL is loopback (`http://localhost:3000`), the client-facing authority must be derived by pairing the hostname dialed in `r.Host` with Grafana's external port (default 3000) and the incoming request scheme (`http` or `https`).
3. **Explicit External Override Sanctity:**
   If the operator explicitly configured `FAK_GRAFANA_URL` (or config `[observability] grafana_url`) to a non-loopback URL (e.g. `https://grafana.internal.net/`), the gateway must use that URL verbatim across all access contexts without mutating the host.
4. **Single-Port Multiplexing via Gateway Reverse Proxy:**
   To eliminate firewall blockers and multi-port SSH tunneling friction (`-L 8080 -L 3000`), the gateway should support transparently reverse-proxying Grafana under `http://<host>:8080/grafana/` when enabled.
5. **Appliance Catalog Awareness:**
   When running in appliance mode or on hardware provisioned with `fak-strix-*` dashboards, the gateway homepage and redirect routes must present and navigate to the live provisioned dashboards (`fak-strix-index`, `fak-strix-serving`, etc.) rather than missing generic development dashboards.
6. **Fail-Safe Auth Handover:**
   When jumping from the gateway (`/?key=...`) to Grafana, the user must either receive transparent view access (via Grafana anonymous viewer configuration on private LANs) or clear, one-click credential visibility in the gateway UI.

---

# Detailed Design & Implementation Seams

### 1. Dynamic Client Base URL Resolver (`internal/gateway/rich_dashboard.go`)
Refactor destination URL calculation to accept the incoming `*http.Request`:

```go
func (m *richDashboardManager) clientBaseURL(r *http.Request) string {
    base := m.baseURL
    if base == "" {
        base = bundledGrafanaURL // "http://localhost:3000"
    }
    u, err := url.Parse(base)
    if err != nil {
        return base
    }

    // If the configured base URL is not loopback, honor it verbatim.
    if !isLoopbackHost(u.Hostname()) {
        return strings.TrimRight(base, "/")
    }

    // If the request itself came in via loopback, localhost is valid.
    reqHost := ""
    if r != nil {
        reqHost = strings.TrimSpace(r.Host)
    }
    if reqHost == "" || isLoopbackHost(stripPort(reqHost)) {
        return strings.TrimRight(base, "/")
    }

    // Client accessed via remote IP, mDNS, or domain.
    // Substitute the client's dialed hostname while preserving Grafana's port and path.
    clientHost := stripPort(reqHost)
    scheme := "http"
    if r.TLS != nil {
        scheme = "https"
    }

    port := u.Port()
    if port == "" {
        port = "3000"
    }

    return fmt.Sprintf("%s://%s:%s%s", scheme, clientHost, port, strings.TrimRight(u.Path, "/"))
}
```

### 2. Client-Facing Link Generation in Homepage (`internal/gateway/home.go`)
Update `homePageData` and template rendering:
- Pass the resolved client base URL or render relative links (`/?dashboard=rich&uid=...`).
- In JavaScript, allow dashboard cards to navigate smoothly, preserving `?key=` and providing direct target URLs.
- Render Grafana connection details and SSH tunnel assistance for remote operators (`ssh -L 8080:localhost:8080 -L 3000:localhost:3000`).

### 3. Gateway Reverse Proxy for Grafana (Optional Single-Port Mode)
Implement `httputil.ReverseProxy` mounted at `/grafana/` in `internal/gateway/`:
- Enabled via flag `--proxy-grafana` or `config.toml [observability] proxy_grafana = true`.
- Transparently proxies requests from `http://192.168.1.208:8080/grafana/` to internal `http://127.0.0.1:3000/`.
- Rewrites headers (`X-Forwarded-Host`, `X-Forwarded-Proto`, and subpath prefix).
- Simplifies firewall rules: **only port 8080 needs to be opened**.

### 4. Catalog Adaptation (`cmd/fak-server` & `cmd/fak-strix`)
- When running on AMD Strix Halo (detected via silicon probe or `--appliance` flag), populate `RichDashboards` with the canonical Strix suite:
  1. `fak-strix-index`: FAK Strix | Observability Index (Master Landing Page)
  2. `fak-strix-appliance`: FAK Strix | Appliance health (CPU, VRAM, DRM GPU, thermals)
  3. `fak-strix-serving`: FAK Strix | Serving performance (tok/s, routes, latency)
  4. `fak-strix-cache`: FAK Strix | Cache and recovery (KV reuse, prefix hits)
  5. `fak-strix-cluster`: FAK Strix | Multi-node cluster fleet (USB4 40G P2P)
  6. `fak-strix-agents`: FAK Strix | Autonomous Agents & Dispatch
- Provide seamless fallback to `fak-gateway-observability` and dev dashboards when running in standard local development mode.

### 5. Grafana Auth Posture & UX Handover
- In `platform/installer/observability/assets/grafana.ini`:
  - Evaluate enabling LAN anonymous viewer:
    ```ini
    [auth.anonymous]
    enabled = true
    org_role = Viewer
    ```
    (Admin mutations remain locked behind user/password).
- In `home.go`: Display the Grafana admin username and hint (`admin`) with a link to copy the appliance key.

---

# Non-Goals
- Do not expose Prometheus TSDB or raw metrics endpoints to the public internet unauthenticated.
- Do not weaken capability-floor security policies (`/etc/fak/policy.json`).
- Do not alter frozen ABI interfaces (`internal/abi`).
- Do not introduce external cloud SaaS dependencies for local appliance monitoring.

---

# Plan

- [ ] 1. **Milestone 1: Dynamic Client Base URL & Request Authority Resolver (#830)**
  - Implement `clientBaseURL(r *http.Request)` in `internal/gateway/rich_dashboard.go` replacing static `snap.URL` redirects.
  - Add loopback detection helpers (`isLoopbackHost`, `stripPort`) handling IPv4 (`127.0.0.1`), IPv6 (`::1`, `[::1]`), and `localhost`.
  - Author unit tests in `internal/gateway/rich_dashboard_test.go` covering all 6 access contexts (localhost, LAN IP, mDNS, Tailscale, reverse proxy, and explicit `FAK_GRAFANA_URL`).
  - Tracked in Issue #830 (`docs/tickets/observability/TICKET-04-dynamic-dashboard-authority-resolution.md`).

- [ ] 2. **Milestone 2: Contextual Link Generation & Template Refinements (#831)**
  - Update `internal/gateway/home.go` to carry request context into dashboard cards.
  - Update `homePageTemplate` JavaScript to handle key retention, direct links, and remote navigation cleanly.
  - Add SSH port-forwarding guidance on dashboard launcher pages when connection errors occur (`ssh -L 8080:localhost:8080 -L 3000:localhost:3000`).
  - Tracked in Issue #831 (`docs/tickets/observability/TICKET-05-homepage-remote-navigation-key-retention.md`).

- [ ] 3. **Milestone 3: Dashboard Catalog Dynamic Registration & Strix Parity (#832)**
  - Refactor `richDashboardLinks` in `internal/gateway/rich_dashboard.go` to support dynamic registration or profile selection (Standard Dev vs Strix Appliance).
  - Register `fak-strix-*` dashboards (`fak-strix-index`, `fak-strix-appliance`, `fak-strix-serving`, `fak-strix-cache`, `fak-strix-cluster`, `fak-strix-agents`) when running under appliance configuration.
  - Ensure default redirect (`/?dashboard=rich`) routes to `fak-strix-index` on Strix appliances.
  - Tracked in Issue #832 (`docs/tickets/observability/TICKET-06-appliance-dashboard-catalog-registration.md`).

- [ ] 4. **Milestone 4: Single-Port Ingress Reverse Proxy (`/grafana/*`) (#833)**
  - Implement optional reverse proxy handler in `internal/gateway/` mounted at `/grafana/` targeting `http://127.0.0.1:3000/`.
  - Wire configuration flag `--proxy-grafana` and TOML setting `[observability] proxy_grafana = true`.
  - Verify seamless single-port operation where browsing `http://<host>:8080/grafana/` renders Grafana through port 8080.
  - Tracked in Issue #833 (`docs/tickets/observability/TICKET-07-single-port-grafana-reverse-proxy.md`).

- [ ] 5. **Milestone 5: Live Hardware Witness & End-to-End Verification**
  - Deploy updated gateway to `strix1` via `go run ./cmd/fak-strix upgrade` or `push`.
  - Execute end-to-end verification from external client browser:
    - Dial `http://192.168.1.208:8080/?key=...`
    - Click dashboard cards and verify HTTP 303 redirects to `http://192.168.1.208:3000/d/fak-strix-index` (0 calls to `localhost:3000`).
  - Run regression test suite: `go test -v ./internal/gateway/... ./cmd/fak-strix/...`.

---

# Verification Witnesses

1. **Unit Test Witness:**
   ```bash
   go test -v ./internal/gateway/ -run "TestRichDashboardClientBaseURL"
   ```
   Must verify that requests with `Host: 192.168.1.208:8080` redirect to `http://192.168.1.208:3000/d/...` while requests with `Host: localhost:8080` redirect to `http://localhost:3000/d/...`.

2. **Hardware Remote Dial Witness:**
   ```bash
   curl -s -i "http://192.168.1.208:8080/?dashboard=rich&uid=fak-strix-index" -H "Host: 192.168.1.208:8080"
   ```
   Must assert `HTTP/1.1 303 See Other` with `Location: http://192.168.1.208:3000/d/fak-strix-index` (zero presence of `localhost:3000`).

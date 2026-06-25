package demoui

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"path"
	"strings"
)

// DemoBasePathEnv lets the web demos share one deployment knob when they are
// mounted behind a reverse proxy path such as /guarddemo on a single HTTPS host.
const DemoBasePathEnv = "FAK_DEMO_BASE_PATH"

// BasePathFlag registers the shared reverse-proxy mount flag used by browser
// demos. The environment default keeps containers and VM service wrappers from
// needing per-demo flag plumbing.
func BasePathFlag(fs *flag.FlagSet, example string) *string {
	return fs.String(
		"base-path",
		os.Getenv(DemoBasePathEnv),
		fmt.Sprintf("URL path prefix when this demo is mounted behind a reverse proxy on a shared HTTPS host (for example %s). Also read from %s.", example, DemoBasePathEnv),
	)
}

// NormalizeBasePath turns a user/deployment path prefix into the canonical form
// used by the demo HTTP helpers: "" for root, otherwise "/prefix" with no
// trailing slash.
func NormalizeBasePath(raw string) string {
	p := strings.TrimSpace(raw)
	if p == "" || p == "/" {
		return ""
	}
	p = "/" + strings.Trim(p, "/")
	p = path.Clean(p)
	if p == "." || p == "/" {
		return ""
	}
	return p
}

// URLBasePath returns the browser path for a mounted demo, always ending in /.
func URLBasePath(raw string) string {
	base := NormalizeBasePath(raw)
	if base == "" {
		return "/"
	}
	return base + "/"
}

// LocalURL formats the URL printed by the demo servers on startup.
func LocalURL(addr, basePath string) string {
	if host, port, err := net.SplitHostPort(addr); err == nil && isWildcardHost(host) {
		addr = net.JoinHostPort("127.0.0.1", port)
	}
	return "http://" + addr + URLBasePath(basePath)
}

func isWildcardHost(host string) bool {
	return host == "" || host == "0.0.0.0" || host == "::" || host == "[::]"
}

// ListenAddr honors the $PORT contract used by container/VM platforms. When
// PORT is set and addr is still the compiled-in loopback default, bind
// 0.0.0.0:$PORT. An explicit non-default -addr still wins.
func ListenAddr(addr, defaultAddr string) string {
	if p := strings.TrimSpace(os.Getenv("PORT")); p != "" && addr == defaultAddr {
		return "0.0.0.0:" + p
	}
	return addr
}

// MountWithBasePath mounts app at root or under basePath. When basePath is set,
// /prefix/... is stripped before app sees the request, /prefix redirects to
// /prefix/, and / redirects to /prefix/ so a container health click opens the
// actual demo.
func MountWithBasePath(mux *http.ServeMux, basePath string, app http.Handler) string {
	base := NormalizeBasePath(basePath)
	if base == "" {
		mux.Handle("/", app)
		return ""
	}
	mux.Handle(base+"/", http.StripPrefix(base, app))
	mux.HandleFunc(base, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, base+"/", http.StatusMovedPermanently)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, base+"/", http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})
	return base
}

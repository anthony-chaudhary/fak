package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardLifecycleSocketEnv = "FAK_GUARD_LIFECYCLE_SOCKET"
	guardLifecycleTokenEnv  = "FAK_GUARD_LIFECYCLE_TOKEN"
)

type guardLifecycleRequest struct {
	Token string `json:"token"`
}

type guardLifecycleResponse struct {
	Signals gateway.LifecycleSignals `json:"signals"`
	Error   string                   `json:"error,omitempty"`
}

type guardLifecycleServer struct {
	listener net.Listener
	path     string
	token    string
	srv      *gateway.Server
	done     chan struct{}
	once     sync.Once
}

func startGuardLifecycleServer(srv *gateway.Server) (*guardLifecycleServer, error) {
	if srv == nil {
		return nil, errors.New("nil gateway server")
	}
	dir, err := guardSessionTempDir("lifecycle")
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "signals.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	g := &guardLifecycleServer{listener: ln, path: path, token: slackoutbox.NewNonce(), srv: srv, done: make(chan struct{})}
	go g.serve()
	return g, nil
}

func (g *guardLifecycleServer) Env() [][2]string {
	if g == nil {
		return nil
	}
	return [][2]string{{guardLifecycleSocketEnv, g.path}, {guardLifecycleTokenEnv, g.token}}
}

func (g *guardLifecycleServer) Close() {
	if g == nil {
		return
	}
	g.once.Do(func() {
		_ = g.listener.Close()
		<-g.done
		_ = os.RemoveAll(filepath.Dir(g.path))
	})
}

func (g *guardLifecycleServer) serve() {
	defer close(g.done)
	for {
		conn, err := g.listener.Accept()
		if err != nil {
			return
		}
		go g.handle(conn)
	}
}

func (g *guardLifecycleServer) handle(conn net.Conn) {
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(time.Second))
	var req guardLifecycleRequest
	if json.NewDecoder(io.LimitReader(conn, 4096)).Decode(&req) != nil || req.Token != g.token {
		_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Error: "unauthorized"})
		return
	}
	_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Signals: g.srv.LifecycleSignalsSnapshot()})
}

func fetchGuardLifecycleSignals(socketPath, token string, timeout time.Duration) (gateway.LifecycleSignals, error) {
	if socketPath == "" || token == "" {
		return gateway.LifecycleSignals{}, errors.New("lifecycle IPC not configured")
	}
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	conn, err := net.DialTimeout("unix", socketPath, timeout)
	if err != nil {
		return gateway.LifecycleSignals{}, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(timeout))
	if err := json.NewEncoder(conn).Encode(guardLifecycleRequest{Token: token}); err != nil {
		return gateway.LifecycleSignals{}, err
	}
	var resp guardLifecycleResponse
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&resp); err != nil {
		return gateway.LifecycleSignals{}, err
	}
	if resp.Error != "" {
		return gateway.LifecycleSignals{}, fmt.Errorf("lifecycle IPC: %s", resp.Error)
	}
	return resp.Signals, nil
}

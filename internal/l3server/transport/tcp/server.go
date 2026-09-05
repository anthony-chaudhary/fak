package tcp

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/dispatch"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// Compile-time assertion: tcp.Server implements transport.Transport.
var _ transport.Transport = (*Server)(nil)

const (
	defaultIdleTimeout  = 300 * time.Second
	defaultWriteTimeout = 30 * time.Second
)

// connState tracks per-connection state.
type connState struct {
	handshaked    bool
	clientVersion string
}

// Server is the TCP transport server.
type Server struct {
	addr             string
	manager          *shard.Manager
	connReg          *metrics.ConnRegistry
	dispatcher       *dispatch.Dispatcher
	listener         net.Listener
	wg               sync.WaitGroup
	quit             chan struct{}
	connsMu          sync.Mutex
	conns            map[net.Conn]struct{}
	connsPeak        int
	maxConns         int   // 0 = unlimited
	maxOpsPerConnSec int64 // M1: per-connection rate limit; 0 = unlimited
}

// NewServer creates a new TCP server.
// manager may be nil for early connection acceptance â€” data ops will return
// RespNotReady until SetManager() is called.
func NewServer(addr string, manager *shard.Manager, connReg *metrics.ConnRegistry, startedAt time.Time, numShards ...int) *Server {
	d := &dispatch.Dispatcher{Manager: manager, ConnReg: connReg, StartedAt: startedAt}
	if manager != nil {
		d.SetManager(manager)
	} else if len(numShards) > 0 {
		// Early-connect mode: Manager is nil but numShards is known for progress responses
		d.SetNumShards(numShards[0])
	}
	return &Server{
		addr:       addr,
		manager:    manager,
		connReg:    connReg,
		dispatcher: d,
		quit:       make(chan struct{}),
		conns:      make(map[net.Conn]struct{}),
	}
}

// SetManager wires the shard Manager into the TCP server after background allocation.
func (s *Server) SetManager(mgr *shard.Manager) {
	s.manager = mgr
	s.dispatcher.SetManager(mgr)
}

// Start begins listening for TCP connections.
func (s *Server) Start() error {
	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("tcp listen on %s: %w", s.addr, err)
	}
	s.listener = ln
	log.Printf("[tcp] listening on %s", s.addr)

	s.wg.Add(1)
	go s.acceptLoop()
	return nil
}

// Stop gracefully shuts down the TCP server.
func (s *Server) Stop() {
	close(s.quit)
	if s.listener != nil {
		s.listener.Close()
	}
	// Force-close all tracked connections to unblock blocked reads
	s.connsMu.Lock()
	for c := range s.conns {
		c.Close()
	}
	s.connsMu.Unlock()
	s.wg.Wait()
}

// SetRDMAEndpoints advertises RDMA endpoints via this server's dispatcher.
func (s *Server) SetRDMAEndpoints(endpoints []dispatch.RDMAEndpointInfo) {
	s.dispatcher.RDMAEndpoints = endpoints
}

// SetClientStatsReg sets the client stats registry on this server's dispatcher.
func (s *Server) SetClientStatsReg(reg *metrics.ClientStatsRegistry) {
	s.dispatcher.ClientStatsReg = reg
}

// SetMetricsAddr sets the metrics/ready endpoint address on this server's dispatcher.
func (s *Server) SetMetricsAddr(addr string) {
	s.dispatcher.MetricsAddr = addr
}

func (s *Server) SetPollerMetrics(p metrics.PollerMetricsProvider) {
	s.dispatcher.PollerMetrics = p
}

func (s *Server) SetOpLatencyMetrics(p metrics.OpLatencyProvider) {
	s.dispatcher.OpLatencyMetrics = p
}

// SetDispatchTimeout sets the batch dispatch timeout on this server's dispatcher.
func (s *Server) SetDispatchTimeout(d time.Duration) {
	s.dispatcher.DispatchTimeout = d
}

// SetCluster sets cluster configuration on this server's dispatcher.
func (s *Server) SetCluster(ring any, replicator any, localID string, snapshotDir string) {
	s.dispatcher.SetClusterConfig(ring, replicator, localID, snapshotDir)
}

// SetMaxConns sets the maximum number of concurrent TCP connections.
// 0 means unlimited.
func (s *Server) SetMaxConns(n int) {
	s.maxConns = n
}

// SetMaxOpsPerConnSec sets the per-connection rate limit (ops/sec).
// 0 means unlimited.
func (s *Server) SetMaxOpsPerConnSec(n int64) {
	s.maxOpsPerConnSec = n
}

// Name returns the transport name.
func (s *Server) Name() string { return "tcp" }

// Addr returns the listener address (useful for tests with ":0").
func (s *Server) Addr() string {
	if s.listener != nil {
		return s.listener.Addr().String()
	}
	return s.addr
}

func (s *Server) acceptLoop() {
	defer s.wg.Done()
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			select {
			case <-s.quit:
				return
			default:
				log.Printf("[tcp] accept error: %v", err)
				continue
			}
		}

		// Reject if at connection limit
		if s.maxConns > 0 {
			s.connsMu.Lock()
			atLimit := len(s.conns) >= s.maxConns
			s.connsMu.Unlock()
			if atLimit {
				conn.Close()
				metrics.IncrTCPConnectionRejections()
				log.Printf("[tcp] WARNING: rejecting connection from %s â€” at limit (%d/%d)", conn.RemoteAddr(), s.maxConns, s.maxConns)
				continue
			}
		}

		// Set TCP options
		if tc, ok := conn.(*net.TCPConn); ok {
			tc.SetNoDelay(true)
			tc.SetReadBuffer(65536)
			tc.SetWriteBuffer(65536)
		}

		s.wg.Add(1)
		go s.handleConn(conn)
	}
}

func (s *Server) handleConn(conn net.Conn) {
	defer s.wg.Done()
	defer conn.Close()

	// H1: Panic recovery â€” catch panics so cleanup defers still run
	defer func() {
		if r := recover(); r != nil {
			buf := make([]byte, 4096)
			n := runtime.Stack(buf, false)
			log.Printf("[tcp] PANIC in handleConn from %s: %v\n%s", conn.RemoteAddr(), r, buf[:n])
			metrics.IncrTCPHandlerPanics()
		}
	}()

	// Track connection for forced close on shutdown
	s.connsMu.Lock()
	s.conns[conn] = struct{}{}
	if n := len(s.conns); n > s.connsPeak {
		s.connsPeak = n
	}
	s.connsMu.Unlock()
	defer func() {
		s.connsMu.Lock()
		delete(s.conns, conn)
		// Compact map when it shrinks well below peak (Go maps never free buckets).
		cur := len(s.conns)
		if s.connsPeak > 64 && cur < s.connsPeak/4 {
			fresh := make(map[net.Conn]struct{}, cur)
			for k, v := range s.conns {
				fresh[k] = v
			}
			s.conns = fresh
			s.connsPeak = cur
		}
		s.connsMu.Unlock()
	}()

	cm := s.connReg.Register("tcp", conn.RemoteAddr().String(), "", "")
	defer s.connReg.Deregister(cm.ID)

	reader := bufio.NewReaderSize(conn, 65536)
	writer := bufio.NewWriterSize(conn, 65536)
	state := connState{}
	firstMessage := true
	rl := transport.NewConnRateLimiter(s.maxOpsPerConnSec) // M1: nil if unlimited

	for {
		select {
		case <-s.quit:
			return
		default:
		}

		conn.SetReadDeadline(time.Now().Add(defaultIdleTimeout))
		msg, err := protocol.ReadMessage(reader)
		if err != nil {
			// Suppress log noise during shutdown
			select {
			case <-s.quit:
				return
			default:
			}
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				log.Printf("[tcp] idle timeout, disconnecting %s", conn.RemoteAddr())
			} else if err.Error() != "EOF" {
				log.Printf("[tcp] read error: %v", err)
			}
			return
		}

		cm.AddBytesRecv(int64(protocol.HeaderSize + len(msg.Body)))
		cm.IncrRequests()

		// Detect legacy client: first message is not a handshake
		if firstMessage {
			firstMessage = false
			if msg.Header.OpCode == protocol.OpHandshake {
				state.handshaked = true
			} else {
				log.Printf("[tcp] legacy client (no handshake) from %s", conn.RemoteAddr())
			}
		}

		// M1: Per-connection rate limiting
		if rl != nil && !rl.Allow() {
			resp := protocol.Message{
				Header: protocol.Header{
					OpCode:    protocol.RespError,
					RequestID: msg.Header.RequestID,
				},
				Body: protocol.EncodeErrorResponse("rate limit exceeded"),
			}
			protocol.PutBodyBuf(msg.Body, msg.BodyPoolIdx) // M3: return pooled body
			conn.SetWriteDeadline(time.Now().Add(defaultWriteTimeout))
			if err := protocol.WriteMessage(writer, resp); err != nil {
				return
			}
			if err := writer.Flush(); err != nil {
				return
			}
			continue
		}

		var resp protocol.Message
		switch msg.Header.OpCode {
		case protocol.OpReportStats:
			resp = s.dispatcher.HandleReportStats(msg, cm.ID, cm.RemoteAddr)
		default:
			resp = s.dispatcher.Dispatch(msg)
		}
		resp.Header.RequestID = msg.Header.RequestID
		protocol.PutBodyBuf(msg.Body, msg.BodyPoolIdx) // M3: return pooled body after dispatch

		cm.AddBytesSent(int64(protocol.HeaderSize + len(resp.Body)))

		conn.SetWriteDeadline(time.Now().Add(defaultWriteTimeout))
		if err := protocol.WriteMessage(writer, resp); err != nil {
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				log.Printf("[tcp] write timeout, disconnecting %s", conn.RemoteAddr())
			} else {
				log.Printf("[tcp] write error: %v", err)
			}
			return
		}
		if err := writer.Flush(); err != nil {
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				log.Printf("[tcp] flush timeout, disconnecting %s", conn.RemoteAddr())
			} else {
				log.Printf("[tcp] flush error: %v", err)
			}
			return
		}
	}
}

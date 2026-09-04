package cluster

import (
	"encoding/json"
	"log"
	"math/rand"
	"net"
	"sync"
	"time"
)

// GossipConfig configures the SWIM gossip protocol.
type GossipConfig struct {
	BindAddr       string        // UDP bind address
	PingInterval   time.Duration // default 500ms
	SuspectTimeout time.Duration // default 3s
	Seeds          []string      // initial peer addresses
	NodeID         string
	NodeAddr       string // TCP address of this node's KV service
}

// DefaultGossipConfig returns sensible defaults.
func DefaultGossipConfig() GossipConfig {
	return GossipConfig{
		PingInterval:   500 * time.Millisecond,
		SuspectTimeout: 3 * time.Second,
	}
}

// Gossip implements the SWIM protocol for cluster membership.
type Gossip struct {
	config  GossipConfig
	ring    *Ring
	conn    *net.UDPConn
	mu      sync.RWMutex
	members map[string]*memberState
	quit    chan struct{}
}

type memberState struct {
	NodeID    string    `json:"id"`
	Addr      string    `json:"addr"`
	State     string    `json:"state"` // "alive", "suspect", "dead"
	LastPing  time.Time `json:"-"`
	SuspectAt time.Time `json:"-"`
}

type gossipMessage struct {
	Type     string         `json:"type"`     // "ping", "ack", "ping-req", "join", "membership"
	SenderID string         `json:"sender"`
	Members  []memberUpdate `json:"members,omitempty"`
}

type memberUpdate struct {
	NodeID string `json:"id"`
	Addr   string `json:"addr"`
	State  string `json:"state"`
}

// NewGossip creates a new SWIM gossip instance.
func NewGossip(cfg GossipConfig, ring *Ring) *Gossip {
	return &Gossip{
		config:  cfg,
		ring:    ring,
		members: make(map[string]*memberState),
		quit:    make(chan struct{}),
	}
}

// Start begins the gossip protocol.
func (g *Gossip) Start() error {
	addr, err := net.ResolveUDPAddr("udp", g.config.BindAddr)
	if err != nil {
		return err
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return err
	}
	g.conn = conn

	// Register self
	g.mu.Lock()
	g.members[g.config.NodeID] = &memberState{
		NodeID: g.config.NodeID,
		Addr:   g.config.NodeAddr,
		State:  "alive",
	}
	g.mu.Unlock()

	g.ring.AddNode(&Node{
		ID:      g.config.NodeID,
		Addr:    g.config.NodeAddr,
		IsLocal: true,
		Alive:   true,
		Weight:  1.0,
	})

	go g.receiveLoop()
	go g.protocolLoop()

	// Join seeds
	for _, seed := range g.config.Seeds {
		g.sendJoin(seed)
	}

	log.Printf("[gossip] started on %s as %s", g.config.BindAddr, g.config.NodeID)
	return nil
}

// Stop halts the gossip protocol.
func (g *Gossip) Stop() {
	close(g.quit)
	if g.conn != nil {
		g.conn.Close()
	}
}

// Members returns current live members.
func (g *Gossip) Members() []memberUpdate {
	g.mu.RLock()
	defer g.mu.RUnlock()
	var members []memberUpdate
	for _, m := range g.members {
		members = append(members, memberUpdate{
			NodeID: m.NodeID,
			Addr:   m.Addr,
			State:  m.State,
		})
	}
	return members
}

func (g *Gossip) receiveLoop() {
	buf := make([]byte, 65536)
	for {
		select {
		case <-g.quit:
			return
		default:
		}
		g.conn.SetReadDeadline(time.Now().Add(1 * time.Second))
		n, remoteAddr, err := g.conn.ReadFromUDP(buf)
		if err != nil {
			continue
		}
		var msg gossipMessage
		if err := json.Unmarshal(buf[:n], &msg); err != nil {
			continue
		}
		g.handleMessage(msg, remoteAddr)
	}
}

func (g *Gossip) protocolLoop() {
	ticker := time.NewTicker(g.config.PingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-g.quit:
			return
		case <-ticker.C:
			g.doPingRound()
			g.checkSuspects()
		}
	}
}

func (g *Gossip) doPingRound() {
	g.mu.RLock()
	var targets []string
	for id, m := range g.members {
		if id != g.config.NodeID && m.State == "alive" {
			targets = append(targets, id)
		}
	}
	g.mu.RUnlock()

	if len(targets) == 0 {
		return
	}

	// Pick random target
	target := targets[rand.Intn(len(targets))]
	g.mu.RLock()
	m := g.members[target]
	g.mu.RUnlock()
	if m == nil {
		return
	}

	g.sendPing(m.Addr)

	g.mu.Lock()
	if ms, ok := g.members[target]; ok {
		ms.LastPing = time.Now()
	}
	g.mu.Unlock()
}

func (g *Gossip) checkSuspects() {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := time.Now()
	for id, m := range g.members {
		if id == g.config.NodeID {
			continue
		}
		if m.State == "alive" && !m.LastPing.IsZero() && now.Sub(m.LastPing) > g.config.SuspectTimeout {
			m.State = "suspect"
			m.SuspectAt = now
			log.Printf("[gossip] node %s is now suspect", id)
		}
		if m.State == "suspect" && now.Sub(m.SuspectAt) > g.config.SuspectTimeout*2 {
			m.State = "dead"
			log.Printf("[gossip] node %s declared dead", id)
			g.ring.RemoveNode(id)
		}
	}
}

func (g *Gossip) handleMessage(msg gossipMessage, from *net.UDPAddr) {
	switch msg.Type {
	case "ping":
		g.sendAck(from)
	case "ack":
		g.mu.Lock()
		if m, ok := g.members[msg.SenderID]; ok {
			m.State = "alive"
			m.LastPing = time.Now()
		}
		g.mu.Unlock()
	case "join":
		g.handleJoin(msg, from)
	case "membership":
		g.mergeMembership(msg.Members)
	}
}

func (g *Gossip) handleJoin(msg gossipMessage, from *net.UDPAddr) {
	for _, u := range msg.Members {
		g.addMember(u)
	}
	// Reply with our membership
	g.sendMembership(from)
}

func (g *Gossip) addMember(u memberUpdate) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.members[u.NodeID]; !exists {
		g.members[u.NodeID] = &memberState{
			NodeID:   u.NodeID,
			Addr:     u.Addr,
			State:    u.State,
			LastPing: time.Now(),
		}
		g.ring.AddNode(&Node{
			ID:    u.NodeID,
			Addr:  u.Addr,
			Alive: u.State == "alive",
		})
		log.Printf("[gossip] added member %s at %s", u.NodeID, u.Addr)
	}
}

func (g *Gossip) mergeMembership(updates []memberUpdate) {
	for _, u := range updates {
		g.addMember(u)
	}
}

func (g *Gossip) sendPing(addr string) {
	msg := gossipMessage{Type: "ping", SenderID: g.config.NodeID}
	g.sendUDP(addr, msg)
}

func (g *Gossip) sendAck(to *net.UDPAddr) {
	msg := gossipMessage{Type: "ack", SenderID: g.config.NodeID}
	data, _ := json.Marshal(msg)
	g.conn.WriteToUDP(data, to)
}

func (g *Gossip) sendJoin(addr string) {
	msg := gossipMessage{
		Type:     "join",
		SenderID: g.config.NodeID,
		Members: []memberUpdate{{
			NodeID: g.config.NodeID,
			Addr:   g.config.NodeAddr,
			State:  "alive",
		}},
	}
	g.sendUDP(addr, msg)
}

func (g *Gossip) sendMembership(to *net.UDPAddr) {
	g.mu.RLock()
	var members []memberUpdate
	for _, m := range g.members {
		members = append(members, memberUpdate{
			NodeID: m.NodeID,
			Addr:   m.Addr,
			State:  m.State,
		})
	}
	g.mu.RUnlock()

	msg := gossipMessage{
		Type:     "membership",
		SenderID: g.config.NodeID,
		Members:  members,
	}
	data, _ := json.Marshal(msg)
	g.conn.WriteToUDP(data, to)
}

func (g *Gossip) sendUDP(addr string, msg gossipMessage) {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return
	}
	g.conn.WriteToUDP(data, udpAddr)
}

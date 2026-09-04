package dispatch

import (
	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/client"
)

// nodeGroup holds keys destined for a single cluster node during batch fan-out.
type nodeGroup struct {
	addr    string
	isLocal bool
	indices []int    // positions in original keys array
	keys    [][]byte
	values  [][]byte // for MSet
}

// isLocalKey checks if a key belongs to this node via rendezvous hashing.
// Returns (primary node, isLocal). When Ring is nil (standalone), always returns (nil, true).
func (d *Dispatcher) isLocalKey(key []byte) (*cluster.Node, bool) {
	if d.Ring == nil {
		return nil, true
	}
	primary := d.Ring.GetPrimary(key)
	if primary == nil {
		return nil, true // no nodes in ring â€” handle locally
	}
	return primary, primary.IsLocal
}

// getProxyConn returns a cached client connection to a remote node.
func (d *Dispatcher) getProxyConn(addr string) (*client.Client, error) {
	d.proxyMu.Lock()
	defer d.proxyMu.Unlock()
	if d.proxyPool == nil {
		d.proxyPool = make(map[string]*client.Client)
	}
	if c, ok := d.proxyPool[addr]; ok {
		return c, nil
	}
	c, err := client.New(addr)
	if err != nil {
		return nil, err
	}
	d.proxyPool[addr] = c
	return c, nil
}

// removeProxyConn evicts a proxy connection on error.
func (d *Dispatcher) removeProxyConn(addr string) {
	d.proxyMu.Lock()
	defer d.proxyMu.Unlock()
	if c, ok := d.proxyPool[addr]; ok {
		c.Close()
		delete(d.proxyPool, addr)
	}
}

// groupKeysByNode partitions keys by their primary node.
func (d *Dispatcher) groupKeysByNode(keys [][]byte, values [][]byte) map[string]*nodeGroup {
	groups := make(map[string]*nodeGroup)
	for i, key := range keys {
		primary, isLocal := d.isLocalKey(key)
		addr := "local"
		if !isLocal && primary != nil {
			addr = primary.Addr
		}
		g, ok := groups[addr]
		if !ok {
			g = &nodeGroup{addr: addr, isLocal: isLocal || addr == "local"}
			groups[addr] = g
		}
		g.indices = append(g.indices, i)
		g.keys = append(g.keys, key)
		if values != nil {
			g.values = append(g.values, values[i])
		}
	}
	return groups
}

// CloseProxyPool closes all cached proxy connections. Called during shutdown.
func (d *Dispatcher) CloseProxyPool() {
	d.proxyMu.Lock()
	defer d.proxyMu.Unlock()
	for addr, c := range d.proxyPool {
		c.Close()
		delete(d.proxyPool, addr)
	}
}

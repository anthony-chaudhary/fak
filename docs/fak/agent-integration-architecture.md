# Agent Integration Architecture for fak

This document describes how external coding agents integrate with the fak (Fused Agent Kernel) - the tool-call firewall and policy boundary that sits between an AI agent and its tools.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENT INTEGRATION LAYERS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐        ┌──────────────────┐        ┌─────────────────┐   │
│  │ External    │        │   fak Gateway    │        │  Tool Backend   │   │
│  │ Coding Agent │◄──────►│   (HTTP/MCP)     │◄──────►│  (Engines)      │   │
│  │ (Claude Code│        │                  │        │                 │   │
│  │  / Custom)  │        │  ┌────────────┐ │        │ ┌─────────────┐ │   │
│  └──────────────┘        │  │   Kernel   │ │        │ │ Local/Mock  │ │   │
│                          │  │            │ │        │ │ Remote      │ │   │
│                          │  │-Adjudicate │ │        │ │ In-Kernel   │ │   │
│                          │  │-VDSO       │ │        │ └─────────────┘ │   │
│                          │  │-MMU        │ │        └─────────────────┘   │
│                          │  └────────────┘ │                            │
│                          │                  │                            │
│                          │  ┌────────────┐ │                            │
│                          │  │   Policy   │ │                            │
│                          │  │   Floor    │ │                            │
│                          │  └────────────┘ │                            │
│                          └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Gateway Entry Points

The `fak serve` gateway provides multiple protocol entry points for agent integration:

| Protocol | Endpoint | Purpose |
|-----------|----------|---------|
| **OpenAI-compatible** | `POST /v1/chat/completions` | Standard tool-call adjudication proxy |
| **Anthropic Messages** | `POST /v1/messages` | Native Claude Code integration |
| **Direct syscall** | `POST /v1/fak/syscall` | Run one adjudicated tool call directly |
| **Adjudication only** | `POST /v1/fak/adjudicate` | Get verdict without dispatching |
| **MCP over stdio** | `fak serve --stdio` | Model Context Protocol integration |
| **MCP over HTTP** | `POST /mcp` | HTTP-based MCP |

### 2. Kernel API (ABI)

The frozen ABI (`internal/abi/types.go`) defines the stable syscall interface:

```go
// The core syscall interface - every agent tool call becomes this
type Kernel interface {
    // Submit adjudicates (folds the Adjudicator chain) and enqueues the call
    Submit(ctx context.Context, c *ToolCall) (SubmissionHandle, Verdict)

    // Reap blocks for the completion of a specific submission
    Reap(ctx context.Context, h SubmissionHandle) (*Result, error)

    // Syscall is the synchronous convenience: Submit then Reap
    Syscall(ctx context.Context, c *ToolCall) (*Result, Verdict)

    // Resolver is the active Ref backend
    Resolver() Resolver

    // Negotiate intersects a caller's advertised caps with what's registered
    Negotiate(advertised []Capability) []Capability
}
```

#### ToolCall Structure

```go
type ToolCall struct {
    Op      OpCode            // Operation selector
    Tool    string            // Logical tool name (training token)
    Engine  string            // Optional per-call engine route
    Args    Ref               // Addressable handle to arguments
    Caps    []Capability      // Caller-advertised capabilities
    Spec    SpeculationContext // For speculative execution
    Txn     TxnID             // For transactional context
    SeqNo   uint64            // Submission identity
    TraceID string            // Correlation ID
    Meta    map[string]string // Open metadata
    Ext     map[ExtKey]Ref    // Typed sidecar payloads
}
```

#### Verdict Types

The kernel returns typed verdicts from a closed, discriminated union:

| Verdict | Meaning |
|---------|---------|
| `Allow` | Call permitted - dispatch to engine |
| `Deny` | Provable refusal - blocked |
| `Transform` | Rewrite Args before dispatch |
| `Quarantine` | Hold result out of agent context (MMU) |
| `RequireWitness` | Gate pending independent verification |
| `Defer` | Not adjudicable here - pass to next link |

### 3. Communication Protocol

#### Request Flow

```
1. Agent proposes tool call
   ↓
2. Gateway receives HTTP/MCP request
   ↓
3. Gateway adjudicates (vDSO → Adjudicator chain)
   ↓
4. Verdict returned:
   - Allow/Transform → Dispatch to engine
   - Deny/Quarantine → Return refusal
   ↓
5. Engine executes and returns result
   ↓
6. Result admitter chain (context-MMU)
   ↓
7. Response returned to agent
```

#### Wire Protocol Example

**OpenAI /v1/chat/completions:**
```json
{
  "model": "agent-model",
  "messages": [
    {"role": "user", "content": "Read the file README.md"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "Read",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

**fak adjudicates and returns:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "Read",
            "arguments": "{\"file_path\":\"README.md\"}"
          }
        }
      ]
    }
  }],
  "_fak": {
    "adjudicated": [
      {"tool": "Read", "verdict": "ALLOW", "by": "monitor"}
    ]
  }
}
```

### 4. Policy / Capability Floor

Agents interact with a declarative capability floor (`fak-policy/v1`):

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["read_file", "write_file", "grep"],
  "allow_prefix": ["read_", "get_", "search_"],
  "deny": {
    "delete_account": "POLICY_BLOCK",
    "rm_rf": "DESTRUCTIVE_OP"
  },
  "self_modify_globs": [".git/", ".dos/", "internal/kernel/"],
  "redact_fields": ["password", "secret", "api_key"]
}
```

**Policy workflow:**
```bash
# Dump built-in default
fak policy --dump > policy.json

# Edit policy.json for your agent's needs

# Validate before deploy
fak policy --check policy.json

# Load at gateway start
fak serve --policy policy.json --addr :8080
```

### 5. Initialization Points

#### For Gateway Deployment

```bash
# Basic gateway with local model
fak serve \
  --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# With in-kernel model
fak serve \
  --addr 127.0.0.1:8080 \
  --engine inkernel \
  --gguf /path/to/model.gguf \
  --tokenizer /path/to/tokenizer \
  --model qwen3.6-27b

# MCP over stdio
fak serve --stdio --policy policy.json
```

#### For Programmatic Integration (Go)

```go
package main

import (
    "context"
    "github.com/anthony-chaudhary/fak/internal/abi"
    "github.com/anthony-chaudhary/fak/internal/kernel"
    "github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func main() {
    // 1. Register your engine
    abi.RegisterEngine("myengine", MyEngine{})

    // 2. Configure policy
    adjudicator.Default.SetPolicy(adjudicator.Policy{
        Allow: map[string]bool{
            "read_file": true,
            "write_file": true,
        },
        Deny: map[string]abi.ReasonCode{
            "delete": abi.ReasonPolicyBlock,
        },
    })

    // 3. Create kernel
    k := kernel.New("myengine")

    // 4. Make syscall
    call := &abi.ToolCall{
        Tool: "read_file",
        Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"x"}`)},
    }
    result, verdict := k.Syscall(context.Background(), call)

    // Handle result based on verdict
    switch verdict.Kind {
    case abi.VerdictAllow:
        // Process result.Payload
    case abi.VerdictDeny:
        // Log refusal
    }
}
```

### 6. Extension Points

#### Custom Adjudicators

```go
type MyAdjudicator struct{}

func (a MyAdjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
    if c.Tool == "dangerous_op" {
        return abi.Verdict{
            Kind:   abi.VerdictDeny,
            Reason: abi.ReasonPolicyBlock,
            By:     "my-policy",
        }
    }
    return abi.Verdict{Kind: abi.VerdictDefer}
}

func (a MyAdjudicator) Caps() []abi.Capability {
    return []abi.Capability{"my.custom.feature"}
}

// Register at init time
func init() {
    abi.RegisterAdjudicator(100, MyAdjudicator{})
}
```

#### Custom Engines

```go
type MyEngine struct{}

func (e MyEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
    // Execute tool call
    args := refBytes(ctx, c.Args)
    result := executeTool(c.Tool, args)
    ref := putBytes(ctx, result)
    return &abi.Result{
        Call:    c,
        Payload: ref,
        Status:  abi.StatusOK,
        Meta:    map[string]string{"engine": "myengine"},
    }, nil
}

func (e MyEngine) Caps() []abi.Capability { return nil }

// Register
func init() {
    abi.RegisterEngine("myengine", MyEngine{})
}
```

#### Custom Fast Paths (vDSO)

```go
type MyFastPath struct{}

func (fp MyFastPath) Lookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
    if c.Tool == "cached_query" && isCached(c.Args) {
        return serveFromCache(c), true
    }
    return nil, false
}

func (fp MyFastPath) Caps() []abi.Capability { return nil }

func init() {
    abi.RegisterFastPath(50, MyFastPath{})
}
```

## Agent Configuration

### Claude Code Integration

The `dogfood-claude.sh` script demonstrates complete Claude Code integration:

```bash
# One-command setup
./scripts/dogfood-claude.sh

# This:
# 1. Builds fak
# 2. Ensures local model (ollama or shim)
# 3. Starts fak serve with capability floor
# 4. Points Claude Code at http://127.0.0.1:8080
# 5. Launches Claude Code
```

**Environment variables for Claude Code:**
- `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` - Point at fak gateway
- `CLAUDE_CONFIG_DIR` - Isolated account directory
- `FAK_DOGFOOD_POLICY` - Capability floor manifest

### Custom Agent Integration

Any agent that speaks OpenAI-compatible or Anthropic Messages protocol can integrate:

1. **Start fak gateway**
   ```bash
   fak serve --addr :8080 --base-url <your-model-url> --policy floor.json
   ```

2. **Configure agent base URL**
   - OpenAI SDK: `baseURL = "http://127.0.0.1:8080/v1"`
   - Anthropic SDK: `baseURL = "http://127.0.0.1:8080"`

3. **Agent makes normal tool calls** - fak transparently adjudicates

## Coherence and Cross-Agent Communication

The vDSO coherence bus enables cross-agent cache invalidation:

```bash
# Query what changed since cursor
curl http://127.0.0.1:8080/v1/fak/changes?since=123

# Revoke a poisoned witness
curl -X POST http://127.0.0.1:8080/v1/fak/revoke \
  -H 'Content-Type: application/json' \
  -d '{"witness":"git-commit-abc123"}'
```

**Coherence events:**
```json
{
  "kind": "mutation",   // or "revocation"
  "seq": 456,
  "tool": "write_file",
  "tags": ["file:config.json"],
  "world_ver": 789,
  "trust_epoch": 10
}
```

## Security Model

### Default-Deny Posture

- **Without a policy**: All tool calls denied
- **With explicit allow**: Only named tools permitted
- **With allow_prefix**: Tools matching prefix patterns permitted
- **With explicit deny**: Named tools refused with reason

### Refusal Reasons (Closed Vocabulary)

```
DEFAULT_DENY    - Tool not in allow list
POLICY_BLOCK    - Explicitly denied tool
SELF_MODIFY     - Agent modifying its own kernel/config
LEASE_HELD      - Resource held by another agent
TRUST_VIOLATION - Witness verification failed
MALFORMED       - Invalid arguments
MISROUTE        - Call routed to wrong system
RATE_LIMITED    - Rate limit exceeded
SECRET_EXFIL    - Potential secret exfiltration
UNWITNESSED     - Claim lacks independent verification
OVERSIZE        - Payload exceeds size limit
UNKNOWN_TOOL    - Tool not recognized
```

### Context-MMU Quarantine

Results flagged with certain taints are held out of agent context:

```go
type TaintLabel uint8

const (
    TaintTainted     TaintLabel = iota // Untrusted
    TaintTrusted                       // Adjudicated trusted
    TaintQuarantined                   // Held from context
)
```

## Performance Characteristics

### Adjudication Cost

```
in-process adjudication p50: ~1,300 ns
vDSO hit p50: ~50 ns
engine dispatch: variable (network/IO)
```

### vDSO Cache Layers

1. **Pure tier** - Pure function, zero allocation
2. **Content-addressed tier** - Cached by content hash
3. **Static tier** - Static answers

## Debugging and Observability

### Metrics Endpoint

```
GET /metrics
```

Prometheus-format metrics for:
- HTTP latency/status
- Verdict counters
- Kernel counters (submits, denies, vDSO hits)
- In-flight requests

### Debug Endpoint

```
GET /debug/vars
```

JSON snapshot of:
- Gateway config/uptime
- Runtime memory/goroutines
- Kernel counters
- Completed operation rows

### Trace Correlation

Every request gets a `TraceID` (or generates one). This threads through:
- HTTP `X-Trace-ID` header
- Kernel operations
- Per-operation verdict logs
- Metrics

## Further Reading

- `../README.md` - Project overview
- `ARCHITECTURE.md` - Extension model and ABI design
- `POLICY.md` - Capability floor schema
- `GETTING-STARTED.md` - Tier 0-2 setup guide
- `DOGFOOD-CLAUDE.md` - Claude Code integration example
- `../explainers/policy-in-the-kernel.md` - Policy architecture
- `../explainers/addressable-kv-cache.md` - Addressable cache design

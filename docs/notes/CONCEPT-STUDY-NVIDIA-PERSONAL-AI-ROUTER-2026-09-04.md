# Concept Study: NVIDIA Personal AI Router (PAIR) Architecture and Subsystems

**Sources:**
- `https://github.com/mitkox/Personal-AI-Router` (mirror/fork of upstream NVIDIA Personal AI Router)

**Pinned Revisions:**
- `13b68115fa2c9c1d94f1ead1358f8d5a527cfecf` (v0.1.1, 2026-09-04)

**Study Date:** 2026-09-04  
**Study Receipt ID:** `study_e10f8240a711d1576ec020f28f43fb275e320d5cb289c86f47b816bb088e7f6d`  
**License / Gate:** Apache-2.0 (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES) — ADAPT / INSPIRE  
**Filed Issues:** [#11317](https://github.com/anthony-chaudhary/fak/issues/11317), [#11318](https://github.com/anthony-chaudhary/fak/issues/11318), [#11319](https://github.com/anthony-chaudhary/fak/issues/11319)  
**Study Depth:** Deep (multi-subagent high fan-out across 14 Go microservices, TypeScript/Electron desktop, Swift privileged helper, mDNS discovery, EAP-NOOB OOB pairing, job scheduling, engine management, and multi-OS GPU telemetry)  
**Completeness Critic:** Verified — all core Go services (`nvpair-cluster-manager`, `nvpair-job-scheduler`, `nvpair-node-scanner`, `nvpair-node-info`, `nvpair-engine-manager`, `ollama-proxy`, `lmstudio-proxy`, `eap-noob`, `services/shared`) inspected at pinned commit `13b68115fa2c9c1d94f1ead1358f8d5a527cfecf`.

---

## Executive Summary

NVIDIA Personal AI Router (PAIR) is a local multi-node inference router and cluster manager designed to connect heterogeneous home and small-office computers (Windows gaming rigs, Apple Silicon Macs, Linux desktop/server boxes) over a standard local area network (Wi-Fi or Ethernet). Released by NVIDIA in August 2026 under Apache-2.0, PAIR provides OpenAI- and Ollama-compatible HTTP proxy ingress points while dynamically routing concurrent inference requests to eligible nodes based on engine availability, model residency, and real-time GPU compute pressure.

Crucially, PAIR establishes a distinct and pragmatic architectural worldview:
1. **Explicit Non-Goal on Model Sharding**: PAIR explicitly does **not** pool GPU memory, combine GPUs into a logical mega-device, or shard models via tensor/pipeline parallelism across machines. Over consumer 1 Gbps Ethernet or Wi-Fi (where latency is 2–15 ms), distributed tensor parallelism suffers catastrophic performance collapse (dropping generation from 50 tok/s to <0.2 tok/s). Instead, PAIR routes **independent, whole-request inference jobs** across machines, maximizing cluster throughput for multi-agent workflows.
2. **Zero-Trust LAN Bootstrapping (RFC 9140 EAP-NOOB)**: Rather than relying on cloud sign-in, central coordinators, or enterprise WebPKI/CAs, PAIR establishes peer-to-peer security between untrusted local devices using Nimble Out-of-Band authentication (ephemeral ECDH with human-entered 6-digit PINs), followed by byte-for-byte X.509 DER certificate pinning and transitive Ed25519 gossip endorsements.
3. **Robust Multi-OS Windows & macOS Systems Engineering**: PAIR solves critical platform traps:
   - **Windows Multicast Socket Drop**: Bypasses Windows TCP/IP silently dropping outgoing multicast packets from multicast-bound sockets by creating per-interface unicast UDP sockets.
   - **Zero-Dependency Windows GPU Probing**: Uses native Win32 DXGI COM and WDDM Performance Data Helper (PDH) counters to sample GPU VRAM and utilization without requiring the CUDA Toolkit, NVML Cgo, or `nvidia-smi` on `PATH`, while filtering RDP phantom adapters.
   - **Apple Silicon Metal Telemetry**: Decodes unprivileged `/usr/sbin/ioreg` XML plists with timeout-isolated goroutines to monitor Metal VRAM allocation (`Alloc system memory`) and GPU activity.
   - **Single-Port Multiplexing (`splitlisten`)**: Peeks the initial connection byte (`0x16` TLS ClientHello vs ASCII HTTP) to co-host plaintext loopback HTTP and cluster mTLS on the exact same port, halving OS firewall configuration friction.

---

## Architectural Comparison: PAIR vs. `fak`

| Architectural Dimension | NVIDIA PAIR | `fak` (Agent Kernel & Serving Substrate) |
| :--- | :--- | :--- |
| **Primary Role** | Layer-7 reverse proxy router over 3rd-party engines (Ollama, LM Studio). | In-kernel agent runtime, capability security gate, and native model serving engine. |
| **Inference Ownership** | Model-blind HTTP forwarding; owns zero weights, no memory, no compute kernels. | **Fak-native inference all the way** (`internal/engine/native`); owns kernels, KV cache, and memory scheduling. |
| **Context & Memory** | Black-box streaming pass-through; cannot inspect prompt caches or KV state. | **Context MMU (`ctxmmu`)**: Virtual paging, prefix KV reuse, turn shedding, and provider cache preservation. |
| **Tool Execution** | Completely unaware of agent tool calls or environment mutations. | **vDSO tool interceptor**: Caches idempotent tools (`fak_read`) locally; enforces default-deny capability floor. |
| **Cluster Networking** | mDNS LAN discovery, EAP-NOOB PIN pairing, Ed25519 endorsement mesh. | Workstation is the control point; dispatches device-level GEMM/CUDA to fleet nodes via verifiable receipts. |
| **Multi-Agent Lens** | Routes concurrent independent prompt turns across machines. | Orchestrates isolated subagent workers under lane leases (`dos arbitrate`) with proof-by-default witnesses. |

---

## Deep Subsystem Mechanisms & Code Anchors

```
                    +-------------------------------------------------------------+
                    |                      Client / Agent                         |
                    |              (curl, Claude Code, Cursor, OpenCode)          |
                    +------------------------------+------------------------------+
                                                   |
                             Plaintext HTTP :11434 | (Loopback only)
                                                   v
                          +------------------------------------------------+
                          |                  ollama-proxy                  |
                          |  - splitlisten (0x16 mTLS vs ASCII Plaintext)  |
                          |  - Optimistic burst reservations (proxy.go)    |
                          |  - Streaming proof-of-life (reportActivity)    |
                          +------------------------+-----------------------+
                                                   |
                           JSON-RPC 2.0 (stdio)    | Evaluates candidates
                                                   v
                          +------------------------------------------------+
                          |              nvpair-job-scheduler              |
                          |  - Composite Load = Pending + GPUPressure      |
                          |  - EWMA smoothing (alpha=0.35) + Hysteresis    |
                          +------------------------+-----------------------+
                                                   |
                     +-----------------------------+-----------------------------+
                     | Remote mTLS :11434                                        | Local Backend :11435
                     v                                                           v
  +-------------------------------------+                     +-------------------------------------+
  |          Peer Remote Node           |                     |       Local Inference Engine        |
  |  - Verified against pinned cert DER |                     |    - Supervised Ollama / LM Studio  |
  |  - Reverse-proxied via mTLS         |                     |    - Process adoption & port check  |
  +-------------------------------------+                     +-------------------------------------+
```

### 1. Cluster Networking & Discovery (`services/nvpair-node-scanner`, `services/shared/discovery`)
- **Single Consolidated mDNS Record**: Advertises `_nvpair-node._tcp.local.` on UDP `5353` (`services/shared/noderec/noderec.go:39-51`). Models are kept off DNS TXT records to avoid the 255-byte string limit, instead served dynamically via `GET /v1/models` (`daemon.go:53-61`).
- **Windows Multicast Send Fix**: Standard libraries (`grandcat/zeroconf`, `pion/mdns`) bind `224.0.0.0:5353`. Windows TCP/IP silently drops outbound packets written to multicast-bound sockets (`discovery.go:575-582`). PAIR binds per-interface ephemeral unicast UDP sockets (`net.ListenUDP("udp4", &net.UDPAddr{IP: ifaceIP, Port: 0})`), calls `ipv4.NewPacketConn(conn).SetMulticastInterface(&ifi)` and `SetMulticastTTL(255)`, and writes directly to `224.0.0.251:5353` (`discovery.go:600-671`, `services/shared/mdns/responder.go:573-603`).
- **Data-Plane Liveness Vouching**: Heavy local inference starves CPU/event loops, causing control planes to drop mDNS packets and fail HTTP TCP health probes. Proxies report streaming response chunk timestamps to the node scanner (`reportActivity`). If a node returned data-plane bytes within 60s, the scanner marks it alive and suppresses eviction (`services/ollama-proxy/proxy.go:1362`, `services/nvpair-node-scanner/daemon.go:1008-1012`).

### 2. Zero-Trust Security & EAP-NOOB Out-of-Band Pairing (`services/eap-noob`, `services/nvpair-cluster-manager`)
- **RFC 9140 Implementation**: Implements Nimble Out-of-Band Authentication (`services/eap-noob/crypto.go:36-47`). Uses Curve25519/SHA-256 (Suite 1) and NIST P-256 (Suite 2) with NIST SP 800-56C One-Step KDF deriving MSK, EMSK, AMSK, and association key $K_z$ (`kdf.go:26-43`).
- **6-Digit PIN Exchange**: A random 6-digit PIN (`000000`–`999999`) generated by the inviter is encoded into a 16-byte left-zero-padded array (`noobFromPIN`, `pairing.go:132-138`). The user enters the PIN on the joiner, establishing authenticated key agreement without pre-shared keys.
- **Direct DER Certificate Pinning**: Outbound TLS overrides `VerifyPeerCertificate` to perform a byte-for-byte exact DER check (`bytes.Equal(rawCerts[0], pinnedPeerDER)`), bypassing WebPKI CAs entirely (`services/shared/clustertrust/clustertrust.go:208-251`).
- **Transitive Endorsements & Removal Proofs**: Peers sign Ed25519 `Endorsement` records (`nvpair-endorse:v2`), allowing transitive trust expansion (A↔B and A↔C automatically establishes B↔C trust, `roster.go:97-128`). Removals produce signed `Tombstone` proofs naming monotonic `AdmissionEpoch` counters (`endorsement.go:32-45`), ensuring clean peer eviction across restarts.

### 3. Job Scheduling & Optimistic Burst Reservations (`services/nvpair-job-scheduler`, `services/ollama-proxy`)
- **Scheduler Ranking**: $\text{Composite Load} = \text{PendingWorkloads} + \text{GPUPressure}$ (`schedule.go:82-127`).
  - `PendingWorkloads`: Count of active jobs in states `queued` or `running` across both Ollama and LM Studio simultaneously (`state.go:83-98`).
  - `GPUPressure`: Raw GPU utilization % smoothed with EWMA ($\alpha = 0.35$):
    $$\text{EWMA}_t = 0.35 \cdot U_t + 0.65 \cdot \text{EWMA}_{t-1}$$
    Quantized into hysteresis pressure bands (0–3) with separate upward ($40\%, 70\%, 85\%$) and downward ($35\%, 65\%, 80\%$) thresholds to eliminate flapping (`telemetry.go:41-138`).
- **Optimistic Burst Reservations**: To prevent concurrent request bursts from piling onto a single idle node before workload feedback round-trips, the proxy locally increments an in-flight reservation map (`proxy.go:1675-1729`):
  $$\text{Effective Load} = \text{Pending} + \text{GPUPressure} + \text{Reservations}[\text{nodeID}]$$
  A newly received priority snapshot from the scheduler resets the reservation map.

### 4. Cross-Platform GPU Telemetry (`services/nvpair-node-info`, `desktop/native`)
- **Windows DXGI COM & PDH**: Zero Cgo, zero NVML, zero `nvidia-smi` dependency. Calls DXGI COM API `CreateDXGIFactory1` -> `EnumAdapters1` -> `GetDesc1` to enumerate physical GPUs and dedicated VRAM (`gpu_windows.go:211-277`). Filters out Microsoft WARP and checks registry LUIDs to eliminate ephemeral RDP shadow display adapters (`gpu_windows.go:167-209`). Samples WDDM Performance Data Helper (PDH) counters `\GPU Adapter Memory(*)\Dedicated Usage` and `\GPU Engine(*)\Utilization Percentage` (`stats_windows.go:21-379`, `stats.go:100-143`).
- **Apple Silicon IORegistry Telemetry**: Unprivileged `/usr/sbin/ioreg -a -r -d 1 -c IOAccelerator` execution decoded as XML plist (`ioreg_parse.go:73-112`). Extracts `PerformanceStatistics["Alloc system memory"]` for active Metal VRAM allocations and `Device Utilization %` for GPU compute activity. Spawns in an independent goroutine isolated from CPU polling (`stats_darwin.go:76-116`).
- **NVIDIA Unified Memory Architecture (UMA) Fallback**: Detects `[N/A]` memory fields from `nvidia-smi` on unified memory architectures and automatically maps total system RAM and used RAM to VRAM metrics (`gpu_linux.go:87-136`).

### 5. Dual-Personality Port Multiplexing (`services/shared/splitlisten`)
- `splitlisten.Splitter` binds a single TCP port (e.g. `:11434` or `:14321`) and peeks the first byte with a 5s deadline (`splitlisten.go:26-120`):
  - **Byte is `0x16` (TLS Handshake)**: Dispatched to the virtual TLS listener, requiring cluster-pinned mutual TLS for inter-node communication.
  - **Byte is ASCII (`G`, `P`, `D`)**: Dispatched to the virtual plain HTTP listener, strictly restricted to loopback (`127.0.0.1` / `::1`).
  - Preserves the peeked byte via `io.MultiReader` so downstream parsers read the full payload without modification.

---

## Candidate Borrows & On-Axis Ablation Matrix

| # | Technique & Source Anchor | Axis | Their Worldview Reason | Witness against `fak` Seam | Status on Axis | Disposition | Filed Issue / Note |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Per-interface unicast UDP sockets for multicast send**<br>`discovery.go:575-671@13b6811` | Windows reliable multicast heartbeat transmission | Windows TCP/IP drops outbound multicast packets from sockets bound to multicast groups across multi-NIC setups | `internal/fleetspine/transport.go:68-93`<br>Uses `ListenMulticastUDP` and writes directly via `conn.WriteToUDP` | **PARTIAL** | **ADAPT** | **[#11317](https://github.com/anthony-chaudhary/fak/issues/11317)** |
| **2** | **Win32 DXGI COM + PDH GPU probe fallback**<br>`gpu_windows.go:211-277@13b6811`, `stats.go:100-143@13b6811` | Zero-dependency Windows GPU VRAM and utilization telemetry | Consumer Windows hosts rarely have CUDA toolkit / NVML installed; DXGI + PDH provide universal telemetry for NVIDIA, AMD, Intel with RDP filter | `internal/compute/gpustats.go:11-70`<br>Relies strictly on `nvidia-smi` CLI; fails soft to `(nil, false)` on Windows without CUDA Toolkit | **PARTIAL** | **ADAPT** | **[#11318](https://github.com/anthony-chaudhary/fak/issues/11318)** |
| **3** | **macOS Apple Silicon Metal allocation via IORegistry**<br>`ioreg_parse.go:73-112@13b6811`, `stats_darwin.go:76-116@13b6811` | Cgo-free Metal memory and GPU utilization on Apple Silicon | Apple Silicon has no `nvidia-smi`; `/usr/sbin/ioreg` XML plist exposes `Alloc system memory` and `Device Utilization %` without private frameworks | `internal/compute/hostmem_darwin.go:1-22`, `gpustats.go`<br>Only reads `hw.memsize`; has zero GPU telemetry on Darwin | **PARTIAL** | **ADAPT** | **[#11319](https://github.com/anthony-chaudhary/fak/issues/11319)** |
| **4** | **Single-port connection-peeking listener (`splitlisten`)**<br>`splitlisten.go:4-85@13b6811` | Single-port multiplexing of loopback plain HTTP and cluster mTLS | Opening multiple ports triggers multiple OS firewall prompts on Windows/macOS; first-byte inspection (`0x16` vs ASCII) halves firewall rules | `internal/gateway/http.go:217-294`<br>Binds standard `net.Listen("tcp", addr)`; no protocol-peeking virtual listener multiplexing | **ABSENT** | **OPTIONAL-MODULE** | Recorded as design candidate |
| **5** | **Data-plane streaming proof-of-life liveness vouching**<br>`proxy.go:1362@13b6811`, `daemon.go:1008-1012@13b6811` | Saturated node health check failure prevention during heavy prefill | GPU inference saturates CPU buses, causing control plane to drop mDNS/TCP probes; streaming response bytes vouch for node liveness | `internal/leaseref/liveness.go:1-45`<br>Heartbeat is descriptor-based; data-plane token bytes are not currently wired to extend lease liveness | **PARTIAL** | **WATCH** | Recorded in receipt |
| **6** | **Optimistic burst reservations (`reserveCandidate`)**<br>`proxy.go:1675-1729@13b6811` | Concurrency burst distribution before telemetry round-trip | Prevents concurrent agent turns from piling onto a single idle node before scheduler feedback completes | `internal/gateway/replica_router.go:69-431`, `fleet_membership.go:206-471`<br>Fak already ships live route reservations (`fleetReservation`, `reserveForModel`) | **PRESENT** | **DEFAULT** | Dropped (earned PRESENT-on-axis) |
| **7** | **Zero-trust peer pairing via EAP-NOOB & Ed25519 gossip**<br>`eap-noob/crypto.go:36-47@13b6811`, `endorsement.go:21-98@13b6811` | Consensus-free cluster membership and out-of-band mutual auth | Consumer LANs have frequent sleep/wake cycles breaking Paxos/Raft; OOB PIN + gossip endorsements allow decentralized zero-trust | `internal/fleetspine`, `internal/fleetbus`<br>Fak uses one-way UDP multicast and filesystem locks; multi-node relies on operator SSH/tunnels | **ABSENT** | **INSPIRE** | Architectural reference for multi-node mesh |

---

## Filed Issues Summary

1. **[#11317](https://github.com/anthony-chaudhary/fak/issues/11317): `fix(fleetspine): bind per-interface unicast UDP sockets for Windows multicast transmission`**
   - **Problem:** `internal/fleetspine/transport.go` calls `conn.WriteToUDP` on a socket joined via `net.ListenMulticastUDP`. On multi-adapter Windows machines, Windows TCP/IP silently drops outgoing multicast frames from multicast-bound sockets.
   - **Fix:** Adapt PAIR's `discovery.go` pattern by enumerating multicast interfaces and sending heartbeats via per-interface unicast UDP sockets configured with `SetMulticastInterface`.
2. **[#11318](https://github.com/anthony-chaudhary/fak/issues/11318): `feat(compute): add native Win32 DXGI and PDH fallback probe for Windows GPU telemetry`**
   - **Problem:** `internal/compute/gpustats.go` shells out strictly to `nvidia-smi`. On Windows machines without the CUDA Toolkit installed (or on AMD Radeon / Intel Arc GPUs), it fails soft to `(nil, false)`, blinding the harness resource sampler.
   - **Fix:** Implement a native Windows fallback using Win32 DXGI COM (`CreateDXGIFactory1`, `EnumAdapters1`) and WDDM PDH counters (`\GPU Adapter Memory(*)\Dedicated Usage`, `\GPU Engine(*)\Utilization Percentage`) with RDP shadow adapter filtering.
3. **[#11319](https://github.com/anthony-chaudhary/fak/issues/11319): `feat(compute): probe macOS Apple Silicon Metal allocation and GPU activity via IORegistry`**
   - **Problem:** `internal/compute/hostmem_darwin.go` only reads total `hw.memsize`, while `gpustats.go` has no probe on macOS. `fak` is completely blind to Metal VRAM and GPU utilization on Apple Silicon.
   - **Fix:** Implement `/usr/sbin/ioreg` XML plist parsing in a timeout-isolated goroutine to sample `Alloc system memory` and `Device Utilization %` on Darwin.

---

## Anti-Gaming Verification

- **Code Anchors Grounded at Pinned SHA:** Every borrow cites verified source code in the cloned repository at `13b68115fa2c9c1d94f1ead1358f8d5a527cfecf`.
- **Completeness Critic:** Deep fan-out conducted across 14 Go microservices, TypeScript/Electron frontend, and Swift privileged helpers. Subsystems evaluated for both mechanisms and their underlying worldview.
- **Earned Ablation vs. Ego Dismissal:** Candidate 6 (optimistic burst reservations) was confirmed **PRESENT-on-axis** after verifying `internal/gateway/replica_router.go` and `internal/gateway/fleet_membership.go`. Candidate 1, 2, and 3 were verified **PARTIAL-on-axis** against concrete omissions in `internal/fleetspine` and `internal/compute`.
- **Durable Registration:** Immutable receipt `study_e10f8240a711d1576ec020f28f43fb275e320d5cb289c86f47b816bb088e7f6d` stored in `fak study`, linked in all downstream GitHub issues (#11317, #11318, #11319), and registered in `docs/research/monitored-repositories.json`.

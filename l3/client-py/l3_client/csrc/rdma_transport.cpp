/**
 * L3 RDMA transport — pybind11 extension wrapping libibverbs/librdmacm.
 *
 * Mirrors the Go RDMA client (pkg/client/rdma_client.go) in C++ for use by
 * the Python RDMAClient. Provides RDMATransport class with connect, roundtrip,
 * RDMA Read, and memory registration.
 *
 * GIL RELEASE: All methods that perform RDMA I/O release the Python GIL
 * before entering busy-poll loops. Without this, 64 Python threads serialize
 * into effectively single-threaded execution (~2 GB/s instead of 10+ GB/s).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;

/**
 * Human-readable IB work completion status names (like cm_event_name for CM events).
 * Maps ibv_wc_status enum values to strings so logs say "REM_ACCESS_ERR" not "status=5".
 */
static const char* wc_status_name(int status) {
    switch (status) {
        case IBV_WC_SUCCESS:          return "SUCCESS";
        case IBV_WC_LOC_LEN_ERR:     return "LOC_LEN_ERR";
        case IBV_WC_LOC_QP_OP_ERR:   return "LOC_QP_OP_ERR";
        case IBV_WC_LOC_EEC_OP_ERR:  return "LOC_EEC_OP_ERR";
        case IBV_WC_LOC_PROT_ERR:    return "LOC_PROT_ERR";
        case IBV_WC_WR_FLUSH_ERR:    return "WR_FLUSH_ERR";
        case IBV_WC_MW_BIND_ERR:     return "MW_BIND_ERR";
        case IBV_WC_BAD_RESP_ERR:    return "BAD_RESP_ERR";
        case IBV_WC_LOC_ACCESS_ERR:  return "LOC_ACCESS_ERR";
        case IBV_WC_REM_INV_REQ_ERR: return "REM_INV_REQ_ERR";
        case IBV_WC_REM_ACCESS_ERR:  return "REM_ACCESS_ERR";
        case IBV_WC_REM_OP_ERR:      return "REM_OP_ERR";
        case IBV_WC_RETRY_EXC_ERR:   return "RETRY_EXC_ERR";
        case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR_RETRY_EXC_ERR";
        case IBV_WC_LOC_RDD_VIOL_ERR: return "LOC_RDD_VIOL_ERR";
        case IBV_WC_REM_INV_RD_REQ_ERR: return "REM_INV_RD_REQ_ERR";
        case IBV_WC_REM_ABORT_ERR:   return "REM_ABORT_ERR";
        case IBV_WC_INV_EECN_ERR:    return "INV_EECN_ERR";
        case IBV_WC_INV_EEC_STATE_ERR: return "INV_EEC_STATE_ERR";
        case IBV_WC_FATAL_ERR:       return "FATAL_ERR";
        case IBV_WC_RESP_TIMEOUT_ERR: return "RESP_TIMEOUT_ERR";
        case IBV_WC_GENERAL_ERR:     return "GENERAL_ERR";
        default:                     return "UNKNOWN";
    }
}

// Default buffer sizes — 16MB accommodates ~5 × 3MB KV entries per batch.
// Sub-batching handles overflow automatically (extra roundtrips).  32MB was
// the previous default but the per-connection overhead (128MB client, 160MB
// server) was excessive — 16MB halves that while keeping good batch efficiency.
// Must match server defaults (rdma_cgo.go).
static constexpr size_t DEFAULT_SEND_BUF_SIZE  = 16 * 1024 * 1024;  // 16MB
static constexpr size_t DEFAULT_RECV_BUF_SIZE  = 16 * 1024 * 1024;  // 16MB
static constexpr size_t DEFAULT_READ_BUF_SIZE  = 32 * 1024 * 1024;  // 32MB
static constexpr int    CQ_DEPTH       = 256;
static constexpr int    MAX_SEND_WR    = 128;
static constexpr int    MAX_RECV_WR    = 128;
static constexpr int    MAX_INLINE     = 256;

class RDMATransport {
public:
    RDMATransport(size_t send_buf_size = DEFAULT_SEND_BUF_SIZE,
                  size_t recv_buf_size = DEFAULT_RECV_BUF_SIZE,
                  size_t read_buf_size = DEFAULT_READ_BUF_SIZE,
                  int poll_timeout_ms = 30000)
        : send_buf_size_(send_buf_size),
          recv_buf_size_(recv_buf_size),
          read_buf_size_(read_buf_size),
          poll_timeout_ms_(poll_timeout_ms) {}

    ~RDMATransport() {
        try { close_impl(); } catch (...) {}
    }

    /**
     * Connect to an RDMA server via RDMA CM.
     * Mirrors Go client NewRDMA() (rdma_client.go:159-289).
     * GIL released via call_guard at binding — all args are C++ stack values.
     */
    void connect(const std::string& server_ip, int port) {
        if (connected_.load(std::memory_order_acquire))
            throw std::runtime_error("already connected");

        // Pass node (host/IP) and service (port) separately to rdma_getaddrinfo.
        // rdma_getaddrinfo wraps getaddrinfo internally — passing "ip:port" as node
        // with nullptr service causes EAI_NONAME (-2). Matches Go server pattern
        // (rdma_cgo.go:319-329).
        std::string port_str = std::to_string(port);

        // Create event channel
        cm_channel_ = rdma_create_event_channel();
        if (!cm_channel_) {
            int err = errno;
            throw std::runtime_error(
                "rdma_create_event_channel failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        // Create CM ID
        int ret = rdma_create_id(cm_channel_, &cm_id_, nullptr, RDMA_PS_TCP);
        if (ret != 0) {
            int err = errno;
            rdma_destroy_event_channel(cm_channel_);
            cm_channel_ = nullptr;
            throw std::runtime_error(
                "rdma_create_id failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        // Resolve address.
        // REGRESSION RISK — rdma_getaddrinfo() requires node and service as
        // SEPARATE arguments (matches POSIX getaddrinfo).  Passing a combined
        // "ip:port" string as node with nullptr service returns EAI_NONAME (-2),
        // which looks like a DNS failure even when connecting via a raw IP.
        // This split was accidentally reverted in v0.2.3 (because the stale .so
        // was not rebuilt — see Gotcha 1) and required a third fix in v0.2.4.
        // Do NOT merge node and service back into one argument.
        struct rdma_addrinfo hints = {};
        hints.ai_port_space = RDMA_PS_TCP;
        struct rdma_addrinfo* res = nullptr;
        errno = 0;
        ret = rdma_getaddrinfo(
            const_cast<char*>(server_ip.c_str()),   // node: IP only, no ":port"
            const_cast<char*>(port_str.c_str()),    // service: port string only
            &hints, &res);
        if (ret != 0) {
            int err = errno;
            cleanup_cm();
            throw std::runtime_error(
                "rdma_getaddrinfo failed for " + server_ip + ":" + port_str + ": " +
                std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ", ret " + std::to_string(ret) + ")");
        }

        ret = rdma_resolve_addr(cm_id_, nullptr, res->ai_dst_addr, 5000);
        rdma_freeaddrinfo(res);
        if (ret != 0) {
            int err = errno;
            cleanup_cm();
            throw std::runtime_error(
                "rdma_resolve_addr failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        // Wait ADDR_RESOLVED
        wait_for_event(RDMA_CM_EVENT_ADDR_RESOLVED);

        // Resolve route
        ret = rdma_resolve_route(cm_id_, 5000);
        if (ret != 0) {
            cleanup_cm();
            throw std::runtime_error("rdma_resolve_route failed");
        }

        // Wait ROUTE_RESOLVED
        wait_for_event(RDMA_CM_EVENT_ROUTE_RESOLVED);

        // Get verbs context
        ctx_ = cm_id_->verbs;
        if (!ctx_) {
            cleanup_cm();
            throw std::runtime_error("no verbs context after route resolve");
        }

        // Allocate PD
        pd_ = ibv_alloc_pd(ctx_);
        if (!pd_) {
            cleanup_cm();
            throw std::runtime_error("ibv_alloc_pd failed");
        }

        // Create CQ
        cq_ = ibv_create_cq(ctx_, CQ_DEPTH, nullptr, nullptr, 0);
        if (!cq_) {
            ibv_dealloc_pd(pd_); pd_ = nullptr;
            cleanup_cm();
            throw std::runtime_error("ibv_create_cq failed");
        }

        // Create QP
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = cq_;
        qp_attr.recv_cq = cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr  = MAX_SEND_WR;
        qp_attr.cap.max_recv_wr  = MAX_RECV_WR;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = MAX_INLINE;

        ret = rdma_create_qp(cm_id_, pd_, &qp_attr);
        if (ret != 0) {
            ibv_destroy_cq(cq_); cq_ = nullptr;
            ibv_dealloc_pd(pd_); pd_ = nullptr;
            cleanup_cm();
            throw std::runtime_error("rdma_create_qp failed: " + std::to_string(ret));
        }

        // Register internal buffers
        register_buffers();

        // Pre-post recv
        post_recv();

        // Connect
        struct rdma_conn_param conn_param = {};
        conn_param.initiator_depth    = 16;
        conn_param.responder_resources = 16;
        conn_param.rnr_retry_count    = 7;
        conn_param.retry_count        = 7;

        ret = rdma_connect(cm_id_, &conn_param);
        if (ret != 0) {
            cleanup_all();
            throw std::runtime_error("rdma_connect failed: " + std::to_string(ret));
        }

        // Wait ESTABLISHED — if server rejects/is down, cleanup before throwing
        // so the destructor doesn't call rdma_disconnect on a partial CM ID
        // (which can segfault in the RDMA driver).
        try {
            wait_for_event(RDMA_CM_EVENT_ESTABLISHED);
        } catch (...) {
            cleanup_all();
            throw;
        }
        connected_.store(true, std::memory_order_release);
    }

    /**
     * Close the RDMA connection and release all resources.
     * Uses manual GIL release so it works from both Python calls and C++ destructor.
     */
    void close() {
        if (!connected_.load(std::memory_order_acquire) && !cm_id_)
            return;
        py::gil_scoped_release release;
        close_impl();
    }

    /**
     * Send a request and receive a response via RDMA Send/Recv.
     * Returns raw response bytes.
     * GIL is released around the RDMA I/O (roundtrip_impl), re-acquired for py::bytes.
     */
    py::bytes roundtrip(const std::string& request_bytes) {
        std::string result;
        {
            py::gil_scoped_release release;
            result = roundtrip_impl(request_bytes.data(), request_bytes.size());
        }
        return py::bytes(result);
    }

    /**
     * Perform an RDMA Read into the internal 32MB read buffer.
     * Returns the read data as bytes.
     * GIL is released around the RDMA I/O, re-acquired for py::bytes.
     */
    py::bytes rdma_read(uint32_t rkey, uint64_t remote_addr, uint32_t length) {
        std::string result;
        {
            py::gil_scoped_release release;
            result = rdma_read_impl(rkey, remote_addr, length);
        }
        return py::bytes(result);
    }

    /**
     * Zero-copy RDMA Read directly into a user-registered memory region.
     * Used when SGL has a registered MR (e.g., HiCache tensor buffer).
     * GIL released via call_guard at binding — void return, pure C++.
     */
    void rdma_read_into(uint32_t rkey, uint64_t remote_addr, uint32_t length,
                        uint64_t local_addr, uint32_t local_lkey) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");

        bool do_timing = (sample_rate_ > 0 && ++rd_counter_ % sample_rate_ == 0);
        auto t0 = do_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        void* local_buf = reinterpret_cast<void*>(local_addr);
        post_rdma_read(local_buf, local_lkey, rkey, remote_addr, length);
        if (do_timing) {
            auto t1 = std::chrono::steady_clock::now();
            total_rdma_read_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            rdma_read_count_++;
        }
    }

    /**
     * Try RDMA Read into user buffer, returning WC status instead of throwing.
     * Returns 0 on success, nonzero ibv_wc_status on failure.
     * Used by Python retry logic to detect migration-induced failures.
     * GIL released via call_guard at binding.
     */
    int try_rdma_read_into(uint32_t rkey, uint64_t remote_addr, uint32_t length,
                           uint64_t local_addr, uint32_t local_lkey) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");

        void* local_buf = reinterpret_cast<void*>(local_addr);

        struct ibv_sge sge = {};
        sge.addr   = reinterpret_cast<uint64_t>(local_buf);
        sge.length = length;
        sge.lkey   = local_lkey;

        struct ibv_send_wr wr = {};
        struct ibv_send_wr* bad_wr = nullptr;
        wr.wr_id   = 2;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey        = rkey;

        bool do_timing = (sample_rate_ > 0 && ++rd_counter_ % sample_rate_ == 0);
        auto t0 = do_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        int ret = ibv_post_send(cm_id_->qp, &wr, &bad_wr);
        if (ret != 0)
            throw std::runtime_error("ibv_post_send (RDMA Read) failed: " + std::to_string(ret));

        auto deadline = poll_timeout_ms_ > 0
            ? std::chrono::steady_clock::now() + std::chrono::milliseconds(poll_timeout_ms_)
            : std::chrono::steady_clock::time_point::max();
        int poll_iter = 0;
        while (true) {
            struct ibv_wc wc = {};
            int n = ibv_poll_cq(cq_, 1, &wc);
            if (n < 0)
                throw std::runtime_error("ibv_poll_cq error during RDMA Read");
            if (n == 0) {
                if (++poll_iter % POLL_CHECK_INTERVAL == 0)
                    check_poll_deadline(deadline, "try_rdma_read_into");
                continue;
            }
            if (wc.opcode == IBV_WC_RDMA_READ || wc.status != IBV_WC_SUCCESS) {
                if (do_timing) {
                    auto t1 = std::chrono::steady_clock::now();
                    total_rdma_read_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    rdma_read_count_++;
                }
                return static_cast<int>(wc.status);
            }
        }
    }

    /**
     * Post RDMA Read WRs in MAX_SEND_WR chunks, poll CQ for completions.
     * Returns false if QP entered ERROR state (caller should abort remaining work).
     *
     * Parameters are raw pointers so both batch_rdma_read_into and batch_rdma_read
     * can pass data flexibly without extra copies.
     */
    bool post_and_poll_batch_reads(
        size_t n,
        const uint32_t* rkeys,
        const uint64_t* remote_addrs,
        const uint32_t* lengths,
        const uint64_t* local_addrs,
        const uint32_t* lkeys,
        std::vector<uint8_t>& statuses,
        const char* debug_label
    ) {
        bool qp_error = false;
        auto deadline = poll_timeout_ms_ > 0
            ? std::chrono::steady_clock::now() + std::chrono::milliseconds(poll_timeout_ms_)
            : std::chrono::steady_clock::time_point::max();

        for (size_t chunk_start = 0; chunk_start < n && !qp_error; chunk_start += MAX_SEND_WR) {
            size_t chunk_end = std::min(chunk_start + static_cast<size_t>(MAX_SEND_WR), n);
            size_t chunk_size = chunk_end - chunk_start;

            // Build linked WR list
            std::vector<struct ibv_sge> sges(chunk_size);
            std::vector<struct ibv_send_wr> wrs(chunk_size);
            for (size_t i = 0; i < chunk_size; i++) {
                size_t idx = chunk_start + i;
                std::memset(&sges[i], 0, sizeof(struct ibv_sge));
                sges[i].addr   = local_addrs[idx];
                sges[i].length = lengths[idx];
                sges[i].lkey   = lkeys[idx];

                std::memset(&wrs[i], 0, sizeof(struct ibv_send_wr));
                wrs[i].wr_id   = idx;  // Index for out-of-order tracking
                wrs[i].next    = (i < chunk_size - 1) ? &wrs[i + 1] : nullptr;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
                wrs[i].opcode  = IBV_WR_RDMA_READ;
                wrs[i].send_flags = IBV_SEND_SIGNALED;
                wrs[i].wr.rdma.remote_addr = remote_addrs[idx];
                wrs[i].wr.rdma.rkey        = rkeys[idx];
            }

            // Post all WRs — single doorbell
            struct ibv_send_wr* bad_wr = nullptr;
            int ret = ibv_post_send(cm_id_->qp, &wrs[0], &bad_wr);
            if (ret != 0) {
                bool mark_failed = false;
                for (size_t i = 0; i < chunk_size; i++) {
                    if (&wrs[i] == bad_wr)
                        mark_failed = true;
                    if (mark_failed)
                        statuses[chunk_start + i] = 255;
                }
                if (bad_wr == &wrs[0]) {
                    qp_error = true;
                    continue;
                }
                chunk_size = 0;
                for (size_t i = 0; i < wrs.size(); i++) {
                    if (&wrs[i] == bad_wr) break;
                    chunk_size = i + 1;
                }
            }

            // Poll CQ for all chunk completions
            size_t completed = 0;
            int poll_iter = 0;
            while (completed < chunk_size) {
                struct ibv_wc wcs[32];
                int num = ibv_poll_cq(cq_, 32, wcs);
                if (num < 0)
                    throw std::runtime_error(
                        std::string("ibv_poll_cq error during ") + debug_label);
                if (num == 0) {
                    if (++poll_iter % POLL_CHECK_INTERVAL == 0)
                        check_poll_deadline(deadline, debug_label);
                    continue;
                }
                for (int j = 0; j < num; j++) {
                    if (wcs[j].opcode == IBV_WC_RDMA_READ || wcs[j].status != IBV_WC_SUCCESS) {
                        statuses[wcs[j].wr_id] = static_cast<uint8_t>(wcs[j].status);
                        completed++;
                        if (wcs[j].status != IBV_WC_SUCCESS) {
                            qp_error = true;
                            // Mark remaining entries in this chunk as failed —
                            // QP is in ERROR state so pending WRs won't complete.
                            for (size_t k = chunk_start; k < chunk_end; k++) {
                                if (statuses[k] == 0 && k != wcs[j].wr_id)
                                    statuses[k] = 255;
                            }
                            completed = chunk_size;  // break outer while loop
                            break;
                        }
                    }
                }
            }

            // If QP entered ERROR state, mark remaining entries as failed
            if (qp_error) {
                for (size_t i = chunk_end; i < n; i++) {
                    statuses[i] = 255;
                }
            }
        }

        return !qp_error;
    }

    /**
     * Batch RDMA Read into multiple user-registered memory regions.
     * Posts all WRs with a single ibv_post_send (single doorbell), polls CQ
     * for all completions. Processes in chunks of MAX_SEND_WR to avoid
     * exceeding QP send queue depth.
     *
     * Returns vector of WC statuses (0=success, nonzero=ibv_wc_status error).
     * GIL released via call_guard at binding.
     */
    std::vector<uint8_t> batch_rdma_read_into(
        const std::vector<uint32_t>& rkeys,
        const std::vector<uint64_t>& remote_addrs,
        const std::vector<uint32_t>& lengths,
        const std::vector<uint64_t>& local_addrs,
        const std::vector<uint32_t>& lkeys
    ) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");

        size_t n = rkeys.size();
        if (n == 0)
            return {};
        if (remote_addrs.size() != n || lengths.size() != n ||
            local_addrs.size() != n || lkeys.size() != n)
            throw std::runtime_error("batch_rdma_read_into: all vectors must have same length");

        auto t_batch_start = std::chrono::steady_clock::now();
        std::vector<uint8_t> statuses(n, 255);

        post_and_poll_batch_reads(
            n, rkeys.data(), remote_addrs.data(), lengths.data(),
            local_addrs.data(), lkeys.data(), statuses, "batch_rdma_read_into");

        auto t_batch_end = std::chrono::steady_clock::now();
        batch_read_total_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t_batch_end - t_batch_start).count();
        batch_read_count_++;

        return statuses;
    }

    /**
     * Batch RDMA Read into internal read_buf_, returning raw bytes per entry.
     * Used when the caller needs the raw bytes (e.g., for decompression) rather
     * than zero-copy into a registered user buffer.
     *
     * Sub-batches by read_buf_ capacity: lays out values contiguously, processes
     * when cumulative bytes would exceed the buffer, then reuses the buffer.
     *
     * GIL: released during RDMA ops, reacquired to construct py::bytes objects.
     */
    std::vector<py::bytes> batch_rdma_read(
        const std::vector<uint32_t>& rkeys,
        const std::vector<uint64_t>& remote_addrs,
        const std::vector<uint32_t>& lengths
    ) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");
        if (!read_buf_ || !read_mr_)
            throw std::runtime_error("batch_rdma_read: no internal read buffer (skip_read_buf connection?)");

        size_t n = rkeys.size();
        if (n == 0)
            return {};
        if (remote_addrs.size() != n || lengths.size() != n)
            throw std::runtime_error("batch_rdma_read: all vectors must have same length");

        auto t_batch_start = std::chrono::steady_clock::now();
        std::vector<py::bytes> results(n);

        // Sub-batch by read_buf_ capacity
        size_t sub_start = 0;
        while (sub_start < n) {
            // Compute how many entries fit in read_buf_
            size_t sub_end = sub_start;
            size_t cumulative = 0;
            while (sub_end < n) {
                if (cumulative + lengths[sub_end] > read_buf_size_) {
                    if (sub_end == sub_start) {
                        throw std::runtime_error(
                            "batch_rdma_read: entry " + std::to_string(sub_end) +
                            " length " + std::to_string(lengths[sub_end]) +
                            " exceeds read_buf_size " + std::to_string(read_buf_size_));
                    }
                    break;
                }
                cumulative += lengths[sub_end];
                sub_end++;
            }

            size_t sub_n = sub_end - sub_start;

            // Build offset table and local_addrs/lkeys for the helper
            std::vector<size_t> offsets(sub_n);
            std::vector<uint64_t> sub_local_addrs(sub_n);
            std::vector<uint32_t> sub_lkeys(sub_n);
            std::vector<uint32_t> sub_rkeys(sub_n);
            std::vector<uint64_t> sub_remote_addrs(sub_n);
            std::vector<uint32_t> sub_lengths(sub_n);
            size_t off = 0;
            for (size_t i = 0; i < sub_n; i++) {
                size_t gi = sub_start + i;
                offsets[i] = off;
                sub_local_addrs[i] = reinterpret_cast<uint64_t>(
                    static_cast<char*>(read_buf_) + off);
                sub_lkeys[i] = read_mr_->lkey;
                sub_rkeys[i] = rkeys[gi];
                sub_remote_addrs[i] = remote_addrs[gi];
                sub_lengths[i] = lengths[gi];
                off += lengths[gi];
            }

            // RDMA Read this sub-batch (GIL released)
            {
                py::gil_scoped_release release;

                std::vector<uint8_t> statuses(sub_n, 255);
                post_and_poll_batch_reads(
                    sub_n, sub_rkeys.data(), sub_remote_addrs.data(),
                    sub_lengths.data(), sub_local_addrs.data(), sub_lkeys.data(),
                    statuses, "batch_rdma_read");

                // Check for failures
                for (size_t i = 0; i < sub_n; i++) {
                    if (statuses[i] != 0) {
                        throw std::runtime_error(
                            "batch_rdma_read: WC error on entry " + std::to_string(sub_start + i) +
                            ": " + wc_status_name(statuses[i]) +
                            " (status=" + std::to_string(statuses[i]) + ")");
                    }
                }
            }
            // GIL reacquired — extract bytes from read_buf_
            for (size_t i = 0; i < sub_n; i++) {
                size_t gi = sub_start + i;
                results[gi] = py::bytes(
                    static_cast<char*>(read_buf_) + offsets[i],
                    lengths[gi]);
            }

            sub_start = sub_end;
        }

        auto t_batch_end = std::chrono::steady_clock::now();
        batch_read_total_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t_batch_end - t_batch_start).count();
        batch_read_count_++;

        return results;
    }

    /**
     * Check if internal read buffer is available (false for skip_read_buf connections).
     */
    bool has_read_buf() const { return read_buf_ != nullptr; }

    /**
     * Register a user memory region for RDMA access.
     * Returns (lkey, mr_handle) where mr_handle is an opaque integer.
     * GIL released manually around ibv_reg_mr, re-acquired for return value.
     */
    std::pair<uint32_t, uint64_t> reg_mr(uint64_t addr, size_t length) {
        if (!pd_)
            throw std::runtime_error("not connected");

        void* ptr = reinterpret_cast<void*>(addr);
        struct ibv_mr* mr;
        {
            py::gil_scoped_release release;
            mr = ibv_reg_mr(pd_, ptr, length, IBV_ACCESS_LOCAL_WRITE);
        }
        if (!mr)
            throw std::runtime_error("ibv_reg_mr failed for user buffer");

        uint32_t lkey = mr->lkey;
        uint64_t handle = reinterpret_cast<uint64_t>(mr);
        return {lkey, handle};
    }

    /**
     * Deregister a previously registered memory region.
     * GIL released via call_guard at binding — void return, pure C++.
     */
    void dereg_mr(uint64_t mr_handle) {
        struct ibv_mr* mr = reinterpret_cast<struct ibv_mr*>(mr_handle);
        if (mr)
            ibv_dereg_mr(mr);
    }

    /**
     * Return opaque handle to the Protection Domain (for sharing across pool connections).
     */
    uint64_t get_pd_handle() const {
        if (!pd_) throw std::runtime_error("not connected — no PD");
        return reinterpret_cast<uint64_t>(pd_);
    }

    /**
     * Return opaque handle to the RDMA device context (for verifying same-device).
     */
    uint64_t get_ctx_handle() const {
        if (!ctx_) throw std::runtime_error("not connected — no context");
        return reinterpret_cast<uint64_t>(ctx_);
    }

    /**
     * Connect to RDMA server using a shared PD from another transport.
     * The shared PD must belong to the same RDMA device (verified after route resolve).
     * This transport will NOT free the PD on close — the owning transport handles that.
     *
     * skip_read_buf: if true, skip allocating/registering the 32MB internal read buffer
     * (pool connections with registered user buffers never use it).
     */
    void connect_with_shared_pd(const std::string& server_ip, int port,
                                uint64_t pd_handle, uint64_t ctx_handle,
                                bool skip_read_buf = false) {
        if (connected_.load(std::memory_order_acquire))
            throw std::runtime_error("already connected");

        owns_pd_ = false;

        std::string port_str = std::to_string(port);

        // Create event channel
        cm_channel_ = rdma_create_event_channel();
        if (!cm_channel_) {
            int err = errno;
            throw std::runtime_error(
                "rdma_create_event_channel failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        // Create CM ID
        int ret = rdma_create_id(cm_channel_, &cm_id_, nullptr, RDMA_PS_TCP);
        if (ret != 0) {
            int err = errno;
            rdma_destroy_event_channel(cm_channel_);
            cm_channel_ = nullptr;
            throw std::runtime_error(
                "rdma_create_id failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        // Resolve address (same pattern as connect())
        struct rdma_addrinfo hints = {};
        hints.ai_port_space = RDMA_PS_TCP;
        struct rdma_addrinfo* res = nullptr;
        errno = 0;
        ret = rdma_getaddrinfo(
            const_cast<char*>(server_ip.c_str()),
            const_cast<char*>(port_str.c_str()),
            &hints, &res);
        if (ret != 0) {
            int err = errno;
            cleanup_cm();
            throw std::runtime_error(
                "rdma_getaddrinfo failed for " + server_ip + ":" + port_str + ": " +
                std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ", ret " + std::to_string(ret) + ")");
        }

        ret = rdma_resolve_addr(cm_id_, nullptr, res->ai_dst_addr, 5000);
        rdma_freeaddrinfo(res);
        if (ret != 0) {
            int err = errno;
            cleanup_cm();
            throw std::runtime_error(
                "rdma_resolve_addr failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        wait_for_event(RDMA_CM_EVENT_ADDR_RESOLVED);

        ret = rdma_resolve_route(cm_id_, 5000);
        if (ret != 0) {
            cleanup_cm();
            throw std::runtime_error("rdma_resolve_route failed");
        }

        wait_for_event(RDMA_CM_EVENT_ROUTE_RESOLVED);

        // Verify same RDMA device as the owning transport
        ctx_ = cm_id_->verbs;
        if (!ctx_) {
            cleanup_cm();
            throw std::runtime_error("no verbs context after route resolve");
        }
        auto* expected_ctx = reinterpret_cast<struct ibv_context*>(ctx_handle);
        if (ctx_ != expected_ctx) {
            cleanup_cm();
            throw std::runtime_error(
                "connect_with_shared_pd: route resolved to a different local HCA — "
                "PD is scoped to one physical device and cannot be shared across HCAs "
                "(expected in multi-NIC topologies with different subnets)");
        }

        // Use shared PD (do NOT allocate a new one)
        pd_ = reinterpret_cast<struct ibv_pd*>(pd_handle);

        // Create CQ
        cq_ = ibv_create_cq(ctx_, CQ_DEPTH, nullptr, nullptr, 0);
        if (!cq_) {
            pd_ = nullptr;  // don't free shared PD in cleanup
            cleanup_cm();
            throw std::runtime_error("ibv_create_cq failed");
        }

        // Create QP on shared PD
        struct ibv_qp_init_attr qp_attr = {};
        qp_attr.send_cq = cq_;
        qp_attr.recv_cq = cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr  = MAX_SEND_WR;
        qp_attr.cap.max_recv_wr  = MAX_RECV_WR;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = MAX_INLINE;

        ret = rdma_create_qp(cm_id_, pd_, &qp_attr);
        if (ret != 0) {
            ibv_destroy_cq(cq_); cq_ = nullptr;
            pd_ = nullptr;  // don't free shared PD
            cleanup_cm();
            throw std::runtime_error("rdma_create_qp failed: " + std::to_string(ret));
        }

        // Register internal buffers (optionally skip read buffer)
        if (skip_read_buf) {
            // Only send + recv buffers
            int access = IBV_ACCESS_LOCAL_WRITE;
            send_buf_ = std::malloc(send_buf_size_);
            if (!send_buf_) throw std::runtime_error("malloc send_buf failed");
            std::memset(send_buf_, 0, send_buf_size_);
            send_mr_ = ibv_reg_mr(pd_, send_buf_, send_buf_size_, access);
            if (!send_mr_) {
                std::free(send_buf_); send_buf_ = nullptr;
                throw std::runtime_error("ibv_reg_mr send_buf failed");
            }

            recv_buf_ = std::malloc(recv_buf_size_);
            if (!recv_buf_) {
                ibv_dereg_mr(send_mr_); send_mr_ = nullptr;
                std::free(send_buf_); send_buf_ = nullptr;
                throw std::runtime_error("malloc recv_buf failed");
            }
            std::memset(recv_buf_, 0, recv_buf_size_);
            recv_mr_ = ibv_reg_mr(pd_, recv_buf_, recv_buf_size_, access);
            if (!recv_mr_) {
                std::free(recv_buf_); recv_buf_ = nullptr;
                ibv_dereg_mr(send_mr_); send_mr_ = nullptr;
                std::free(send_buf_); send_buf_ = nullptr;
                throw std::runtime_error("ibv_reg_mr recv_buf failed");
            }
            // read_buf_ and read_mr_ remain nullptr — caller must use
            // try_rdma_read_into with registered user buffers only
        } else {
            register_buffers();
        }

        // Pre-post recv
        post_recv();

        // Connect
        struct rdma_conn_param conn_param = {};
        conn_param.initiator_depth    = 16;
        conn_param.responder_resources = 16;
        conn_param.rnr_retry_count    = 7;
        conn_param.retry_count        = 7;

        ret = rdma_connect(cm_id_, &conn_param);
        if (ret != 0) {
            cleanup_all();
            throw std::runtime_error("rdma_connect failed: " + std::to_string(ret));
        }

        // Wait ESTABLISHED — cleanup on rejection so destructor is safe
        try {
            wait_for_event(RDMA_CM_EVENT_ESTABLISHED);
        } catch (...) {
            cleanup_all();
            throw;
        }
        connected_.store(true, std::memory_order_release);
    }

    // --- Timing stats (public for pybind11 lambda access) ---
    uint64_t total_roundtrip_ns_ = 0;
    uint64_t total_rdma_read_ns_ = 0;
    uint64_t roundtrip_count_ = 0;
    uint64_t rdma_read_count_ = 0;

    // --- Batch RDMA Read timing (always-on, not sampled) ---
    uint64_t batch_read_total_ns_ = 0;
    uint64_t batch_read_count_ = 0;

    // --- Timing sample rate (0=off, 1=every op, N=every Nth op) ---
    //
    // Default: 64 — sample 1 in every 64 roundtrip/RDMA-read operations.
    // This keeps timing overhead negligible (<0.1%) during production
    // traffic while still providing representative latency averages.
    //
    // Tradeoff: lower values give more accurate timing stats but add
    // overhead from clock_gettime calls on every sampled op.
    //   set_sample_rate(1)  — full sampling, useful for debugging or
    //                         micro-benchmarks where every op matters.
    //   set_sample_rate(0)  — disables sampling entirely; get_stats()
    //                         returns zeroed averages (counts still tick).
    uint64_t sample_rate_ = 64;
    uint64_t rt_counter_ = 0;   // roundtrip_impl sampling counter
    uint64_t rd_counter_ = 0;   // rdma_read_impl / rdma_read_into / try_rdma_read_into sampling counter

    // --- Poll timeout (public for pybind11 lambda access) ---
    int poll_timeout_ms_ = 30000;

    // --- Buffer sizes (public for pybind11 property access) ---
    size_t send_buf_size_ = DEFAULT_SEND_BUF_SIZE;
    size_t recv_buf_size_ = DEFAULT_RECV_BUF_SIZE;

private:
    /**
     * Internal close — no GIL manipulation; called from close() and destructor.
     */
    void close_impl() {
        // Serialize concurrent close calls — the pool's force-close path
        // (rebuild lock timeout) can race with _reconnect_non_owner calling
        // close() on the same transport.  Without this lock both threads
        // enter cleanup_all() and double-free RDMA resources → segfault.
        std::lock_guard<std::mutex> guard(close_mu_);

        if (!connected_.load(std::memory_order_acquire) && !cm_id_)
            return;

        // 1. Signal all poll loops to exit
        bool was_connected = connected_.exchange(false, std::memory_order_acq_rel);

        // 2. Flush outstanding WRs — rdma_disconnect transitions QP to ERROR
        //    state, causing all pending WRs to complete with WR_FLUSH_ERR.
        //    This wakes any thread stuck in ibv_poll_cq.
        //    IMPORTANT: only disconnect if the connection was fully established.
        //    Calling rdma_disconnect on a partial CM ID (e.g. after REJECTED)
        //    can segfault in the RDMA driver.
        if (was_connected && cm_id_)
            rdma_disconnect(cm_id_);

        // 3. Give poll loops time to observe connected_==false / flushed WCs
        //    and exit before we destroy the CQ underneath them.
        if (was_connected)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 4. Now safe to destroy CQ, MRs, PD, CM resources
        cleanup_all();
    }

    /**
     * Internal roundtrip — pure C++, no Python objects.
     * Extracted from roundtrip() so GIL can be released around this call.
     */
    std::string roundtrip_impl(const char* data, size_t len) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");
        if (len > send_buf_size_)
            throw std::runtime_error("request too large: " + std::to_string(len));

        bool do_timing = (sample_rate_ > 0 && ++rt_counter_ % sample_rate_ == 0);
        auto t0 = do_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        // Copy request into send buffer
        std::memcpy(send_buf_, data, len);

        // Post send
        struct ibv_sge sge = {};
        sge.addr   = reinterpret_cast<uint64_t>(send_buf_);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = send_mr_->lkey;

        struct ibv_send_wr wr = {};
        struct ibv_send_wr* bad_wr = nullptr;
        wr.wr_id   = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        if (len <= MAX_INLINE)
            wr.send_flags |= IBV_SEND_INLINE;

        int ret = ibv_post_send(cm_id_->qp, &wr, &bad_wr);
        if (ret != 0)
            throw std::runtime_error("ibv_post_send failed: " + std::to_string(ret));

        // Poll for SEND and RECV completions
        bool send_done = false;
        bool recv_done = false;
        uint32_t recv_len = 0;
        auto deadline = poll_timeout_ms_ > 0
            ? std::chrono::steady_clock::now() + std::chrono::milliseconds(poll_timeout_ms_)
            : std::chrono::steady_clock::time_point::max();
        int poll_iter = 0;

        while (!send_done || !recv_done) {
            struct ibv_wc wc = {};
            int n = ibv_poll_cq(cq_, 1, &wc);
            if (n < 0)
                throw std::runtime_error("ibv_poll_cq error");
            if (n == 0) {
                if (++poll_iter % POLL_CHECK_INTERVAL == 0)
                    check_poll_deadline(deadline, "roundtrip_impl");
                continue;
            }

            if (wc.status != IBV_WC_SUCCESS)
                throw std::runtime_error(
                    std::string("WC error: ") + wc_status_name(wc.status) +
                    " (status=" + std::to_string(wc.status) + ")");

            if (wc.opcode == IBV_WC_SEND) {
                send_done = true;
            } else if (wc.opcode == IBV_WC_RECV) {
                recv_done = true;
                // Clamp to buffer size — NIC may report full message length
                // even if RDMA layer truncated the write to recv_buf_size_.
                recv_len = std::min(wc.byte_len, static_cast<uint32_t>(recv_buf_size_));
            }
        }

        // Copy response bytes
        std::string result(static_cast<char*>(recv_buf_), recv_len);

        // Re-post recv for next response
        post_recv();

        if (do_timing) {
            auto t1 = std::chrono::steady_clock::now();
            total_roundtrip_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            roundtrip_count_++;
        }

        return result;
    }

    /**
     * Internal RDMA read — pure C++, no Python objects.
     * Extracted from rdma_read() so GIL can be released around this call.
     */
    std::string rdma_read_impl(uint32_t rkey, uint64_t remote_addr, uint32_t length) {
        if (!connected_.load(std::memory_order_acquire))
            throw std::runtime_error("not connected");
        if (length > read_buf_size_)
            throw std::runtime_error("RDMA Read too large");

        bool do_timing = (sample_rate_ > 0 && ++rd_counter_ % sample_rate_ == 0);
        auto t0 = do_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        post_rdma_read(read_buf_, read_mr_->lkey, rkey, remote_addr, length);

        std::string result(static_cast<char*>(read_buf_), length);

        if (do_timing) {
            auto t1 = std::chrono::steady_clock::now();
            total_rdma_read_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            rdma_read_count_++;
        }

        return result;
    }

    // RDMA CM resources
    struct rdma_event_channel* cm_channel_ = nullptr;
    struct rdma_cm_id*         cm_id_      = nullptr;
    struct ibv_context*        ctx_        = nullptr;
    struct ibv_pd*             pd_         = nullptr;
    struct ibv_cq*             cq_         = nullptr;

    // Pre-registered buffers
    void*          send_buf_ = nullptr;
    struct ibv_mr* send_mr_  = nullptr;
    void*          recv_buf_ = nullptr;
    struct ibv_mr* recv_mr_  = nullptr;
    void*          read_buf_ = nullptr;
    struct ibv_mr* read_mr_  = nullptr;

    std::atomic<bool> connected_{false};
    std::mutex close_mu_;  // serializes concurrent close() calls (force-close race)
    bool owns_pd_ = true;  // false when PD is shared from another transport

    // Read buffer size (private — not exposed to Python)
    size_t read_buf_size_ = DEFAULT_READ_BUF_SIZE;

    void register_buffers() {
        int access = IBV_ACCESS_LOCAL_WRITE;

        // Send buffer
        send_buf_ = std::malloc(send_buf_size_);
        if (!send_buf_) throw std::runtime_error("malloc send_buf failed");
        std::memset(send_buf_, 0, send_buf_size_);
        send_mr_ = ibv_reg_mr(pd_, send_buf_, send_buf_size_, access);
        if (!send_mr_) throw std::runtime_error("ibv_reg_mr send_buf failed");

        // Recv buffer
        recv_buf_ = std::malloc(recv_buf_size_);
        if (!recv_buf_) throw std::runtime_error("malloc recv_buf failed");
        std::memset(recv_buf_, 0, recv_buf_size_);
        recv_mr_ = ibv_reg_mr(pd_, recv_buf_, recv_buf_size_, access);
        if (!recv_mr_) throw std::runtime_error("ibv_reg_mr recv_buf failed");

        // Read buffer (for RDMA Read of large values)
        read_buf_ = std::malloc(read_buf_size_);
        if (!read_buf_) throw std::runtime_error("malloc read_buf failed");
        std::memset(read_buf_, 0, read_buf_size_);
        read_mr_ = ibv_reg_mr(pd_, read_buf_, read_buf_size_, access);
        if (!read_mr_) throw std::runtime_error("ibv_reg_mr read_buf failed");
    }

    static constexpr int POLL_CHECK_INTERVAL = 1024;

    void check_poll_deadline(std::chrono::steady_clock::time_point deadline, const char* ctx) {
        if (!connected_.load(std::memory_order_relaxed))
            throw std::runtime_error(std::string(ctx) + ": transport closed during poll");
        if (poll_timeout_ms_ > 0 && std::chrono::steady_clock::now() > deadline) {
            connected_.store(false, std::memory_order_release);
            throw std::runtime_error(std::string(ctx) + ": poll timeout after " +
                std::to_string(poll_timeout_ms_) + "ms");
        }
    }

    void post_recv() {
        struct ibv_sge sge = {};
        sge.addr   = reinterpret_cast<uint64_t>(recv_buf_);
        sge.length = static_cast<uint32_t>(recv_buf_size_);
        sge.lkey   = recv_mr_->lkey;

        struct ibv_recv_wr wr = {};
        struct ibv_recv_wr* bad_wr = nullptr;
        wr.wr_id   = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        int ret = ibv_post_recv(cm_id_->qp, &wr, &bad_wr);
        if (ret != 0)
            throw std::runtime_error("ibv_post_recv failed: " + std::to_string(ret));
    }

    void post_rdma_read(void* local_buf, uint32_t lkey,
                        uint32_t rkey, uint64_t remote_addr, uint32_t length) {
        struct ibv_sge sge = {};
        sge.addr   = reinterpret_cast<uint64_t>(local_buf);
        sge.length = length;
        sge.lkey   = lkey;

        struct ibv_send_wr wr = {};
        struct ibv_send_wr* bad_wr = nullptr;
        wr.wr_id   = 2;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey        = rkey;

        int ret = ibv_post_send(cm_id_->qp, &wr, &bad_wr);
        if (ret != 0)
            throw std::runtime_error("ibv_post_send (RDMA Read) failed: " + std::to_string(ret));

        // Poll for RDMA Read completion
        auto deadline = poll_timeout_ms_ > 0
            ? std::chrono::steady_clock::now() + std::chrono::milliseconds(poll_timeout_ms_)
            : std::chrono::steady_clock::time_point::max();
        int poll_iter = 0;
        while (true) {
            struct ibv_wc wc = {};
            int n = ibv_poll_cq(cq_, 1, &wc);
            if (n < 0)
                throw std::runtime_error("ibv_poll_cq error during RDMA Read");
            if (n == 0) {
                if (++poll_iter % POLL_CHECK_INTERVAL == 0)
                    check_poll_deadline(deadline, "post_rdma_read");
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS)
                throw std::runtime_error(
                    std::string("RDMA Read WC error: ") + wc_status_name(wc.status) +
                    " (status=" + std::to_string(wc.status) + ")");
            if (wc.opcode == IBV_WC_RDMA_READ)
                break;
        }
    }

    static const char* cm_event_name(enum rdma_cm_event_type ev) {
        switch (ev) {
            case RDMA_CM_EVENT_ADDR_RESOLVED:    return "ADDR_RESOLVED";
            case RDMA_CM_EVENT_ADDR_ERROR:       return "ADDR_ERROR";
            case RDMA_CM_EVENT_ROUTE_RESOLVED:   return "ROUTE_RESOLVED";
            case RDMA_CM_EVENT_ROUTE_ERROR:      return "ROUTE_ERROR";
            case RDMA_CM_EVENT_CONNECT_REQUEST:  return "CONNECT_REQUEST";
            case RDMA_CM_EVENT_CONNECT_RESPONSE: return "CONNECT_RESPONSE";
            case RDMA_CM_EVENT_CONNECT_ERROR:    return "CONNECT_ERROR";
            case RDMA_CM_EVENT_UNREACHABLE:      return "UNREACHABLE";
            case RDMA_CM_EVENT_REJECTED:         return "REJECTED";
            case RDMA_CM_EVENT_ESTABLISHED:      return "ESTABLISHED";
            case RDMA_CM_EVENT_DISCONNECTED:     return "DISCONNECTED";
            default:                             return "UNKNOWN";
        }
    }

    void wait_for_event(enum rdma_cm_event_type expected) {
        struct rdma_cm_event* event = nullptr;
        int ret = rdma_get_cm_event(cm_channel_, &event);
        if (ret != 0) {
            int err = errno;
            throw std::runtime_error(
                "rdma_get_cm_event failed: " + std::string(std::strerror(err)) +
                " (errno " + std::to_string(err) + ")");
        }

        enum rdma_cm_event_type ev_type = event->event;
        int status = event->status;
        rdma_ack_cm_event(event);

        if (ev_type != expected) {
            std::string msg = std::string("RDMA CM event: got ") +
                cm_event_name(ev_type) + " (expected " +
                cm_event_name(expected) + ")";
            if (status != 0)
                msg += ", status=" + std::to_string(status);
            if (ev_type == RDMA_CM_EVENT_ADDR_ERROR)
                msg += ". Hint: the address may not be reachable via RDMA. "
                       "Use the RDMA interface IP (e.g. from 'ibdev2netdev' or "
                       "server startup log), not a hostname or management IP.";
            else if (ev_type == RDMA_CM_EVENT_REJECTED)
                msg += ". The server rejected the connection.";
            else if (ev_type == RDMA_CM_EVENT_UNREACHABLE)
                msg += ". The server is unreachable via RDMA.";
            throw std::runtime_error(msg);
        }
    }

    void cleanup_cm() {
        if (cm_id_) {
            rdma_destroy_id(cm_id_);
            cm_id_ = nullptr;
        }
        if (cm_channel_) {
            rdma_destroy_event_channel(cm_channel_);
            cm_channel_ = nullptr;
        }
    }

    void cleanup_all() {
        // Deregister and free buffers
        if (read_mr_)  { ibv_dereg_mr(read_mr_);  read_mr_  = nullptr; }
        if (read_buf_) { std::free(read_buf_);     read_buf_ = nullptr; }
        if (recv_mr_)  { ibv_dereg_mr(recv_mr_);  recv_mr_  = nullptr; }
        if (recv_buf_) { std::free(recv_buf_);     recv_buf_ = nullptr; }
        if (send_mr_)  { ibv_dereg_mr(send_mr_);  send_mr_  = nullptr; }
        if (send_buf_) { std::free(send_buf_);     send_buf_ = nullptr; }

        // Destroy QP
        if (cm_id_ && cm_id_->qp) {
            rdma_destroy_qp(cm_id_);
        }
        // Destroy CQ
        if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
        // Only free PD if we own it (not shared from another transport)
        if (pd_ && owns_pd_) { ibv_dealloc_pd(pd_); }
        pd_ = nullptr;

        cleanup_cm();
        ctx_ = nullptr;
    }
};

/**
 * Check if RDMA devices are available on this machine.
 */
static bool is_available() {
    int num_devices = 0;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list || num_devices == 0)
        return false;
    ibv_free_device_list(device_list);
    return true;
}

// L3_VERSION is set by setup.py via -DL3_VERSION="x.y.z".
// If building manually without the define, fall back to "unknown".
#ifndef L3_VERSION
#define L3_VERSION "unknown"
#endif

PYBIND11_MODULE(_l3_rdma, m) {
    m.doc() = "L3 RDMA transport extension (libibverbs/librdmacm)";

    m.attr("__version__") = L3_VERSION;

    // Feature flag: Python code can check this to verify the GIL-releasing
    // build is loaded (guards against stale .so from editable installs).
    m.attr("GIL_RELEASED") = true;

    // Buffer size defaults — Python reads these instead of hardcoding.
    m.attr("DEFAULT_SEND_BUF_SIZE") = DEFAULT_SEND_BUF_SIZE;
    m.attr("DEFAULT_RECV_BUF_SIZE") = DEFAULT_RECV_BUF_SIZE;
    m.attr("DEFAULT_READ_BUF_SIZE") = DEFAULT_READ_BUF_SIZE;

    m.def("is_available", &is_available,
          "Check if RDMA devices are available on this machine.");

    m.def("wc_status_name", &wc_status_name,
          py::arg("status"),
          "Return human-readable name for an IB work completion status code.");

    py::class_<RDMATransport>(m, "RDMATransport")
        .def(py::init<size_t, size_t, size_t, int>(),
             py::arg("send_buf_size") = DEFAULT_SEND_BUF_SIZE,
             py::arg("recv_buf_size") = DEFAULT_RECV_BUF_SIZE,
             py::arg("read_buf_size") = DEFAULT_READ_BUF_SIZE,
             py::arg("poll_timeout_ms") = 30000)
        .def("connect", &RDMATransport::connect,
             py::arg("server_ip"), py::arg("port"),
             py::call_guard<py::gil_scoped_release>(),
             "Connect to an RDMA server via RDMA CM.")
        .def("close", &RDMATransport::close,
             "Close the RDMA connection and release all resources.")
        .def("roundtrip", &RDMATransport::roundtrip,
             py::arg("request_bytes"),
             "Send a request and receive a response via RDMA Send/Recv.")
        .def("rdma_read", &RDMATransport::rdma_read,
             py::arg("rkey"), py::arg("remote_addr"), py::arg("length"),
             "Perform an RDMA Read into internal buffer, return bytes.")
        .def("rdma_read_into", &RDMATransport::rdma_read_into,
             py::arg("rkey"), py::arg("remote_addr"), py::arg("length"),
             py::arg("local_addr"), py::arg("local_lkey"),
             py::call_guard<py::gil_scoped_release>(),
             "Zero-copy RDMA Read into a user-registered buffer.")
        .def("try_rdma_read_into", &RDMATransport::try_rdma_read_into,
             py::arg("rkey"), py::arg("remote_addr"), py::arg("length"),
             py::arg("local_addr"), py::arg("local_lkey"),
             py::call_guard<py::gil_scoped_release>(),
             "Try RDMA Read into user buffer, returning WC status (0=success).")
        .def("batch_rdma_read_into", &RDMATransport::batch_rdma_read_into,
             py::arg("rkeys"), py::arg("remote_addrs"), py::arg("lengths"),
             py::arg("local_addrs"), py::arg("lkeys"),
             py::call_guard<py::gil_scoped_release>(),
             "Batch RDMA Read into multiple registered buffers. Returns list of WC statuses.")
        .def("batch_rdma_read", &RDMATransport::batch_rdma_read,
             py::arg("rkeys"), py::arg("remote_addrs"), py::arg("lengths"),
             "Batch RDMA Read into internal read buffer. Returns list of raw bytes.")
        .def("has_read_buf", &RDMATransport::has_read_buf,
             "Check if internal read buffer is available.")
        .def("get_pd_handle", &RDMATransport::get_pd_handle,
             "Return opaque PD handle for sharing across pool connections.")
        .def("get_ctx_handle", &RDMATransport::get_ctx_handle,
             "Return opaque RDMA device context handle.")
        .def("connect_with_shared_pd", &RDMATransport::connect_with_shared_pd,
             py::arg("server_ip"), py::arg("port"),
             py::arg("pd_handle"), py::arg("ctx_handle"),
             py::arg("skip_read_buf") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Connect using a shared PD from another transport (same RDMA device).")
        .def("reg_mr", &RDMATransport::reg_mr,
             py::arg("addr"), py::arg("length"),
             "Register a user memory region. Returns (lkey, mr_handle).")
        .def("dereg_mr", &RDMATransport::dereg_mr,
             py::arg("mr_handle"),
             py::call_guard<py::gil_scoped_release>(),
             "Deregister a previously registered memory region.")
        .def_property_readonly("send_buf_size", [](const RDMATransport& self) { return self.send_buf_size_; },
             "Actual send buffer size in bytes.")
        .def_property_readonly("recv_buf_size", [](const RDMATransport& self) { return self.recv_buf_size_; },
             "Actual recv buffer size in bytes.")
        .def("get_stats", [](RDMATransport& self) -> py::dict {
            py::dict d;
            d["roundtrip_count"] = self.roundtrip_count_;
            d["rdma_read_count"] = self.rdma_read_count_;
            d["avg_roundtrip_us"] = self.roundtrip_count_ > 0
                ? self.total_roundtrip_ns_ / self.roundtrip_count_ / 1000.0 : 0.0;
            d["avg_rdma_read_us"] = self.rdma_read_count_ > 0
                ? self.total_rdma_read_ns_ / self.rdma_read_count_ / 1000.0 : 0.0;
            d["sample_rate"] = self.sample_rate_;
            d["avg_batch_read_us"] = self.batch_read_count_ > 0
                ? static_cast<double>(self.batch_read_total_ns_) / self.batch_read_count_ / 1000.0 : 0.0;
            d["batch_read_count"] = self.batch_read_count_;
            return d;
        }, "Return timing stats: avg roundtrip/rdma_read latency in microseconds.")
        .def("reset_stats", [](RDMATransport& self) {
            self.total_roundtrip_ns_ = 0;
            self.total_rdma_read_ns_ = 0;
            self.roundtrip_count_ = 0;
            self.rdma_read_count_ = 0;
            self.rt_counter_ = 0;
            self.rd_counter_ = 0;
            self.batch_read_total_ns_ = 0;
            self.batch_read_count_ = 0;
        }, "Reset timing counters to zero.")
        .def("set_sample_rate", [](RDMATransport& self, uint64_t rate) {
            self.sample_rate_ = rate;
        }, py::arg("rate"),
           "Set timing sample rate: 0=off, 1=every op, N=every Nth op (default 64).")
        .def("set_poll_timeout", [](RDMATransport& self, int ms) {
            self.poll_timeout_ms_ = ms;
        }, py::arg("ms"),
           "Set poll timeout in milliseconds: 0=infinite, >0=throw after N ms (default 30000).");
}

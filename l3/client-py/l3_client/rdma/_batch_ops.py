"""Shared batch operation helpers for RDMAClient and RDMAClientPool.

Provides TransportHandle adapter and free functions that deduplicate
the mget_rdma and mget_rdma_raw flows across single-connection and
pooled-connection modes.
"""

from __future__ import annotations

import time

from l3_client import protocol
from l3_client.errors import CamaNotReadyError, CamaOOMError, CamaServerOverloadError
from l3_client.rdma._constants import logger


class TransportHandle:
    """Unifies single-conn vs pool-conn transport access for shared batch logic."""
    __slots__ = ("transport", "lock", "next_req_id_fn", "mr_lookup_fn")

    def __init__(self, transport, lock, next_req_id_fn, mr_lookup_fn):
        self.transport = transport
        self.lock = lock
        self.next_req_id_fn = next_req_id_fn
        self.mr_lookup_fn = mr_lookup_fn


def _mget_rdma_control(keys, do_roundtrip):
    """Send OP_MGET_RDMA and parse response.

    Args:
        keys: list of key strings
        do_roundtrip: callable(body: bytes) -> raw_response_bytes.
            Caller handles locking, req_id, header packing.

    Returns:
        ("multi_value", (values_data, founds))  -- migration fallback
        ("read_ready", entries)                  -- normal RDMA path
    """
    body = protocol.encode_mget_body([k.encode() for k in keys])
    raw_resp = do_roundtrip(body)
    resp = protocol.read_message_from_bytes(raw_resp)

    if resp.header.opcode == protocol.RESP_MULTI_VALUE:
        values_data, founds = protocol.decode_multi_value_response(resp.body)
        return ("multi_value", (values_data, founds))

    if resp.header.opcode == protocol.RESP_OOM:
        raise CamaOOMError.from_body(resp.body)
    if resp.header.opcode == protocol.RESP_NOT_READY:
        raise CamaNotReadyError.from_body(resp.body)
    if resp.header.opcode == protocol.RESP_ERROR:
        body_str = resp.body.decode(errors='replace')
        if "server overloaded" in body_str:
            raise CamaServerOverloadError(body_str)
        raise RuntimeError(f"CAMA error: {body_str}")

    if resp.header.opcode != protocol.OP_MGET_READ_READY:
        raise RuntimeError(f"unexpected opcode {resp.header.opcode:#x} in mget_rdma")

    entries = protocol.decode_mget_read_ready(resp.body)
    return ("read_ready", entries)


def _send_batch_read_ack(statuses, do_roundtrip):
    """Best-effort batch ReadAck."""
    ack_body = protocol.encode_batch_read_ack(statuses)
    try:
        do_roundtrip(ack_body)
    except Exception:
        pass


def mget_rdma_flow(handle: TransportHandle, keys: list[str], sgls, sizes=None):
    """Core mget_rdma flow: control roundtrip + batch RDMA Read + ReadAck.

    Args:
        handle: TransportHandle adapter for the target connection.
        keys: list of key strings.
        sgls: list of SGL objects (one per key).
        sizes: optional list of value sizes (unused, API compat).

    Returns:
        (results, timing_dict | None, read_count, total_bytes)
        results: list[int] — 0=found, -1=miss per key.
        timing_dict: sub-phase timing dict, or None if no RDMA reads.
        read_count: number of successful RDMA reads.
        total_bytes: total bytes read via RDMA.
    """
    n = len(keys)
    if n == 0:
        return [], None, 0, 0

    t0 = time.perf_counter()

    def _do_rt(body):
        with handle.lock:
            req_id = handle.next_req_id_fn()
            hdr = protocol._pack_header(protocol.OP_MGET_RDMA, 0, req_id, body)
            return handle.transport.roundtrip(hdr + body)

    tag, payload = _mget_rdma_control(keys, _do_rt)

    t_ctrl = time.perf_counter()

    # Handle migration fallback (server returns inline values)
    if tag == "multi_value":
        values_data, founds = payload
        results = [-1] * n
        for i in range(n):
            if founds[i] and values_data[i] is not None:
                sgls[i].from_bytes(values_data[i])
                results[i] = 0
        return results, None, 0, 0

    entries = payload

    # Build RDMA Read parameters for found entries
    rkeys, remote_addrs, lengths, local_addrs, lkeys_list = [], [], [], [], []
    rdma_to_key_idx = []
    results = [-1] * n
    total_bytes = 0

    unreg_count = 0
    found_count = 0
    for i, (found, rkey, addr, length) in enumerate(entries):
        if not found:
            continue
        found_count += 1
        mr_entry = handle.mr_lookup_fn(sgls[i].reg_handle)
        if mr_entry is None:
            unreg_count += 1
            continue
        rkeys.append(rkey)
        remote_addrs.append(addr)
        lengths.append(length)
        local_addrs.append(sgls[i].ptr)
        lkeys_list.append(mr_entry.lkey)
        rdma_to_key_idx.append(i)
        total_bytes += length

    if unreg_count > 0:
        logger.warning(
            "mget_rdma: %d/%d found keys skipped — SGL not registered. "
            "These appear as cache misses.", unreg_count, found_count)

    t_meta = time.perf_counter()

    # Batch RDMA Reads (GIL released in C++)
    timing_dict = None
    if rdma_to_key_idx:
        with handle.lock:
            wc_statuses = handle.transport.batch_rdma_read_into(
                rkeys, remote_addrs, lengths, local_addrs, lkeys_list
            )

        t_read = time.perf_counter()

        for j, wc_st in enumerate(wc_statuses):
            key_idx = rdma_to_key_idx[j]
            results[key_idx] = 0 if wc_st == 0 else -1

        # Send batch ReadAck
        def _do_ack(body):
            with handle.lock:
                req_id = handle.next_req_id_fn()
                hdr = protocol._pack_header(protocol.OP_BATCH_READ_ACK, 0, req_id, body)
                return handle.transport.roundtrip(hdr + body)

        _send_batch_read_ack(list(wc_statuses), _do_ack)

        t_ack = time.perf_counter()

        timing_dict = {
            "t_ctrl_ms": (t_ctrl - t0) * 1000,
            "t_meta_ms": (t_meta - t_ctrl) * 1000,
            "t_read_ms": (t_read - t_meta) * 1000,
            "t_ack_ms": (t_ack - t_read) * 1000,
            "n_keys": n,
            "n_found": len(rdma_to_key_idx),
            "total_bytes": total_bytes,
        }

    return results, timing_dict, len(rdma_to_key_idx), total_bytes


def mget_rdma_raw_flow(handle: TransportHandle, keys: list[str]):
    """Core mget_rdma_raw flow: control roundtrip + batch RDMA Read into
    internal buffer + ReadAck.

    Args:
        handle: TransportHandle adapter for the target connection.
        keys: list of key strings.

    Returns:
        (results, timing_dict | None, found_count)
        results: list[(int, bytes | None)] — (0, data) or (-1, None) per key.
        timing_dict: timing dict, or None if no RDMA reads.
        found_count: number of found entries.
    """
    n = len(keys)
    if n == 0:
        return [], None, 0

    t0 = time.perf_counter()

    def _do_rt(body):
        with handle.lock:
            req_id = handle.next_req_id_fn()
            hdr = protocol._pack_header(protocol.OP_MGET_RDMA, 0, req_id, body)
            return handle.transport.roundtrip(hdr + body)

    tag, payload = _mget_rdma_control(keys, _do_rt)

    t_ctrl = time.perf_counter()

    # Handle migration fallback (server returns inline values)
    if tag == "multi_value":
        values_data, founds = payload
        results = []
        for i in range(n):
            if founds[i] and values_data[i] is not None:
                results.append((0, values_data[i]))
            else:
                results.append((-1, None))
        return results, None, 0

    entries = payload

    # Collect RDMA Read params for found entries
    rkeys_list, addrs_list, lens_list = [], [], []
    rdma_to_key_idx = []
    results: list[tuple[int, bytes | None]] = [(-1, None)] * n

    for i, (found, rkey, addr, length) in enumerate(entries):
        if not found:
            continue
        rkeys_list.append(rkey)
        addrs_list.append(addr)
        lens_list.append(length)
        rdma_to_key_idx.append(i)

    t_meta = time.perf_counter()

    # Batch RDMA Read into internal buffer
    timing_dict = None
    found_count = 0
    if rdma_to_key_idx:
        with handle.lock:
            raw_bytes_list = handle.transport.batch_rdma_read(
                rkeys_list, addrs_list, lens_list
            )

        t_read = time.perf_counter()

        for j, raw_data in enumerate(raw_bytes_list):
            key_idx = rdma_to_key_idx[j]
            results[key_idx] = (0, raw_data)

        # Send batch ReadAck (all success since batch_rdma_read throws on failure)
        def _do_ack(body):
            with handle.lock:
                req_id = handle.next_req_id_fn()
                hdr = protocol._pack_header(protocol.OP_BATCH_READ_ACK, 0, req_id, body)
                return handle.transport.roundtrip(hdr + body)

        _send_batch_read_ack([0] * len(rdma_to_key_idx), _do_ack)

        t_ack = time.perf_counter()
        found_count = sum(1 for rc, _ in results if rc == 0)
        total_bytes = sum(lens_list)
        timing_dict = {
            "t_ctrl_ms": (t_ctrl - t0) * 1000,
            "t_meta_ms": (t_meta - t_ctrl) * 1000,
            "t_read_ms": (t_read - t_meta) * 1000,
            "t_ack_ms": (t_ack - t_read) * 1000,
            "t_total_ms": (t_ack - t0) * 1000,
            "n_keys": n,
            "n_found": found_count,
            "total_bytes": total_bytes,
            "mode": "raw",
        }

    return results, timing_dict, found_count

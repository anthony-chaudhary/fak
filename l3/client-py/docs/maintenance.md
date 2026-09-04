# Maintenance & Auto-Tune

## Quick Reference

```python
from cama_client import CamaClient

# TCP connection
c = CamaClient("208.0.0.13", 18000)

# Check current auto-tune and vacuum status
status = c.maintenance_status()
print(status)

# Force auto-tune (all shards)
result = c.autotune(force=True)
print(result)

# Trigger vacuum rebalance
result = c.vacuum()
print(result)

c.close()
```

## RDMA Connection

```python
from cama_client import PriskvClient

# RDMA path (auto-selects RDMA when available)
c = PriskvClient("208.0.0.13", 18001)

status = c.maintenance_status()
print(status)

c.close()
```

---

## Auto-Tune

Auto-tune detects the dominant value size from SET operations and rebuilds the slab allocator to minimize internal fragmentation.

### How It Works

1. The server records every SET value size in a per-shard tracker
2. After `warmup_ops` SETs (default: 1000), detection fires
3. The dominant size is identified and the slab allocator is rebuilt with optimized class sizes
4. After rebuild, the tracker freezes (stops recording)

### Default Configuration

| Setting | Default | Description |
|---|---|---|
| `auto_tune_slabs` | `true` | Enable auto-tune |
| `warmup_ops` | `1000` | SETs before detection fires |
| `allocator_mode` | `"slab"` | Must be `"slab"` for auto-tune (offset allocator handles variable sizes natively) |

### Status Values

| Status | Meaning |
|---|---|
| `disabled` | `warmup_ops = 0` or `auto_tune_slabs = false` |
| `warming_up` | Collecting value sizes, not enough SETs yet |
| `detected` | Dominant size found, rebuild pending |
| `rebuilt` | Slab allocator rebuilt with optimal sizes |

### Trigger On-Demand

```python
# Force auto-tune even if warmup threshold not reached
result = c.autotune(force=True)
print(result["shards_rebuilt"])         # [0, 1, 2, ...]
print(result["detection_snapshots"])    # per-shard detection state

# Target specific shards only
result = c.autotune(force=True, shard_ids=[0, 1])
```

---

## Vacuum

Vacuum evaluates slab utilization and triggers ZeroLatencyBalance rebalancing for shards with high fragmentation.

```python
# Evaluate all shards
result = c.vacuum()
print(result["shards_rebalanced"])
print(result["shards_skipped"])

# Force rebalance even if shards appear healthy
result = c.vacuum(force=True)

# Target specific shards
result = c.vacuum(shard_ids=[0, 2])
```

---

## Maintenance Status

Query the current state without triggering any action:

```python
status = c.maintenance_status()
```

Returns a dict with:

| Field | Description |
|---|---|
| `vacuum_config` | Current vacuum configuration (interval, threshold, etc.) |
| `vacuum_stats` | Rebalance counts, pressure evaluations |
| `shard_detections` | Per-shard auto-tune detection snapshots |

### Example Output

```json
{
  "vacuum_config": {
    "enabled": true,
    "interval_seconds": 300,
    "utilization_threshold": 0.85
  },
  "vacuum_stats": {
    "total_evaluations": 12,
    "total_rebalances": 2
  },
  "shard_detections": {
    "0": {
      "status": "rebuilt",
      "warmup_progress": "1000/1000",
      "dominant_value_size": 5242880,
      "dominant_frequency_percent": 94.2,
      "current_slot_utilization": 99.8,
      "top_sizes": [
        {"size": 5242880, "count": 942, "percent": 94.2}
      ]
    }
  }
}
```

---

## Troubleshooting

**Auto-tune shows empty results / status is "disabled":**
- Check that `auto_tune_slabs = true` and `warmup_ops > 0` in `cama.toml`
- If `allocator_mode = "offset"`, auto-tune is not used (offset handles all sizes natively)

**Status is "warming_up":**
- Not enough SETs yet. Check `warmup_progress` (e.g., `"342/1000"`)
- Use `autotune(force=True)` to trigger early

**Status is "rebuilt" but `top_sizes` is empty:**
- Normal. After a successful rebuild, the tracker freezes and stops recording
- The `dominant_value_size` and `current_slot_utilization` fields still show what was detected

**`ImportError: cannot import CamaClient`:**
- Use `from cama_client import CamaClient` or `from cama_client.client import CamaClient`

**`TypeError: missing self argument`:**
- You're calling the method on the class, not an instance. Make sure to create an instance first:
  ```python
  c = CamaClient("208.0.0.13", 18000)  # note the parentheses
  c.maintenance_status()
  ```

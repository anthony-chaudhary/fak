# fak Server Troubleshooting

Common startup failures, port conflicts, and resource issues when running `fak serve` or the in-kernel model engine.

## Table of Contents

- [Port Conflicts](#port-conflicts)
- [Memory Issues](#memory-issues)
- [GPU/CUDA Issues](#gpucuda-issues)
- [Model Loading Failures](#model-loading-failures)
- [Policy and Configuration Issues](#policy-and-configuration-issues)
- [Startup Failures](#startup-failures)
- [Debugging Tools](#debugging-tools)

---

## Port Conflicts

### Symptom: "bind: Only one usage of each socket address"

**Example error:**
```
listen tcp 127.0.0.1:8080: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```

**Diagnosis:**
```bash
# Check what's using the port (Windows)
netstat -ano | findstr :8080

# Check what's using the port (Linux/macOS)
lsof -i :8080
```

**Solutions:**
1. **Kill the conflicting process:**
   - Windows: `taskkill /PID <pid> /F`
   - Linux/macOS: `kill -9 <pid>`

2. **Use a different port:**
   ```bash
   fak serve --addr 127.0.0.1:8081
   ```

3. **Check for multiple fak instances:**
   ```powershell
   Get-Process fak
   ```

---

## Memory Issues

### Symptom: Out of memory during model load

**Example errors:**
- `cannot allocate memory`
- `alloc.*failed`
- Process termination with OOM

**Common causes:**

1. **Model too large for available RAM:**
   - Qwen3.6-27B requires ~26 GB RSS with KV cache
   - SmolLM2-135M requires ~500 MB
   - Qwen2.5-0.5B requires ~2 GB

2. **Context window too large:**
   - Larger context windows require more KV cache memory
   - Reduce context length or use a smaller model

**Solutions:**

1. **Check available memory:**
   ```powershell
   # Windows
   Get-ComputerInfo | select CsTotalPhysicalMemory, CsFreePhysicalMemory

   # Linux
   free -h
   ```

2. **Use a smaller model:**
   ```bash
   # Instead of 27B
   fak serve --gguf models/qwen2.5-0.5b-q8.gguf --tokenizer ~/.cache/fak-models/tokenizers/qwen2.5

   # Or SmolLM2-135M
   fak serve --gguf internal/model/.cache/smollm2-135m
   ```

3. **Reduce concurrent sessions:**
   - Each session maintains its own KV cache
   - Process concurrent requests sequentially or use fewer agents

4. **Check for memory leaks:**
   ```bash
   # Monitor memory usage
   watch -n 1 'ps aux | grep fak'
   ```

### Symptom: WSLL / FSL OOM during tests

**Issue:** Model tests may intermittently OOM on the 538MB weights.f32 test data.

**Solution:** Run weight-backed tests in isolation:
```powershell
.\fak\test.ps1 ./internal/model -run TestWeight
```

---

## GPU/CUDA Issues

### Symptom: CUDA initialization failures

**Example errors:**
- `compute: cuda device allocation failed`
- `cudaGetLastError() returned non-zero`
- CUDA driver/library not found

**Diagnosis:**

1. **Check NVIDIA GPU availability:**
   ```bash
   nvidia-smi
   ```

2. **Check CUDA toolkit:**
   ```bash
   nvcc --version
   ```

3. **Verify WSL2 GPU passthrough (Windows):**
   ```bash
   # In WSL
   ls /usr/lib/wsl/lib/libcuda.so
   ```

**Solutions:**

1. **Install CUDA toolkit (no sudo required):**
   ```bash
   # From fak/
   bash internal/compute/setup_cuda_wsl.sh
   ```

2. **Build with CUDA support:**
   ```bash
   bash internal/compute/build_cuda.sh
   ```

3. **Use CPU backend instead:**
   ```bash
   fak serve --engine inkernel
   # Or explicitly
   fak serve --engine cpu-ref
   ```

### Symptom: Vulkan device allocation failed

**Example error:**
```
fak-vulkan: device-local alloc(X bytes) failed VkResult=...
```

**Diagnosis:**
```bash
# Check Vulkan support
vulkaninfo
```

**Solutions:**
1. Check GPU driver is up to date
2. Verify Vulkan runtime is installed
3. Try CPU backend: `fak serve --engine cpu-ref`

---

## Model Loading Failures

### Symptom: GGUF file not found or invalid

**Example errors:**
- `open models/qwen.gguf: no such file or directory`
- `invalid GGUF magic`
- `unsupported GGUF version`

**Diagnosis:**
```bash
# Verify file exists and is readable
ls -lh models/qwen.gguf
file models/qwen.gguf
```

**Solutions:**

1. **Download model using provided script:**
   ```powershell
   # From repo root
   python fak/scripts/fetch_model.ps1
   ```

2. **Use correct model path:**
   ```bash
   # Relative to current directory
   fak serve --gguf ./models/qwen.gguf

   # Absolute path
   fak serve --gguf /full/path/to/model.gguf
   ```

3. **Verify GGUF format:**
   - Use `llama.cpp` tools to inspect/convert
   - Ensure model architecture is supported (Llama, Qwen, etc.)

### Symptom: GGUF embeds no usable BPE tokenizer (rare; SPM-only checkpoints)

`fak serve --gguf X` (no `--base-url`) serves real in-kernel chat using the tokenizer
**embedded in the GGUF** — no separate `--tokenizer` is needed for the common case
(Qwen, Gemma, Phi, and other byte-level BPE models). Only a checkpoint that embeds no
usable BPE tokenizer (e.g. an SPM-only model) falls back to the offline mock planner,
with this stderr note:

```
fak serve: --gguf set without --tokenizer and no embedded BPE tokenizer (...);
  /v1/chat/completions will use the offline mock planner. Pass --tokenizer <dir|file> for real chat.
```

**Solution:** point `--tokenizer` at a `tokenizer.json` (or its directory) for that model:
```bash
fak serve --gguf models/qwen.gguf --tokenizer ~/.cache/fak-models/tokenizers/qwen3.6
```

### Symptom: FAK_Q4K model load fails

**Example error:**
```
q4k-direct-load failed
```

**Diagnosis:**
- FAK_Q4K path is for direct Q4_K matmul tensors
- Requires compatible model (Qwen3.6-27B q4_k_m)

**Solutions:**
1. **Verify model compatibility:**
   ```bash
   # Check if model is Qwen3.6-27B q4_k_m
   ```

2. **Use default Q8 path:**
   ```bash
   unset FAK_Q4K
   fak serve --gguf models/qwen.gguf
   ```

---

## Policy and Configuration Issues

### Symptom: Policy validation failure

**Example error:**
```
fak policy: <policy-file>: validation error
```

**Diagnosis:**
```bash
# Validate policy before using
fak policy --check policy.json
```

**Solutions:**

1. **Dump default policy for reference:**
   ```bash
   fak policy --dump > default-policy.json
   ```

2. **Check policy syntax:**
   - Verify JSON is valid
   - Check tool names match registered tools
   - Ensure reason classes are from closed vocabulary

3. **Use built-in policy:**
   ```bash
   fak serve  # Uses DefaultPolicy
   ```

### Symptom: API key not configured

**Example error:**
```
fak serve: env OPENAI_API_KEY is empty
```

**Solutions:**

1. **Set API key:**
   ```powershell
   $env:OPENAI_API_KEY="sk-..."
   fak serve --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
   ```

2. **Use offline mode (no API key):**
   ```bash
   fak serve  # Uses mock planner with no --base-url
   ```

---

## Startup Failures

### Symptom: Gateway fails to start

**Example error:**
```
fak serve: gateway.New: ...
```

**Common causes:**

1. **Invalid engine ID:**
   ```bash
   # Check available engines
   fak run --trace testdata/tau2/smoke.json --engine invalid
   ```

2. **Invalid invalidation granularity:**
   ```bash
   # Must be: global | namespace | resource
   fak serve --invalidation global  # correct
   fak serve --invalidation invalid  # fails
   ```

3. **Engine cache misconfiguration:**
   ```bash
   # --engine-cache-base-url required when --engine-cache-engine is set
   fak serve --engine-cache-engine sglang --engine-cache-base-url http://localhost:10000
   ```

### Symptom: Model load hangs or takes very long

**Diagnosis:**

1. **Check model size and I/O speed:**
   ```bash
   # Large models (27B) can take 30+ seconds to load
   ```

2. **Monitor progress:**
   - Metrics endpoint shows load phases: `GET /metrics`
   - Look for `fak_model_load_phase_duration_seconds`

**Solutions:**

1. **Use smaller model for testing:**
   ```bash
   fak serve --gguf internal/model/.cache/smollm2-135m
   ```

2. **Pre-load weights:**
   - Gateway eager-loads by default
   - First request is fast

---

## Debugging Tools

### Health check endpoint

```bash
curl http://localhost:8080/healthz
```

Returns HTTP 200 when gateway is ready.

### Metrics endpoint

```bash
curl http://localhost:8080/metrics
```

Key metrics for troubleshooting:
- `fak_gateway_time_to_ready_seconds` - Total startup time
- `fak_gateway_startup_phase_duration_seconds` - Per-phase boot cost
- `fak_model_load_duration_seconds` - Model load time
- `fak_model_load_bytes` - Bytes loaded

### Verbose logging

```bash
# Enable debug logging
FAK_LOG=debug fak serve
```

### Test kernel in isolation

```bash
# Test adjudication without model
fak run --trace testdata/tau2/smoke.json

# Test with mock planner
fak serve  # No --base-url = offline mode
```

### Check registered engines

```bash
# View available engines
fak run --trace testdata/tau2/smoke.json --engine ?
```

---

## Quick Reference: Common Commands

```bash
# Minimal server (no model, offline mode)
fak serve

# With local GGUF model
fak serve --gguf models/qwen.gguf --tokenizer ~/.cache/fak-models/tokenizers/qwen

# Proxy to external model
fak serve --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY

# With custom policy
fak serve --policy policy.json

# Check policy before using
fak policy --check policy.json

# Verify model load
fak serve --gguf models/qwen.gguf --policy-check
```

---

## Additional Resources

- [Getting Started](../../fak/GETTING-STARTED.md) - Install and basic usage
- [GPU Support](../../fak/GPU.md) - CUDA and Vulkan setup
- [README](../../fak/README.md) - Project overview
- [Architecture](../../fak/ARCHITECTURE.md) - System design

---

## Still stuck?

1. Check the logs: `fak serve` writes to stderr by default
2. Verify prerequisites: Go 1.26+, sufficient RAM, compatible model
3. Try minimal config first: `fak serve` (no model, offline)
4. Check GitHub issues: https://github.com/anthony-chaudhary/fak/issues

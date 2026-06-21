# WebBench End-to-End Runner

`webbench-run` is a reproducible end-to-end webbench runner that executes web agent tasks with a specified model API and measures actual token usage.

## Build

```bash
cd fak/cmd/webbench-run
go build -o webbench-run.exe .
```

## Usage

```bash
webbench-run --dataset <path> --api-key <key> [options]
```

### Required Arguments

- `--dataset`: Path to dataset file (JSONL format)
- `--api-key`: GLM API key (or set `GLM_API_KEY` environment variable)

### Options

- `--model`: Model to use (default: `glm-4`)
  - Supported: `glm-4`, `glm-4-plus`, `glm-4-0520`, etc.
- `--max-turns`: Maximum turns per task (default: 10)
- `--output`: Output file for results (default: `results.json`)
- `--max-tasks`: Maximum tasks to run, 0 = all (default: 0)
- `--timeout`: Timeout per task in seconds (default: 300)

## Dataset Format

The dataset must be JSONL with one task per line:

```json
{"task_id":"wb-001","benchmark":"webbench","source_url":"https://example.com","description":"Find the contact email","instructions":"Navigate to https://example.com and find the contact email address."}
```

## Getting a GLM API Key

1. Visit [https://open.bigmodel.cn/](https://open.bigmodel.cn/)
2. Register/login to your account
3. Navigate to API Keys section
4. Create a new API key
5. Set the `GLM_API_KEY` environment variable or use `--api-key`

## Browser Automation Setup

### Option 1: Windows (Node.js + Playwright)
```bash
npm install -g playwright
npx playwright install chromium
```

### Option 2: WSL (recommended)
```bash
wsl -e bash -lc "pip install playwright && playwright install chromium --with-deps"
```

## Example Run

```bash
# Using environment variable
export GLM_API_KEY="your-api-key-here"
webbench-run --dataset sample-dataset.jsonl --output results.json

# Using command-line argument
webbench-run --dataset sample-dataset.jsonl --api-key "your-api-key-here" --output results.json
```

## Output Format

The output JSON contains:

```json
{
  "tasks_total": 3,
  "tasks_success": 2,
  "success_rate": 66.7,
  "total_tokens": 15000,
  "total_prefill": 12000,
  "total_decode": 3000,
  "results": [
    {
      "task_id": "wb-001",
      "success": true,
      "turn_results": [
        {
          "turn_number": 1,
          "prefill_tokens": 5000,
          "decode_tokens": 150,
          "total_tokens": 5150,
          "latency_ms": 1234,
          "timestamp": "2026-06-20T08:22:40.7670389-07:00"
        }
      ],
      "total_prefill": 15000,
      "total_decode": 450,
      "total_tokens": 15450,
      "turn_count": 3
    }
  ],
  "config": { ... },
  "start_time": "2026-06-20T08:22:40Z",
  "end_time": "2026-06-20T08:25:00Z",
  "duration_sec": 140.5
}
```

## Current Status

- ✅ GLM-5.2 API integration (removed simulated fallback, proper error handling)
- ✅ Environment variable support for API key
- ✅ Detailed error messages for API failures
- ✅ Sample dataset created
- ✅ JSON output with per-turn token measurements

## Next Steps

For full web agent benchmarks, the runner would need:
1. Browser automation integration (Playwright/Selenium)
2. DOM state capture and tokenization
3. Action execution (click, type, navigate)
4. Task success evaluation

Currently, `webbench-run` demonstrates the token measurement framework with model API integration.

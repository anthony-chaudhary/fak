---
title: "fak — Cài đặt và bắt đầu (Bản nhập môn tiếng Việt / Vietnamese introduction)"
description: "Trang cài đặt fak bằng tiếng Việt: một Go binary tĩnh duy nhất kiểm duyệt mọi tool call trước khi chạy và tái sử dụng công việc lặp lại trong phiên dài — cùng một agent loop trở nên an toàn, rẻ và nhanh hơn; self-host, phù hợp Nghị định 13/2023/NĐ-CP, Apache-2.0."
---

# fak — Cài đặt và bắt đầu (Bản nhập môn tiếng Việt)

> Đây là một **điểm vào (entry point) đã bản địa hóa**, không phải bản dịch đầy đủ.
> Toàn bộ tài liệu là tiếng Anh — trang này đưa bạn từ một bản checkout sạch đến một
> kernel đang chạy, rồi dẫn tới [tài liệu tiếng Anh](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Lưu ý:** Bản dịch này do máy tạo và đang chờ người bản ngữ soát lại — nếu phát hiện
> lỗi, hãy mở issue/PR.
> Các ngôn ngữ khác: xem [i18n hub](../README.md).

Đây là **cửa trước để cài đặt và chạy**. Phần giới thiệu chi tiết nằm ở
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md).

**fak là một Go binary duy nhất** — một artifact tĩnh, không phụ thuộc bên ngoài — nằm
giữa AI agent của bạn và các tool mà nó gọi. Nó kiểm duyệt mọi tool call *trước khi* chạy,
và tái sử dụng công việc chung lặp lại trong một phiên dài. Kết quả: cùng một agent loop
trở nên **an toàn, rẻ và nhanh hơn**, mà không phải viết lại gì. Bạn không viết lại agent —
bạn chỉ trỏ một base URL về `fak serve`, hoặc bọc agent hiện có bằng một lệnh:

```bash
fak manage claude
```

## Vì sao điều này quan trọng ở Việt Nam

- **Dữ liệu ở lại trong nước (Nghị định 13/2023/NĐ-CP — PDPD).** fak ưu tiên self-host:
  một binary tĩnh đặt trước một **model local** hoặc provider trong nước, với residency
  fail-closed trên mọi backend, một capability floor mặc-định-từ-chối (default-deny), và
  mọi tool call đều được kiểm duyệt *trước khi* chạy. Dữ liệu không rời khỏi máy của bạn.
- **Chi phí tính bằng VND, doanh thu cũng vậy.** fak tái sử dụng công việc chung (KV cache
  của system prompt và tool list) trong các phiên dài — làm **ít hơn khoảng 4.1× công
  việc** so với một warm-cache stack đã được tinh chỉnh, đo trên một phiên 50 lượt × 5
  agent. Con số ~60× (khoảng 19 giờ xuống còn khoảng 19 phút) **chỉ đúng khi so với vòng
  lặp "gửi lại tất cả" ngây thơ (naive)**, không phải là con số headline. Phần tái sử dụng
  này **chỉ áp dụng cho self-host** và cho các fleet đọc-nhiều (read-heavy). Đây là một đòn
  bẩy trực tiếp lên biên lợi nhuận tính bằng VND.
- **Không vượt qua rào thanh toán nào.** fak là **Apache-2.0**, miễn phí, self-host — không
  thẻ tín dụng, không hóa đơn xuyên biên giới, không cần pháp nhân. `git clone` cùng
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` là toàn bộ con đường.

> **Về prompt-cache của provider.** Chiết khấu prompt-cache chỉ còn hiệu lực khi cached
> prefix giữ nguyên từng byte; fak vẫn giữ prefix **byte-identical** trong khi loại bỏ các
> lượt cũ ở giữa. fak **bảo đảm** tính byte-identical của prefix; còn việc provider có thật
> sự tái dùng cache hay không là quyết định của provider — fak chuyển tiếp (relay) điều đó
> chứ không tự tuyên bố.

## 0. Điều kiện tiên quyết

- **Go 1.26+.** `fak/go.mod` khai báo `go 1.26`. Với `GOTOOLCHAIN=auto` mặc định của Go,
  một `go` cũ hơn sẽ tự tải đúng toolchain trong lần build đầu (cần mạng một lần); nếu không,
  cài Go 1.26 từ <https://go.dev/dl/>. Kiểm tra bằng `go version`.
- **Tier 0 chỉ cần thế**: không GPU, không API key, không mạng.

## Các tier (theo chi phí thiết lập tăng dần)

| Tier | Bạn nhận được | Thiết lập |
|---|---|---|
| **0 — Thử kernel** | Chạy/đo ranh giới kiểm duyệt (adjudication) offline | `go build` |
| **1 — Đặt trước một model thật** | Đặt kernel trước một model bạn phục vụ ở nơi khác (Ollama / vLLM / llama.cpp / cloud) | + một OpenAI-compatible server đang chạy |
| **1b — Model local bằng một lệnh** | Chạy một model GGUF local in-kernel với agent hiện có — không key, không mạng, không cần terminal thứ hai | `fak guard --gguf qwen2.5:7b -- claude` |
| **2 — Model hợp nhất trong kernel** | Forward pass thuần Go do kernel sở hữu (proven bit-exact) | + export (real weights) |

Nếu bạn chỉ muốn **phục vụ một model hữu ích với fak đứng trước nó**, hãy chọn **Tier 1**.
Model in-kernel ở Tier 2 là một *reference forward pass* được chứng minh khớp từng bit với
HuggingFace, không phải một serving engine chất lượng chat.

## Cài đặt binary

Build từ bản clone:

```bash
go build -o fak ./cmd/fak
```

Hoặc cài trực tiếp bằng Go (module path chính là gốc repository):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Ghi chú cho Windows.** Trên Windows hãy build bằng `go build -o fak.exe ./cmd/fak`;
> `-o fak` (không đuôi) để lại một file `fak` mà cmd.exe / PowerShell không gọi được theo
> tên. `go build` / `go vet` / `go run` chạy natively; nếu cần chạy `go test ./...`, hãy
> chạy dưới WSL.

## Bằng chứng 60 giây (không key, không model, không GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

Vì sao điều này an toàn theo *cấu trúc*, không phải nhờ một classifier:

- **Capability floor mặc-định-từ-chối**, được kiểm bên *trong* kernel trên cùng call path —
  fail-closed. Một hành động không nằm trong allow-list thì không thể chạy, dù model bị dụ
  đến đâu.
- **Cách ly kết quả (result quarantine):** các *kết quả* tool đáng ngờ bị giữ hoàn toàn
  ngoài context của model — đây là cấu trúc, không phải bộ dò tìm. Bộ dò gắn cờ chúng được
  xem là ~100% có thể né tránh theo thiết kế: một phần thưởng, không bao giờ là sàn.
- **Kiểm thử thực tế:** prompt injection chạm tới baseline không được bảo vệ 5/5; fak chặn
  được 5/5.

## Chạy với model của bạn

fak **govern và cache** model của bạn; nó không thay thế model. **Qwen2/Qwen3 và GLM-MoE**
đã được chứng minh bit-exact trong reference engine in-kernel. Mọi thứ khác (DeepSeek,
Mistral, bất kỳ open-weights model nào) đứng phía trước qua OpenAI-compatible wire: Ollama /
vLLM / SGLang / llama.cpp / LM Studio hoặc bất kỳ OpenAI-compatible API nào.

## Đi tiếp đâu

- [Getting Started — hướng dẫn cài đặt đầy đủ (tiếng Anh)](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Guided first session — phiên đầu tiên có hướng dẫn từng bước](../../fak/tutorial.md)
- [README — tổng quan đầy đủ](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — chạy model local trong 10 phút](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Integrations — kết nối agent của bạn](../../integrations/README.md)
- [Data residency và tuân thủ — cho Nghị định 13/2023/NĐ-CP](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — nguồn của từng con số](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — cái gì là shipped / simulated / stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

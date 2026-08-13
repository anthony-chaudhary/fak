---
title: "fak — nhân hợp nhất cho AI agent (giới thiệu tiếng Việt / Vietnamese introduction)"
description: "Trang vào tiếng Việt của fak: một binary Go tĩnh nằm giữa AI agent và các tool nó gọi — xét duyệt mọi tool call trước khi chạy, tái dùng công việc lặp lại trong phiên dài; tự lưu trữ, hợp quy Nghị định 13/2023/NĐ-CP (PDPD), Apache-2.0."
---

# fak — nhân hợp nhất cho AI agent (giới thiệu tiếng Việt)

> Đây là một **trang vào (entry point) đã bản địa hóa**, không phải bản dịch đầy đủ.
> Toàn bộ tài liệu gốc bằng tiếng Anh — trang này trao cho bạn phần cốt lõi của fak,
> bản chứng minh 60 giây và đường cài đặt, rồi dẫn bạn tới
> [tài liệu tiếng Anh](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Lưu ý:** bản dịch này do máy tạo và đang chờ người bản ngữ soát lại — nếu thấy sai sót,
> xin mở issue/PR để góp ý.
>
> **Các ngôn ngữ khác:** xem [i18n hub](../README.md).

## fak trong một câu

**fak là một binary Go** nằm giữa AI agent của bạn và các tool mà nó gọi — xét duyệt mọi
tool call *trước khi nó chạy*, và tái dùng công việc dùng chung được lặp lại trong một phiên
dài. Kết quả: vẫn vòng lặp agent đó nhưng **an toàn hơn, rẻ hơn và nhanh hơn**, không phải
viết lại gì cả.

Bạn không viết lại agent — chỉ trỏ lại một base URL về `fak serve`, và mọi tool call sẽ đi
qua capability floor trước tiên.

```bash
fak manage claude      # bọc agent hiện có của bạn chỉ bằng một lệnh
```

## Vì sao các đội ngũ Việt Nam nên quan tâm

- **Chi phí đau ở tiền đồng, hóa đơn token tính bằng đô.** Trong các phiên dài, fak tái dùng
  công việc dùng chung (system prompt, KV cache của danh sách tool): trên một run 50 lượt × 5
  agent, fak **làm ít hơn khoảng 4.1× công việc** so với một stack warm-cache đã được tinh
  chỉnh. (Con số ~60× — khoảng 19 giờ rút còn khoảng 19 phút — chỉ đúng **so với kiểu gửi lại
  toàn bộ ngây thơ (naive)**, không phải con số tiêu đề.) Phần tái dùng này áp dụng cho
  **bản tự lưu trữ** với các fleet đọc nhiều (read-heavy). Đây là đòn bẩy trực tiếp lên biên
  lợi nhuận (VND).
- **Dữ liệu ở lại trong nước (Nghị định 13/2023/NĐ-CP — PDPD).** fak ưu tiên tự lưu trữ:
  một binary tĩnh đứng chắn trước **model local** hoặc provider trong nước, với lưu trú dữ
  liệu **fail-closed** trên mọi backend, capability floor **default-deny**, và nhật ký kiểm
  toán chống giả mạo (tamper-evident) cho mọi tool call. Dữ liệu không rời khỏi máy bạn.
- **Không phải vượt cổng thanh toán xuyên biên giới.** fak là **Apache-2.0**, miễn phí, tự
  lưu trữ — không thẻ tín dụng, không hóa đơn xuyên biên giới, không pháp nhân. `git clone`
  cùng `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` là toàn bộ con đường.
- **Một binary tĩnh, không phụ thuộc bên ngoài.** Vận hành gọn cho đội nhỏ — không sidecar,
  không thành phần cấp quyền riêng. Từ laptop tới fleet vẫn là một artifact; bạn chỉ thêm
  flag, không thêm thành phần.

## fak giải quyết những vấn đề gì

- **Phiên dài vẫn rẻ.** Chiết khấu prompt-cache của provider chỉ còn hiệu lực khi cached
  prefix giữ nguyên **từng byte**; fak lược bỏ các lượt cũ ở giữa mà vẫn giữ prefix
  **giống hệt từng byte**, nên chiết khấu không bị vỡ. fak **bảo đảm** tính đồng nhất từng
  byte của prefix; còn việc provider có thật sự tái dùng cache hay không là quyết định của
  provider — fak chuyển tiếp (relay) chứ không tự nhận.
- **Bảo mật default-deny.** Chính sách quyền chạy *bên trong* kernel, trên cùng đường gọi đó
  — **fail-closed**. Một hành động không nằm trong allow-list thì không thể chạy, dù model có
  bị dụ đến đâu.
- **Cách ly prompt injection / kết quả bị đầu độc.** Các *kết quả* tool đáng ngờ được giữ lại
  hoàn toàn khỏi ngữ cảnh của model (quarantine) — bằng **cấu trúc**, không phải bằng bộ phát
  hiện. Bộ phát hiện gắn cờ chúng được xem là **~100% có thể né tránh** theo thiết kế: đó là
  phần thưởng thêm, không bao giờ là mức sàn. Trong kiểm thử trực tiếp, prompt injection lọt
  qua baseline không được bảo vệ 5/5 lần; fak chặn kín 5/5 lần.

## Bản chứng minh 60 giây (không key, không model, không GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection bị chặn, tác vụ vẫn hoàn tất
```

## Với model của bạn

fak không thay model của bạn — nó **govern và cache** model đó. **Qwen2/Qwen3 và GLM-MoE**
đã được chứng minh bit-exact trong reference engine in-kernel; mọi thứ còn lại (DeepSeek,
Mistral, bất kỳ model open-weights nào) đứng phía trước qua giao thức tương thích OpenAI —
Ollama / vLLM / SGLang / llama.cpp / LM Studio hoặc bất kỳ OpenAI-compatible API nào.

## Đi tiếp từ đâu

- [Quickstart (tiếng Việt)](./quickstart.md)
- [Cài đặt (tiếng Việt)](./install.md)
- [FAQ (tiếng Việt)](./faq.md)
- [README (tổng quan đầy đủ)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — chạy local model trong 10 phút](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — cài đặt binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — kết nối agent của bạn](../../integrations/README.md)
- [Lưu trú dữ liệu và hợp quy — cho Nghị định 13/2023/NĐ-CP](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — nguồn của mọi con số](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — cái gì đã shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

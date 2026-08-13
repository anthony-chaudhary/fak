---
title: "fak — Câu hỏi thường gặp khi lần đầu tiếp xúc (Giới thiệu tiếng Việt / Vietnamese introduction)"
description: "Trang FAQ tiếng Việt cho fak: một Go binary tĩnh đứng trước AI agent của bạn, kiểm tra mọi tool call trước khi chạy và tái sử dụng công việc lặp lại trong phiên dài — self-host, phù hợp Nghị định 13/2023/NĐ-CP (PDPD), Apache-2.0."
---

# fak — Câu hỏi thường gặp khi lần đầu tiếp xúc

> Đây là một **trang khởi đầu (entry point) được bản địa hóa**, không phải bản dịch
> toàn bộ tài liệu. Tài liệu đầy đủ nằm bằng tiếng Anh — trang này trả lời vài câu hỏi
> đầu tiên rồi dẫn bạn tới [tài liệu tiếng Anh](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Lưu ý:** Bản dịch này do máy tạo và đang chờ người bản ngữ soát lại — nếu thấy sai
> sót, xin mở issue/PR.
>
> **Các ngôn ngữ khác:** xem [i18n hub](../README.md).

## Câu hỏi thường gặp

### Q1. fak là gì?

fak là **một Go binary tĩnh duy nhất** mà bạn đặt *phía trước* AI agent bạn đã chạy sẵn
(Claude Code, Codex, Cursor, hay bất kỳ client OpenAI/Anthropic/MCP nào) bằng cách trỏ
lại **một base URL duy nhất**, không phải viết lại gì cả. Nó làm các phiên dài rẻ hơn
(loại bỏ các lượt cũ trong khi vẫn giữ prefix của provider prompt-cache **y hệt từng
byte**), định tuyến mỗi tool call, có thể chạy các model GGUF cục bộ ngay trong tiến
trình, và ghi lại một phán quyết **có thể kiểm toán** cho mỗi lần gọi.

*(Về prompt-cache: fak **bảo đảm** prefix giữ nguyên bit-identical khi lược bỏ các lượt
cũ ở giữa; còn việc provider có thực sự tái dùng cache hay không là quyết định của
provider — fak chuyển tiếp lời gọi đó chứ không tự nhận thay.)*

### Q2. Tôi có phải đổi model hay viết lại agent không?

Không. fak **govern và cache** chính model bạn đang dùng — nó không thay thế model. Bọc
nó bằng một câu lệnh:

```bash
fak manage claude
```

hoặc trỏ một base URL sang `fak serve`.

### Q3. Dữ liệu của tôi đi đâu — có tuân thủ không?

**Self-host trước tiên:** một binary tĩnh đứng trước một model cục bộ hoặc một nhà cung
cấp trong nước, với **residency fail-closed**, một **capability floor mặc-định-từ-chối
(default-deny)**, và một **audit log có thể kiểm toán** cho mỗi tool call. Dữ liệu của bạn
**không rời khỏi máy của bạn**. Điều này giúp bạn giữ dữ liệu cá nhân trong lãnh thổ theo
yêu cầu của **Nghị định 13/2023/NĐ-CP (PDPD)** về bảo vệ dữ liệu cá nhân.

### Q4. Chi phí bao nhiêu? Có thực sự miễn phí không?

**Apache-2.0, miễn phí, self-host.** Không thẻ tín dụng, không hóa đơn xuyên biên giới,
không cần pháp nhân. `git clone` cộng với `go install` là toàn bộ con đường. Không có
khoản phí bản quyền nào tính bằng VND — chi phí tiền mặt duy nhất của bạn là hạ tầng tự
vận hành, chứ không phải phí license.

### Q5. Rẻ hơn hay nhanh hơn bao nhiêu?

Trên một phiên **đo được 50 lượt x 5 agent**, ít hơn **khoảng 4.1x công việc** so với một
stack warm-cache **đã được tinh chỉnh (TUNED)**. Con số **~60x** (khoảng 19 giờ xuống
khoảng 19 phút) **chỉ** đúng khi so với một vòng lặp gửi-lại-toàn-bộ **ngây thơ (naive)**
— không bao giờ dùng làm con số tiêu đề. Phần thắng nhờ tái sử dụng **chỉ áp dụng cho
self-host**, với các fleet đọc-nhiều (read-heavy).

Vì hóa đơn token thường tính bằng USD trong khi doanh thu của bạn tính bằng **VND**, việc
giảm khối lượng công việc này tác động trực tiếp lên **biên lợi nhuận (margin) tính bằng
VND**.

### Q6. Những model nào chạy được?

**Qwen2/Qwen3 và GLM-MoE** đã được chứng minh **bit-exact** trong reference engine
in-kernel. Mọi thứ khác (DeepSeek, Mistral, bất kỳ model open-weights nào) đứng phía
trước qua wire **tương thích OpenAI**: Ollama / vLLM / SGLang / llama.cpp / LM Studio.

### Q7. Nó chặn prompt injection bằng cách nào?

**Hai cổng mang tính cấu trúc, không phải một classifier:** một **capability floor
mặc-định-từ-chối** (một tool nguy hiểm không bao giờ nằm trong allow-list) và **result
quarantine** (kết quả tool bị đầu độc không bao giờ tới được context của model). Trong các
bài kiểm tra trực tiếp, injection đạt tới baseline không được bảo vệ **5/5**, còn fak chặn
đứng nó **5/5**.

### Q8. Cài đặt thế nào?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

hoặc `go build -o fak ./cmd/fak` từ một bản clone. *(Nếu mạng của bạn không truy cập được
`proxy.golang.org`, hãy đặt biến `GOPROXY` sang một Go module proxy thay thế trước khi
cài.)*

### Q9. Đi tiếp đâu?

Xem mục **Đi tiếp** ở cuối trang — nó dẫn tới README, START-HERE, GETTING-STARTED,
BENCHMARK-AUTHORITY và CLAIMS bằng tiếng Anh.

## Bằng chứng 60 giây (không cần key, không cần model, không cần GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Đi tiếp

- [README (tổng quan đầy đủ)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — chạy model cục bộ trong 10 phút](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — cài đặt binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — nguồn của từng con số](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — cái gì là shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Dữ liệu residency và tuân thủ — cho Nghị định 13/2023/NĐ-CP (PDPD)](../../explainers/data-residency-and-compliance.md)
- [Integrations — kết nối agent của bạn](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

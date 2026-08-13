---
title: "fak — quickstart: 10 phút để có một model chạy local (bản tiếng Việt)"
description: "Trang khởi đầu tiếng Việt cho fak: từ số không đến một AI local được kiểm soát trong khoảng 10 phút — offline, không cần key, không hóa đơn cloud, dữ liệu ở lại máy bạn; self-host phù hợp Nghị định 13/2023/NĐ-CP (PDPD)."
---

# fak — quickstart: 10 phút để có một model local (bản tiếng Việt)

> Đây là một **trang khởi đầu (entry point) được bản địa hóa**, không phải bản dịch đầy đủ.
> Toàn bộ tài liệu gốc bằng tiếng Anh — trang này chỉ đưa bạn nhanh vào fak rồi dẫn tới
> [tài liệu tiếng Anh](https://github.com/anthony-chaudhary/fak/blob/main/README.md), vốn là
> nguồn chuẩn (source of truth).
> **Lưu ý:** bản dịch này do máy tạo và đang chờ người bản ngữ rà soát — nếu thấy lỗi, hãy
> mở issue hoặc PR.
> Các ngôn ngữ khác và trang tổng: [i18n hub](../README.md).

## Lời hứa: khoảng 10 phút, từ số không đến một AI local được kiểm soát

Sau các bước dưới đây, bạn có một AI chạy ngay trên máy của mình: **offline**, **không cần
API key**, **không hóa đơn cloud**, **dữ liệu ở lại trên máy bạn**, và **CPU là đủ** cho các
model nhỏ — không cần GPU.

Với thị trường Việt Nam, việc dữ liệu không rời khỏi máy là đòn bẩy tuân thủ trực tiếp cho
**Nghị định 13/2023/NĐ-CP (PDPD)** về bảo vệ dữ liệu cá nhân: fak là self-host trước tiên, nên
dữ liệu người dùng không đi qua biên giới.

## Đường nhanh nhất: bọc agent hiện có bằng một model local, trong một câu lệnh

Bạn không phải viết lại agent — chỉ cần bọc nó lại:

```bash
fak manage claude
```

fak là **một binary Go tĩnh** nằm giữa AI agent và các tool nó gọi. Nó **xét duyệt mọi tool
call *trước khi* chạy**, và **tái sử dụng phần việc chung lặp lại** trong một session dài. Kết
quả: cùng một vòng lặp agent trở nên **an toàn hơn, rẻ hơn, nhanh hơn** — không cần viết lại
gì cả. Cách còn lại: trỏ **một** base URL sang `fak serve`.

## Kiểm chứng trong 60 giây (không key, không model, không GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection bị chặn, task vẫn hoàn thành
```

Vì sao đáng tin: tầng năng lực (capability floor) **mặc-định-từ-chối** được kiểm ngay *bên
trong* kernel, trên cùng một call path — **fail-closed**. Một hành động không nằm trong
allow-list thì không thể chạy, dù model có bị dụ thế nào. Và các tool *result* đáng ngờ bị giữ
trong **quarantine**, hoàn toàn không đưa vào context của model — đây là **cấu trúc**, không
phải một classifier. Trong kiểm thử trực tiếp, prompt injection xuyên qua baseline không bảo vệ
5/5 lần; fak chặn được 5/5.

## fak là gì và nhanh tới đâu (nói thật)

fak **govern và cache** model của bạn, chứ không thay thế nó. **Qwen2/Qwen3 và GLM-MoE** đã
được chứng minh bit-exact trong reference engine in-kernel. Mọi model khác (DeepSeek, Mistral,
bất kỳ open-weights nào) đi qua wire tương thích OpenAI: Ollama / vLLM / SGLang / llama.cpp /
LM Studio hoặc bất kỳ API tương thích OpenAI nào.

Về tốc độ, xin nói thẳng bằng con số trung thực: trên một session đo được **50 lượt × 5 agent**,
mức lợi so với một stack warm-cache *đã được tinh chỉnh (tuned)* là khoảng **~4.1× ít việc hơn**.
Con số **~60×** (khoảng 19 giờ xuống còn khoảng 19 phút) **chỉ đúng khi so với vòng lặp gửi-lại-
tất-cả kiểu naive** — đừng dùng nó làm tiêu đề. Phần lợi từ tái sử dụng là **chỉ self-host** và
áp dụng cho các fleet đọc-nhiều (read-heavy).

Về chi phí: token tính bằng đô-la nhưng biên lợi nhuận bạn cảm nhận bằng **VND**. fak tái dùng
phần việc chung (system prompt, tool list) qua các session dài, nên đây là đòn bẩy trực tiếp lên
margin. Với prompt-cache của provider: fak giữ **prefix đã cache y hệt từng byte** trong khi
loại bỏ các lượt cũ ở giữa, nên phần giảm giá còn nguyên. fak **bảo đảm** prefix byte-identity;
còn provider có thực sự tái dùng cache hay không là quyết định của provider — fak chuyển tiếp
(relay) chứ không tự nhận.

Giấy phép: **Apache-2.0**, miễn phí, self-host. Không thẻ tín dụng, không hóa đơn xuyên biên
giới, không pháp nhân — `git clone` cùng
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` là toàn bộ con đường.

## Đi tiếp đâu

- [README (tổng quan đầy đủ)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 phút tới một model local](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — cài đặt binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — nguồn của từng con số](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — cái gì đã shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency & tuân thủ — cho Nghị định 13/2023/NĐ-CP (PDPD)](../../explainers/data-residency-and-compliance.md)
- [Integrations — nối agent của bạn vào](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

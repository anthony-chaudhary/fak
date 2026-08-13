---
title: "fak — hızlı başlangıç (10 dakikada yerel bir model)"
description: "fak'ın Türkçe giriş sayfası: sıfırdan yaklaşık 10 dakikada, tek komutla çalışan yönetimli bir yerel AI — çevrimdışı, anahtar yok, bulut faturası yok, veriler makinenizde kalır (KVKK); küçük modeller için CPU yeterli."
---

# fak — hızlı başlangıç (10 dakikada yerel bir model)

> Bu, tam bir çeviri değil, **yerelleştirilmiş bir giriş noktasıdır (entry point)**.
> Dokümantasyonun tamamı İngilizcedir — bu sayfa size fak'ın özünü, 60 saniyelik kanıtı
> ve kurulum yolunu verip sizi
> [İngilizce doğruluk kaynağına](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> yönlendirir.
> **Not:** Bu sayfa makine tarafından yazılmıştır ve ana dil incelemesi beklemektedir —
> hataları düzeltmek için issue/PR açmaktan çekinmeyin.
> **Diğer diller:** [i18n hub](../README.md).

## Söz: sıfırdan ~10 dakikada yönetimli bir yerel AI

Bu adımların sonunda kendi makinenizde çalışan bir AI'ınız olur: **çevrimdışı** çalışır,
hiçbir maliyeti yoktur (API anahtarı yok, bulut faturası yok), **verileriniz makinenizde
kalır** ve küçük modeller için **CPU yeterlidir** — GPU şart değil. Veriler makineden
çıkmadığı için bu, KVKK kapsamındaki veri yerelliği yükümlülüklerine de uygun bir
başlangıç noktasıdır.

## En hızlı yol

Mevcut ajanınızı tek komutla yerel bir modelin arkasına alın (anahtar yok, ağ yok):

```bash
fak manage claude
```

Ajanınızı yeniden yazmazsınız — tek bir base URL'i `fak serve`'e yönlendirir ya da
mevcut ajanı `fak guard -- claude` ile sararsınız. Aynı ajan döngüsü olduğu gibi kalır;
sadece daha güvenli, daha ucuz ve daha hızlı olur.

## 60 saniyelik kanıt (anahtar yok, model yok, GPU yok)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## fak nedir

**fak**, AI ajanınız ile onun çağırdığı araçlar arasında oturan **tek bir statik Go
binary**'sidir. Her tool call'ı *çalışmadan önce* denetler ve uzun bir oturumda
tekrarlanan ortak işi yeniden kullanır. Sonuç: aynı ajan döngüsü, hiçbir yeniden yazım
olmadan daha güvenli, daha ucuz ve daha hızlı hale gelir.

- **Yapısal güvenlik.** Kernel'in *içinde*, aynı call path üzerinde çalışan bir
  default-deny yetenek tabanı (capability floor) vardır — **fail-closed**. Allow-list'te
  olmayan bir eylem, model ne kadar kandırılırsa kandırılsın çalışamaz. Ayrıca şüpheli
  tool *sonuçları* model bağlamının tamamen dışında tutulur (quarantine) — bunu bir
  sınıflandırıcı değil, yapı sağlar. Canlı testlerde prompt injection korumasız temele
  5/5 ulaştı; fak onu 5/5 duvarla kapattı.
- **Modelinizi değiştirmez, yönetir ve cache'ler.** Qwen2/Qwen3 ve GLM-MoE, in-kernel
  referans motorunda bit-exact olarak kanıtlanmıştır; geri kalan her şey (DeepSeek,
  Mistral, herhangi bir open-weights model) OpenAI uyumlu wire üzerinden bağlanır —
  Ollama / vLLM / SGLang / llama.cpp / LM Studio veya herhangi bir OpenAI uyumlu API.

## Ne kadar hızlı (dürüst çerçeve)

Ölçülmüş bir 50-tur × 5-ajan oturumunda, *ayarlanmış* (tuned) bir warm-cache yığınına
kıyasla dürüst kazanç **yaklaşık 4.1× daha az iş**tir. Maliyet TRY cinsinden hissedildiği
için bu doğrudan bir marj kaldıracıdır: uzun oturumlarda paylaşılan iş (system prompt,
tool list'in KV cache'i) yeniden kullanılır.

**~19 saatten ~19 dakikaya (~60×)** rakamı ise **yalnızca naif, her şeyi yeniden gönderen
bir döngüye kıyasla** doğrudur; bunu asla başlık rakamı olarak sunmayın. Bu yeniden
kullanım kazancı **yalnızca self-host** içindir ve okuma ağırlıklı filolar için geçerlidir.

Provider'ın prompt-cache indirimi yalnızca cached prefix byte-for-byte aynı kaldığında
korunur; fak eski orta turları atarken bile prefix'i byte-identical tutar. fak prefix'in
byte-özdeşliğini **garanti eder**; provider'ın cache'i gerçekten yeniden kullanıp
kullanmadığı ise provider'ın kararıdır — fak bunu iddia etmez, yalnızca aktarır.

## Lisans / maliyet

fak **Apache-2.0**, ücretsiz ve self-host'tur. Kredi kartı yok, sınır ötesi fatura yok,
tüzel kişilik gerekmez. `git clone` ve
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` tüm yol budur.

## Nereye gitmeli

- [README (tam genel bakış)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 dakikada yerel model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary'i kurun](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Kurulum — binary'i indirin/derleyin](./install.md)
- [SSS — sık sorulan sorular](./faq.md)
- [Integrations — ajanınızı bağlayın](../../integrations/README.md)
- [Veri yerelliği ve uyum — KVKK için](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — her rakamın kaynağı](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — neyin shipped/simulated/stub olduğu](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

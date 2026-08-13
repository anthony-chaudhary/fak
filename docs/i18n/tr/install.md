---
title: "fak — kurulum ve başlangıç (Türkçe giriş / Turkish introduction)"
description: "fak'in Türkçe kurulum giriş sayfası: her tool call'u çalışmadan önce inceleyen tek statik Go binary — aynı ajan loop'u daha güvenli, daha ucuz, daha hızlı; self-host, KVKK'ya uygun, Apache-2.0."
---

# fak — kurulum ve başlangıç (Türkçe giriş)

> Bu bir **yerelleştirilmiş giriş noktasıdır (entry point)**, dokümantasyonun tam çevirisi
> değildir. Tüm dokümantasyon İngilizcedir — bu sayfa size fak'in kurulum ve çalıştırma
> yolunu verip sizi [İngilizce dokümana](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> yönlendirir.
> **Not:** Bu çeviri makine tarafından üretilmiştir ve ana dilde inceleme beklemektedir —
> hata görürseniz issue/PR açın. Diğer diller için: [i18n hub](../README.md).

## Tek satırda

Bu, **kurulum-ve-çalıştırma giriş kapısıdır**; yoğun tanıtım
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md)'dedir. fak, AI
ajanınız ile çağırdığı araçlar arasında oturan **tek bir statik Go binary**'sidir: her tool
call'u *çalışmadan önce* inceler ve uzun bir oturumda tekrarlanan paylaşılan işi yeniden
kullanır. Sonuç: aynı ajan loop'u **daha güvenli, daha ucuz ve daha hızlı** — hiçbir şeyi
yeniden yazmadan. Ajanınızı yeniden yazmazsınız; tek bir base URL'i `fak serve`'e yöneltir
ya da mevcut ajanı tek komutla sararsınız:

```bash
fak manage claude      # mevcut ajanınızı tek komutla sarar
```

## Türkiye için neden önemli

- **Veri ülkede kalır (KVKK).** fak önce self-host'tur: bir **local model** veya yerel bir
  provider'ın önünde oturan statik bir binary, her backend'de fail-closed veri yerleşimi,
  default-deny capability floor ve her tool call için kurcalama-kanıtlı (tamper-evident)
  audit log. Veri makinenizden çıkmaz.
- **Maliyet TL ile acıtır, token faturası döviz cinsinden gelir.** fak, uzun oturumlarda
  paylaşılan işi (system prompt, tool listesinin KV cache'i) yeniden kullanır: tuned bir
  warm-cache stack'e kıyasla 50×5 (50 tur × 5 ajan) oturumda **~4.1× daha az iş** yapar.
  (Bu yeniden-kullanım kazancı yalnızca **self-host** içindir ve okuma-yoğun fleet'lerde
  geçerlidir. ~60× rakamı — yaklaşık 19 saatten yaklaşık 19 dakikaya — yalnızca **naive**
  "her şeyi yeniden gönder" loop'una kıyasla doğrudur; başlık rakamı 4.1× olandır.) Bu,
  doğrudan marja etki eden bir kaldıraçtır.
- **Ödeme rayı yok.** fak **Apache-2.0**, ücretsiz, self-host'tur — kredi kartı yok, sınır
  ötesi fatura yok, tüzel kişilik yok. `git clone` ve `go install` tüm yoldur.

## fak neyi çözer

- **Uzun oturumlar giderek pahalılaşmaz.** provider'ın prompt-cache indirimi yalnızca
  cached prefix bayt-bayt aynı kaldığında korunur; fak, aradaki eski turları atarken bile
  prefix'i **bayt-bayt aynı** tutar. fak prefix'in bayt-bayt aynılığını **garanti eder**;
  provider'ın cache'i gerçekten yeniden kullanıp kullanmaması provider'ın kararıdır — fak
  bunu iddia etmez, provider'a aktarır (relay).
- **Default-deny güvenlik (yapısal, sınıflandırıcı değil).** İzin politikası kernel'in
  *içinde*, aynı call path üzerinde çalışır — **fail-closed**. Allow-list'te olmayan bir
  eylem, model ne kadar ikna edilirse edilsin çalışamaz.
- **Sonuç karantinası (result quarantine).** Şüpheli tool *sonuçları* model context'ine
  hiç girmeden ayrı bir karantinada tutulur — bir dedektörle değil, yapıyla. Sonuçları
  işaretleyen dedektör tasarımı gereği ~%100 atlatılabilir kabul edilir: bir bonus, asla
  taban değil.
- **Canlı testler:** prompt injection korumasız temel hatta 5/5 ulaştı; fak 5/5 duvarla
  kapattı.

## Ön koşullar

- **Go 1.26+.** `fak/go.mod` `go 1.26` bildirir. Go'nun varsayılan `GOTOOLCHAIN=auto`
  ayarıyla eski bir `go`, ilk build'de doğru araç zincirini otomatik indirir (bir kez ağ
  gerekir); `go version` ile kontrol edin.
- **Tier 0 için hepsi bu:** GPU yok, API key yok, ağ yok.

## Katmanlar (Tier'lar)

Yükselen kurulum maliyeti sırasına göre — **aralarında yeni bir şey kurulmaz**:

| Tier | Ne elde edersiniz | Kurulum |
|---|---|---|
| **0 — Çekirdeği deneyin** | Adjudication sınırını çevrimdışı çalıştırın/ölçün | `go build` |
| **1 — Gerçek bir model'in önüne geçin** | Kernel'i, başka yerde sunduğunuz bir model'in önüne koyun (Ollama / vLLM / llama.cpp / bulut) | + çalışan bir OpenAI-compatible sunucu |
| **1b — Tek komutla local model** | Mevcut ajanınızla in-kernel bir local GGUF model — key yok, ağ yok, ikinci terminal yok | `fak guard --gguf qwen2.5:7b -- claude` |
| **2 — Kaynaşık in-kernel model** | Kernel'in sahip olduğu saf-Go forward pass | + (gerçek weight) Python export |

Yalnızca **önünde fak olan işe yarar bir model sunmak** istiyorsanız, istediğiniz
**Tier 1**'dir.

## Model'inizle

fak model'inizi değiştirmez — onu govern eder ve cache'ler. **Qwen2/Qwen3 ve GLM-MoE**,
in-kernel referans motorunda bit-bazında (bit-exact) kanıtlanmıştır; geri kalan her şey
(DeepSeek, Mistral, herhangi bir open-weights model) OpenAI-compatible wire üzerinden öne
geçer — Ollama / vLLM / SGLang / llama.cpp / LM Studio veya herhangi bir OpenAI-compatible
API.

## Binary'yi edinin

**Katkıda bulunan (clone'dan build):**

```bash
go build -o fak ./cmd/fak
```

**Go ile kurulum** (modül yolu `github.com/anthony-chaudhary/fak` repo köküdür, doğrudan
kurulur):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Windows notu.** `go build -o fak.exe ./cmd/fak` ile derleyin; binary'yi `.\fak.exe`
> olarak çağırın.

## 60 saniyelik kanıt (key yok, model yok, GPU yok)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Sonraki adım

- [README (tam genel bakış)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 dakikada local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — binary'yi kurun (İngilizce tam rehber)](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — her rakamın kaynağı](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — neyin shipped/simulated/stub olduğu](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Veri yerleşimi ve uyum — KVKK için](../../explainers/data-residency-and-compliance.md)
- [Integrations — ajanınızı bağlayın](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

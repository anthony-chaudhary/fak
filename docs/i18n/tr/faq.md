---
title: "fak — İlk temas için SSS (Türkçe giriş / Turkish introduction)"
description: "fak'ın Türkçe giriş sayfası: her tool call'u çalışmadan önce inceleyen tek bir Go binary'si — aynı ajan loop'u daha güvenli, daha ucuz, daha hızlı; KVKK'ya uygun self-host."
---

# fak — İlk temas için SSS (Türkçe giriş)

> Bu bir **yerelleştirilmiş giriş noktasıdır (entry point)**, dokümantasyonun tam
> çevirisi değil. Tam dokümantasyon İngilizcedir — bu sayfa fak'ın özünü, en sık
> sorulan soruları ve kurulum yolunu verip sizi
> [İngilizce dokümana](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> yönlendirir.
> **Not:** Bu çeviri makine tarafından üretilmiştir ve ana dili konuşan biri
> tarafından gözden geçirilmeyi beklemektedir — hata görürseniz issue/PR açın.
>
> **Diğer diller:** [i18n hub](../README.md).

## Sık sorulan sorular

### S1. fak nedir?

Zaten kullandığınız AI ajanının (Claude Code, Codex, Cursor; herhangi bir
OpenAI/Anthropic/MCP istemcisi) önüne, tek bir base URL'i yeniden yönlendirerek
koyduğunuz tek bir statik Go binary'si — kodu yeniden yazmadan. Uzun oturumları
ucuzlatır (eski turn'leri atarken provider'ın prompt-cache prefix'ini bayt-bayt
aynı tutar), her tool call'u çalışmadan önce inceler, yerel modelleri çekirdek-içi
referans motorunda çalıştırabilir ve her çağrı için denetlenebilir bir karar
(verdict) kaydeder.

### S2. Model değiştirmem ya da ajanımı yeniden yazmam gerekir mi?

Hayır. fak zaten kullandığınız modeli yönetir (govern) ve önbelleğe alır (cache).
Şununla sararsınız:

```bash
fak manage claude
```

veya tek bir base URL'i `fak serve`'e yönlendirirsiniz.

### S3. Verilerim nereye gider — bu uyumlu mu?

Önce self-host: yerel bir modelin ya da yurt içi bir provider'ın önünde tek bir
statik binary; her backend'de fail-closed veri ikametgahı (residency), varsayılan
olarak reddeden (default-deny) bir yetki tabanı ve her tool call için değiştirilmeye
karşı korumalı (tamper-evident) bir denetim kaydı. Verileriniz makinenizden çıkmaz.
Bu, **KVKK** (6698 sayılı Kişisel Verilerin Korunması Kanunu) kapsamında verileri
yurt içinde ve kendi kontrolünüzde tutmayı doğrudan destekler.

### S4. Maliyeti nedir? Gerçekten ücretsiz mi?

**Apache-2.0**, ücretsiz, self-host. Kredi kartı yok, sınır ötesi fatura yok, tüzel
kişilik yok. Lisans için tek bir TL bile yurt dışına çıkmaz; kur farkı ya da sınır
ötesi ödeme yok. `git clone` ve `go install` tüm yoldur.

### S5. Ne kadar daha ucuz ya da hızlı?

Ölçülmüş bir 50-turn x 5-ajanlık oturumda, **iyi ayarlanmış (tuned) bir warm-cache
yığınına göre yaklaşık 4.1x daha az iş**. ~60x rakamı (yaklaşık 19 saatten yaklaşık
19 dakikaya) YALNIZCA her şeyi yeniden gönderen naif (naive) loop'a karşı geçerlidir
— asla ana başlık olarak kullanılmaz. Yeniden kullanım kazancı yalnızca self-host
içindir ve okuma ağırlıklı (read-heavy) filolar için geçerlidir. Bu tasarruf
doğrudan TL cinsinden token faturanıza yansır.

### S6. Hangi modeller çalışır?

**Qwen2/Qwen3 ve GLM-MoE**, çekirdek-içi (in-kernel) referans motorunda bit-bazında
aynı (bit-exact) olduğu kanıtlanmıştır. Diğer her şey (DeepSeek, Mistral, herhangi
bir open-weights model) OpenAI-uyumlu wire üzerinden önden bağlanır:
Ollama / vLLM / SGLang / llama.cpp / LM Studio.

### S7. Prompt injection'ı nasıl durdurur?

Bir sınıflandırıcı (classifier) ile değil, iki yapısal geçit ile: varsayılan olarak
reddeden bir yetki tabanı (tehlikeli bir araç allow-list'te asla yer almaz) ve sonuç
karantinası (zehirlenmiş tool sonuçları model bağlamına hiç ulaşmaz). Canlı
testlerde, injection korumasız temel hattı (baseline) 5/5 aştı; fak ise 5/5 duvarla
engelledi.

### S8. Nasıl kurarım?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

veya bir clone içinden `go build -o fak ./cmd/fak`. Modül proxy'sine erişim sorun
oluyorsa `GOPROXY`'yi yerel bir aynaya ayarlayabilirsiniz.

## 60 saniyelik kanıt (key yok, model yok, GPU yok)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## S9. Sırada ne var?

- [README (tam genel bakış)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 dakikada yerel model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary'yi kurun](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — her rakamın kaynağı](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — neyin shipped/simulated/stub olduğu](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Veri ikametgahı ve uyumluluk — KVKK için](../../explainers/data-residency-and-compliance.md)
- [Integrations — ajanınızı bağlayın](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

---
title: "fak — kaynaşık ajan çekirdeği (Türkçe giriş / Turkish introduction)"
description: "fak'ın Türkçe giriş sayfası: her tool call'u çalışmadan önce denetleyen tek bir Go binary'si — aynı ajan döngüsü daha güvenli, daha ucuz, daha hızlı; KVKK uyumlu self-host."
---

# fak — kaynaşık ajan çekirdeği (Türkçe giriş)

> Bu bir **yerelleştirilmiş giriş sayfasıdır (entry point)**, tüm dokümantasyonun
> çevirisi değil. Tam dokümantasyon İngilizcedir — bu sayfa size fak'ın özünü,
> 60 saniyelik kanıtı ve kurulum yolunu verip sizi
> [İngilizce dokümana](https://github.com/anthony-chaudhary/fak/blob/main/README.md) yönlendirir.
> **Not:** Bu çeviri makine tarafından üretilmiştir ve ana dil konuşuru
> incelemesi beklemektedir — hata görürseniz issue/PR açın.
>
> **Diğer diller:** tüm liste [i18n hub](../README.md) üzerinde.

## Tek satırda fak

**fak, bir Go binary'sidir** — AI ajanınız ile onun çağırdığı araçlar (tool)
arasında durur; her tool call'u *çalışmadan önce* denetler ve uzun bir oturumda
tekrar eden paylaşılan işi yeniden kullanır. Sonuç: aynı ajan döngüsü **daha
güvenli, daha ucuz ve daha hızlı** olur — başka hiçbir şeyi değiştirmeden.

Ajanınızı yeniden yazmazsınız — tek bir base URL'i `fak serve`'e yöneltirsiniz,
ve her tool call önce capability floor'dan geçer.

```bash
fak manage claude      # mevcut ajanınızı tek bir komutla sarar
```

## Türkiye'deki ekipler için neden önemli

- **Maliyet TL ile can yakar, token faturası döviz ile gelir.** fak uzun
  oturumlarda paylaşılan işi (system prompt, tool list'in KV cache'i) yeniden
  kullanır — tuned bir warm-cache yığınına kıyasla 50×5'lik bir oturumda
  **~4.1× daha az iş** (naive re-send döngüsüne kıyasla ~60×, ancak dürüst
  rakam 4.1× olanıdır). Bu, doğrudan marja etki eden bir kaldıraçtır: kur
  dalgalanmasına açık bir döviz kalemini küçültür.
- **Veri yurt içinde kalsın (KVKK).** fak önce self-host'tur: bir **local
  model** veya yerli bir provider'ın önünde duran tek bir static binary; her
  backend için fail-closed veri yerleşimi (residency) ve default-deny
  capability floor sunar. Veri makinenizden dışarı çıkmaz.
- **Sınır ötesi ödeme yok.** fak **Apache-2.0**, ücretsiz ve self-host'tur — ne
  kart, ne sınır ötesi fatura, ne de tüzel kişilik gerekir. `git clone` ve
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` tüm yoldur.
- **Tek static binary, sıfır harici bağımlılık.** Küçük ekipler için kolay ops —
  ne sidecar ne de ayrı bir authorizer. Laptop'tan fleet'e aynı artifact;
  bileşen değil yalnızca flag eklersiniz.

## fak hangi sorunları çözer

- **Uzun oturumlar ucuz kalır.** Provider'ın prompt-cache indirimi ancak cached
  prefix bayt-bayt aynı kaldığında geçerlidir; fak ortadaki eski turn'leri
  atarken bile prefix'i byte-identical tutar. fak prefix'in bayt-eşliğini
  **garanti eder**; provider'ın cache'i gerçekten yeniden kullanıp kullanmayacağı
  provider'ın kararıdır — fak bunu iddia etmez, yalnızca aktarır. (Bu yeniden
  kullanım kazancı yalnızca self-host içindir ve okuma-yoğun fleet'lere uygulanır.)
- **Default-deny güvenlik.** İzin politikası kernel'in *içinde*, aynı call
  path üzerinde çalışır — **fail-closed**. Allow-list'te hiç yer almayan bir
  eylem, model ne kadar kandırılırsa kandırılsın çalışamaz. Bu, bir saldırıyı
  "yakalamaya" bağlı değildir; yapısaldır.
- **Prompt injection / zehirlenmiş sonuç karantinası.** Şüpheli tool
  *sonuçları*, model context'ine hiç girmeyecek şekilde ayrı bir karantinada
  tutulur — bir classifier ile değil, yapı ile. Onları işaretleyen dedektör
  tasarım gereği ~%100 atlatılabilir sayılır: bir bonustur, asla zemin değil.
  Canlı testlerde prompt injection korumasız temele 5/5 ulaştı; fak 5/5 duvarla
  kapattı.

## 60 saniyelik kanıt (key yok, model yok, GPU yok)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection engellendi, görev yine de tamamlanır
```

## Kendi modelinizle

fak modelinizi değiştirmez — onu govern eder ve cache'ler. **Qwen2/Qwen3 ve
GLM-MoE**, in-kernel referans motorunda bit-exact olarak kanıtlanmıştır; geri
kalan her şey (DeepSeek, Mistral, herhangi bir open-weights model) OpenAI-uyumlu
wire üzerinden öne geçer — Ollama / vLLM / SGLang / llama.cpp / LM Studio ya da
herhangi bir OpenAI-uyumlu API aracılığıyla.

## Sonraki adım

- [Hızlı başlangıç (Türkçe) — 10 dakikada yerel model](./quickstart.md)
- [Kurulum (Türkçe) — binary'yi kurun](./install.md)
- [SSS (Türkçe) — sık sorulan sorular](./faq.md)
- [README (tam genel bakış)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 dakikada local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — binary'yi kurun](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — ajanınızı bağlayın](../../integrations/README.md)
- [Veri yerleşimi ve uyum — KVKK için](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — her rakamın kaynağı](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — neyin shipped/simulated/stub olduğu](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

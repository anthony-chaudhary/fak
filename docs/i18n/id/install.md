---
title: "fak — instalasi & mulai cepat (pengantar Bahasa Indonesia / Indonesian getting started)"
description: "Halaman pintu masuk instalasi fak: satu binary Go yang meninjau setiap tool call sebelum berjalan dan memakai ulang kerja bersama di sesi panjang — loop agen yang sama jadi lebih aman, murah, dan cepat. Self-host, ramah UU PDP 2022, Apache-2.0."
---

# fak — instalasi & mulai cepat (pengantar Bahasa Indonesia)

> Ini adalah **halaman pintu masuk (entry point) terlokalkan**, bukan terjemahan
> lengkap. Dokumentasi lengkap tersedia dalam bahasa Inggris — halaman ini hanya
> mengantar Anda memasang dan menjalankan fak, memetakan tier, lalu meneruskan ke
> [dokumentasi Inggris](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Catatan:** terjemahan ini dihasilkan mesin dan masih menunggu tinjauan penutur
> asli — silakan buka issue/PR untuk perbaikan.
> Pintu masuk bahasa lain ada di [i18n hub](../README.md).

Ini adalah pintu masuk **pasang-dan-jalan**. Pitch padatnya ada di
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md); halaman ini
membawa Anda dari checkout bersih ke kernel yang berjalan.

**fak adalah satu binary Go** — satu artefak statis tanpa dependensi eksternal — yang
duduk di antara agen AI Anda dan tool yang dipanggilnya: meninjau setiap tool call
*sebelum* berjalan, dan memakai ulang kerja bersama yang berulang di sesi panjang.
Hasilnya: loop agen yang sama menjadi **lebih aman, lebih murah, dan lebih cepat**,
tanpa menulis ulang apa pun. Anda tidak menulis ulang agen — cukup arahkan satu base
URL ke `fak serve`, atau bungkus agen yang sudah ada dengan satu perintah:

```bash
fak manage claude
```

## Mengapa ini penting untuk Indonesia

- **Tagihan token dalam dolar, margin Anda dalam rupiah (IDR).** fak memakai ulang
  kerja bersama (KV cache dari system prompt + daftar tool) di sesi panjang — **~4.1×
  lebih sedikit kerja** dibanding tumpukan warm-cache yang sudah disetel, diukur pada
  sesi **50-turn × 5-agen**. (Angka ~60× — sekitar 19 jam menjadi sekitar 19 menit —
  hanya benar dibanding loop *naive* yang mengirim ulang semuanya; angka jujur untuk
  dijadikan headline tetap 4.1×.) Kemenangan pemakaian ulang ini **khusus self-host**
  dan berlaku untuk armada baca-berat (read-heavy). Ini tuas langsung ke margin IDR.
- **Data tetap di dalam negeri (UU PDP 2022).** fak mengutamakan self-host: satu binary
  statis di depan **model lokal** atau provider domestik, dengan capability floor
  default-deny yang fail-closed dan audit log setiap tool call. Data tidak keluar dari
  mesin Anda.
- **Diskon prompt-cache provider bertahan.** fak menjaga cached prefix **identik
  byte-per-byte** sambil membuang turn tengah yang lama. fak **menjamin** prefix
  byte-identik; apakah provider benar-benar memakai ulang cache tersebut adalah
  keputusan provider — fak meneruskannya, bukan mengklaimnya.
- **Tanpa gesekan pembayaran lintas batas.** fak **Apache-2.0**, gratis, self-host —
  tanpa kartu kredit, tanpa invoice lintas negara, tanpa badan hukum. `git clone` plus
  `go install` adalah seluruh jalurnya.

## Keamanan: struktural, bukan classifier

- **Capability floor default-deny**, diperiksa di *dalam* kernel pada jalur panggilan
  yang sama — **fail-closed**. Aksi yang tidak pernah ada di allow-list tidak bisa
  berjalan, sekuat apa pun model dibujuk.
- **Karantina hasil.** Tool *result* yang mencurigakan ditahan sepenuhnya dari konteks
  model — ini struktur, bukan detektor. Detektor yang menandainya dianggap ~100% dapat
  dielakkan by design: bonus, bukan lantai.
- **Uji langsung:** prompt injection menembus baseline tanpa proteksi 5/5; fak
  menemboknya 5/5.

## Prasyarat

- **Go 1.26+.** `go.mod` mendeklarasikan `go 1.26`; dengan `GOTOOLCHAIN=auto` (default),
  toolchain yang tepat terunduh otomatis dari `go.mod` pada build pertama (perlu jaringan
  sekali). Selain itu pasang Go 1.26 dari <https://go.dev/dl/>. Cek dengan `go version`.
- **Tier 0 tidak butuh apa-apa lagi:** tanpa GPU, tanpa API key, tanpa jaringan.

## Peta tier

| Tier | Yang Anda dapat | Setup |
|---|---|---|
| **0 — Coba kernel** | Jalankan/ukur batas adjudikasi secara offline | `go build` |
| **1 — Fronting model nyata** | Taruh kernel di depan model yang Anda layani sendiri (Ollama / vLLM / llama.cpp / provider cloud) | + server OpenAI-compatible |
| **1b — Model lokal satu perintah** | Model GGUF lokal in-kernel bersama agen Anda — tanpa key, tanpa jaringan | `fak guard --gguf qwen2.5:7b -- claude` |
| **2 — Model fused in-kernel** | Forward pass murni-Go yang dimiliki kernel | + (bobot nyata) ekspor Python |

Kalau Anda hanya ingin **melayani model yang berguna dengan fak di depannya**, pilih
**Tier 1**. Model in-kernel Tier 2 adalah *reference forward pass* yang terbukti
bit-exact, bukan mesin serving kualitas-chat.

## Model Anda

fak **mengatur dan meng-cache** model Anda; ia tidak menggantinya. **Qwen2/Qwen3 dan
GLM-MoE** terbukti bit-exact di reference engine in-kernel. Selebihnya (DeepSeek,
Mistral, model open-weights apa pun) di-front lewat wire OpenAI-compatible: Ollama /
vLLM / SGLang / llama.cpp / LM Studio atau API OpenAI-compatible mana pun.

## Pasang binary-nya

```bash
go build -o fak ./cmd/fak
```

Atau pasang langsung dengan Go (module path adalah root repository, jadi resolve
langsung):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Catatan Windows.** Build dengan `go build -o fak.exe ./cmd/fak`, lalu ketik binary
> sebagai `.\fak.exe` di tiap tempat panduan ini menulis `./fak`.

## Bukti 60 detik (tanpa key, tanpa model, tanpa GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Ke mana selanjutnya

**Halaman Bahasa Indonesia lain:** [Mulai cepat](./quickstart.md) · [FAQ](./faq.md) · [Indeks Bahasa Indonesia](./README.md)

- [README — gambaran lengkap](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — model lokal dalam 10 menit](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — referensi instalasi & tutorial berpandu](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — sambungkan agen Anda](../../integrations/README.md)
- [Data residency & kepatuhan — untuk UU PDP 2022](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — sumber tiap angka](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — apa yang shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

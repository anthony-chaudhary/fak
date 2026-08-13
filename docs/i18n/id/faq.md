---
title: "fak — FAQ Kontak Pertama (pengantar Bahasa Indonesia / Indonesian introduction)"
description: "Halaman masuk Bahasa Indonesia untuk fak: satu Go binary yang duduk di antara agen AI dan tool yang dipanggilnya — memeriksa setiap tool call sebelum dijalankan dan memakai ulang kerja berulang. Loop agen yang sama jadi lebih aman, lebih murah, lebih cepat; self-host, ramah UU PDP 2022, Apache-2.0."
---

# fak — FAQ Kontak Pertama (Bahasa Indonesia)

> Ini adalah **halaman masuk (entry point) yang dilokalkan**, bukan terjemahan lengkap.
> Dokumentasi lengkap tersedia dalam bahasa Inggris — halaman ini memberi Anda inti fak,
> bukti 60 detik, dan jalur pemasangan, lalu mengarahkan Anda ke
> [dokumentasi Inggris](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Catatan:** terjemahan ini dibuat oleh mesin dan masih menunggu tinjauan penutur asli —
> jika menemukan kesalahan, silakan buka issue/PR.
> **Bahasa lain:** lihat [i18n hub](../README.md).

## Q1. Apa itu fak?

Satu Go binary statis yang Anda tempatkan **di depan** agen AI yang sudah Anda jalankan
(Claude Code, Codex, Cursor, klien OpenAI/Anthropic/MCP mana pun) dengan mengarahkan ulang
satu base URL — tanpa menulis ulang apa pun. fak membuat sesi panjang lebih murah (membuang
turn lama sambil menjaga prefix prompt-cache provider tetap **byte-identical**), mengarahkan
setiap tool call, dapat menjalankan model GGUF lokal secara in-process, dan mencatat sebuah
verdict yang dapat diaudit untuk setiap panggilan.

## Q2. Apakah saya harus ganti model atau menulis ulang agen saya?

Tidak. fak **govern** dan **cache** model yang sudah Anda pakai — bukan menggantinya.
Bungkus agen Anda dengan satu perintah, atau arahkan ulang satu base URL ke `fak serve`:

```bash
fak manage claude
```

## Q3. Ke mana data saya pergi — apakah ini patuh?

Self-host lebih dulu: satu binary statis di depan sebuah model lokal atau provider dalam
negeri, dengan residency yang **fail-closed**, sebuah **default-deny capability floor**, dan
audit log yang **tamper-evident** untuk setiap tool call. Data Anda tidak keluar dari mesin
Anda. Ini memudahkan pemenuhan kewajiban perlindungan data domestik di bawah
**UU PDP (UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi)**: pemrosesan tetap berada
di lingkungan yang Anda kendalikan, dan setiap akses tool meninggalkan jejak yang dapat
diperiksa.

## Q4. Berapa biayanya? Apakah benar-benar gratis?

**Apache-2.0, gratis, self-host.** Tidak perlu kartu kredit, tidak ada faktur lintas-batas
dalam dolar, dan tidak perlu badan hukum. Anda tidak menandatangani kontrak berlangganan
yang ditagih dalam mata uang asing — seluruh jalurnya hanyalah `git clone` ditambah
`go install`.

## Q5. Seberapa lebih murah atau lebih cepat?

Pada sesi terukur **50 turn x 5 agen**, sekitar **4,1x lebih sedikit kerja** dibandingkan
sebuah tumpukan warm-cache yang sudah **di-tuning**. Angka ~60x (sekitar 19 jam menjadi
sekitar 19 menit) **hanya** berlaku dibandingkan loop naif yang mengirim ulang semuanya —
jangan pernah dijadikan angka utama. Keuntungan pemakaian-ulang ini **khusus self-host**,
untuk fleet yang banyak membaca (read-heavy).

Kenapa ini penting dalam Rupiah: tagihan token dihitung dalam dolar sementara pendapatan
Anda dalam Rupiah, sehingga setiap kerja yang tidak dikirim ulang langsung menjadi tuas
margin — bukan diskon yang menunggu negosiasi vendor.

## Q6. Model mana yang bisa dipakai?

**Qwen2/Qwen3 dan GLM-MoE** telah terbukti bit-exact di reference engine in-kernel. Semua
lainnya (DeepSeek, Mistral, model open-weights mana pun) masuk melalui wire yang kompatibel
dengan OpenAI: Ollama / vLLM / SGLang / llama.cpp / LM Studio.

## Q7. Bagaimana fak menghentikan prompt injection?

Dua gerbang **struktural**, bukan sebuah classifier:

- **Default-deny capability floor** — sebuah tool berbahaya tidak pernah ada di allow-list,
  jadi ia tidak bisa berjalan betapapun model dibujuk. Pemeriksaan terjadi di dalam kernel
  pada jalur panggilan yang sama, **fail-closed**.
- **Result quarantine** — tool *result* yang mencurigakan ditahan sepenuhnya di luar konteks
  model (struktur, bukan detektor). Detektor yang menandainya dianggap ~100% bisa dielakkan
  secara desain — sebuah bonus, bukan lantai keamanan.

Dalam pengujian langsung, injeksi menembus baseline yang tidak terlindungi **5/5**, dan fak
menembloknya **5/5**.

## Q8. Bagaimana cara memasangnya?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Atau build dari hasil clone: `go build -o fak ./cmd/fak`. Jika `proxy.golang.org` sulit
dijangkau dari jaringan Anda, setel sebuah module proxy lewat `GOPROXY` terlebih dahulu.

### Bukti 60 detik (tanpa key, tanpa model, tanpa GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. Ke mana selanjutnya?

- [README (gambaran lengkap)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — model lokal dalam 10 menit](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — pasang binary-nya](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — sumber setiap angka](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — apa yang shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency & kepatuhan — untuk UU PDP 2022](../../explainers/data-residency-and-compliance.md)
- [Integrations — hubungkan agen Anda](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).

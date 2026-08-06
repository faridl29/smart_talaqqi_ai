# Smart Talaqqi AI Server (Tencent VPS Ready)

Server backend **Python WebSocket Real-Time ASR & Makhraj AI** yang didesain khusus untuk berjalan secara efisien di VPS Tencent (2 vCPU / 4 GB RAM / 30 Mbps).

---

## 🚀 Fitur Utama
1. **Real-Time WebSocket Audio & Transcript Streaming**: Mengembalikan status kata & makhraj secara instan dalam hitungan milidetik.
2. **Diagnosis Makhraj Spesifik**: Mendeteksi kesalahan artikulasi organ (Wasathul Halq, Aqshal Halq, Isti'la, dll.) seperti `ح` vs `ه`, `ع` vs `ء`, `ط` vs `ت`, `ق` vs `ك`.
3. **Bebas Auto-Correct**: Memeriksa akurasi fonetis secara murni tanpa *Language Model auto-correct*.

---

## 🛠️ Cara Menjalankan Secara Lokal

```bash
# 1. Masuk ke direktori server
cd /Users/miftahfaridlal-anshari/Projects/ai/money_management/servers/talaqqi_ai_server

# 2. Buat virtual environment & install dependency
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Jalankan server FastAPI
python main.py
# atau
uvicorn main:app --reload --port 8000
```

Server akan berjalan di `http://localhost:8000` dan WebSocket di `ws://localhost:8000/ws/talaqqi/stream`.

---

## ☁️ Cara Deploy 1-Click di VPS Tencent (2 vCPU / 4GB RAM)

### Opsi A: Menggunakan Docker Compose (Direkomendasikan)
1. Clone / upload folder `talaqqi_ai_server` ke VPS Tencent Anda.
2. Jalankan perintah:
   ```bash
   docker compose up -d --build
   ```
3. Cek status:
   ```bash
   docker compose logs -f
   ```

### Opsi B: Menggunakan Systemd / PM2 di VPS
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```
# smart_talaqqi_ai

---

## 🔐 Keamanan Server

### 1. API Key Authentication (wajib untuk production)

Semua endpoint (HTTP + WebSocket) butuh header `X-API-Key`. Tanpa key valid → HTTP `401`, WebSocket ditutup `4401`.

**Setup:**
1. Generate key:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Server — tambah di `.env` (folder server):
   ```env
   API_KEYS=<key_yang_sama>
   ```
   Bisa lebih dari satu key, pisahkan koma: `API_KEYS=key1,key2`
3. App Flutter — tambah di `apps/smart_talaqqi/.env`:
   ```env
   API_KEY=<key_yang_sama>
   ```
4. Restart server (pastikan `compose.yaml` punya `env_file: - .env`):
   ```bash
   docker compose up -d --build
   ```

> [!NOTE]
> `API_KEYS` kosong = auth nonaktif (mode dev). Health check `/` tetap publik untuk monitoring.

**Verifikasi:**
```bash
# Tanpa key → harus 401
curl -X POST http://localhost:8000/api/v1/talaqqi/evaluate \
  -H "Content-Type: application/json" -d '{}'

# Dengan key → bukan 401
curl -X POST http://localhost:8000/api/v1/talaqqi/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <key>" \
  -d '{"target_ayah_text":"x","recognized_speech_text":"x"}'
```

### 2. Firewall UFW (blokir port 8000 dari internet)

Port API tidak perlu terbuka ke publik — hanya akses via localhost / reverse proxy.

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp          # SSH
sudo ufw allow 80/tcp          # HTTP (Caddy/Let's Encrypt)
sudo ufw allow 443/tcp         # HTTPS
sudo ufw enable
sudo ufw status verbose
```

Setelah ini, port 8000 **tidak bisa diakses dari internet** — hanya dari dalam VPS.

### 3. HTTPS + Reverse Proxy (Caddy, auto-TLS)

Enkripsi audio user di transit (suara = data biometrik, wajib HTTPS).

```bash
# Install Caddy
sudo apt install -y caddy
```

Buat `/etc/caddy/Caddyfile`:
```
talaqqi.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

```bash
# Point DNS A record ke IP VPS, lalu:
sudo systemctl reload caddy
```

Caddy auto-provision Let's Encrypt TLS. WebSocket (`wss://`) ditangani otomatis.

**Update URL di app `.env`:**
```env
DEFAULT_AI_HTTP_URL=https://talaqqi.example.com/
DEFAULT_AI_WS_URL=wss://talaqqi.example.com/ws/talaqqi/stream
```

> [!IMPORTANT]
> API key di APK bisa diekstrak (rooted device) — ini anti-abuse bar, bukan keamanan absolut. Kombinasi firewall + HTTPS menutup sisanya. Untuk proteksi penuh butuh per-user auth (backend user system).

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

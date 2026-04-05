# Gunshot Triangulation System (GTS) v1.0

## What It Does
Three microphones (your phones) listen for a sound. When all three hear it, the server measures the millisecond-level time differences, solves the TDOA hyperbola intersection, and drops a GPS pin on the map — in under 2 seconds.

- **DEMO mode**: clap your hands to trigger. Identical algorithm to production.
- **REAL mode**: gunshot detection with YAMNet CNN classifier + physical anti-spoofing checks.

---

## 1. Start the Server

```bash
# One command — installs deps, runs tests, starts server
bash run.sh

# Or manually:
cd backend
pip install fastapi uvicorn numpy scipy librosa soundfile
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 2. Open Dashboard
On your laptop, open: **http://localhost:8000/**

---

## 3. Connect Phones
All phones must be on the **same WiFi** as the laptop.

Find your laptop IP (shown in terminal), then open these URLs on each phone's browser:
- Phone A: `http://[YOUR_IP]:8000/phone?mic=A&triad=triad_1`
- Phone B: `http://[YOUR_IP]:8000/phone?mic=B&triad=triad_1`
- Phone C: `http://[YOUR_IP]:8000/phone?mic=C&triad=triad_1`

Tap **CONNECT MIC** on each phone.

---

## 4. Place Mics on Map
1. Measure the **physical distances** between your phones in centimetres/metres.
2. In the dashboard, click **📍 Place Mic**, then click the map where each phone is.
3. Enter the local X,Y coordinates (in metres) matching your physical setup.

---

## 5. Demo: Run a Detection
- **DEMO mode**: Clap sharply once. All 3 mics trigger → pin drops on map.
- **Simulation**: Click **⬡ SIMULATE SHOT** — runs the full math without needing phones.

---

## Architecture

```
Phone A (browser mic)  ──┐
Phone B (browser mic)  ──┼──► WebSocket PCM stream ──► FastAPI backend
Phone C (browser mic)  ──┘         │
                                   ▼
                         [VAD: peak > threshold?]
                                   │
                         [YAMNet CNN Classifier]
                         DEMO: Clapping? / REAL: Gunshot?
                                   │
                         [TDOA Solver — Levenberg-Marquardt]
                         Residuals: r_i - r_ref - Δd_i = 0
                                   │
                         [local (x,y) → GPS (lat,lng)]
                                   │
                         [WebSocket broadcast to dashboard]
                                   │
                         [Leaflet.js pin drop on map]
```

---

## TDOA Math

```
Sound speed:   c = 343 m/s

Time difference:   Δt_AB = t_B - t_A  (milliseconds)
Distance difference:   Δd_AB = Δt_AB × c / 1000  (metres)

Hyperbola equation:
  dist(P, A) - dist(P, B) = Δd_AB

Residual (minimised via Levenberg-Marquardt):
  f_AB(x,y) = sqrt((x-xA)²+(y-yA)²) - sqrt((x-xB)²+(y-yB)²) - Δd_AB

Position error bound (Cramér-Rao):
  σ_pos ≈ c × σ_t = 343 × 0.001 = 0.343 m  (at 1ms clock precision)

GPS conversion:
  Δlat = y_metres / 111320
  Δlng = x_metres / (111320 × cos(lat_origin))
```

---

## Phone Client Protocol

The phone sends **binary frames** over WebSocket:
```
Bytes [0-7]   Float64:  client timestamp (ms since epoch)
Bytes [8-11]  Uint32:   sample rate (Hz)
Bytes [12+]   Float32[]: PCM audio samples (mono)
```
No audio leaves the local network. Only timestamps travel to the TDOA solver.

---

## Files
```
gunshot-system/
├── backend/
│   ├── main.py           # FastAPI app, WebSocket handlers, pipeline
│   ├── tdoa_solver.py    # TDOA math (Levenberg-Marquardt nonlinear solver)
│   ├── classifier.py     # YAMNet + fallback audio classifier
│   ├── session_manager.py# Triad/mic state management
│   └── models.py         # Pydantic data models
├── frontend/
│   └── dashboard.html    # Admin dashboard (Leaflet map, live logs)
├── phone/
│   └── index.html        # Phone mic client (Web Audio API)
├── tests/
│   └── test_all.py       # 56 test cases (run: pytest tests/test_all.py -v)
├── requirements.txt
└── run.sh                # One-command startup
```

---

## Adding More Triads
In the dashboard left panel → **+ ADD TRIAD** → enter ID and name → place mics on map.
Each triad is independent — multiple triads can run simultaneously.

---

## Anti-Spoofing (REAL mode)
| Attack | Counter |
|--------|---------|
| Speaker playing audio | Rise time check < 100ms; speakers clip shockwave |
| Thunder | Duration check < 500ms; thunder rolls 2–5s |
| Voice/speech | YAMNet rejects; spectral centroid too low |
| Vehicle backfire | Frequency profile check; energy 800–4kHz band |

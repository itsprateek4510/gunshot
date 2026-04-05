"""
Gunshot Triangulation System — FastAPI Backend
================================================
Serves:
  GET  /              → Dashboard HTML
  GET  /phone         → Phone mic client HTML
  WS   /ws/dashboard  → Admin dashboard live updates
  WS   /ws/mic/{mic_id}/{triad_id} → Phone mic audio stream
  GET  /api/status    → JSON system status
  GET  /api/triads    → JSON triad configs
  POST /api/triad     → Add a triad
  POST /api/simulate  → Simulate a detection (no phones needed)

Audio pipeline per mic:
  Phone PCM stream → VAD threshold → Ring buffer
  → On trigger: YAMNet classify → if target → record timestamp
  → When all 3 mics of triad triggered within 500ms:
    → TDOA solve → lat/lng → broadcast to dashboard
"""

import asyncio
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from classifier import classifier
from models import SystemMode, TriadConfig, MicPosition
from session_manager import SessionManager
from tdoa_solver import solve_tdoa, local_to_latlon

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Gunshot Triangulation System", version="1.0.0")

BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
PHONE_DIR = BASE_DIR / "phone"

# Serve static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Global state ───────────────────────────────────────────────────────────────
session = SessionManager()
current_mode: SystemMode = SystemMode.DEMO
system_active: bool = True
sensitivity_threshold: float = 0.45   # Peak amplitude trigger threshold

# Connected WebSocket clients
dashboard_sockets: List[WebSocket] = []
mic_sockets: Dict[str, WebSocket] = {}   # {mic_id: ws}

SAMPLE_RATE = 16000


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("═" * 60)
    logger.info("  GUNSHOT TRIANGULATION SYSTEM v1.0")
    logger.info("═" * 60)
    local_ip = _get_local_ip()
    logger.info(f"  Server: http://{local_ip}:8000")
    logger.info(f"  Dashboard: http://{local_ip}:8000/")
    logger.info(f"  Phone URL: http://{local_ip}:8000/phone?mic=A&triad=triad_1")
    logger.info("─" * 60)

    # Load YAMNet in background (don't block startup)
    asyncio.create_task(_load_classifier())


async def _load_classifier():
    """Load YAMNet asynchronously so server responds immediately."""
    loop = asyncio.get_event_loop()
    loaded = await loop.run_in_executor(None, classifier.load)
    status = "YAMNet loaded ✓" if loaded else "Using fallback classifier ⚠"
    logger.info(status)
    await broadcast({
        "type": "classifier_status",
        "loaded": loaded,
        "message": status
    })


# ── HTTP routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_dashboard():
    path = FRONTEND_DIR / "dashboard.html"
    if path.exists():
        return FileResponse(str(path))
    return JSONResponse({"error": "dashboard.html not found"}, status_code=404)


@app.get("/phone")
async def serve_phone():
    path = PHONE_DIR / "index.html"
    if path.exists():
        return FileResponse(str(path))
    return JSONResponse({"error": "phone/index.html not found"}, status_code=404)


@app.get("/api/status")
async def api_status():
    return {
        "ok": True,
        "mode": current_mode.value,
        "system_active": system_active,
        "classifier_loaded": classifier.loaded,
        "stats": session.get_stats(),
        "connected_mics": session.get_connected_mics(),
        "server_ip": _get_local_ip(),
        "triads": session.get_triads_dict(),
        "sensitivity": sensitivity_threshold
    }


@app.get("/api/triads")
async def api_triads():
    return session.get_triads_dict()


@app.post("/api/simulate")
async def api_simulate(data: dict):
    """Simulate a detection without phones — for testing."""
    triad_id = data.get("triad_id", "triad_1")
    shooter_x = data.get("x", None)
    shooter_y = data.get("y", None)
    await _simulate_detection(triad_id, shooter_x, shooter_y)
    return {"ok": True}


# ── Dashboard WebSocket ────────────────────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    dashboard_sockets.append(ws)
    logger.info("Dashboard connected")

    # Send full initial state
    await ws.send_json({
        "type": "init",
        "mode": current_mode.value,
        "system_active": system_active,
        "classifier_loaded": classifier.loaded,
        "triads": session.get_triads_dict(),
        "connected_mics": session.get_connected_mics(),
        "stats": session.get_stats(),
        "server_ip": _get_local_ip(),
        "sensitivity": sensitivity_threshold,
        "detections": session.detections[-50:]  # Last 50 detections
    })

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_dashboard_msg(ws, msg)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Dashboard WS error: {e}")
    finally:
        if ws in dashboard_sockets:
            dashboard_sockets.remove(ws)
        logger.info("Dashboard disconnected")


async def _handle_dashboard_msg(ws: WebSocket, msg: dict):
    global current_mode, system_active, sensitivity_threshold
    t = msg.get("type", "")

    if t == "set_mode":
        current_mode = SystemMode(msg["mode"])
        await broadcast({"type": "mode_changed", "mode": current_mode.value})
        logger.info(f"Mode → {current_mode.value}")

    elif t == "toggle_system":
        system_active = msg.get("active", True)
        await broadcast({"type": "system_toggled", "active": system_active})

    elif t == "set_sensitivity":
        sensitivity_threshold = float(msg.get("value", 0.45))
        await broadcast({"type": "sensitivity_changed", "value": sensitivity_threshold})

    elif t == "add_triad":
        triad_data = msg["triad"]
        triad = TriadConfig(**triad_data)
        session.add_triad(triad)
        await broadcast({"type": "triad_added", "triad": session.get_triads_dict()[triad.triad_id]})

    elif t == "remove_triad":
        session.remove_triad(msg["triad_id"])
        await broadcast({"type": "triad_removed", "triad_id": msg["triad_id"]})

    elif t == "update_mic_position":
        session.update_mic_position(
            msg["triad_id"], msg["mic_id"],
            msg["lat"], msg["lng"],
            msg["x"], msg["y"]
        )
        await broadcast({"type": "mic_position_updated", **msg})

    elif t == "simulate":
        await _simulate_detection(
            msg.get("triad_id", "triad_1"),
            msg.get("x"), msg.get("y")
        )

    elif t == "ping":
        await ws.send_json({"type": "pong", "server_time_ms": time.time() * 1000})

    elif t == "clear_log":
        session.detections.clear()
        await broadcast({"type": "log_cleared"})

    elif t == "get_stats":
        await ws.send_json({"type": "stats", **session.get_stats()})


# ── Mic WebSocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws/mic/{mic_id}/{triad_id}")
async def mic_ws(ws: WebSocket, mic_id: str, triad_id: str):
    await ws.accept()
    mic_sockets[mic_id] = ws

    logger.info(f"Mic '{mic_id}' connected (triad: {triad_id})")

    # Register mic state
    mic_state = session.register_mic(mic_id, triad_id, SAMPLE_RATE)

    # Send clock sync
    await ws.send_json({
        "type": "connected",
        "server_time_ms": time.time() * 1000,
        "mic_id": mic_id,
        "triad_id": triad_id
    })

    # Notify dashboards
    await broadcast({
        "type": "mic_connected",
        "mic_id": mic_id,
        "triad_id": triad_id
    })

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                if len(raw) >= 12:
                    # Protocol: [float64: client_timestamp_ms][uint32: sampleRate][float32[]: PCM]
                    client_ts = float(np.frombuffer(raw[:8], dtype=np.float64)[0])
                    sr = int(np.frombuffer(raw[8:12], dtype=np.uint32)[0])
                    pcm = np.frombuffer(raw[12:], dtype=np.float32).copy()

                    if sr != SAMPLE_RATE and sr > 0:
                        # Simple decimation for common rates
                        pcm = _resample(pcm, sr, SAMPLE_RATE)

                    server_ts = time.time() * 1000
                    offset = server_ts - client_ts
                    mic_state.clock_offset_ms = offset * 0.1 + mic_state.clock_offset_ms * 0.9

                    if system_active:
                        await _process_chunk(mic_id, triad_id, pcm, server_ts)

            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "sync_reply":
                        # NTP-style clock sync
                        mic_state.clock_offset_ms = (
                            data["server_time_ms"] - data["client_send_ms"]
                        )
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Mic {mic_id} WS error: {e}")
    finally:
        session.disconnect_mic(mic_id)
        mic_sockets.pop(mic_id, None)
        logger.info(f"Mic '{mic_id}' disconnected")
        await broadcast({"type": "mic_disconnected", "mic_id": mic_id, "triad_id": triad_id})


# ── Audio processing pipeline ──────────────────────────────────────────────────
async def _process_chunk(
    mic_id: str,
    triad_id: str,
    pcm: np.ndarray,
    server_ts_ms: float
):
    """Process one 100ms PCM chunk from a microphone."""
    mic_state = session.get_mic_state(mic_id)
    if not mic_state or len(pcm) == 0:
        return

    # Push into ring buffer
    mic_state.push_samples(pcm)
    mic_state.last_seen_ms = server_ts_ms

    # Compute levels
    peak = float(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    mic_state.peak_level = peak
    mic_state.rms_level = rms

    # Broadcast audio level to dashboard (throttled per chunk — fine at 100ms)
    await broadcast({
        "type": "audio_level",
        "mic_id": mic_id,
        "triad_id": triad_id,
        "peak": round(peak, 4),
        "rms": round(rms, 4)
    })

    # VAD: trigger if peak crosses threshold AND cooldown elapsed
    if peak >= sensitivity_threshold and mic_state.should_trigger(server_ts_ms):
        mic_state.mark_triggered(server_ts_ms)
        audio_window = mic_state.get_window(1.0)
        logger.info(f"Mic {mic_id}: Trigger (peak={peak:.3f})")

        # Classify in background — don't block the audio stream
        asyncio.create_task(
            _classify_and_register(mic_id, triad_id, audio_window, server_ts_ms)
        )


async def _classify_and_register(
    mic_id: str,
    triad_id: str,
    audio: np.ndarray,
    timestamp_ms: float
):
    """Classify audio then register TDOA trigger if target class detected."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, classifier.classify_pcm, audio, SAMPLE_RATE, current_mode.value
    )

    # Broadcast classification to dashboard
    await broadcast({
        "type": "classification",
        "mic_id": mic_id,
        "triad_id": triad_id,
        "result": result,
        "timestamp_ms": timestamp_ms
    })

    if not result["is_target"]:
        logger.info(f"Mic {mic_id}: Rejected — {result['top_class']} (reason: {result['reason']})")
        return

    logger.info(
        f"Mic {mic_id}: TARGET DETECTED — '{result['top_class']}' "
        f"conf={result['confidence']:.1f}%"
    )

    # Record trigger in session
    all_triggers = session.record_trigger(triad_id, mic_id, timestamp_ms, result)

    if all_triggers is None:
        return   # Waiting for other mics

    # All mics triggered — run TDOA
    await _run_tdoa(triad_id, all_triggers)


async def _run_tdoa(triad_id: str, triggers: dict):
    """Run TDOA solver and broadcast detection to dashboard."""
    triad = session.get_triad(triad_id)
    if not triad:
        return

    try:
        timestamps = {mic_id: v["timestamp_ms"] for mic_id, v in triggers.items()}
        mic_positions = {m.mic_id: np.array([m.x, m.y]) for m in triad.mics}

        # TDOA solve
        local_pos, tdoa_details = solve_tdoa(timestamps, mic_positions)

        # Convert to GPS
        origin = triad.mics[0]
        shooter_lat, shooter_lng = local_to_latlon(
            local_pos, origin.lat, origin.lng
        )

        # Aggregate confidence
        confs = [v["classification"]["confidence"] for v in triggers.values()]
        confidence = round(sum(confs) / len(confs), 1)
        top_class = list(triggers.values())[0]["classification"]["top_class"]

        import datetime
        detection = {
            "type": "detection",
            "triad_id": triad_id,
            "triad_name": triad.name,
            "shooter_lat": round(shooter_lat, 8),
            "shooter_lng": round(shooter_lng, 8),
            "shooter_x": round(float(local_pos[0]), 2),
            "shooter_y": round(float(local_pos[1]), 2),
            "confidence": confidence,
            "sound_class": top_class,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": current_mode.value,
            "tdoa_details": tdoa_details,
            "mic_timestamps": {k: round(v["timestamp_ms"], 3) for k, v in triggers.items()}
        }

        logger.info(
            f"DETECTION: triad={triad_id} pos=({local_pos[0]:.1f},{local_pos[1]:.1f})m "
            f"GPS=({shooter_lat:.6f},{shooter_lng:.6f}) conf={confidence}%"
        )

        session.add_detection(detection)
        await broadcast(detection)

    except Exception as e:
        logger.error(f"TDOA failed for triad {triad_id}: {e}")
        await broadcast({
            "type": "tdoa_error",
            "triad_id": triad_id,
            "error": str(e)
        })


# ── Simulation ────────────────────────────────────────────────────────────────
async def _simulate_detection(
    triad_id: str,
    shooter_x: Optional[float] = None,
    shooter_y: Optional[float] = None
):
    """
    Simulate a full detection pipeline without physical phones.
    Uses exact TDOA math — same as production.
    """
    triad = session.get_triad(triad_id)
    if not triad or len(triad.mics) < 3:
        await broadcast({"type": "error", "message": f"Triad '{triad_id}' needs 3 mics"})
        return

    mics = triad.mics

    # Default: random position within the triad bounding box + 20% padding
    if shooter_x is None:
        cx = np.mean([m.x for m in mics])
        cy = np.mean([m.y for m in mics])
        spread = max(
            max(m.x for m in mics) - min(m.x for m in mics),
            max(m.y for m in mics) - min(m.y for m in mics)
        ) * 0.7
        shooter_x = cx + np.random.uniform(-spread, spread)
        shooter_y = cy + np.random.uniform(-spread, spread)

    SPEED = 343.0
    base_t = time.time() * 1000

    # Calculate true arrival times
    triggers = {}
    for mic in mics:
        dist = np.sqrt((shooter_x - mic.x)**2 + (shooter_y - mic.y)**2)
        cls = "Clapping [sim]" if current_mode == SystemMode.DEMO else "Gunshot [sim]"
        triggers[mic.mic_id] = {
            "timestamp_ms": base_t + (dist / SPEED) * 1000,
            "classification": {"confidence": 96.0, "top_class": cls}
        }

    logger.info(
        f"Simulating detection: triad={triad_id} "
        f"shooter=({shooter_x:.1f},{shooter_y:.1f})"
    )

    # Notify dashboard about simulated classification per mic
    for mic_id, data in triggers.items():
        await broadcast({
            "type": "classification",
            "mic_id": mic_id,
            "triad_id": triad_id,
            "result": {
                "is_target": True,
                "confidence": data["classification"]["confidence"],
                "top_class": data["classification"]["top_class"],
                "top5": [{"class": data["classification"]["top_class"],
                          "confidence": data["classification"]["confidence"]}],
                "reason": "simulated",
                "mode": current_mode.value
            },
            "timestamp_ms": data["timestamp_ms"]
        })
        await asyncio.sleep(0.05)  # Brief visual delay

    await _run_tdoa(triad_id, triggers)


# ── Helpers ───────────────────────────────────────────────────────────────────
async def broadcast(message: dict):
    """Send JSON message to all connected dashboard clients."""
    if not dashboard_sockets:
        return
    dead = []
    for ws in dashboard_sockets:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in dashboard_sockets:
            dashboard_sockets.remove(ws)


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple decimation/interpolation resampling."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Crude decimation fallback
        ratio = orig_sr / target_sr
        indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

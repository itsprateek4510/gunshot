"""
Session Manager
===============
Manages mic triads, their positions, connection states, and audio buffers.
Supports multiple simultaneous triads (each with 3 mics).
"""

import time
import numpy as np
from typing import Dict, List, Optional
from models import TriadConfig, MicPosition


class MicState:
    """Runtime state for a single connected microphone."""
    def __init__(self, mic_id: str, triad_id: str, sample_rate: int = 16000):
        self.mic_id = mic_id
        self.triad_id = triad_id
        self.sample_rate = sample_rate
        self.connected = False
        self.last_seen_ms = 0.0

        # Audio ring buffer: 2 seconds of audio
        self.max_samples = sample_rate * 2
        self.samples: List[float] = []

        # Trigger control
        self.last_trigger_ms = 0.0
        self.trigger_cooldown_ms = 1500.0

        # Clock sync
        self.clock_offset_ms = 0.0   # server_time - client_time

        # Display
        self.peak_level = 0.0
        self.rms_level = 0.0

    def push_samples(self, chunk: np.ndarray):
        self.samples.extend(chunk.tolist())
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

    def get_window(self, duration_s: float = 1.0) -> np.ndarray:
        """Get the last `duration_s` seconds of audio."""
        n = int(self.sample_rate * duration_s)
        return np.array(self.samples[-n:], dtype=np.float32)

    def should_trigger(self, server_time_ms: float) -> bool:
        return (server_time_ms - self.last_trigger_ms) > self.trigger_cooldown_ms

    def mark_triggered(self, server_time_ms: float):
        self.last_trigger_ms = server_time_ms


class SessionManager:
    """Manages all triads and mic states."""

    def __init__(self):
        # Triad configurations {triad_id: TriadConfig}
        self.triads: Dict[str, TriadConfig] = {}

        # Runtime mic states {mic_id: MicState}
        self.mic_states: Dict[str, MicState] = {}

        # Pending TDOA triggers per triad {triad_id: {mic_id: timestamp_ms}}
        self.pending_triggers: Dict[str, Dict] = {}

        # Detection history
        self.detections: List[dict] = []

        # Stats
        self.total_triggers = 0
        self.total_detections = 0
        self.start_time = time.time()

        # Create a default triad
        self._create_default_triad()

    def _create_default_triad(self):
        """Default 1-metre equilateral triangle triad for demo."""
        triad = TriadConfig(
            triad_id="triad_1",
            name="Demo Triad",
            mics=[
                MicPosition(mic_id="A", lat=28.6139, lng=77.2090,
                            x=0.0, y=0.0),
                MicPosition(mic_id="B", lat=28.6139, lng=77.2091,
                            x=1.0, y=0.0),
                MicPosition(mic_id="C", lat=28.6140, lng=77.2090,
                            x=0.5, y=0.866),
            ]
        )
        self.triads["triad_1"] = triad

    # ── Triad management ───────────────────────────────────────────────────────

    def add_triad(self, triad: TriadConfig):
        self.triads[triad.triad_id] = triad
        self.pending_triggers[triad.triad_id] = {}

    def remove_triad(self, triad_id: str):
        self.triads.pop(triad_id, None)
        self.pending_triggers.pop(triad_id, None)
        # Disconnect mics belonging to this triad
        to_remove = [k for k, v in self.mic_states.items()
                     if v.triad_id == triad_id]
        for k in to_remove:
            del self.mic_states[k]

    def get_triad(self, triad_id: str) -> Optional[TriadConfig]:
        return self.triads.get(triad_id)

    def update_mic_position(
        self,
        triad_id: str,
        mic_id: str,
        lat: float, lng: float,
        x: float, y: float
    ):
        triad = self.triads.get(triad_id)
        if not triad:
            return
        for mic in triad.mics:
            if mic.mic_id == mic_id:
                mic.lat = lat
                mic.lng = lng
                mic.x = x
                mic.y = y
                return

    def get_triads_dict(self) -> dict:
        result = {}
        for tid, triad in self.triads.items():
            result[tid] = {
                "triad_id": triad.triad_id,
                "name": triad.name,
                "mics": [
                    {"mic_id": m.mic_id,
                     "lat": m.lat, "lng": m.lng,
                     "x": m.x, "y": m.y}
                    for m in triad.mics
                ]
            }
        return result

    # ── Mic state management ───────────────────────────────────────────────────

    def register_mic(self, mic_id: str, triad_id: str, sample_rate: int = 16000):
        state = MicState(mic_id, triad_id, sample_rate)
        state.connected = True
        state.last_seen_ms = time.time() * 1000
        self.mic_states[mic_id] = state
        return state

    def disconnect_mic(self, mic_id: str):
        state = self.mic_states.get(mic_id)
        if state:
            state.connected = False
        # Clear any pending trigger from this mic
        triad_id = state.triad_id if state else None
        if triad_id and triad_id in self.pending_triggers:
            self.pending_triggers[triad_id].pop(mic_id, None)

    def get_mic_state(self, mic_id: str) -> Optional[MicState]:
        return self.mic_states.get(mic_id)

    def get_connected_mics(self) -> List[str]:
        return [k for k, v in self.mic_states.items() if v.connected]

    # ── TDOA trigger management ────────────────────────────────────────────────

    def record_trigger(
        self,
        triad_id: str,
        mic_id: str,
        timestamp_ms: float,
        classification: dict
    ) -> Optional[dict]:
        """
        Record a sound trigger from a mic.
        Returns triggers dict if all mics in triad have triggered, else None.
        """
        if triad_id not in self.pending_triggers:
            self.pending_triggers[triad_id] = {}

        triggers = self.pending_triggers[triad_id]
        triggers[mic_id] = {
            "timestamp_ms": timestamp_ms,
            "classification": classification
        }
        self.total_triggers += 1

        triad = self.triads.get(triad_id)
        if not triad:
            return None

        required = {m.mic_id for m in triad.mics}
        triggered = set(triggers.keys())

        if not required.issubset(triggered):
            return None   # Still waiting

        # Check time window: all triggers within 500ms
        ts_list = [triggers[m]["timestamp_ms"] for m in required]
        time_span = max(ts_list) - min(ts_list)

        if time_span > 500:
            # Too spread out — stale data, reset
            self.pending_triggers[triad_id] = {}
            return None

        # Ready for TDOA
        result = dict(triggers)
        self.pending_triggers[triad_id] = {}
        self.total_detections += 1
        return result

    # ── Detection history ──────────────────────────────────────────────────────

    def add_detection(self, detection: dict):
        self.detections.append(detection)
        if len(self.detections) > 200:   # Keep last 200
            self.detections = self.detections[-200:]

    def get_stats(self) -> dict:
        uptime_s = time.time() - self.start_time
        return {
            "total_triggers": self.total_triggers,
            "total_detections": self.total_detections,
            "uptime_s": round(uptime_s, 1),
            "connected_mics": len(self.get_connected_mics()),
            "total_triads": len(self.triads)
        }

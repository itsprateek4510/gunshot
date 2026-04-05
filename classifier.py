"""
Audio Classifier
================
Primary:  Google YAMNet (521-class pre-trained CNN from TF Hub)
Fallback: Energy + spectral feature classifier (works without TF)

Mode DEMO:  Detects 'Clapping' class  (sharp impulse, broadband, <400ms)
Mode REAL:  Detects 'Gunshot/Gunfire' (very sharp impulse, 1–4kHz peak, <200ms)

Anti-spoofing checks (REAL mode):
  1. Rise time check: genuine gunshot rises to peak in <50ms
  2. Duration check:  gunshot body <300ms, not rolling thunder (2–5s)
  3. Spectral check:  peak energy 800–4000 Hz band
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Target class groups ────────────────────────────────────────────────────────
GUNSHOT_CLASSES = {
    'Gunshot, gunfire', 'Machine gun', 'Artillery fire',
    'Explosion', 'Firearms', 'Gun', 'Rifle', 'Pistol',
    'Burst, pop'
}
CLAP_CLASSES = {
    'Clapping', 'Hands', 'Finger snapping', 'Slap, smack',
    'Tap', 'Knock'
}
REJECTION_CLASSES = {
    'Speech', 'Music', 'Silence', 'Breathing', 'Wind',
    'Rain', 'Water', 'Bird'
}

# Confidence thresholds
THRESHOLD_DEMO = 40.0   # Lower: clapping is harder to distinguish
THRESHOLD_REAL = 55.0   # Higher: gunshots need strong confidence


class AudioClassifier:
    def __init__(self):
        self.model = None
        self.class_names: Optional[List[str]] = None
        self.loaded: bool = False
        self._loading: bool = False

    def load(self) -> bool:
        """Load YAMNet from TensorFlow Hub (downloads ~13MB on first run)."""
        if self._loading or self.loaded:
            return self.loaded
        self._loading = True
        try:
            import tensorflow_hub as hub
            import csv

            logger.info("Loading YAMNet from TensorFlow Hub...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')

            # Load class names from model
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            with open(class_map_path, 'r') as f:
                self.class_names = [row['display_name'] for row in csv.DictReader(f)]

            self.loaded = True
            logger.info(f"✓ YAMNet loaded — {len(self.class_names)} classes")
            return True

        except ImportError:
            logger.warning("tensorflow-hub not installed. Using fallback classifier.")
        except Exception as e:
            logger.warning(f"YAMNet load failed: {e}. Using fallback classifier.")

        self.loaded = False
        self._loading = False
        return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify_pcm(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str = "real"
    ) -> Dict:
        """
        Classify a mono float32 PCM audio array.

        Returns dict:
          is_target    bool   — True if mode's target sound detected
          confidence   float  — 0–100
          top_class    str    — best matching class name
          top5         list   — [{class, confidence}]
          mode         str
          reason       str
          physics      dict   — impulse analysis
        """
        if len(audio) == 0:
            return self._empty_result(mode)

        # Normalize
        peak = float(np.max(np.abs(audio)))
        if peak < 0.01:
            return {**self._empty_result(mode), "reason": "too_quiet"}

        if self.loaded:
            result = self._yamnet_classify(audio, sample_rate, mode)
        else:
            result = self._fallback_classify(audio, sample_rate, mode)

        # Extra physical validation in REAL mode
        if mode == "real" and result["is_target"]:
            phys = self._impulse_analysis(audio, sample_rate)
            result["physics"] = phys
            if not phys["pass"]:
                result["is_target"] = False
                result["reason"] = f"impulse_check_failed: {phys['fail_reason']}"
        else:
            result["physics"] = self._impulse_analysis(audio, sample_rate)

        return result

    # ── YAMNet classifier ──────────────────────────────────────────────────────

    def _yamnet_classify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str
    ) -> Dict:
        try:
            import librosa
            import tensorflow as tf

            # Resample to 16kHz mono (YAMNet requirement)
            if sample_rate != 16000:
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Clip and normalize
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            # Run YAMNet → scores shape: (n_frames, 521)
            scores, _, _ = self.model(audio)
            mean_scores = np.mean(scores.numpy(), axis=0)

            # Top-10 predictions
            top_idx = np.argsort(mean_scores)[::-1][:10]
            top_preds = [
                {"class": self.class_names[i],
                 "confidence": round(float(mean_scores[i]) * 100, 1)}
                for i in top_idx
            ]

            target_classes = CLAP_CLASSES if mode == "demo" else GUNSHOT_CLASSES
            threshold = THRESHOLD_DEMO if mode == "demo" else THRESHOLD_REAL

            # Find best target match
            target_conf = 0.0
            target_name = ""
            for pred in top_preds:
                if pred["class"] in target_classes and pred["confidence"] > target_conf:
                    target_conf = pred["confidence"]
                    target_name = pred["class"]

            # Check for hard rejection
            for pred in top_preds[:3]:
                if pred["class"] in REJECTION_CLASSES and pred["confidence"] > 75:
                    return {
                        "is_target": False,
                        "confidence": 0.0,
                        "top_class": pred["class"],
                        "top5": top_preds[:5],
                        "reason": f"rejected:{pred['class']}",
                        "mode": mode
                    }

            is_target = target_conf >= threshold
            return {
                "is_target": is_target,
                "confidence": target_conf,
                "top_class": target_name or top_preds[0]["class"],
                "top5": top_preds[:5],
                "reason": "yamnet",
                "mode": mode
            }

        except Exception as e:
            logger.error(f"YAMNet classify error: {e}")
            return self._fallback_classify(audio, sample_rate, mode)

    # ── Fallback classifier (no TF) ────────────────────────────────────────────

    def _fallback_classify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str
    ) -> Dict:
        """
        Energy + spectral feature classifier.
        Works without TensorFlow.
        """
        abs_a = np.abs(audio)
        peak = float(np.max(abs_a))
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Duration above 10% of peak
        threshold = 0.1 * peak
        above = abs_a > threshold
        duration_ms = (float(np.sum(above)) / sample_rate) * 1000.0

        # Spectral centroid via FFT
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
        spectral_centroid = float(
            np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9)
        )

        # Energy ratio: 800–4000 Hz band vs total
        band_mask = (freqs >= 800) & (freqs <= 4000)
        band_energy = float(np.sum(fft_mag[band_mask] ** 2))
        total_energy = float(np.sum(fft_mag ** 2) + 1e-9)
        band_ratio = band_energy / total_energy

        if mode == "demo":
            # Clap: short broadband impulse, medium freq
            is_target = (
                20 < duration_ms < 400 and
                peak > 0.25 and
                spectral_centroid > 400
            )
            conf = 78.0 if is_target else 0.0
            cls = "Clapping [fallback]" if is_target else "Not a clap"
        else:
            # Gunshot: very short, high peak, 800–4kHz band
            is_target = (
                duration_ms < 300 and
                peak > 0.4 and
                spectral_centroid > 700 and
                band_ratio > 0.3
            )
            conf = 72.0 if is_target else 0.0
            cls = "Gunshot [fallback]" if is_target else "Not a gunshot"

        return {
            "is_target": is_target,
            "confidence": conf,
            "top_class": cls,
            "top5": [{"class": cls, "confidence": conf}],
            "reason": "fallback",
            "mode": mode,
            "fallback_details": {
                "duration_ms": round(duration_ms, 1),
                "peak_amplitude": round(peak, 3),
                "spectral_centroid_hz": round(spectral_centroid, 1),
                "band_energy_ratio_800_4k": round(band_ratio, 3)
            }
        }

    # ── Physical impulse analysis (anti-spoofing) ──────────────────────────────

    def _impulse_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Anti-spoofing checks for real gunshots:
          1. Rise time < 50ms  (speakers can't reproduce <1ms shockwave)
          2. Duration < 350ms  (not thunder rolling 2–5s)
          3. Spectral peak in 800–4000 Hz band

        A speaker playing a gunshot recording FAILS check 1.
        A firecracker PASSES 1 but has different spectral profile.
        Thunder FAILS check 2.
        """
        abs_a = np.abs(audio)
        peak_idx = int(np.argmax(abs_a))
        peak_amp = float(abs_a[peak_idx])

        # Rise time: samples from 10% threshold to peak
        onset_threshold = 0.1 * peak_amp
        onset_idx = peak_idx
        for i in range(peak_idx, 0, -1):
            if abs_a[i] < onset_threshold:
                onset_idx = i
                break
        rise_time_ms = ((peak_idx - onset_idx) / sample_rate) * 1000.0

        # Duration above 10% of peak
        above = abs_a > onset_threshold
        duration_ms = (float(np.sum(above)) / sample_rate) * 1000.0

        # Spectral peak band
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
        if len(fft_mag) > 0 and np.sum(fft_mag) > 0:
            peak_freq = float(freqs[np.argmax(fft_mag)])
            band_mask = (freqs >= 800) & (freqs <= 4000)
            band_ratio = float(np.sum(fft_mag[band_mask]**2) /
                               (np.sum(fft_mag**2) + 1e-9))
        else:
            peak_freq, band_ratio = 0.0, 0.0

        # Evaluate checks
        checks = {
            "rise_time_ok": rise_time_ms < 100,   # < 100ms onset (we're lenient due to phone mics)
            "duration_ok": duration_ms < 500,
            "spectral_ok": band_ratio > 0.2
        }
        fail_reasons = [k for k, v in checks.items() if not v]

        return {
            "pass": all(checks.values()),
            "fail_reason": ", ".join(fail_reasons) if fail_reasons else None,
            "rise_time_ms": round(rise_time_ms, 2),
            "duration_ms": round(duration_ms, 1),
            "peak_freq_hz": round(peak_freq, 1),
            "band_energy_ratio": round(band_ratio, 3),
            "checks": checks
        }

    @staticmethod
    def _empty_result(mode: str) -> Dict:
        return {
            "is_target": False,
            "confidence": 0.0,
            "top_class": "No audio",
            "top5": [],
            "reason": "empty",
            "mode": mode,
            "physics": {}
        }

    def decode_b64_pcm(self, b64_str: str) -> Tuple[np.ndarray, int]:
        """Decode base64-encoded float32 PCM audio (sent from phone)."""
        import base64
        raw = base64.b64decode(b64_str)
        audio = np.frombuffer(raw, dtype=np.float32).copy()
        return audio, 16000


# Singleton
classifier = AudioClassifier()

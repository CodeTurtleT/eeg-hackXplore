"""
Neurofeedback Game — Unicorn Hybrid Black
==========================================
Channels : FZ, C3, CZ, C4, PZ, PO7, OZ, PO8  (indices 0–7)
Samplerate: 250 Hz
Protocol  : Theta/Beta ratio at CZ (ADHD standard)
            Theta  = 4–8  Hz  (suppress → ratio ↓ = good)
            Beta   = 13–21 Hz (enhance  → ratio ↓ = good)

Modes
-----
1. OFFLINE  – reads a .bdf or .fif file (for development / replay)
2. REALTIME – streams live from the Unicorn via UnicornPy (Windows only)
3. SIM      – interactive sine wave simulation 

Dependencies
------------
    pip install mne numpy scipy pygame

UnicornPy (real-time only, Windows):
    Install from the Unicorn Suite or:
    pip install UnicornPy   # if available via g.tec
"""

import sys
import queue
import threading
import argparse
import numpy as np
import pygame
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Optional imports ────────────────────────────────────────────────────────
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    import UnicornPy
    UNICORN_AVAILABLE = True
except ImportError:
    UNICORN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

FS          = 250           # Hz
CH_NAMES    = ["FZ","C3","CZ","C4","PZ","PO7","OZ","PO8"]
CZ_IDX      = 2             # index of CZ in the 8-channel array
N_EEG_CH    = 8

# Frequency bands (Hz)
THETA       = (4,  8)
BETA        = (13, 21)

# Processing
WINDOW_SEC  = 2.0           # power estimation window length
WINDOW_SAMP = int(WINDOW_SEC * FS)
STEP_SAMP   = int(0.1 * FS) # update every 100 ms

# Neurofeedback thresholds
# ratio < RATIO_GOOD  → full reward;  ratio > RATIO_BAD → punishment
RATIO_GOOD  = 1.5
RATIO_BAD   = 4.0

# Display
FPS         = 30
W, H        = 1000, 700

# Colors
BG          = (10,  12,  30)
ACCENT_GOOD = (50, 220, 120)
ACCENT_BAD  = (220, 60,  60)
NEUTRAL     = (180, 180, 220)
WHITE       = (255, 255, 255)
GREY        = ( 80,  80, 100)


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def butter_bandpass(low, high, fs=FS, order=4):
    """Return second-order-sections for a Butterworth bandpass."""
    nyq = fs / 2.0
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos


class BandpowerEstimator:
    """
    Maintains a rolling buffer and computes band power via
    variance of the filtered signal (fast, low-latency).
    """
    def __init__(self):
        self.buf   = np.zeros(WINDOW_SAMP)
        self._ptr  = 0
        self._full = False

        self.sos_theta = butter_bandpass(*THETA)
        self.sos_beta  = butter_bandpass(*BETA)

        # initial filter states
        self._zi_theta = sosfilt_zi(self.sos_theta) * 0
        self._zi_beta  = sosfilt_zi(self.sos_beta)  * 0

        # ratio history for smoothing
        self._ratio_hist = np.full(10, 2.5)

    def push(self, sample_uv: float):
        """Push one new CZ sample (µV). Returns smoothed theta/beta ratio."""
        self.buf[self._ptr % WINDOW_SAMP] = sample_uv
        self._ptr += 1

        # filter sample through both bands (causal, stateful)
        out_t, self._zi_theta = sosfilt(
            self.sos_theta, [sample_uv], zi=self._zi_theta)
        out_b, self._zi_beta  = sosfilt(
            self.sos_beta,  [sample_uv], zi=self._zi_beta)

        # accumulate power using variance of the rolling window
        if self._ptr >= WINDOW_SAMP:
            # full window available — compute power properly
            seg = np.roll(self.buf, -self._ptr)[:]
            p_theta = np.var(sosfilt(self.sos_theta, seg)[-WINDOW_SAMP//2:])
            p_beta  = np.var(sosfilt(self.sos_beta,  seg)[-WINDOW_SAMP//2:])
        else:
            # bootstrap: use running variance estimate
            p_theta = out_t[0]**2 + 1e-9
            p_beta  = out_b[0]**2 + 1e-9

        ratio = (p_theta + 1e-9) / (p_beta + 1e-9)
        # exponential smoothing
        self._ratio_hist = np.roll(self._ratio_hist, -1)
        self._ratio_hist[-1] = ratio
        return float(np.median(self._ratio_hist))


# ═══════════════════════════════════════════════════════════════════════════
#  DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════

class OfflineSource:
    """
    Reads a .bdf or .fif file, extracts CZ, and
    feeds samples into a queue at real-time speed.
    """
    def __init__(self, filepath: str, data_q: queue.Queue):
        assert MNE_AVAILABLE, "pip install mne"
        self.filepath = filepath
        self.q        = data_q
        self._stop    = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        import time
        ext = self.filepath.lower()
        if ext.endswith('.bdf') or ext.endswith('.edf'):
            raw = mne.io.read_raw_bdf(self.filepath, preload=True, verbose=False)
        elif ext.endswith('.fif'):
            raw = mne.io.read_raw_fif(self.filepath, preload=True, verbose=False)
        else:
            raise ValueError("Unsupported format. Use .bdf or .fif")

        # pick CZ channel (case-insensitive)
        ch_map = {ch.upper(): i for i, ch in enumerate(raw.ch_names)}
        cz_name = next((c for c in raw.ch_names if c.upper() in ('CZ','EEG CZ','CZ-REF')), None)
        if cz_name is None:
            raise RuntimeError(f"CZ channel not found. Available: {raw.ch_names}")

        data, times = raw[cz_name, :]
        data_uv = data[0] * 1e6  # V → µV

        dt = 1.0 / FS
        for sample in data_uv:
            if self._stop.is_set():
                break
            self.q.put(float(sample))
            time.sleep(dt)


class RealtimeUnicornSource:
    """
    Streams live data from the Unicorn Hybrid Black via UnicornPy.
    Pushes CZ samples into a queue.
    """
    def __init__(self, data_q: queue.Queue):
        assert UNICORN_AVAILABLE, "UnicornPy not installed (Windows only)"
        self.q     = data_q
        self._stop = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # discover device
        available = UnicornPy.GetAvailableDevices(True)
        if not available:
            raise RuntimeError("No Unicorn device found. Is it paired via Bluetooth?")
        device = UnicornPy.Unicorn(available[0])
        config = device.GetConfiguration()

        # total frame length: 8 EEG + 3 acc + 1 counter + 1 battery + validation
        total_ch = device.GetNumberOfAcquiredChannels()
        frame_size = total_ch * 4  # float32

        device.StartAcquisition(False)  # False = no test signal
        buf = bytearray(frame_size)

        try:
            while not self._stop.is_set():
                device.GetData(1, buf, frame_size)  # 1 frame
                frame = np.frombuffer(buf, dtype=np.float32)
                # CZ is channel index 2 (FZ=0, C3=1, CZ=2, ...)
                cz_sample = float(frame[CZ_IDX]) * 1e6  # already µV from device
                self.q.put(cz_sample)
        finally:
            device.StopAcquisition()
            device.Dispose()


class SineSimSource:
    """
    Simulated signal for testing without hardware.
    Generates theta + beta sine waves with adjustable ratio.
    """
    def __init__(self, data_q: queue.Queue, theta_amp=5.0, beta_amp=3.5):
        self.q         = data_q
        self.theta_amp = theta_amp  # mutable at runtime
        self.beta_amp  = beta_amp   # mutable at runtime
        self._stop     = threading.Event()
        self.t         = 0.0

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        import time
        dt = 1.0 / FS
        while not self._stop.is_set():
            # mix theta + beta + noise
            theta_f = 6.0
            beta_f  = 17.0
            val = (self.theta_amp * np.sin(2*np.pi*theta_f*self.t)
                 + self.beta_amp  * np.sin(2*np.pi*beta_f *self.t)
                 + np.random.normal(0, 0.5))
            self.q.put(val)
            self.t += dt
            time.sleep(dt)


# ═══════════════════════════════════════════════════════════════════════════
#  PYGAME NEUROFEEDBACK GAME
# ═══════════════════════════════════════════════════════════════════════════

class NeurofeedbackGame:
    """
    Simple but functional neurofeedback game:
    - A "focus beam" expands when theta/beta ratio is low (good focus)
    - A score bar fills up with sustained good performance
    - Animated background pulses with brain state
    """

    def __init__(self, data_q: queue.Queue, source=None):
        pygame.init()
        pygame.display.set_caption("Neurofeedback — Theta/Beta @ CZ")
        self.screen  = pygame.display.set_mode((W, H))
        self.clock   = pygame.time.Clock()
        self.q       = data_q
        self.source  = source
        self.estimator = BandpowerEstimator()

        self.ratio   = 2.5     # current smoothed ratio
        self.score   = 0.0     # 0–100
        self.running = True

        # font
        pygame.font.init()
        self.font_lg = pygame.font.SysFont("monospace", 52, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 22)
        self.font_xs = pygame.font.SysFont("monospace", 16)

        self._tick   = 0       # animation counter

    # ── helpers ──────────────────────────────────────────────────────────

    def _ratio_to_norm(self, ratio):
        """Map ratio [RATIO_GOOD, RATIO_BAD] → [1.0, 0.0] (higher = better)."""
        return float(np.clip(
            1.0 - (ratio - RATIO_GOOD) / (RATIO_BAD - RATIO_GOOD), 0.0, 1.0))

    def _lerp_color(self, t, c0, c1):
        return tuple(int(c0[i] + t*(c1[i]-c0[i])) for i in range(3))

    # ── drawing ──────────────────────────────────────────────────────────

    def _draw_background(self, norm):
        """Animated dark background with subtle radial pulse."""
        self.screen.fill(BG)
        pulse_r = int(200 + 80 * norm * np.sin(self._tick * 0.05))
        col     = self._lerp_color(norm, GREY, ACCENT_GOOD)
        alpha_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for r in range(pulse_r, 0, -30):
            a = max(0, int(30 * (1 - r/pulse_r) * norm))
            pygame.draw.circle(alpha_surf, (*col, a), (W//2, H//2), r, 2)
        self.screen.blit(alpha_surf, (0, 0))

    def _draw_beam(self, norm):
        """Central 'focus beam' — grows with good performance."""
        max_r = 180
        r     = int(30 + max_r * norm)
        col   = self._lerp_color(norm, ACCENT_BAD, ACCENT_GOOD)

        # outer glow rings
        for i in range(4):
            glow_r = r + (4-i)*12
            glow_a = 40 - i*8
            s = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.circle(s, (*col, glow_a), (W//2, H//2-60), glow_r)
            self.screen.blit(s, (0,0))

        pygame.draw.circle(self.screen, col, (W//2, H//2-60), r)
        pygame.draw.circle(self.screen, WHITE, (W//2, H//2-60), r, 3)

        # label inside beam
        lbl = self.font_sm.render("FOKUS", True, BG if norm > 0.5 else GREY)
        self.screen.blit(lbl, lbl.get_rect(center=(W//2, H//2-60)))

    def _draw_ratio_bar(self, norm):
        """Horizontal ratio bar at the bottom."""
        bar_w, bar_h = W - 120, 28
        bx, by = 60, H - 90
        # background
        pygame.draw.rect(self.screen, GREY, (bx, by, bar_w, bar_h), border_radius=14)
        # fill
        fill_w = int(bar_w * norm)
        col    = self._lerp_color(norm, ACCENT_BAD, ACCENT_GOOD)
        if fill_w > 8:
            pygame.draw.rect(self.screen, col,
                             (bx, by, fill_w, bar_h), border_radius=14)
        # border
        pygame.draw.rect(self.screen, WHITE, (bx, by, bar_w, bar_h),
                         2, border_radius=14)
        # labels
        lbl = self.font_xs.render(
            f"Theta/Beta Ratio: {self.ratio:.2f}   (gut < {RATIO_GOOD:.1f})",
            True, NEUTRAL)
        self.screen.blit(lbl, (bx, by - 24))

    def _draw_score(self):
        """Score display top-right."""
        sc = self.font_lg.render(f"{int(self.score):3d}", True, WHITE)
        self.screen.blit(sc, (W - 130, 30))
        lbl = self.font_xs.render("PUNKTE", True, GREY)
        self.screen.blit(lbl, (W - 120, 90))

    def _draw_channel_strip(self):
        """Small channel labels so the user knows what's active."""
        y = H - 42
        for i, ch in enumerate(CH_NAMES):
            col = ACCENT_GOOD if ch == "CZ" else GREY
            t   = self.font_xs.render(ch, True, col)
            self.screen.blit(t, (30 + i * 120, y))

    def _draw_header(self):
        title = self.font_sm.render(
            "Unicorn Hybrid Black  ·  Neurofeedback  ·  CZ", True, NEUTRAL)
        self.screen.blit(title, (30, 20))
        proto = self.font_xs.render(
            "Protokoll: Theta (4–8 Hz) ↓  Beta (13–21 Hz) ↑", True, GREY)
        self.screen.blit(proto, (30, 48))
        
        # Display hints if we are in interactive simulation mode
        if self.source and hasattr(self.source, 'theta_amp'):
            hint = self.font_xs.render(
                "↑ Fokus simulieren   ↓ Ablenkung simulieren   R Reset",
                True, GREY)
            self.screen.blit(hint, (30, 68))

    # ── main loop ────────────────────────────────────────────────────────

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # ── keyboard control for sim mode ────────────────────────────
            keys = pygame.key.get_pressed()
            if self.source and hasattr(self.source, 'theta_amp'):
                if keys[pygame.K_UP]:
                    # simulate focus (beta ↑)
                    self.source.beta_amp  = min(self.source.beta_amp  + 0.05, 12.0)
                if keys[pygame.K_DOWN]:
                    # simulate distraction (theta ↑)
                    self.source.theta_amp = min(self.source.theta_amp + 0.05, 12.0)
                if keys[pygame.K_r]:
                    # reset to neutral
                    self.source.theta_amp = 6.0
                    self.source.beta_amp  = 2.5

            # drain all available samples
            processed = 0
            while not self.q.empty() and processed < STEP_SAMP:
                sample = self.q.get_nowait()
                self.ratio = self.estimator.push(sample)
                processed += 1

            norm = self._ratio_to_norm(self.ratio)

            # update score: +1/s when good, -0.5/s when bad
            dt_s = 1.0 / FPS
            if norm > 0.6:
                self.score = min(100, self.score + norm * 2.0 * dt_s)
            elif norm < 0.3:
                self.score = max(0,   self.score - 1.0 * dt_s)

            # draw
            self._draw_background(norm)
            self._draw_header()
            self._draw_beam(norm)
            self._draw_score()
            self._draw_ratio_bar(norm)
            self._draw_channel_strip()

            pygame.display.flip()
            self.clock.tick(FPS)
            self._tick += 1

        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  CHANNEL INSPECTOR  (quick offline diagnostic)
# ═══════════════════════════════════════════════════════════════════════════

def inspect_file(filepath: str):
    """Print channel info and PSD summary for all 8 EEG channels."""
    assert MNE_AVAILABLE, "pip install mne"
    ext = filepath.lower()
    if ext.endswith('.bdf') or ext.endswith('.edf'):
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)

    print(f"\n{'='*55}")
    print(f"File    : {filepath}")
    print(f"Channels: {raw.ch_names}")
    print(f"Fs      : {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} s")
    print(f"{'='*55}")

    # PSD per EEG channel
    from mne.time_frequency import psd_array_welch
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data_v, _ = raw[eeg_picks, :]
    data_uv   = data_v * 1e6

    freqs, psds = psd_array_welch(
        data_uv, sfreq=raw.info['sfreq'],
        fmin=1, fmax=40, n_fft=512, verbose=False)

    def band_power(psd, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return float(np.mean(psd[idx]))

    print(f"{'Channel':<10} {'Theta(µV²)':<14} {'Beta(µV²)':<14} {'Θ/β ratio'}")
    print("-"*55)
    for i, ch in enumerate([raw.ch_names[p] for p in eeg_picks]):
        pt = band_power(psds[i], freqs, *THETA)
        pb = band_power(psds[i], freqs, *BETA)
        print(f"{ch:<10} {pt:<14.3f} {pb:<14.3f} {pt/pb:.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unicorn Neurofeedback — Theta/Beta @ CZ")
    parser.add_argument("--mode", choices=["offline","realtime","sim"],
                        default="sim",
                        help="Data source: offline .bdf/.fif, realtime Unicorn, or sim")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to .bdf or .fif file (--mode offline)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print channel/PSD summary and exit")
    args = parser.parse_args()

    if args.inspect:
        if not args.file:
            print("Specify --file path.bdf for inspection.")
            sys.exit(1)
        inspect_file(args.file)
        sys.exit(0)

    data_q = queue.Queue(maxsize=2000)

    if args.mode == "offline":
        assert args.file, "Provide --file path.bdf or path.fif"
        source = OfflineSource(args.file, data_q)
    elif args.mode == "realtime":
        source = RealtimeUnicornSource(data_q)
    else:
        print("[SIM] Kein Gerät — synthetisches Theta+Beta-Signal.")
        source = SineSimSource(data_q, theta_amp=6.0, beta_amp=2.5)

    source.start()
    # Pass the source object so the game can access source.theta_amp
    game = NeurofeedbackGame(data_q, source=source)
    try:
        game.run()
    finally:
        source.stop()


if __name__ == "__main__":
    main()
    
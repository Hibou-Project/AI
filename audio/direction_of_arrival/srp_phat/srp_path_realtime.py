"""
=============================================================================
  Drone DOA — Real-Time SRP-PHAT via GStreamer RTP Multicast
=============================================================================

Receives 4-channel L24 audio at 48 kHz from the same multicast RTP stream
used to record WAV files, routes 2 channels through appsink into Python,
and continuously estimates drone direction with a live polar plot.

GStreamer pipeline (Python side, mirrors your recording pipeline):

    udpsrc multicast
        → rtpjitterbuffer
        → rtpL24depay
        → audioconvert
        → deinterleave
            src_0 → appsink (channel 0)
            src_1 → appsink (channel 1)

Processing loop (separate thread):
    1. Drain ring buffers from both appsinks
    2. Align and window into analysis frames
    3. GCC-PHAT per mic pair
    4. SRP map over 0-360° with beam weighting
    5. Push result to display queue

Display (main thread):
    - Live updating matplotlib polar plot
    - Needle tracks current DOA
    - History arc shows last N estimates
    - Terminal line shows bearing + confidence

Dependencies:
    pip install numpy scipy matplotlib
    apt install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

Usage:
    python drone_doa_realtime.py
    python drone_doa_realtime.py --address 239.69.250.255 --port 5005
    python drone_doa_realtime.py --iface enp3s0 --latency 100
    python drone_doa_realtime.py --no-gui          # terminal-only output
=============================================================================
"""

import sys
import argparse
import time
import threading
import collections
import queue

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

try:
    from direction_of_arrival.srp_phat.srp_engine import SRPEngine as SharedSRPEngine
except ModuleNotFoundError:
    # Allows running this file directly from inside the folder.
    from srp_engine import SRPEngine as SharedSRPEngine


# =============================================================================
# ─── ARRAY CONFIGURATION  (edit to match your hardware) ──────────────────────
# =============================================================================

SPEED_OF_SOUND = 343.0  # m/s  (~331 + 0.6 * T_celsius)
SAMPLE_RATE = 48000  # Hz   — must match RTP stream clock-rate

# ── 2-mic linear array, 50 cm baseline ──────────────────────────────────────
#
#  Coordinate system (top-down view):
#
#          User  (at 90° = North)
#            ↑
#            |
#    Mic2    |    Mic1
#  (left)    |  (right)     ← as seen by the user facing the array
#    ×───────┼───────×
#  -25cm     0    +25cm   (x axis, East = positive)
#
#  Both mics face North (90°) — pointing toward the user.
#
#  Consequence for bearings (as heard from the user's position):
#    Drone to the user's RIGHT  → arrives from  East  (  ~0°)
#    Drone straight ahead       → arrives from  North ( ~90°)
#    Drone to the user's LEFT   → arrives from  West  (~180°)
#    Drone behind the array     → arrives from  South (~270°)
#
#  TDOA sign convention:
#    Source on the RIGHT (East):  hits Mic1 first  (positive TDOA on pair 0→1)
#    Source on the LEFT  (West):  hits Mic2 first  (negative TDOA on pair 0→1)
#
MIC_DISTANCE = 0.50  # metres  (50 cm baseline)

MIC_POSITIONS = np.array(
    [
        [0.25, 0.0],  # Mic 1 — RIGHT as seen by user  (East,  x = +25 cm)
        [-0.25, 0.0],  # Mic 2 — LEFT  as seen by user  (West,  x = -25 cm)
    ]
)

# Both parabolic dishes point toward the user (North = 90°)
MIC_DIRECTIONS = np.array([90.0, 90.0])

# Parabolic beam half-width (degrees). Gain tapers from 1 (on-axis) to
# GAIN_FLOOR beyond this angle.
#
# With both mics pointing North (90°), sources at East (0°) or West (180°)
# are 90° off-axis. Set BEAM_HALF_WIDTH > 90° so those directions still
# receive meaningful weight in the SRP map.
# Rule of thumb: set to 90° + half your desired side-coverage margin.
BEAM_HALF_WIDTH = 100.0  # degrees — covers full forward hemisphere + 10° behind
GAIN_FLOOR = 0.05  # residual gain beyond BEAM_HALF_WIDTH (models sidelobes)

# Angular resolution of DOA search grid
THETA_RESOLUTION = 0.5  # degrees

# Frequency band relevant for drone acoustics
FREQ_MIN = 50  # Hz
FREQ_MAX = 4000  # Hz

# Analysis window per SRP frame
FRAME_DURATION = 0.05  # seconds  (50 ms)
FRAME_HOP = 0.025  # seconds  (25 ms, 50% overlap)

# How many frames to accumulate per DOA estimate (more = smoother but slower)
FRAMES_PER_ESTIMATE = 4

# Ring buffer length in seconds (how much audio to keep per channel)
RING_BUFFER_SECONDS = 2.0


# =============================================================================
# ─── DISPLAY CONFIGURATION ───────────────────────────────────────────────────
# =============================================================================

HISTORY_LENGTH = 60  # number of past estimates shown on the polar plot
UPDATE_INTERVAL = 80  # ms between plot refreshes
SMOOTHING_ALPHA = 0.35  # EMA smoothing for the needle (0=frozen, 1=raw)


# =============================================================================
# ─── BEAM PATTERN ────────────────────────────────────────────────────────────
# =============================================================================


def parabolic_gain(theta_source: float, mic_direction: float) -> float:
    """Cosine-tapered parabolic beam gain, with a floor for back-lobe."""
    off_axis = abs((theta_source - mic_direction + 180) % 360 - 180)
    if off_axis >= BEAM_HALF_WIDTH:
        return GAIN_FLOOR
    main_lobe = np.cos(np.pi * off_axis / (2 * BEAM_HALF_WIDTH))
    return float(GAIN_FLOOR + (1.0 - GAIN_FLOOR) * main_lobe)


def pair_weight(theta: float, dir_a: float, dir_b: float) -> float:
    return float(np.sqrt(parabolic_gain(theta, dir_a) * parabolic_gain(theta, dir_b)))


# =============================================================================
# ─── GCC-PHAT ────────────────────────────────────────────────────────────────
# =============================================================================


def gcc_phat(x1: np.ndarray, x2: np.ndarray, n_fft: int, fs: int) -> np.ndarray:
    """
    Generalised Cross-Correlation with Phase Transform.
    Returns the time-domain correlation (length n_fft).
    Peak at index k means mic1 leads mic2 by k/fs seconds.
    """
    X1 = np.fft.rfft(x1, n=n_fft)
    X2 = np.fft.rfft(x2, n=n_fft)

    cross = X1 * np.conj(X2)
    cross /= np.abs(cross) + 1e-8  # PHAT whitening

    # Bandpass — only drone-relevant frequencies
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    cross[(freqs < FREQ_MIN) | (freqs > FREQ_MAX)] = 0.0

    return np.fft.irfft(cross, n=n_fft).real


# =============================================================================
# ─── SRP-PHAT ────────────────────────────────────────────────────────────────
# =============================================================================


class SRPEngine:
    """
    Precomputes steering tables once at startup, then processes frames fast.
    """

    def __init__(self, fs: int = SAMPLE_RATE):
        self.fs = fs
        self.n_fft = int(FRAME_DURATION * fs)
        self.hop = int(FRAME_HOP * fs)
        self.theta_grid = np.arange(0.0, 360.0, THETA_RESOLUTION)
        self.n_mics = len(MIC_POSITIONS)
        self.pairs = [
            (i, j) for i in range(self.n_mics) for j in range(i + 1, self.n_mics)
        ]
        self.window = np.hanning(self.n_fft)
        self._build_tables()

    def _build_tables(self):
        """Precompute TDOA sample indices and beam weights for all pairs × directions."""
        n_pairs = len(self.pairs)
        n_theta = len(self.theta_grid)

        self.tdoa_table = np.zeros((n_pairs, n_theta), dtype=int)
        self.weight_table = np.zeros((n_pairs, n_theta))

        for p, (i, j) in enumerate(self.pairs):
            for t, theta in enumerate(self.theta_grid):
                u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
                baseline = MIC_POSITIONS[i] - MIC_POSITIONS[j]
                tau = -np.dot(baseline, u) / SPEED_OF_SOUND
                delay_s = int(round(tau * self.fs)) % self.n_fft

                self.tdoa_table[p, t] = delay_s
                self.weight_table[p, t] = pair_weight(
                    theta, MIC_DIRECTIONS[i], MIC_DIRECTIONS[j]
                )

    def process(self, signals: np.ndarray) -> np.ndarray:
        """
        Run SRP-PHAT over all available frames in `signals` (n_mics × n_samples).
        Returns the normalised SRP map (shape: n_theta).
        """
        n_samples = signals.shape[1]
        srp_acc = np.zeros(len(self.theta_grid))
        n_frames = 0

        start = 0
        while start + self.n_fft <= n_samples:
            end = start + self.n_fft
            frames = [signals[m, start:end] * self.window for m in range(self.n_mics)]

            # GCC-PHAT per pair
            gccs = [
                gcc_phat(frames[i], frames[j], self.n_fft, self.fs)
                for i, j in self.pairs
            ]

            # Accumulate SRP
            srp_frame = np.zeros(len(self.theta_grid))
            for p, gcc in enumerate(gccs):
                srp_frame += self.weight_table[p] * gcc[self.tdoa_table[p]]

            srp_acc += srp_frame
            n_frames += 1
            start += self.hop

        if n_frames == 0:
            return srp_acc
        return srp_acc / n_frames

    def find_doa(self, srp_map: np.ndarray) -> tuple[float, float]:
        """
        Returns (doa_degrees, confidence).
        Confidence = ratio of peak power to mean power (>1 = clear peak).
        Uses parabolic interpolation for sub-grid precision.
        """
        idx = int(np.argmax(srp_map))
        n = len(srp_map)
        y0 = srp_map[(idx - 1) % n]
        y1 = srp_map[idx]
        y2 = srp_map[(idx + 1) % n]

        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-12:
            delta = (y0 - y2) / denom
        else:
            delta = 0.0

        step = self.theta_grid[1] - self.theta_grid[0]
        doa = float((self.theta_grid[idx] + delta * step) % 360)

        mean_power = float(np.mean(np.abs(srp_map)))
        confidence = float(y1 / (mean_power + 1e-12))

        return doa, confidence


# =============================================================================
# ─── RING BUFFER ─────────────────────────────────────────────────────────────
# =============================================================================


class RingBuffer:
    """Thread-safe ring buffer for streaming audio samples (float32)."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._head = 0  # write position
        self._size = 0  # valid samples
        self._lock = threading.Lock()
        self.total_written = 0  # diagnostic counter (never wraps in practice)

    def write(self, data: np.ndarray):
        with self._lock:
            self.total_written += len(data)
            n = len(data)
            if n >= self._cap:
                # Overwrite entirely with the latest chunk
                self._buf[:] = data[-self._cap :]
                self._head = 0
                self._size = self._cap
                return
            end = (self._head + n) % self._cap
            if end > self._head:
                self._buf[self._head : end] = data
            else:
                split = self._cap - self._head
                self._buf[self._head :] = data[:split]
                self._buf[:end] = data[split:]
            self._head = end
            self._size = min(self._size + n, self._cap)

    def read_latest(self, n: int) -> np.ndarray | None:
        """Return the latest `n` samples, or None if not enough data."""
        with self._lock:
            if self._size < n:
                return None
            out = np.empty(n, dtype=np.float32)
            start = (self._head - n) % self._cap
            if start + n <= self._cap:
                out[:] = self._buf[start : start + n]
            else:
                split = self._cap - start
                out[:split] = self._buf[start:]
                out[split:] = self._buf[: n - split]
            return out


# =============================================================================
# ─── GSTREAMER PIPELINE ──────────────────────────────────────────────────────
# =============================================================================


class GstReceiver:
    """
    Builds and runs the GStreamer receive pipeline.
    Pushes decoded float32 samples into per-channel RingBuffers.

    Design: NO GLib.MainLoop is used.
    ──────────────────────────────────────────────────────────────
    The GLib main loop tries to acquire the default GLib main context.
    When matplotlib is running in the main thread (it also uses an event
    loop), both fight over the same context → deadlock / segfault.

    The fix: appsink callbacks are invoked directly from GStreamer's
    internal streaming threads — they don't need a GLib main loop at
    all.  Bus messages (errors, EOS) are handled by a tiny polling
    thread that calls bus.timed_pop_filtered() in a loop.
    """

    def __init__(
        self,
        address: str,
        port: int,
        iface: str,
        latency: int,
        ring_buffers: list[RingBuffer],
        n_channels: int = 2,
    ):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except ImportError:
            raise ImportError(
                "PyGObject not found.\n"
                "Install with:  sudo apt install python3-gi gir1.2-gstreamer-1.0"
            )

        Gst.init(None)
        self._Gst = Gst
        self._buffers = ring_buffers
        self._stop_evt = threading.Event()
        self._sample_counts = [0] * n_channels  # diagnostic: samples received per ch

        # ── Build pipeline string ─────────────────────────────────────────────
        caps = (
            f"application/x-rtp, media=(string)audio, "
            f"clock-rate=(int){SAMPLE_RATE}, channels=(int)4, "
            f"encoding-name=(string)L24, payload=(int)99"
        )
        sink_caps = (
            f"audio/x-raw, format=F32LE, "
            f"rate={SAMPLE_RATE}, channels=1, layout=interleaved"
        )

        src_pads = ""
        for ch in range(n_channels):
            src_pads += (
                f" d.src_{ch} ! queue max-size-time=200000000"
                f" ! audioconvert"
                f" ! audio/x-raw,format=F32LE,rate={SAMPLE_RATE},channels=1"
                f" ! appsink name=sink{ch} emit-signals=true sync=false"
                f' caps="{sink_caps}"'
            )

        pipe_str = (
            f"udpsrc address={address} port={port} multicast-iface={iface} "
            f'caps="{caps}"'
            f" ! rtpjitterbuffer latency={latency}"
            f" ! rtpL24depay"
            f" ! audioconvert"
            f" ! deinterleave name=d" + src_pads
        )

        print(f"[GStreamer] Pipeline:\n  {pipe_str}\n")
        self._pipeline = Gst.parse_launch(pipe_str)

        # ── Connect appsink new-sample callbacks (fire from GStreamer threads) ─
        for ch in range(n_channels):
            sink = self._pipeline.get_by_name(f"sink{ch}")
            if sink is None:
                raise RuntimeError(f"appsink 'sink{ch}' not found in pipeline")
            sink.connect("new-sample", self._on_new_sample, ch)

        # ── Bus: get handle for polling (no add_signal_watch — avoids GLib ctx) ─
        self._bus = self._pipeline.get_bus()

    def _on_new_sample(self, sink, channel: int):
        """
        Called directly by GStreamer's streaming thread — no GLib loop needed.
        Reads the buffer and writes float32 samples into the ring buffer.
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            return self._Gst.FlowReturn.OK

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(self._Gst.MapFlags.READ)
        if not ok:
            return self._Gst.FlowReturn.ERROR

        data = np.frombuffer(mapinfo.data, dtype=np.float32).copy()
        buf.unmap(mapinfo)
        self._buffers[channel].write(data)
        self._sample_counts[channel] += len(data)
        return self._Gst.FlowReturn.OK

    def start(self):
        ret = self._pipeline.set_state(self._Gst.State.PLAYING)
        if ret == self._Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                "GStreamer failed to enter PLAYING state. "
                "Check multicast address, interface, and that "
                "the sender is running."
            )
        print("[GStreamer] Pipeline PLAYING", flush=True)

        # Start bus polling thread
        self._bus_thread = threading.Thread(
            target=self._bus_poll_loop, daemon=True, name="GStreamer-Bus"
        )
        self._bus_thread.start()

    def _bus_poll_loop(self):
        """
        Polls the GStreamer bus for error / EOS messages.
        Uses timed_pop_filtered() — purely a C call, no GLib main context.
        """
        Gst = self._Gst
        msg_types = (
            Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
        )

        while not self._stop_evt.is_set():
            # Block for up to 200 ms waiting for a bus message
            msg = self._bus.timed_pop_filtered(200 * Gst.MSECOND, msg_types)
            if msg is None:
                continue  # timeout — loop and check stop flag

            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"\n[GStreamer ERROR] {err.message}", flush=True)
                if dbg:
                    print(f"  Debug: {dbg}", flush=True)
                self._stop_evt.set()

            elif t == Gst.MessageType.WARNING:
                warn, _ = msg.parse_warning()
                print(f"\n[GStreamer WARNING] {warn.message}", flush=True)

            elif t == Gst.MessageType.EOS:
                print("\n[GStreamer] End of stream.", flush=True)
                self._stop_evt.set()

    def stop(self):
        self._stop_evt.set()
        # Transition to NULL flushes all buffers and joins streaming threads
        self._pipeline.set_state(self._Gst.State.NULL)
        # Wait for state change to complete before Python tears down GLib objects
        self._pipeline.get_state(timeout=2 * self._Gst.SECOND)
        print("[GStreamer] Pipeline stopped.", flush=True)

    def run_loop(self):
        """
        Kept for API compatibility — just waits on the stop event.
        The real work happens in GStreamer's own threads + bus poll thread.
        """
        self._stop_evt.wait()


# =============================================================================
# ─── PROCESSING THREAD ───────────────────────────────────────────────────────
# =============================================================================


class ProcessingThread(threading.Thread):
    """
    Continuously reads from ring buffers, runs SRP-PHAT, pushes results.
    """

    def __init__(
        self,
        ring_buffers: list[RingBuffer],
        result_queue: queue.Queue,
        engine: SRPEngine,
    ):
        super().__init__(daemon=True, name="SRP-Worker")
        self._bufs = ring_buffers
        self._queue = result_queue
        self._engine = engine
        self._stop = threading.Event()

        # How many samples we need to fill FRAMES_PER_ESTIMATE frames
        n_fft = engine.n_fft
        hop = engine.hop
        self._needed = n_fft + hop * (FRAMES_PER_ESTIMATE - 1)

        self._interval = hop * FRAMES_PER_ESTIMATE / SAMPLE_RATE  # seconds

    def run(self):
        print(
            f"[SRP Worker] Started — "
            f"need {self._needed} samples "
            f"({self._needed / SAMPLE_RATE * 1000:.0f} ms), "
            f"running every {self._interval * 1000:.0f} ms",
            flush=True,
        )

        while not self._stop.is_set():
            t0 = time.perf_counter()

            # Read latest N samples from each channel
            chunks = [b.read_latest(self._needed) for b in self._bufs]

            if any(c is None for c in chunks):
                # Not enough data yet — wait and retry
                time.sleep(0.01)
                continue

            signals = np.stack(chunks, axis=0)  # (n_mics, n_samples)

            try:
                srp_map = self._engine.process(signals)
                doa, confidence = self._engine.find_doa(srp_map)
                self._queue.put_nowait(
                    {
                        "doa": doa,
                        "confidence": confidence,
                        "srp_map": srp_map,
                        "timestamp": time.time(),
                    }
                )
            except Exception as exc:
                print(f"[SRP Worker] Error: {exc}", flush=True)

            # Sleep for remainder of interval
            elapsed = time.perf_counter() - t0
            sleep = max(0.0, self._interval - elapsed)
            time.sleep(sleep)

    def stop(self):
        self._stop.set()


# =============================================================================
# ─── LIVE DISPLAY ────────────────────────────────────────────────────────────
# =============================================================================


class LiveDisplay:
    """
    Matplotlib polar plot that updates in real time.

    Layout:
      - Blue shaded area:  current SRP map (normalised)
      - Red needle:        current DOA estimate (EMA smoothed)
      - Fading dots:       history of last N estimates
      - Orange lines:      mic pointing directions
      - Text box:          bearing, confidence, update rate
    """

    def __init__(self, result_queue: queue.Queue, engine: SRPEngine):
        self._queue = result_queue
        self._engine = engine
        self._history = collections.deque(maxlen=HISTORY_LENGTH)
        self._smoothed_doa = None
        self._last_update = time.time()
        self._fps_counter = collections.deque(maxlen=20)

        matplotlib.rcParams["toolbar"] = "None"
        self._fig = plt.figure(figsize=(9, 9), facecolor="#0d1117")
        self._fig.canvas.manager.set_window_title("Drone DOA — Real-Time")

        self._ax = self._fig.add_subplot(111, projection="polar", facecolor="#0d1117")
        self._setup_axes()
        self._init_artists()

        self._ani = animation.FuncAnimation(
            self._fig,
            self._update,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False,
        )

    def _setup_axes(self):
        ax = self._ax
        theta_grid = np.radians(self._engine.theta_grid)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0, 1.15)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], color="#404040")
        ax.tick_params(colors="#606060")
        ax.grid(color="#1e2a38", linewidth=0.8, linestyle="--")

        # Cardinal labels
        for deg, label in [(0, "E"), (90, "N"), (180, "W"), (270, "S")]:
            ax.text(
                np.radians(deg),
                1.12,
                label,
                ha="center",
                va="center",
                color="#aaaaaa",
                fontsize=11,
                fontweight="bold",
            )

        # Degree ticks every 30°
        for deg in range(0, 360, 30):
            ax.text(
                np.radians(deg),
                1.06,
                f"{deg}°",
                ha="center",
                va="center",
                color="#555555",
                fontsize=7,
            )

        ax.set_title(
            "DRONE DIRECTION OF ARRIVAL",
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

    def _init_artists(self):
        ax = self._ax
        theta_grid = np.radians(self._engine.theta_grid)
        n_theta = len(theta_grid)
        zero_map = np.zeros(n_theta)

        # SRP fill area
        (self._srp_fill,) = ax.fill(theta_grid, zero_map, color="#1a6faf", alpha=0.25)
        # SRP line
        (self._srp_line,) = ax.plot(
            theta_grid, zero_map, color="#3a9fd8", linewidth=1.2, alpha=0.7
        )

        # Mic direction markers
        self._mic_lines = []
        for mic_dir in MIC_DIRECTIONS:
            rad = np.radians(mic_dir)
            (line,) = ax.plot(
                [rad, rad],
                [0, 0.28],
                color="#e8a020",
                linewidth=2.5,
                alpha=0.85,
                solid_capstyle="round",
            )
            ax.text(
                rad,
                0.32,
                f"M{len(self._mic_lines)+1}\n{mic_dir:.0f}°",
                ha="center",
                va="center",
                color="#e8a020",
                fontsize=8,
                fontweight="bold",
            )
            self._mic_lines.append(line)

        # History dots (fading)
        self._history_scatter = ax.scatter(
            [], [], s=12, c=[], cmap="Blues", vmin=0, vmax=1, alpha=0.7, zorder=3
        )

        # DOA needle (filled triangle marker at tip)
        (self._needle,) = ax.plot(
            [0, 0],
            [0, 0],
            color="#e8192c",
            linewidth=3,
            solid_capstyle="round",
            zorder=5,
        )
        (self._needle_tip,) = ax.plot(
            [0], [0], "o", color="#e8192c", markersize=10, zorder=6
        )
        (self._needle_base,) = ax.plot(
            [0], [0], "o", color="#ff6b6b", markersize=5, zorder=6
        )

        # Info text box
        self._info_text = ax.text(
            np.radians(270),
            1.38,
            "",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1a1a2e",
                edgecolor="#3a3a5c",
                alpha=0.9,
            ),
        )

        # Status dot (green=receiving data, red=no data)
        self._status_dot = ax.text(
            np.radians(0),
            1.38,
            "●",
            ha="center",
            va="center",
            fontsize=14,
            color="#ff4444",
        )

    def _compass_label(self, deg: float) -> str:
        dirs = [
            "E",
            "ENE",
            "NE",
            "NNE",
            "N",
            "NNW",
            "NW",
            "WNW",
            "W",
            "WSW",
            "SW",
            "SSW",
            "S",
            "SSE",
            "SE",
            "ESE",
        ]
        return dirs[int((deg + 11.25) / 22.5) % 16]

    def _update(self, frame):
        """Called by FuncAnimation every UPDATE_INTERVAL ms."""
        # Drain result queue, keep only the latest
        result = None
        while True:
            try:
                result = self._queue.get_nowait()
            except queue.Empty:
                break

        if result is None:
            return  # nothing new — keep last frame

        # ── Update smoothed DOA ───────────────────────────────────────────────
        doa = result["doa"]
        confidence = result["confidence"]
        srp_map = result["srp_map"]

        if self._smoothed_doa is None:
            self._smoothed_doa = doa
        else:
            # EMA on the unit circle to handle 0°/360° wraparound
            alpha = SMOOTHING_ALPHA
            prev = np.radians(self._smoothed_doa)
            curr = np.radians(doa)
            sin_avg = (1 - alpha) * np.sin(prev) + alpha * np.sin(curr)
            cos_avg = (1 - alpha) * np.cos(prev) + alpha * np.cos(curr)
            self._smoothed_doa = np.degrees(np.arctan2(sin_avg, cos_avg)) % 360

        self._history.append(self._smoothed_doa)

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        self._fps_counter.append(now)
        if len(self._fps_counter) >= 2:
            dt = self._fps_counter[-1] - self._fps_counter[0]
            fps = (len(self._fps_counter) - 1) / (dt + 1e-9)
        else:
            fps = 0.0

        # ── SRP map ───────────────────────────────────────────────────────────
        theta_rad = np.radians(self._engine.theta_grid)
        norm_srp = srp_map / (np.max(np.abs(srp_map)) + 1e-12)
        norm_srp = np.clip(norm_srp, 0, None)

        # Update fill (requires re-creating path vertices)
        verts = np.column_stack([theta_rad, norm_srp])
        self._srp_fill.set_xy(verts)
        self._srp_line.set_ydata(norm_srp)

        # ── Needle ────────────────────────────────────────────────────────────
        needle_rad = np.radians(self._smoothed_doa)
        self._needle.set_data([needle_rad, needle_rad], [0, 1.0])
        self._needle_tip.set_data([needle_rad], [1.0])
        self._needle_base.set_data([needle_rad], [0.0])

        # ── History dots ──────────────────────────────────────────────────────
        if len(self._history) > 1:
            hist_rads = np.radians(list(self._history))
            hist_radii = np.ones(len(hist_rads)) * 0.92
            hist_colors = np.linspace(0.2, 0.9, len(hist_rads))
            self._history_scatter.set_offsets(np.column_stack([hist_rads, hist_radii]))
            self._history_scatter.set_array(hist_colors)

        # ── Info text ─────────────────────────────────────────────────────────
        self._info_text.set_text(
            f"DOA:  {self._smoothed_doa:6.1f}°  "
            f"({self._compass_label(self._smoothed_doa)})\n"
            f"Conf: {confidence:5.1f}x        \n"
            f"Rate: {fps:4.1f} est/s   "
        )

        # Status dot: green if data is fresh (<1s old), red otherwise
        age = now - result["timestamp"]
        self._status_dot.set_color("#44ff88" if age < 1.0 else "#ff4444")

        # ── Terminal line ─────────────────────────────────────────────────────
        bar = "█" * int(min(confidence, 20) / 20 * 30)
        print(
            f"\r  DOA: {self._smoothed_doa:6.1f}°  "
            f"({self._compass_label(self._smoothed_doa):3s})  "
            f"conf={confidence:5.1f}x  [{bar:<30}]  "
            f"{fps:.1f} est/s   ",
            end="",
            flush=True,
        )

    def show(self):
        plt.tight_layout()
        plt.show()


# =============================================================================
# ─── TERMINAL-ONLY DISPLAY (no GUI) ──────────────────────────────────────────
# =============================================================================


class TerminalDisplay:
    """Prints DOA results to stdout when --no-gui is used."""

    def __init__(self, result_queue: queue.Queue):
        self._queue = result_queue
        self._stop = threading.Event()

    def _compass_label(self, deg):
        dirs = [
            "E",
            "ENE",
            "NE",
            "NNE",
            "N",
            "NNW",
            "NW",
            "WNW",
            "W",
            "WSW",
            "SW",
            "SSW",
            "S",
            "SSE",
            "SE",
            "ESE",
        ]
        return dirs[int((deg + 11.25) / 22.5) % 16]

    def run(self):
        print("  Waiting for audio...", flush=True)
        while not self._stop.is_set():
            try:
                r = self._queue.get(timeout=1.0)
                doa, conf = r["doa"], r["confidence"]
                bar = "█" * int(min(conf, 20) / 20 * 40)
                ts = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
                print(
                    f"[{ts}]  DOA: {doa:6.1f}°  "
                    f"({self._compass_label(doa):3s})  "
                    f"conf={conf:5.1f}x  [{bar:<40}]",
                    flush=True,
                )
            except queue.Empty:
                pass

    def stop(self):
        self._stop.set()


# =============================================================================
# ─── MAIN ────────────────────────────────────────────────────────────────────
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Real-time drone DOA via GStreamer RTP multicast"
    )
    parser.add_argument(
        "--address",
        default="239.69.250.255",
        help="RTP multicast address (default: 239.69.250.255)",
    )
    parser.add_argument(
        "--port", type=int, default=5005, help="RTP UDP port (default: 5005)"
    )
    parser.add_argument(
        "--iface",
        default="enp3s0",
        help="Network interface for multicast (default: enp3s0)",
    )
    parser.add_argument(
        "--latency",
        type=int,
        default=100,
        help="rtpjitterbuffer latency in ms (default: 100)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Print results to terminal only (no matplotlib)",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Print per-stage diagnostics every second "
        "(use this if DOA is not updating to find where data stops)",
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   Drone DOA — Real-Time  (SRP-PHAT / GStreamer)  ║")
    print("╚══════════════════════════════════════════════════╝\n")
    print(f"  Multicast : {args.address}:{args.port}  iface={args.iface}")
    print(f"  Mics      : {len(MIC_POSITIONS)}  " f"directions={list(MIC_DIRECTIONS)}°")
    print(f"  Sample rate: {SAMPLE_RATE} Hz\n")

    # ── Shared objects ────────────────────────────────────────────────────────
    ring_capacity = int(RING_BUFFER_SECONDS * SAMPLE_RATE)
    ring_buffers = [RingBuffer(ring_capacity) for _ in range(len(MIC_POSITIONS))]
    result_queue = queue.Queue(maxsize=10)
    engine = SharedSRPEngine(
        fs=SAMPLE_RATE,
        mic_positions=MIC_POSITIONS,
        mic_directions=MIC_DIRECTIONS,
        speed_of_sound=SPEED_OF_SOUND,
        beam_half_width=BEAM_HALF_WIDTH,
        gain_floor=GAIN_FLOOR,
        theta_resolution=THETA_RESOLUTION,
        frame_duration=FRAME_DURATION,
        frame_hop=FRAME_HOP,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
    )

    print(
        f"  SRP engine ready  "
        f"({len(engine.theta_grid)} directions, "
        f"{len(engine.pairs)} pairs, "
        f"frame={engine.n_fft} samples)\n"
    )

    # ── GStreamer receiver ────────────────────────────────────────────────────
    try:
        receiver = GstReceiver(
            address=args.address,
            port=args.port,
            iface=args.iface,
            latency=args.latency,
            ring_buffers=ring_buffers,
            n_channels=len(MIC_POSITIONS),
        )
    except ImportError as exc:
        print(f"\n[ERROR] {exc}\n")
        sys.exit(1)

    # ── Processing thread ─────────────────────────────────────────────────────
    proc_thread = ProcessingThread(ring_buffers, result_queue, engine)

    # ── Start everything ─────────────────────────────────────────────────────
    # GStreamer receiver.start() launches its own bus-poll daemon thread.
    # run_loop() is no longer blocking — we don't need a wrapper thread.
    receiver.start()
    proc_thread.start()

    # ── Optional diagnostic thread ────────────────────────────────────────────
    if args.diag:

        def _diag_loop():
            import time as _time

            print(
                "\n[DIAG] Watching data flow every 1s  (Ctrl-C to stop)\n"
                "       Stage 1: GStreamer → appsink callbacks firing?\n"
                "       Stage 2: RingBuffers filling up?\n"
                "       Stage 3: SRP Worker producing estimates?\n",
                flush=True,
            )
            prev_gst = list(receiver._sample_counts)
            prev_rb = [b.total_written for b in ring_buffers]
            prev_queue = result_queue.qsize()
            while True:
                _time.sleep(1.0)
                cur_gst = list(receiver._sample_counts)
                cur_rb = [b.total_written for b in ring_buffers]
                cur_q = result_queue.qsize()

                gst_delta = [cur_gst[i] - prev_gst[i] for i in range(len(cur_gst))]
                rb_delta = [cur_rb[i] - prev_rb[i] for i in range(len(cur_rb))]

                ok1 = any(d > 0 for d in gst_delta)
                ok2 = any(d > 0 for d in rb_delta)
                ok3 = cur_q > 0 or prev_queue != cur_q

                print(
                    f"[DIAG] "
                    f"Stage1(GST callbacks) {'OK' if ok1 else 'NO DATA'} "
                    f"{[f'ch{i}:{d}smp' for i,d in enumerate(gst_delta)]}  "
                    f"Stage2(RingBuf fill) {'OK' if ok2 else 'NO DATA'} "
                    f"{[f'ch{i}:{rb._size}/{rb._cap}' for i,rb in enumerate(ring_buffers)]}  "
                    f"Stage3(result_queue) size={cur_q}",
                    flush=True,
                )

                if not ok1:
                    print(
                        "[DIAG] ❌ GStreamer callbacks not firing. "
                        "Check: multicast address, iface, sender running.",
                        flush=True,
                    )
                elif not ok2:
                    print(
                        "[DIAG] ❌ Samples not reaching RingBuffer. "
                        "Possible appsink format mismatch.",
                        flush=True,
                    )
                elif cur_q == 0:
                    sizes = [rb._size for rb in ring_buffers]
                    needed = proc_thread._needed
                    print(
                        f"[DIAG] ❌ SRP Worker waiting. "
                        f"Buffer sizes={sizes}, need={needed}. "
                        + (
                            "Enough data — check SRP errors above."
                            if all(s >= needed for s in sizes)
                            else f"Not enough yet, need {needed} samples."
                        ),
                        flush=True,
                    )

                prev_gst = cur_gst
                prev_rb = cur_rb
                prev_queue = cur_q

        diag_thread = threading.Thread(target=_diag_loop, daemon=True, name="Diag")
        diag_thread.start()

    try:
        if args.no_gui:
            display = TerminalDisplay(result_queue)
            display.run()  # blocks in main thread (Ctrl-C to exit)
        else:
            display = LiveDisplay(result_queue, engine)
            display.show()  # matplotlib blocks main thread here
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        print("\n  Shutting down...")
        proc_thread.stop()
        proc_thread.join(timeout=2.0)
        receiver.stop()  # must come AFTER proc_thread stops
        # (stop() blocks until pipeline is NULL)
        print("  Done.")


if __name__ == "__main__":
    main()

from time import sleep
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for better interactivity
import matplotlib.pyplot as plt
import threading
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

channel_id = 0
# Fixed pipeline: explicitly specify F32LE output format
pipeline_str = f'udpsrc address=239.69.250.255 port=5005 multicast-iface=enp2s0 caps="application/x-rtp, media=(string)audio, clock-rate=(int)48000, channels=(int)4, encoding-name=(string)L24, payload=(int)99" ! rtpjitterbuffer latency=100 ! rtpL24depay ! queue ! audioconvert ! audio/x-raw, format=F32LE ! volume volume=1.0 ! deinterleave name=d d.src_{channel_id} ! queue ! appsink name=app_sink sync=false'
alignment = 4  # Because all our data is F32LE.
# Process samples directly without buffering (set to 0 means process immediately)
required_buffer_size = 0
sinks_data = b""

# Visualization parameters
SAMPLE_RATE = 48000
DISPLAY_DURATION = 0.1  # seconds to display (adjust for better visualization)
DISPLAY_SAMPLES = int(SAMPLE_RATE * DISPLAY_DURATION)
signal_buffer = np.zeros(DISPLAY_SAMPLES)
buffer_lock = threading.Lock()
running = True
is_paused = False  # Flag for pause functionality

print(pipeline_str)

if not Gst.init_check(None):
    raise RuntimeError("Could not initialize GStreamer")

pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("app_sink")


def bytes_to_audio(raw_bytes):
    """
    Convert float 32 LE raw bytes to a normalized NumPy float32 array.
    """
    # Remove NaNs to prevent audio spikes and calculation errors
    return np.nan_to_num(np.frombuffer(raw_bytes, dtype=np.float32))


def on_new_sample(data, reset: bool):
    global sinks_data, signal_buffer

    if reset:
        sinks_data = b""

    # store data per channel
    sinks_data += data

    if required_buffer_size == 0:
        if len(sinks_data) > 0:
            buff = sinks_data
            sinks_data = b""

            audio_samples = bytes_to_audio(buff)

            if len(audio_samples) > 0:
                with buffer_lock:
                    if len(audio_samples) >= DISPLAY_SAMPLES:
                        signal_buffer[:] = audio_samples[-DISPLAY_SAMPLES:]
                    else:
                        signal_buffer[:] = np.roll(signal_buffer, -len(audio_samples))
                        signal_buffer[-len(audio_samples) :] = audio_samples
    else:
        while len(sinks_data) >= required_buffer_size:
            buff = sinks_data[:required_buffer_size]
            sinks_data = sinks_data[required_buffer_size:]

            audio_samples = bytes_to_audio(buff)

            if len(audio_samples) > 0:
                with buffer_lock:
                    if len(audio_samples) >= DISPLAY_SAMPLES:
                        signal_buffer[:] = audio_samples[-DISPLAY_SAMPLES:]
                    else:
                        signal_buffer[:] = np.roll(signal_buffer, -len(audio_samples))
                        signal_buffer[-len(audio_samples) :] = audio_samples


def handle_new_sample(sink):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    if not buf:
        return Gst.FlowReturn.ERROR

    reset = buf.has_flags(Gst.BufferFlags.DISCONT)

    try:
        data = buf.extract_dup(0, buf.get_size())
        on_new_sample(data, reset)
    except Exception as e:
        print(f"Error in on_sample callback for channel {channel_id}: {e}")
        import traceback

        traceback.print_exc()
        return Gst.FlowReturn.ERROR

    return Gst.FlowReturn.OK


def check_bus_messages():
    """Check GStreamer bus for messages"""
    global running
    bus = pipeline.get_bus()
    if bus:
        message = bus.pop_filtered(
            Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED
        )
        while message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"GStreamer error: {err.message}")
                if debug:
                    print(f"Debug info: {debug}")
                running = False
            elif message.type == Gst.MessageType.EOS:
                print("End of stream")
                running = False
            message = bus.pop_filtered(
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED
            )


# --- Plot Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
time_axis = np.linspace(0, DISPLAY_DURATION * 1000, DISPLAY_SAMPLES)
freq_axis = np.fft.rfftfreq(DISPLAY_SAMPLES, 1 / SAMPLE_RATE)

# Time domain plot
(line1,) = ax1.plot(time_axis, signal_buffer, "b-", linewidth=0.5)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Time Domain Signal")
ax1.set_xlim(0, DISPLAY_DURATION * 1000)
ax1.set_ylim(-0.4, 0.4)  # Initial Y-limits
ax1.grid(True, alpha=0.3)

# Annotations
max_amp_text = ax1.text(
    0.02,
    0.95,
    "",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
min_amp_text = ax1.text(
    0.02,
    0.85,
    "",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
peak_amp_text = ax1.text(
    0.02,
    0.75,
    "",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Frequency plot
magnitude_db = np.zeros(len(freq_axis))
(line2,) = ax2.plot(freq_axis, magnitude_db, "r-", linewidth=0.5)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.set_title("Frequency Spectrum")
ax2.set_xlim(0, min(5000, SAMPLE_RATE / 2))
ax2.set_ylim(-100, 20)  # Initial Y-limits
ax2.grid(True, alpha=0.3)

freq_text = ax2.text(
    0.02,
    0.95,
    "",
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.ion()
plt.show(block=False)
plt.pause(0.1)


def on_key_press(event):
    global is_paused
    # Pause/Resume
    if event.key == " ":
        is_paused = not is_paused
        if is_paused:
            print("\n*** PAUSED (Press SPACE to resume) ***")
            line1.set_color("gray")
            ax1.set_title("Time Domain Signal (PAUSED)")
            fig.canvas.draw_idle()
        else:
            print("*** RESUMED ***")
            line1.set_color("blue")
            ax1.set_title("Time Domain Signal")
            fig.canvas.draw_idle()

    # Zoom Logic
    elif event.key in ["+", "=", "up"]:
        # Zoom In
        for ax in [ax1, ax2]:
            ylims = ax.get_ylim()
            y_range = ylims[1] - ylims[0]
            new_range = y_range * 0.8  # Reduce range by 20%
            center = (ylims[0] + ylims[1]) / 2
            ax.set_ylim(center - new_range / 2, center + new_range / 2)
        fig.canvas.draw_idle()
        print("Zoom In")

    elif event.key in ["-", "_", "down"]:
        # Zoom Out
        for ax in [ax1, ax2]:
            ylims = ax.get_ylim()
            y_range = ylims[1] - ylims[0]
            new_range = y_range * 1.2  # Increase range by 20%
            center = (ylims[0] + ylims[1]) / 2
            ax.set_ylim(center - new_range / 2, center + new_range / 2)
        fig.canvas.draw_idle()
        print("Zoom Out")

    elif event.key == "r":
        # Reset Zoom
        ax1.set_ylim(-0.4, 0.4)
        ax2.set_ylim(-100, 20)
        fig.canvas.draw_idle()
        print("Zoom Reset")


fig.canvas.mpl_connect("key_press_event", on_key_press)

appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)
appsink.connect("new-sample", handle_new_sample)

if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
    raise RuntimeError("Failed to start pipeline")

print("Visualization running. Close the plot window or press Ctrl+C to stop.")
print("Controls:")
print("  [SPACE] : Pause/Resume")
print("  [+ / Up]   : Zoom In (Amplitude)")
print("  [- / Down] : Zoom Out (Amplitude)")
print("  [r]     : Reset Zoom")
print("Waiting for audio data...")

try:
    while running and plt.fignum_exists(fig.number):
        check_bus_messages()

        if not is_paused:
            # 1. Fetch Data
            with buffer_lock:
                local_buffer = signal_buffer.copy()

            # 2. Update Time Domain Plot Data
            line1.set_ydata(local_buffer)

            # 3. Update Text Stats
            if len(local_buffer) > 0:
                max_amp = np.max(local_buffer)
                min_amp = np.min(local_buffer)
                peak_amp = np.max(np.abs(local_buffer))

                max_amp_text.set_text(f"Max: {max_amp:.4f}")
                min_amp_text.set_text(f"Min: {min_amp:.4f}")
                peak_amp_text.set_text(f"Peak: {peak_amp:.4f}")

            # 4. Compute FFT & Update Frequency Plot
            if np.any(local_buffer != 0):
                windowed_signal = local_buffer * np.hanning(len(local_buffer))
                fft = np.fft.rfft(windowed_signal)
                magnitude = np.abs(fft)
                magnitude_db = 20 * np.log10(magnitude + 1e-10)
                line2.set_ydata(magnitude_db)

                if len(magnitude) > 1:
                    max_idx = np.argmax(magnitude[1:]) + 1
                    dominant_freq = freq_axis[max_idx]
                    dominant_mag_db = magnitude_db[max_idx]

                    if dominant_mag_db > -80:
                        freq_text.set_text(
                            f"Dominant: {dominant_freq:.1f} Hz ({dominant_mag_db:.1f} dB)"
                        )
                    else:
                        freq_text.set_text("Dominant: (below noise floor)")
                else:
                    freq_text.set_text("Dominant: N/A")

            # 5. Draw
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            sleep(0.01)
        else:
            # Paused state: Keep window responsive, don't fetch/process data
            fig.canvas.flush_events()
            sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    running = False
    plt.close("all")

    if pipeline.set_state(Gst.State.NULL) == Gst.StateChangeReturn.FAILURE:
        print("Warning: Failed to cleanly stop pipeline")

    print("Stopped.")

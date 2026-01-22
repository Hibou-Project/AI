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

print(pipeline_str)

if not Gst.init_check(None):
    raise RuntimeError("Could not initialize GStreamer")

pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("app_sink")


def bytes_to_audio(raw_bytes):
    """
    Convert float 32 LE raw bytes to a normalized NumPy float32 array.

    Args:
        raw_bytes (bytes): Raw 32 LE float audio data.

    Returns:
        np.ndarray: 1D array of float32 samples in range [-1.0, 1.0].
    """
    # EOSs coming from the GST pipelines generates NaNs, that can be played as audio spikes,
    # and can also cause problems in the downstream code (e.g.: librosa can complain).
    # Thus, we need to remove them.
    return np.nan_to_num(np.frombuffer(raw_bytes, dtype=np.float32))


def on_new_sample(data, reset: bool):
    global sinks_data, signal_buffer

    if reset:  # It may be requested by a discontinuity, or something else.
        sinks_data = b""

    # store data per channel
    sinks_data += data

    # If required_buffer_size is 0, process samples directly without buffering
    if required_buffer_size == 0:
        # Process all available data immediately
        if len(sinks_data) > 0:
            buff = sinks_data
            sinks_data = b""

            # Convert buffer to audio samples
            audio_samples = bytes_to_audio(buff)

            # Update the signal buffer for visualization
            if len(audio_samples) > 0:
                print(
                    f"Processing {len(audio_samples)} audio samples, max amplitude: {np.max(np.abs(audio_samples)):.4f}"
                )
                with buffer_lock:
                    # Roll the buffer and add new samples
                    if len(audio_samples) >= DISPLAY_SAMPLES:
                        signal_buffer[:] = audio_samples[-DISPLAY_SAMPLES:]
                    else:
                        signal_buffer[:] = np.roll(signal_buffer, -len(audio_samples))
                        signal_buffer[-len(audio_samples) :] = audio_samples
    else:
        # Use received bytes instead of duration. This is more accurate.
        while len(sinks_data) >= required_buffer_size:
            buff = sinks_data[:required_buffer_size]
            sinks_data = sinks_data[required_buffer_size:]

            # Convert buffer to audio samples
            audio_samples = bytes_to_audio(buff)

            # Update the signal buffer for visualization
            if len(audio_samples) > 0:
                with buffer_lock:
                    # Roll the buffer and add new samples
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
        print(f"Received {len(data)} bytes of audio data")
        on_new_sample(data, reset)  # pass raw data upward
    except Exception as e:
        # Log error or handle appropriately
        print(f"Error in on_sample callback for channel {channel_id}: {e}")
        import traceback

        traceback.print_exc()
        return Gst.FlowReturn.ERROR

    return Gst.FlowReturn.OK


def check_bus_messages():
    """Check GStreamer bus for messages (call periodically)"""
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
            elif message.type == Gst.MessageType.STATE_CHANGED:
                if message.src == pipeline:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    print(
                        f"Pipeline state changed: {old_state.value_nick} -> {new_state.value_nick}"
                    )
            message = bus.pop_filtered(
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED
            )


# Set up the plot with fixed axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
time_axis = np.linspace(0, DISPLAY_DURATION * 1000, DISPLAY_SAMPLES)  # in milliseconds
freq_axis = np.fft.rfftfreq(DISPLAY_SAMPLES, 1 / SAMPLE_RATE)

# Time domain plot
(line1,) = ax1.plot(time_axis, signal_buffer, "b-", linewidth=0.5)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Time Domain Signal")
ax1.set_xlim(0, DISPLAY_DURATION * 1000)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, alpha=0.3)

# FFT plot
magnitude_db = np.zeros(len(freq_axis))
(line2,) = ax2.plot(freq_axis, magnitude_db, "r-", linewidth=0.5)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.set_title("Frequency Spectrum")
ax2.set_xlim(0, min(5000, SAMPLE_RATE / 2))  # Show up to 5kHz or Nyquist
ax2.set_ylim(-100, 20)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.ion()  # Turn on interactive mode
plt.show(block=False)
plt.pause(0.1)


appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)  # Don't sync to clock for real-time processing
appsink.connect("new-sample", handle_new_sample)

if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
    raise RuntimeError("Failed to start pipeline")

print("Visualization running. Close the plot window or press Ctrl+C to stop.")
print("Waiting for audio data...")

try:
    while running and plt.fignum_exists(fig.number):
        # Check for GStreamer bus messages
        check_bus_messages()

        with buffer_lock:
            local_buffer = signal_buffer.copy()

        # Update time domain plot
        line1.set_ydata(local_buffer)

        # Compute and update FFT
        if np.any(local_buffer != 0):
            # Apply window to reduce spectral leakage
            windowed_signal = local_buffer * np.hanning(len(local_buffer))
            fft = np.fft.rfft(windowed_signal)
            magnitude_db = 20 * np.log10(
                np.abs(fft) + 1e-10
            )  # Convert to dB, avoid log(0)
            line2.set_ydata(magnitude_db)

        # Redraw the canvas
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        sleep(0.05)  # Update rate ~20 FPS

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    running = False
    plt.close("all")

    if pipeline.set_state(Gst.State.NULL) == Gst.StateChangeReturn.FAILURE:
        print("Warning: Failed to cleanly stop pipeline")

    print("Stopped.")

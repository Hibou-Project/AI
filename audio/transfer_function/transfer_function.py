import numpy as np
import scipy.io.wavfile as wavfile
from utils import display_over_time, remove_pulses_robust
import methods.welch as welch
import methods.welch_2 as welch2

"""
Load audio files
Verify Sampling Rate and Convert to Mono if needed
"""

reference_path = "./data/sweep.wav"
acquired_path = "./data/acquired.0.wav"
ref_sr, reference = wavfile.read(reference_path)
acq_sr, acquired = wavfile.read(acquired_path)
if ref_sr != acq_sr:
    raise ValueError(f"Sample rates don't match: {ref_sr} vs {acq_sr}")

SR = ref_sr
# Converting to mono if stereo
if reference.ndim > 1:
    reference = reference.mean(axis=1)
if acquired.ndim > 1:
    acquired = acquired.mean(axis=1)

# Convert to better precision
acquired = acquired.astype(np.float64)
reference = reference.astype(np.float64)

"""
Cut to same length
"""
min_length = np.min([len(reference), len(acquired)])
print(min_length)
reference = reference[:min_length]
acquired = acquired[:min_length]

"""
Remove unwanted spikes
"""
# n = 150000
# segment_to_clean = acquired[:n]
# cleaned_segment = remove_pulses_robust(segment_to_clean)
# acquired[:n] = cleaned_segment

"""
Display signals over time
"""
display_over_time(reference, acquired, SR)

# """
# Estimate using welch method
# """
# welch.run_optimized(reference, acquired, SR)

#
welch2.run(reference, acquired, SR)

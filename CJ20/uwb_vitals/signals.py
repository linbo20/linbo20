import numpy as np
from scipy import signal, ndimage


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def bandpass(x, band, fs):
    return signal.sosfiltfilt(signal.butter(4, band, fs=fs, btype="bandpass", output="sos"), x, axis=0)


def mdacm(z):
    """Discrete MDACM: integrate (I*dQ-Q*dI)/(I**2+Q**2).

    Starting phase is zero; no arctangent or unwrap. Requires sufficiently small
    inter-frame phase changes and nonzero IQ amplitude. Restart after gaps.
    """
    z = np.asarray(z, dtype=np.complex128)
    i, q = z.real, z.imag
    increments = (i[1:] * np.diff(q) - q[1:] * np.diff(i)) / np.maximum(np.abs(z[1:])**2, 1e-12)
    return np.r_[0., np.cumsum(increments)]


def block_filter(phase, band, cfg):
    """Five-second processing blocks, carrying SOS state across boundaries.

    Reset only at invalid segments. A 0.1 Hz respiration cannot be reliably
    independently filtered in 5 seconds; state preserves its longer history.
    """
    sos = signal.butter(4, band, fs=cfg.fs, btype='bandpass', output='sos')
    state = signal.sosfilt_zi(sos) * phase[0]
    result = np.empty(len(phase))
    block = max(1, round(cfg.filter_block_s * cfg.fs))
    for a in range(0, len(phase), block):
        b = min(len(phase), a+block)
        result[a:b], state = signal.sosfilt(sos, phase[a:b], zi=state)
    return result


def extract(cir, cfg):
    n = len(cir)
    if n == 0:
        raise ValueError('没有可处理帧')
    # All processing uses second receiver (zero-based RX1).
    channel = cir[:, cfg.selected_rx, :]
    energy = np.mean(np.abs(channel)**2, axis=1)
    delta = np.mean(np.abs(np.diff(channel, axis=0))**2, axis=1)
    motion = np.r_[0., np.sqrt(delta / np.maximum((energy[1:]+energy[:-1])/2, 1.))]
    motion = ndimage.uniform_filter1d(motion, size=max(1, round(cfg.fs*.2)), mode='nearest')
    baseline = np.median(motion)
    mad = 1.4826 * np.median(np.abs(motion-baseline))
    threshold = max(cfg.motion_threshold, baseline + cfg.motion_mad_factor * mad)
    moving = (motion > threshold) | (motion > cfg.motion_threshold*3)
    bad = ndimage.maximum_filter1d(moving.astype(int), size=2*round(cfg.guard_s*cfg.fs)+1) > 0
    # Complex temporal std retains phase-only micro-motion; amplitude std would miss it.
    quiet = ~bad
    std = np.std(channel[quiet].astype(np.complex128), axis=0) if quiet.any() else np.zeros(cfg.taps)
    amp = np.median(np.abs(channel), axis=0)
    scores = std.copy()
    scores[(amp < max(5., float(amp.max())*.03)) | (std < 1e-6)] = -np.inf
    selected = int(np.argmax(scores)) if np.isfinite(scores).any() else 0
    z = channel[:, selected]
    good = quiet & (np.abs(z) >= max(5., amp[selected]*.05))
    if not np.isfinite(scores).any():
        good[:] = False
    phase = np.full(n, np.nan)
    resp = np.full(n, np.nan)
    heart = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    for a,b in runs(good):
        phase[a:b] = mdacm(z[a:b])
        resp[a:b] = block_filter(phase[a:b], cfg.respiration_band, cfg)
        heart[a:b] = block_filter(phase[a:b], cfg.heart_band, cfg)
        # Preserve early transient waveform values, explicitly label warm-up.
        valid[min(b,a+round(cfg.minimum_s*cfg.fs)):b] = True
    return dict(motion_score=motion, motion_flag=moving, artifact_flag=bad,
                waveform_valid=valid, phase_rad=phase, respiration_rad=resp,
                heartbeat_rad=heart), dict(selected_rx=cfg.selected_rx,
                selected_tap=cfg.first_tap+selected, selected_tap_offset=selected,
                tap_std=[float(v) for v in std], motion_threshold=float(threshold),
                phase_method='MDACM current-sample denominator, relative phase per clean segment',
                filter_method='causal SOS with continuous state; 5-second processing blocks')

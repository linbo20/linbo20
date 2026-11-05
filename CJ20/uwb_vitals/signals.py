import numpy as np
from scipy import signal, ndimage


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def bandpass(x, band, fs):
    return signal.sosfiltfilt(signal.butter(4, band, fs=fs, btype="bandpass", output="sos"), x, axis=0)


def extract(cir, cfg):
    n = len(cir)
    # Complex change normalized by received energy; median across RX resists one bad receiver.
    energy = np.mean(np.abs(cir)**2, axis=2)
    delta = np.mean(np.abs(np.diff(cir, axis=0))**2, axis=2)
    motion = np.r_[0., np.median(np.sqrt(delta / np.maximum((energy[1:]+energy[:-1])/2, 1.)), axis=1)]
    motion = ndimage.uniform_filter1d(motion, size=max(1, round(cfg.fs*.2)), mode="nearest")
    baseline = np.median(motion)
    mad = 1.4826 * np.median(np.abs(motion-baseline))
    threshold = max(cfg.motion_threshold, baseline + cfg.motion_mad_factor * mad)
    moving = (motion > threshold) | (motion > cfg.motion_threshold * 3)
    # Expand on both sides: offline zero-phase filtering can smear motion backwards.
    bad = ndimage.maximum_filter1d(moving.astype(int), size=2*round(cfg.guard_s*cfg.fs)+1) > 0
    usable = ~bad
    edge = min(n, round(cfg.edge_s*cfg.fs))
    usable[:edge] = False
    usable[max(0,n-edge):] = False
    flat = cir.reshape(n, -1)
    # Fixed recording-wide range selection avoids switches and phase discontinuities.
    amp = np.median(np.abs(flat), axis=0)
    dynamic = np.median(np.abs(np.diff(flat, axis=0)), axis=0)
    score = dynamic / np.maximum(amp, 1)
    score[amp < max(5., np.max(amp)*.03)] = -np.inf
    if not np.isfinite(score).any():
        selected = 0
        usable[:] = False
    else:
        selected = int(np.argmax(score))
    raw_phase = np.full(n, np.nan)
    resp = np.full(n, np.nan)
    heart = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    # Filter independent motion-free segments; never interpolate through motion.
    for a,b in runs(~bad):
        if b-a < round(2*cfg.edge_s*cfg.fs + 5*cfg.fs):
            continue
        phase = np.unwrap(np.angle(flat[a:b,selected])).astype(float)
        phase = signal.detrend(phase)
        raw_phase[a:b] = phase
        r = bandpass(phase, cfg.respiration_band, cfg.fs)
        h = bandpass(phase, cfg.heart_band, cfg.fs)
        lo, hi = a+edge, b-edge
        if hi <= lo:
            continue
        resp[lo:hi] = r[edge:len(r)-edge]
        heart[lo:hi] = h[edge:len(h)-edge]
        valid[lo:hi] = usable[lo:hi]
    resp[~valid] = np.nan
    heart[~valid] = np.nan
    return dict(motion_score=motion, motion_flag=moving, artifact_flag=bad,
                waveform_valid=valid, phase_rad=raw_phase, respiration_rad=resp,
                heartbeat_rad=heart), dict(selected_rx=selected//cfg.taps,
                selected_tap=cfg.first_tap+selected%cfg.taps, motion_threshold=float(threshold))

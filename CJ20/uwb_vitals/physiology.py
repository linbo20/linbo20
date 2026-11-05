import numpy as np
from scipy import signal
from .signals import runs


def spectral_rate(x, band, fs, minimum):
    if not np.all(np.isfinite(x)) or np.std(x) < 1e-6:
        return np.nan, 0.
    # Full-window periodogram: true frequency resolution is 1/window_s.
    f, p = signal.periodogram(x, fs=fs, window="hann", nfft=len(x))
    region = (f >= band[0]) & (f <= band[1])
    indices = np.flatnonzero(region)
    k = indices[np.argmax(p[region])]
    concentration = float(p[max(indices[0],k-1):min(indices[-1]+1,k+2)].sum()/max(p[region].sum(),1e-30))
    freq = f[k]
    if 0 < k < len(p)-1:
        y = np.log(p[k-1:k+2]+1e-30)
        denominator = y[0]-2*y[1]+y[2]
        if abs(denominator) > 1e-12:
            freq += np.clip(.5*(y[0]-y[2])/denominator, -.5, .5)*(f[1]-f[0])
    return (float(freq*60) if concentration >= minimum else np.nan), concentration


def rates(data, cfg):
    n = len(data['motion_score'])
    w, step = round(cfg.window_s*cfg.fs), round(cfg.step_s*cfg.fs)
    output = []
    starts = list(range(0, max(0,n-w+1), step))
    if n >= w and (not starts or starts[-1] != n-w):
        starts.append(n-w)
    for a in starts:
        b = a+w
        row = dict(start_frame=a, end_frame_exclusive=b, center_s=(a+b-1)/(2*cfg.fs),
                   valid_fraction=float(data['waveform_valid'][a:b].mean()),
                   motion_fraction=float(data['motion_flag'][a:b].mean()))
        # Strict full clean window: no concatenation of separated clean fragments.
        for name, key, band in [('respiration','respiration_rad',cfg.respiration_band),
                                 ('heart','heartbeat_rad',cfg.heart_band)]:
            rate, quality = spectral_rate(data[key][a:b], band, cfg.fs, cfg.min_spectral_fraction)
            row[name+'_rate_bpm'] = rate
            row[name+'_spectral_fraction'] = quality
            row[name+'_valid'] = bool(np.isfinite(rate))
        # Flag possible respiratory harmonics; a bandpass alone cannot establish cardiac origin.
        rr, hr = row['respiration_rate_bpm'], row['heart_rate_bpm']
        ambiguous = bool(np.isfinite(rr) and np.isfinite(hr) and any(abs(hr-k*rr)<3 for k in range(2,9)))
        row['heart_harmonic_ambiguous'] = ambiguous
        if ambiguous:
            row['heart_rate_bpm'] = np.nan
            row['heart_valid'] = False
        output.append(row)
    return output


def candidate_beats(data, windows, cfg):
    # Only clean windows with accepted heart spectrum support candidate intervals.
    supported = np.zeros(len(data['heartbeat_rad']),bool)
    for row in windows:
        if row['heart_valid']:
            supported[row['start_frame']:row['end_frame_exclusive']] = True
    result = []
    for segment_id,(a,b) in enumerate(runs(data['waveform_valid'] & supported)):
        x = data['heartbeat_rad'][a:b]
        if len(x) < cfg.fs*3:
            continue
        peaks, _ = signal.find_peaks(x, distance=round(cfg.fs/cfg.heart_band[1]), prominence=.5*np.std(x))
        for previous,current in zip(peaks,peaks[1:]):
            ibi = (current-previous)/cfg.fs
            valid = bool(1/cfg.heart_band[1] <= ibi <= 1/cfg.heart_band[0])
            result.append(dict(segment_id=segment_id, previous_frame=int(a+previous), frame=int(a+current),
                               ibi_ms=ibi*1000, interval_plausible=valid, validated_for_hrv=False))
    return result

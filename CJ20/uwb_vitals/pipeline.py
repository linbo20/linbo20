import csv
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import numpy as np
from .config import Config
from .io import load_recording
from .signals import extract
from .physiology import rates, candidate_beats


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(folder, cfg=None):
    cfg = cfg or Config()
    cir, start, manifest = load_recording(folder, cfg)
    data, selection = extract(cir, cfg)
    windows = rates(data, cfg)
    beats = candidate_beats(data, windows, cfg)
    out = Path(folder)/'vitals_output'
    out.mkdir(exist_ok=True)
    def timestamp(i):
        return (start+timedelta(seconds=float(i)/cfg.fs)).isoformat(timespec='milliseconds')
    frame_fields = ['frame','time_s','timestamp',*data]
    write_csv(out/'frames.csv', (dict(frame=i,time_s=i/cfg.fs,timestamp=timestamp(i),
              **{k:v[i].item() for k,v in data.items()}) for i in range(len(cir))), frame_fields)
    window_fields = ['start_frame','end_frame_exclusive','center_s','valid_fraction','motion_fraction',
                     'respiration_rate_bpm','respiration_spectral_fraction','respiration_valid',
                     'heart_rate_bpm','heart_spectral_fraction','heart_valid','heart_harmonic_ambiguous',
                     'start_time','end_time_exclusive','center_time']
    for row in windows:
        row.update(start_time=timestamp(row['start_frame']),end_time_exclusive=timestamp(row['end_frame_exclusive']),
                   center_time=timestamp(row['center_s']*cfg.fs))
    write_csv(out/'rates_30s.csv',windows,window_fields)
    for row in beats:
        row['timestamp'] = timestamp(row['frame'])
    write_csv(out/'candidate_ibi.csv',beats,['segment_id','previous_frame','frame','ibi_ms','interval_plausible','validated_for_hrv','timestamp'])
    np.savez_compressed(out/'waveforms.npz',frame=np.arange(len(cir)),time_s=np.arange(len(cir))/cfg.fs,**data)
    metadata = dict(config=asdict(cfg),**selection,start_time=start.isoformat(),
                    duration_s=len(cir)/cfg.fs,frames=len(cir),files=manifest,
                    time_assumption='Asia/Shanghai folder start; fixed 100 Hz; no hardware timing validation',
                    waveform_units='radians; band-limited phase proxy, not calibrated displacement or ECG',
                    limitations=['Single subject, stationary range assumption; no real data validation.',
                    'Band-limited cardiac proxy may contain respiratory harmonics and multipath artifacts.',
                    'Candidate IBI is not validated NN intervals; do not directly interpret as clinical HRV.'])
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    plot(out,data,windows,cfg)
    print(f"完成: {len(cir)} 帧，{len(cir)/cfg.fs:.2f} 秒，输出 {out}")
    return out


def plot(out,data,windows,cfg):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cj20-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(data['motion_score']))/cfg.fs
    fig, axes = plt.subplots(4,1,figsize=(14,10),sharex=True)
    axes[0].plot(t,data['motion_score']); axes[0].set_ylabel('Motion score')
    axes[0].fill_between(t,0,1,where=data['artifact_flag'],transform=axes[0].get_xaxis_transform(),alpha=.2,color='red')
    axes[1].plot(t,data['respiration_rad']); axes[1].set_ylabel('Respiration (rad)')
    axes[2].plot(t,data['heartbeat_rad']); axes[2].set_ylabel('Cardiac proxy (rad)')
    for key,label in [('respiration_rate_bpm','Respiration'),('heart_rate_bpm','Heart')]:
        axes[3].plot([r['center_s'] for r in windows],[r[key] for r in windows],label=label)
    axes[3].legend(); axes[3].set_ylabel('Rate (/min)'); axes[3].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(out/'overview.png',dpi=150); plt.close(fig)

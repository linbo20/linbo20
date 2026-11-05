import tempfile
import unittest
from pathlib import Path
import numpy as np
from uwb_vitals.config import Config
from uwb_vitals.io import load_recording
from uwb_vitals.signals import extract, mdacm, block_filter
from uwb_vitals.physiology import rates
from uwb_vitals.pipeline import run


class PipelineTests(unittest.TestCase):
    def make_data(self, seconds=80):
        rng = np.random.default_rng(10)
        t = np.arange(seconds*100)/100
        phase = .35*np.sin(2*np.pi*.23*t)+.045*np.sin(2*np.pi*1.31*t)
        cir = 2500*np.exp(1j*(phase[:,None,None]+np.zeros((1,8,32))))
        cir += rng.normal(0,.5,cir.shape)+1j*rng.normal(0,.5,cir.shape)
        return cir.astype(np.complex64)

    def test_known_rates_and_motion(self):
        cfg = Config()
        cir = self.make_data()
        data,_ = extract(cir,cfg)
        rows = rates(data,cfg)
        clean = [r for r in rows if r['heart_valid']]
        self.assertTrue(clean)
        self.assertAlmostEqual(np.median([r['respiration_rate_bpm'] for r in clean]),13.8,delta=.5)
        self.assertAlmostEqual(np.median([r['heart_rate_bpm'] for r in clean]),78.6,delta=.8)
        cir[3900:4100] *= np.exp(1j*np.arange(200)[:,None,None]*.8)
        data,_ = extract(cir,cfg)
        self.assertTrue(data['motion_flag'][3900:4100].any())
        self.assertTrue(np.isnan(data['heartbeat_rad'][3900:4100]).all())
        self.assertEqual(len(data['motion_score']),8000)

    def test_harmonic_and_short_record(self):
        cfg = Config()
        t = np.arange(6000)/cfg.fs
        phase = .35*np.sin(2*np.pi*.23*t)+.045*np.sin(2*np.pi*1.38*t)
        cir = np.broadcast_to(2500*np.exp(1j*phase[:,None,None]), (6000,8,32)).copy()
        data,_ = extract(cir,cfg)
        rows = rates(data,cfg)
        ambiguous = [r for r in rows if r['heart_harmonic_ambiguous']]
        self.assertTrue(ambiguous)
        self.assertTrue(all(not r['heart_valid'] and np.isnan(r['heart_rate_bpm']) for r in ambiguous))
        short,_ = extract(cir[:2000],cfg)
        self.assertEqual([r['end_frame_exclusive'] for r in rates(short,cfg)], [501,1000,1500,2000])
        self.assertTrue(all(r['warmup_window'] for r in rates(short,cfg)))
        self.assertEqual(len(short['respiration_rad']),2000)

    def test_mdacm_and_second_channel_std(self):
        cfg = Config()
        t = np.arange(6000)/cfg.fs
        truth = .4*np.sin(2*np.pi*.23*t)
        np.testing.assert_allclose(mdacm(2000*np.exp(1j*truth)),truth,atol=1e-5)
        cir = np.full((6000,8,32), 2000+0j, dtype=np.complex64)
        cir[:,1,7] = 2000*np.exp(1j*truth)
        cir[:,0,4] = 10000*np.exp(1j*2*truth)
        data,selection = extract(cir,cfg)
        self.assertEqual(selection['selected_rx'],1)
        self.assertEqual(selection['selected_tap_offset'],7)
        np.testing.assert_allclose(data['phase_rad'],truth,atol=1e-5)
        rows = rates(data,cfg)
        self.assertEqual([r['end_frame_exclusive'] for r in rows],
                         [501,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000])
        self.assertEqual(rows[6]['start_frame'],500)
        self.assertEqual(rows[6]['estimate_s'],34.99)
        self.assertFalse(rows[5]['warmup_window'])

    def test_filter_blocks_preserve_state(self):
        from scipy import signal
        cfg = Config()
        t = np.arange(2345)/cfg.fs
        x = np.sin(2*np.pi*.23*t)
        sos = signal.butter(4,cfg.respiration_band,fs=cfg.fs,btype='bandpass',output='sos')
        expected,_ = signal.sosfilt(sos,x,zi=signal.sosfilt_zi(sos)*x[0])
        np.testing.assert_allclose(block_filter(x,cfg.respiration_band,cfg),expected)

    def test_flat_is_invalid(self):
        data,_ = extract(np.zeros((4000,8,32),np.complex64),Config())
        self.assertFalse(data['waveform_valid'].any())
        self.assertFalse(any(r['heart_valid'] for r in rates(data,Config())))

    def test_io_qi_time_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'radar_2026-11-05-19-36-34'
            root.mkdir()
            iq = np.zeros((100,8,32,2),dtype='<i2'); iq[...,0]=3; iq[...,1]=4
            iq.tofile(root/'COM21_radar_0.bin')
            cir,start,manifest = load_recording(root,Config())
            self.assertEqual(cir[0,0,0],4+3j)
            self.assertEqual(start.isoformat(),'2026-11-05T19:36:34+08:00')
            iq.tofile(root/'COM21_radar_2.bin')
            with self.assertRaisesRegex(ValueError,'缺号'):
                load_recording(root,Config())

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'radar_2026-11-05-19-36-34'; root.mkdir()
            cir = self.make_data(50)
            iq = np.stack((cir.imag,cir.real),axis=-1).astype('<i2')
            for i in range(50):
                iq[i*100:(i+1)*100].tofile(root/f'COM21_radar_{i}.bin')
            out = run(root)
            data = np.load(out/'waveforms.npz')
            self.assertEqual(len(data['time_s']),5000)
            self.assertAlmostEqual(data['time_s'][-1],49.99)
            for name in ['frames.csv','rates_30s.csv','candidate_ibi.csv','metadata.json','overview.png']:
                self.assertTrue((out/name).stat().st_size > 0)

if __name__ == '__main__':
    unittest.main()

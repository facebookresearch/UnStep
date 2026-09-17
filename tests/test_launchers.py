# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only launcher integration tests; workers never import torch or generate video."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
STUB = r'''
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

args = sys.argv[1:]
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default

scoring = '--dimension' in args
key = ('dim0' if 'subject_consistency' in args else 'dim1') if scoring else (
    'shard' + value('--shard-index', '0'))
control = Path(os.environ['STUB_CONTROL'])
if os.environ.get('STUB_BLOCK') == key:
    if os.environ.get('STUB_IGNORE_TERM'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (control / 'started').write_text(str(os.getpid()))
    time.sleep(30)
if os.environ.get('STUB_FAIL') == key:
    for _ in range(100):
        if (control / 'started').exists():
            break
        time.sleep(0.02)
    raise SystemExit(7)

record = {'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'argv': args,
          'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
if scoring:
    if os.environ.get('STUB_IMPORT_PROBE'):
        import unstep_overlay_probe
        record['overlay_origin'] = unstep_overlay_probe.__file__
        record['pythonpath'] = os.environ['PYTHONPATH'].split(os.pathsep)
    out = Path(value('--output_path'))
    dims = args[args.index('--dimension') + 1:args.index('--output_path')]
    score = float('nan') if os.environ.get('STUB_NAN') == key else 0.8
    (out / 'stub_eval_results.json').write_text(json.dumps({d: [score, []] for d in dims}))
    (out / 'worker.json').write_text(json.dumps(record))
else:
    run = value('--run-name')
    work = Path(value('--workdir')) / value('--workdir-name', run)
    if '--reuse-workdir' in args and not (work / 'src').is_dir():
        raise SystemExit('missing reused source')
    (work / 'src').mkdir(parents=True, exist_ok=True)
    count, shard = int(value('--num-shards')), int(value('--shard-index'))
    selected = list(range(946))[946 * shard // count:946 * (shard + 1) // count]
    raw = value('--prompt-indices')
    if raw == 'shard_first':
        selected = selected[:1]
    elif raw != 'all':
        selected = [int(x) for x in raw.split(',')]
    if os.environ.get('STUB_EMPTY'):
        selected = []
    record['measurements'] = [{'prompt_index': i, 'finite_video_fps': 20.0} for i in selected]
    out = Path(value('--out-json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    if '--save-videos' in args and value('--video-dir'):
        videos = Path(value('--video-dir'))
        videos.mkdir(parents=True, exist_ok=True)
        for i in selected:
            (videos / f'{i}-0_ema.mp4').touch()
    if '--keep-workdir' not in args:
        shutil.rmtree(work)
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        for file in ('generation/run.sh', 'generation/path_safety.sh',
                     'benchmarks/run_speed_probe.sh', 'evaluation/run.sh',
                     'evaluation/merge_vbench_scores.py', 'evaluation/install_detectron2.sh'):
            dest = self.repo / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / file, dest)
        (self.repo / 'unstep').mkdir()
        (self.repo / 'unstep/run_wrapper_generation.py').write_text(STUB)
        (self.repo / 'evaluation/vbench_eval_entry.py').write_text(STUB)
        python = self.repo / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.home = self.base / 'home'
        self.home.mkdir()
        self.control = self.base / 'control'
        self.control.mkdir()
        self.run_name = 'launchtest_' + uuid.uuid4().hex
        self.out = self.repo / 'results' / self.run_name
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('UNSTEP_', 'STUB_')) and k not in
                    ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONHOME', 'TORCH_CUDA_ARCH_LIST')}
        self.env.update(HOME=str(self.home), USER=self.run_name,
                        UNSTEP_CUDA=str(self.base / 'cuda'), UNSTEP_FFMPEG='/bin/true',
                        UNSTEP_WORK_ROOT=str(self.base / 'staging'), STUB_CONTROL=str(self.control),
                        PYTHONDONTWRITEBYTECODE='1')
        for gpu in range(2):
            self.addCleanup(shutil.rmtree, f'/tmp/unstep_{self.run_name}_gpu{gpu}_work', True)

    def invoke(self, script, *, name=None, **env):
        return subprocess.run(['bash', str(self.repo / script), name or self.run_name],
                              env=self.env | env, text=True, capture_output=True, timeout=15)

    def start(self, script, **env):
        process = subprocess.Popen(['bash', str(self.repo / script), self.run_name],
                                   env=self.env | env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)
        self.addCleanup(cleanup)
        return process

    def wait_started(self):
        for _ in range(200):
            if (self.control / 'started').exists():
                return int((self.control / 'started').read_text())
            time.sleep(0.02)
        self.fail('stub worker did not start')

    def assert_stopped(self, pid):
        for _ in range(100):
            status = Path(f'/proc/{pid}/status')
            if not status.exists() or '\nState:\tZ' in status.read_text():
                return
            time.sleep(0.02)
        self.fail(f'worker {pid} still running')

    def prepare_videos(self):
        info = self.repo / 'assets/source/VBench/vbench/VBench_full_info.json'
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps([{'prompt_en': 'duplicate'}] * 2))
        videos = self.out / 'indexed_videos'
        videos.mkdir(parents=True)
        for i in range(2):
            (videos / f'{i}-0_ema.mp4').touch()
        return videos

    def test_full_protocol_backend_mask_warmup_and_empty_healthy_stats(self):
        cache_marker = self.home / '.triton/preserve'
        cache_marker.parent.mkdir()
        cache_marker.touch()
        result = self.invoke('generation/run.sh', CUDA_VISIBLE_DEVICES='GPU-a,GPU-b',
                             UNSTEP_ATTENTION_BACKEND='fa2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('healthy_ge30_fps=n/a', result.stdout)
        self.assertTrue(cache_marker.exists())
        jsons = self.out / 'generation_jsons'
        self.assertEqual(len(list(jsons.glob('*.warmup.json'))), 2)
        indices = []
        for i in range(8):
            data = json.loads((jsons / f'shard{i}.json').read_text())
            self.assertEqual(data['gpu'], ('GPU-a', 'GPU-b')[i % 2])
            self.assertTrue(data['cache'].endswith(f'/gpu{i % 2}/torchinductor'))
            args = data['argv']
            for flag, expected in (('--fa3-import-backend', 'fa2'), ('--num-shards', '8'),
                                   ('--seed-mode', 'sequential_global'),
                                   ('--rng-skip-mode', 'initial_only')):
                self.assertEqual(args[args.index(flag) + 1], expected)
            indices.extend(row['prompt_index'] for row in data['measurements'])
        self.assertEqual(indices, list(range(946)))

    def test_probe_one_or_zero_measurements_and_first_visible_gpu(self):
        for empty in ('', '1'):
            with self.subTest(empty=empty):
                result = self.invoke('benchmarks/run_speed_probe.sh', name=self.run_name + empty,
                                     PROMPT_INDICES='0', STUB_EMPTY=empty,
                                     CUDA_VISIBLE_DEVICES='2,3', UNSTEP_ATTENTION_BACKEND='fa2')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('warm_mean_excluding_first=n/a', result.stdout)
                data = json.loads((self.repo / 'results' / (self.run_name + empty) /
                                   (self.run_name + empty + '.json')).read_text())
                self.assertEqual(data['gpu'], '2')

    def test_invalid_controls_fail_before_creating_outputs(self):
        for script, settings in (
            ('generation/run.sh', {'UNSTEP_FULL_WARMUP_RUNS': '2'}),
            ('generation/run.sh', {'UNSTEP_NUM_GPUS': '0'}),
            ('generation/run.sh', {'UNSTEP_EXTRA_ARGS': '--speed-only'}),
            ('generation/run.sh', {'UNSTEP_PROMPT_INDICES': '0,1'}),
            ('generation/run.sh', {'CUDA_VISIBLE_DEVICES': ''}),
            ('benchmarks/run_speed_probe.sh', {'UNSTEP_EXTRA_ARGS': '--num-shards=8'}),
            ('benchmarks/run_speed_probe.sh', {'UNSTEP_EXTRA_ARGS': '--clean-sigma-override'}),
        ):
            with self.subTest(script=script, settings=settings):
                result = self.invoke(script, **settings)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.out.exists())

    def test_subset_is_explicit_and_saves_only_selected_videos(self):
        result = self.invoke('generation/run.sh', UNSTEP_NUM_SHARDS='1', UNSTEP_NUM_GPUS='1',
                             UNSTEP_PROMPT_INDICES='4,9')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in (self.out / 'indexed_videos').iterdir()),
                         ['4-0_ema.mp4', '9-0_ema.mp4'])

    def test_probe_honors_repeated_separate_warmups(self):
        result = self.invoke('benchmarks/run_speed_probe.sh', WARMUP_REPEATS='2', PROMPT_INDICES='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.out.glob('*_WARMUP*.json'))), 2)
        second = json.loads((self.out / f'{self.run_name}_WARMUP1.json').read_text())
        self.assertIn('--reuse-workdir', second['argv'])

    def test_symlinked_lock_does_not_truncate_input(self):
        protected = self.repo / 'unstep/protected.txt'
        protected.write_text('keep source')
        Path(self.env['UNSTEP_WORK_ROOT'] + '.lock').symlink_to(protected)
        result = self.invoke('benchmarks/run_speed_probe.sh')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(protected.read_text(), 'keep source')

    def test_unsafe_cleanup_paths_are_rejected_before_find(self):
        # A broken guard must fail this test, never run real find against broad paths.
        bin_dir = self.base / 'bin'
        bin_dir.mkdir()
        find = bin_dir / 'find'
        find.write_text('#!/bin/sh\ntouch "$STUB_CONTROL/find_called"\nexit 93\n')
        find.chmod(0o755)
        path = str(bin_dir) + ':' + self.env['PATH']
        alias = self.base / 'source_alias'
        alias.symlink_to(self.repo / 'unstep')
        targets = [self.repo, self.repo / 'unstep', self.repo / '.git', self.repo / 'results',
                   self.repo / 'assets', self.repo / 'assets/nested', self.home,
                   self.base, Path('/'), Path('/tmp'), Path('/usr/bin'), Path('/etc'),
                   alias, self.repo / '.venv']
        for target in targets:
            with self.subTest(target=target):
                result = self.invoke('generation/run.sh', UNSTEP_OUT_DIR=str(target),
                                     UNSTEP_OVERWRITE_OUTPUT='1', PATH=path)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.control / 'find_called').exists())
        for script, variable in (('benchmarks/run_speed_probe.sh', 'UNSTEP_WORK_ROOT'),
                                 ('benchmarks/run_speed_probe.sh', 'UNSTEP_RUN_STATE_ROOT'),
                                 ('generation/run.sh', 'UNSTEP_WORK_ROOT')):
            result = self.invoke(script, **{variable: str(alias)}, UNSTEP_CACHE_MODE='shared', PATH=path)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((self.control / 'find_called').exists())
        checkpoint = self.base / 'model/checkpoint.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_text('preserve checkpoint')
        result = self.invoke('generation/run.sh', UNSTEP_OUT_DIR=str(checkpoint.parent),
                             UNSTEP_EXTRA_ARGS=f'--checkpoint-uri {checkpoint}',
                             UNSTEP_OVERWRITE_OUTPUT='1', PATH=path)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(checkpoint.read_text(), 'preserve checkpoint')
        self.assertFalse((self.control / 'find_called').exists())

    def test_staging_lock_and_interrupt_cleanup(self):
        first = self.start('benchmarks/run_speed_probe.sh', STUB_BLOCK='shard0')
        pid = self.wait_started()
        second = self.invoke('benchmarks/run_speed_probe.sh', name=self.run_name + '_second')
        self.assertEqual(second.returncode, 2, second.stderr)
        self.assertIn('staging is in use', second.stderr)
        first.terminate()
        first.communicate(timeout=10)
        self.assertEqual(first.returncode, 143)
        self.assert_stopped(pid)

    def test_failure_of_either_generation_worker_stops_sibling(self):
        for fail in (0, 1):
            with self.subTest(fail=fail):
                (self.control / 'started').unlink(missing_ok=True)
                result = self.invoke('generation/run.sh', UNSTEP_FULL_WARMUP_RUNS='0',
                                     UNSTEP_OVERWRITE_OUTPUT='1', STUB_FAIL=f'shard{fail}',
                                     STUB_BLOCK=f'shard{1 - fail}', STUB_IGNORE_TERM='1')
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assert_stopped(self.wait_started())

    def test_scoring_mask_duplicates_delete_and_rerun_preserves_results(self):
        videos = self.prepare_videos()
        result = self.invoke('evaluation/run.sh', CUDA_VISIBLE_DEVICES='3,7')
        self.assertEqual(result.returncode, 0, result.stderr)
        scores = self.out / 'vbench_score'
        self.assertEqual(len(list((scores / 'standard_video_links').iterdir())), 1)
        for i, gpu in enumerate(('3', '7')):
            self.assertEqual(json.loads((scores / f'dim{i}/worker.json').read_text())['gpu'], gpu)
        report = scores / f'{self.run_name}_table1_score.json'
        before = report.read_bytes()
        self.assertFalse(list(videos.glob('*.mp4')))
        again = self.invoke('evaluation/run.sh')
        self.assertEqual(again.returncode, 2, again.stderr)
        self.assertEqual(report.read_bytes(), before)

    def test_scoring_overlay_precedes_cuda_site_packages(self):
        self.prepare_videos()
        scoring, cuda, extra = [self.base / name for name in
                                ('scoring_overlay', 'generation_site_packages', 'extra_overlay')]
        for overlay in (scoring, cuda, extra):
            overlay.mkdir()
            (overlay / 'unstep_overlay_probe.py').write_text('')
        result = self.invoke('evaluation/run.sh', UNSTEP_VBENCH_PYTHONPATH=str(scoring),
                             UNSTEP_CUDA_OVERLAY=str(cuda), UNSTEP_EXTRA_PYTHONPATH=str(extra),
                             STUB_IMPORT_PROBE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = [self.repo, self.repo / 'assets/source/VBench', scoring, cuda, extra]
        for i in range(2):
            record = json.loads((self.out / f'vbench_score/dim{i}/worker.json').read_text())
            self.assertEqual(record['overlay_origin'], str(scoring / 'unstep_overlay_probe.py'))
            self.assertEqual(record['pythonpath'], [str(path) for path in expected])

    def test_nonfinite_merge_preserves_videos(self):
        videos = self.prepare_videos()
        result = self.invoke('evaluation/run.sh', STUB_NAN='dim1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonfinite VBench score', result.stderr)
        self.assertEqual(len(list(videos.glob('*.mp4'))), 2)
        self.assertFalse((self.out / 'vbench_score' / f'{self.run_name}_table1_score.json').exists())

    def test_scoring_failure_stops_sibling_and_preserves_videos(self):
        videos = self.prepare_videos()
        result = self.invoke('evaluation/run.sh', STUB_FAIL='dim1', STUB_BLOCK='dim0')
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assert_stopped(self.wait_started())
        self.assertEqual(len(list(videos.glob('*.mp4'))), 2)

    def test_keep_videos(self):
        videos = self.prepare_videos()
        result = self.invoke('evaluation/run.sh', UNSTEP_DELETE_VIDEOS_AFTER_SCORE='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(videos.glob('*.mp4'))), 2)


if __name__ == '__main__':
    unittest.main()

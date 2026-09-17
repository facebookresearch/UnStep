# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for score aggregation and Detectron2 build-command selection."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MERGER = ROOT / 'evaluation/merge_vbench_scores.py'
SPEC = importlib.util.spec_from_file_location('merge_vbench_scores', MERGER)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('UNSTEP_', 'STUB_')) and k not in
                    ('PYTHONPATH', 'PYTHONHOME', 'TORCH_CUDA_ARCH_LIST')}
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'

    def merge(self, scores):
        source = self.base / 'dim'
        source.mkdir(exist_ok=True)
        (source / 'stub_eval_results.json').write_text(json.dumps(scores))
        return subprocess.run([sys.executable, str(MERGER), 'test',
                               str(self.base / 'score.json'), str(self.base / 'score.md'), str(source)],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def maximum_scores(self):
        return {dim.replace(' ', '_'): [bounds['Max'], []] for dim, bounds in MERGE.NORMALIZE.items()}

    def test_official_scores_and_historical_norm16_are_unchanged(self):
        result = self.merge(self.maximum_scores())
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((self.base / 'score.json').read_text())
        for key in ('vbench_quality', 'vbench_semantic', 'vbench_total'):
            self.assertAlmostEqual(data[key], 100.0)
        self.assertAlmostEqual(data['vbench_historical_weighted_norm16'], 96.875)
        self.assertEqual(data['vbench_historical_weighted_norm16'],
                         data['vbench_unweighted_normalized_16dim_mean'])
        self.assertIn('Historical weighted Norm16 (auxiliary)', (self.base / 'score.md').read_text())

    def test_nonfinite_and_missing_dimensions_fail_without_report(self):
        for value in (float('nan'), float('inf'), -float('inf'), 1e308, None):
            with self.subTest(value=value):
                scores = self.maximum_scores()
                if value is None:
                    del scores['scene']
                else:
                    scores['scene'][0] = value
                result = self.merge(scores)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.base / 'score.json').exists())
                self.assertFalse((self.base / 'score.md').exists())

    def test_gb200_architecture_detection_override_and_no_gpu_failure(self):
        python = self.base / 'env/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text(f'#!{sys.executable}\n' + r'''
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

if sys.argv[1:3] == ['-m', 'pip']:
    Path(os.environ['STUB_BUILD']).write_text(json.dumps({
        'arch': os.environ['TORCH_CUDA_ARCH_LIST'], 'args': sys.argv[1:]}))
else:
    sys.modules['torch'] = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: not os.environ.get('STUB_NO_GPU'),
        device_count=lambda: 2, get_device_capability=lambda i: (10, 0)))
    sys.modules['detectron2'] = SimpleNamespace(__file__='detectron2/__init__.py',
                                               _C=SimpleNamespace(__file__='detectron2/_C.so'))
    exec(sys.stdin.read())
''')
        python.chmod(0o755)
        source = self.base / 'detectron_source'
        source.mkdir()
        build = self.base / 'build.json'
        env = self.env | {'UNSTEP_PYENV': str(python.parents[1]),
                          'UNSTEP_CUDA': str(self.base / 'cuda'),
                          'UNSTEP_VBENCH_DEPS': str(self.base / 'overlay'),
                          'STUB_BUILD': str(build)}
        script = ['bash', str(ROOT / 'evaluation/install_detectron2.sh'), str(source)]
        for override in ({}, {'TORCH_CUDA_ARCH_LIST': '10.0', 'STUB_NO_GPU': '1'}):
            result = subprocess.run(script, env=env | override, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(build.read_text())
            self.assertEqual(record['arch'], '10.0')
            self.assertIn('--no-deps', record['args'])
            self.assertIn('--no-build-isolation', record['args'])
            self.assertIn('detectron2/_C.so', result.stdout)
        build.unlink()
        result = subprocess.run(script, env=env | {'STUB_NO_GPU': '1'},
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('set TORCH_CUDA_ARCH_LIST explicitly', result.stderr)
        self.assertFalse(build.exists())

    def test_readme_shell_examples_parse_and_overlay_never_lists_torch(self):
        for file in ('generation/README.md', 'benchmarks/README.md', 'evaluation/README.md'):
            text = (ROOT / file).read_text()
            for block in text.split('```bash\n')[1:]:
                result = subprocess.run(['bash', '-n'], input=block.split('```')[0],
                                        text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, (file, result.stderr))
        reqs = (ROOT / 'evaluation/requirements.txt').read_text().splitlines()
        self.assertIn('setuptools==80.9.0', reqs)
        names = {line.split('==')[0].lower() for line in reqs if line and not line.startswith('#')}
        self.assertFalse(names & {'torch', 'torchvision', 'triton', 'open_clip_torch', 'opencv-python'})
        readme = (ROOT / 'evaluation/README.md').read_text()
        self.assertIn('pip install --no-deps --no-build-isolation', readme)
        self.assertNotIn('--allow-pattern', readme)


if __name__ == '__main__':
    unittest.main()

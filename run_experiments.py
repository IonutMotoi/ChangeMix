"""List or run the training runs of an experiment file in configs/experiments/.

    python run_experiments.py configs/experiments/frameworks.yaml          # print the commands
    python run_experiments.py configs/experiments/frameworks.yaml --run    # run them one after another

    # a subset, e.g. UniMatch V2 + ChangeMix on LEVIR-CD 20%, seed 42
    python run_experiments.py configs/experiments/frameworks.yaml --run \
        --filter script=unimatch_v2_changemix dataset=levir split=20% seed=42

Each entry of `runs` expands to all the combinations of its list-valued fields. A run trains
<script>.py with configs/<dataset>.yaml on splits/<dataset>/<split>/ (split "full": the whole
training set) and passes every other field to the script as --<field> <value>. Checkpoints,
TensorBoard logs and out.log are written to exp/<dataset>/<split>/<script>/<tag>/seed<seed>.
"""

import argparse
import itertools
import os
import shlex
import subprocess
import sys

import yaml

PATH_FIELDS = ('script', 'dataset', 'split', 'seed')


def expand(entry):
    keys = list(entry)
    values = [v if isinstance(v, list) else [v] for v in entry.values()]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def build_command(run, nproc, port):
    script, dataset, split, seed = (run[k] for k in PATH_FIELDS)
    extra = {k: v for k, v in run.items() if k not in PATH_FIELDS}

    if split == 'full':
        labeled, unlabeled = f'splits/{dataset}/train.txt', None
    else:
        labeled, unlabeled = f'splits/{dataset}/{split}/labeled.txt', f'splits/{dataset}/{split}/unlabeled.txt'

    tag = [str(extra[k]) for k in ('model', 'backbone') if k in extra]
    tag += [f'{k}-{v}' for k, v in extra.items() if k not in ('model', 'backbone')]
    save_path = os.path.join('exp', dataset, split, script, '_'.join(tag) or 'default', f'seed{seed}')

    cmd = ['torchrun', f'--nproc_per_node={nproc}', f'--master_port={port}', f'{script}.py',
           '--config', f'configs/{dataset}.yaml', '--labeled-id-path', labeled]
    if unlabeled and script != 'supervised':
        cmd += ['--unlabeled-id-path', unlabeled]
    cmd += ['--save-path', save_path, '--seed', str(seed)]
    for key, value in extra.items():
        cmd += [f'--{key}', str(value)]
    return cmd, save_path


def main():
    parser = argparse.ArgumentParser(description='List or run the training runs of an experiment file.')
    parser.add_argument('experiments', help='experiment file, e.g. configs/experiments/frameworks.yaml')
    parser.add_argument('--run', action='store_true', help='run the commands (default: only print them)')
    parser.add_argument('--filter', nargs='+', default=[], metavar='FIELD=VALUE',
                        help='keep only the runs matching all the given field values')
    parser.add_argument('--nproc', type=int, default=1, help='GPUs per run (all results of the paper use 1)')
    parser.add_argument('--port', type=int, default=29500, help='torchrun master port')
    args = parser.parse_args()

    with open(args.experiments, 'r') as f:
        entries = yaml.safe_load(f)['runs']
    filters = [item.split('=', 1) for item in args.filter]
    runs = [run for entry in entries for run in expand(entry)
            if all(str(run.get(key)) == value for key, value in filters)]

    for i, run in enumerate(runs, 1):
        cmd, save_path = build_command(run, args.nproc, args.port)
        print(f'[{i}/{len(runs)}] {shlex.join(cmd)}', flush=True)
        if args.run:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'out.log'), 'a') as log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                if process.wait() != 0:
                    sys.exit(f'Run failed: {shlex.join(cmd)} (log: {save_path}/out.log)')

    print(f'{len(runs)} run(s)' + ('' if args.run else ' (use --run to train)'))


if __name__ == '__main__':
    main()

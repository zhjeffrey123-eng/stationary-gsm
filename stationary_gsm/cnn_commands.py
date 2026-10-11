"""公开周期 CNN 入口。Torch 只在调用这些命令时加载。

原 NumPy 多层感知机与电磁求解模块保持不变。这里调用 stationary_cnn
的同一训练核心，不重新实现卷积、相位、全场损失或检查点逻辑。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


DATA_SHA256 = 'd31d313533f8c5c3e7b9947a8b2005f9f9afca821a3bd3469cafc5b1914a4d9a'


def initialize_torch(device='cuda', seed=120026, threads=4):
    """先设置确定性环境，再显式载入用户自行安装的 Torch。

    CPU 与 CUDA 是两种实际运行平台。CUDA 不可用时直接报错，不能悄悄
    换成 CPU 后仍把结果标记为 GPU。跨平台重训不承诺逐位相同权重。
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', '20')
    import torch
    if not str(torch.__version__).startswith('2.7.1'):
        raise RuntimeError('此配置固定使用 Torch 2.7.1。请按 README 安装对应 CPU/CUDA 版本。')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用。请安装 CUDA 11.8 的 Torch，或明确指定 --device cpu。')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device 必须为 cpu 或 cuda')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def _active_rows(mask):
    from .physics import active
    return active(mask, 24)


def train_cnn(data_path, output_dir, training_count=600, epochs=20,
              checkpoint_epochs=(5, 10, 15, 20), device='cuda', threads=4,
              seed=120026, sampler_seed=2026101102, resume=None):
    """完整矩阵监督，train-only RMS，固定一百幅验证图案。

    count 是训练拓扑数，不是频率条件数。一个 epoch 完整遍历 count 幅图案。
    每批最多四幅拓扑，每拓扑包含三频率乘十八端口的完整场。末批保留余下拓扑。
    """
    from stationary_cnn import trainer
    torch = initialize_torch(device, seed, threads)
    protocol = {
        'architecture': 'CNN32_periodic_matrix', 'training_count': int(training_count),
        'epochs': int(epochs), 'checkpoint_epochs': list(checkpoint_epochs),
        'batch_topologies': 4, 'fields_per_topology': 54,
        'loss': trainer.LOSS_NAME, 'initial_seed': int(seed),
        'sampler_seed': int(sampler_seed), 'learning_rate': .001,
        'minimum_learning_rate': .0001, 'device': device, 'threads': int(threads),
        'data_sha256': DATA_SHA256,
        'validation_start': 5000, 'validation_count': 100,
        'numerical_core_sha256': {name: trainer.file_sha256(Path(trainer.__file__).with_name(name))
                                 for name in ('indexing.py', 'model.py', 'objective.py', 'data.py', 'trainer.py')},
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    protocol_sha = hashlib.sha256(canonical).hexdigest()
    return trainer.train(torch, data_path, output_dir, training_count, epochs=epochs,
        checkpoint_epochs=checkpoint_epochs, active_rows=_active_rows,
        expected_data_sha256=DATA_SHA256, resume_path=resume,
        metadata={'protocol_sha256': protocol_sha, 'protocol': protocol},
        device=device, initial_seed=seed, sampler_seed=sampler_seed, threads=threads)


def train_cnn_series(data_path, output_dir, counts=(50, 100, 200, 1000, 5000),
                     epochs=20, checkpoint_epochs=(5, 10, 15, 20), device='cuda', threads=4):
    """串行启动独立 Python 进程，禁止重用上一规模的进程累计峰值或标签缓存。

    每个规模分别建立自己的缓存。数据源只读共用，测试集没有输入入口。
    这里不按测试误差挑选规模，也不保证误差对规模严格单调。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(set(counts)) != len(counts) or any(count < 1 or count > 5000 for count in counts):
        raise ValueError('counts 必须互异，且为1至5000的整数')
    runs = []
    for count in counts:
        target = output_dir / f'N{count}'
        command = [sys.executable, '-m', 'stationary_gsm', 'train-cnn',
                   '--data', str(Path(data_path).resolve()), '--out', str(target.resolve()),
                   '--training-count', str(count), '--epochs', str(epochs),
                   '--device', device, '--threads', str(threads),
                   '--checkpoint-epochs', *(str(value) for value in checkpoint_epochs)]
        # check=True 保留真实失败，不跳过失败规模或替换为更有利的随机种子。
        subprocess.run(command, check=True)
        runs.append({'training_count': count, 'directory': target.name,
                     'summary': json.loads((target / 'training_summary.json').read_text(encoding='utf-8'))})
    result = {'status': 'complete', 'runs': runs, 'test_data_accessed': False,
              'fresh_process_per_size': True, 'independent_cache_per_size': True}
    (output_dir / 'series_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def cnn_demo(model_path, target, frequency=7., tolerance=.1, maximum_iterations=512,
             device='cuda', threads=4, data_path=None, training_index=0):
    """现场环形像素示例。三种 GSM 算子使用同一份预测/校正电流。

    提取算子比较与神经网络训练成本是两个问题。示例计算参考解只是为了
    验证本次输出，不代表使用网络进行每次设计评价仍必须重算该参考解。
    """
    import numpy as np
    from stationary_cnn.model import load_predictor
    from .physics import solve
    from .correction import refine
    from .extraction import extract
    if frequency <= 0 or tolerance <= 0 or maximum_iterations < 1:
        raise ValueError('frequency/tolerance/maximum_iterations 必须为正')
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    if data_path is None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        mask[3:5, 3:5] = False
        example_name = 'procedurally_generated_pixel_ring'
    else:
        # 只允许公开训练分区的一幅几何，不读取电流/响应标签或验证/测试图案。
        import h5py
        from stationary_cnn.data import file_sha256
        if not 0 <= training_index < 5000:
            raise ValueError('training-index 必须落在公开训练分区0:5000')
        if file_sha256(data_path) != DATA_SHA256:
            raise ValueError('公开训练H5的SHA-256不符')
        with h5py.File(data_path, 'r') as source:
            if int(source['split'][training_index]) != 1:
                raise ValueError('所选行不是训练图案')
            mask = source['masks'][training_index].astype(bool)
        example_name = 'public_training_pattern'
    torch = initialize_torch(device, threads=threads)
    predictor = load_predictor(model_path, torch, _active_rows, device)
    reference, exact_current, _, diagnostic, operator = solve(mask, frequency, N=24, Q=192, M=1, method='dense')
    predicted = predictor.predict(mask, [frequency])[:, :, 0]
    _, correction, snapshots = refine(operator, predicted, tolerance=tolerance,
        maximum_iterations=maximum_iterations, levels=(tolerance,), fixed=())
    # 只使用逐端口首次达标快照，不能用更晚、更准确的末次电流替代。
    corrected = snapshots[0]
    arrays = dict(mask=mask, reference_current=exact_current, raw_current=predicted,
                  corrected_current=corrected, reference_GSM=reference)
    errors = {}
    for name, current in (('raw', predicted), ('corrected', corrected)):
        errors[name] = {}
        if not np.all(np.isfinite(current)):
            errors[name] = {'criterion_attained': False, 'reason': 'Some incoming ports did not reach the specified true residual'}
            continue
        for method in ('projection', 'reciprocal', 'stationary'):
            matrix, _, residual = extract(operator, current, method=method)
            if not np.all(np.isfinite(matrix)):
                raise ArithmeticError('Nonfinite extracted GSM')
            arrays[f'{name}_{method}_GSM'] = matrix
            errors[name][method] = {'relative_GSM_error': float(np.linalg.norm(matrix - reference) / np.linalg.norm(reference)),
                                  'relative_EFIE_residual': residual['relative_plus']}
    np.savez_compressed(target / 'example.npz', **arrays)
    result = dict(example=example_name, frequency_GHz=frequency,
                  training_index=int(training_index) if data_path is not None else None,
                  N=24, Q=192, M=1, device=device,
                  same_currents_for_all_three_methods=True, errors=errors,
                  COCG_tolerance=tolerance, COCG_maximum_iterations=maximum_iterations,
                  corrected_snapshot='first_true_residual_hit_per_port',
                  all_ports_attained=bool(np.all(correction['level_hit'][:,0])),
                  COCG_failed_ports=int(np.count_nonzero(correction['broken'])),
                  COCG_unconverged_ports=int(np.count_nonzero(~correction['converged'])),
                  reference_residual=diagnostic['plus']['maximum_column_residual'])
    (target / 'example.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result

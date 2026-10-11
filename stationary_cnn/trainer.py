"""周期 CNN 的完整电流场训练、固定验证和真实成本记录。

本模块不在导入时加载 Torch，也不查找本机路径。调用方显式提供 Torch、
H5 路径、输出目录和电磁有效行回调。私有实验入口及公开命令行调用此同一核心。
每次参数更新最多含四幅图案的全部三频率、十八端口及全部有效电流行。
末批保留剩余图案，例如 N=50 的末批含两幅，不丢弃也不补重复图案。
一个 epoch 是训练图案的完整遍历，不能与系数随机抽样的轮数混称。
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .data import file_sha256, load_dataset
from .model import ARCHITECTURE, Predictor, export_model, initial_state, make_model, parameter_count
from .objective import loss_torch, target_from_demod


LOSS_NAME = 'full_case_matrix_relative_squared_current'
COST_SCOPE = (
    'cache construction/loading, train-only RMS, initialization, training, '
    'scheduled full validation, weights/full-state saves, synchronized regular '
    'CSV and summary writes; excludes imports, source hash/preflight, testing '
    'and the final cost-stamp write itself'
)


def updates_per_epoch(train_count, batch_topologies=4):
    """完整遍历的实际更新数。N=50 为 13 次，N=100 为 25 次。"""
    if train_count < 1 or batch_topologies < 1:
        raise ValueError('Positive topology count and batch capacity required')
    return (int(train_count) + int(batch_topologies) - 1) // int(batch_topologies)


def epoch_groups(train_count, rng, batch_topologies=4):
    """仅生成一次置换，然后按容量切片，最后一组可小于容量。

    所有图案每轮恰好出现一次。N 可被 4 整除时，组内顺序及连接后的
    little-endian int64 字节流与原 permutation(N).reshape(-1,4) 相同。
    """
    updates_per_epoch(train_count, batch_topologies)
    order = rng.permutation(train_count)
    return tuple(order[first:first + batch_topologies]
                 for first in range(0, train_count, batch_topologies))


def update_group_digest(digest, groups):
    """以实际组顺序记录全部索引，不为短组补零或重复索引。

    恢复重放使用同一函数，继续核对完整置换的 SHA 和采样 RNG 状态。
    """
    for group in groups:
        digest.update(np.asarray(group, dtype='<i8').tobytes())


def batch_arrays(dataset, group):
    """构造当前真实组的完整 54 场输入与标签，规模只由 len(group) 决定。

    四图案组产生 216 场、12 个图案频率 case，两图案末组产生 108 场、
    6 个 case。每 case 的全部有效行和 18 端口都保留，损失公式不变。
    返回顺序对应网络 mask/query/active 及完整 target/denominator。
    """
    group = np.asarray(group, dtype=np.int64)
    if group.ndim != 1 or not 1 <= len(group) <= 4 or len(np.unique(group)) != len(group):
        raise ValueError('One to four distinct topologies required')
    targets, denominators = zip(*(target_from_demod(dataset['demod_views'][int(t)],
        dataset['rows'][int(t)], dataset['scale'], dataset['norm_squared'][int(t)]) for t in group))
    masks = np.repeat((dataset['masks'][group].astype(np.float32) * 2 - 1)[:, None], 54, axis=0)
    active = np.repeat(dataset['host_active'][group], 54, axis=0)
    query = np.tile(dataset['query54'], (len(group), 1))
    return masks, query, active, np.concatenate(targets), np.stack(denominators).astype(np.float32)


def case_weighted_epoch_loss(losses, case_counts):
    """按实际图案频率 case 数汇总训练日志，不将两图案末批与四图案批等权。

    每 update 的反向传播仍使用原 loss_torch 对当前 case 的均匀平均。
    此函数只定义全轮报告量。所有组相同时沿用原 fsum(loss)/组数算式，
    因此 N=100 的旧四图案日志结果也保持完全相同。
    """
    if not losses or len(losses) != len(case_counts) or any(count <= 0 for count in case_counts):
        raise ValueError('Positive matching loss/case records required')
    if len(set(case_counts)) == 1:
        return math.fsum(losses) / len(losses)
    return math.fsum(value * count for value, count in zip(losses, case_counts)) / sum(case_counts)


def _jsonable(value):
    """将运行元信息转换为 JSON 基本类型，不把张量或大缓存写入日志。"""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    """原子替换并同步到磁盘，避免中断留下半个 JSON。"""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_csv(path, records, fields):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sync_file(path):
    # NumPy/Torch 的保存函数返回后再同步实际文件，计入检查点成本。
    with Path(path).open('r+b') as stream:
        os.fsync(stream.fileno())


def _cpu_tree(torch, value):
    """完整 Adam 状态保存为 CPU 张量，恢复时交由 optimizer.load_state_dict。

    不只保存网络权重：一阶/二阶矩、步数与随机状态决定续训轨迹。
    """
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, item) for item in value)
    return value


def process_resources():
    """实际进程内存，而非按数组形状估算。所有内存字段以 MiB 为单位。

    Windows 返回操作系统累计 PeakWorkingSetSize 和当前 PrivateUsage。
    其他平台返回 getrusage 累计最大 RSS，不能冒充 Windows 私有提交内存。
    """
    mib = 2 ** 20
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                        ('PrivateUsage', ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'os_peak_working_set_MiB': counters.PeakWorkingSetSize / mib,
                'os_working_set_MiB': counters.WorkingSetSize / mib,
                'os_private_commit_MiB': counters.PrivateUsage / mib,
                'os_peak_private_commit_MiB': counters.PeakPagefileUsage / mib}
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'os_peak_working_set_MiB': rss / mib if sys.platform == 'darwin' else rss / 1024.0}


def _resources(torch, device, sampler):
    result = dict((sampler or process_resources)())
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
        result.update(gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / (2 ** 20),
                      gpu_peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / (2 ** 20))
    return _jsonable(result)


def _validate(torch, net, dataset, callback, device, epoch):
    """在固定验证图案上报告真实复电流误差，三频率各占同样权重。

    训练目标是相对误差的平方，报告值是每 case 相对 Frobenius 范数本身。
    不取 pooled MSE 的平方根，不删除大误差条件，不把频率当作独立拓扑。
    """
    net.eval()
    predictor = Predictor(torch, net, dataset['scale'], callback, device)
    records = []
    ntrain = dataset['n_train']
    for local in range(dataset['validation_count']):
        index = ntrain + local
        predicted = predictor.predict(dataset['masks'][index], dataset['frequencies'])
        # 保持原始 complex64 参考范数的统计口径；预测减参考仍按 complex128
        # 计算差值。不能仅在新模型验证时改换参考分母的浮点精度。
        reference = np.asanyarray(dataset['validation_views'][local]).transpose(0, 2, 1)
        if predicted.shape != reference.shape or not np.all(np.isfinite(predicted)):
            raise ArithmeticError('Nonfinite or incorrectly shaped validation current')
        for frequency_index, frequency in enumerate(dataset['frequencies']):
            error = float(np.linalg.norm(predicted[:, :, frequency_index] - reference[:, :, frequency_index]) /
                          np.linalg.norm(reference[:, :, frequency_index]))
            if not math.isfinite(error):
                raise ArithmeticError('Nonfinite full-case relative current error')
            records.append({'training_count': ntrain, 'epoch': epoch,
                            'topology_index': int(dataset['selected'][index]),
                            'frequency_GHz': float(frequency), 'relative_current_error': error})
    errors = np.asarray([row['relative_current_error'] for row in records], dtype=np.float64)
    summary = {'mean_current_error': math.fsum(errors.tolist()) / len(errors),
               'median_current_error': float(np.median(errors)),
               'p95_current_error': float(np.quantile(errors, .95)),
               'maximum_current_error': float(errors.max()),
               'validation_topologies': dataset['validation_count'],
               'validation_conditions': len(records)}
    net.train()
    return records, summary


def train(torch, data_path, output_dir, train_count, epochs=20,
          checkpoint_epochs=(5, 10, 15, 20), active_rows=None,
          expected_data_sha256=None, resume_path=None, metadata=None,
          resource_sampler=None, after_update=None, after_epoch=None,
          device='cuda', initial_seed=120026, sampler_seed=2026101102,
          learning_rate=.001, minimum_learning_rate=.0001, threads=4,
          validation_start=5000, validation_count=100):
    """训练一次独立规模并返回摘要。

    active_rows(mask) 提供 N=24 原电磁编号的有效行。resource_sampler() 返回
    实际资源字典。after_update(epoch, epoch_update, total_updates) 可抛异常停止。
    after_epoch(epoch, total_updates, resources) 返回 None 继续，返回字符串则在
    保存完整状态后正常暂停。暂停记录保留实际成本，不作为一次无中断实验。
    resume_path 只读取使用者自己生成的完整状态，未恢复任何外部分发的权重。
    """
    if active_rows is None:
        raise ValueError('An explicit electromagnetic active_rows callback is required')
    if train_count < 1 or epochs < 1 or threads < 1:
        raise ValueError('train_count, epochs and threads must be positive')
    if not (0 < minimum_learning_rate <= learning_rate):
        raise ValueError('Require 0 < minimum_learning_rate <= learning_rate')
    checkpoints = tuple(sorted(set(int(value) for value in checkpoint_epochs) | {int(epochs)}))
    if checkpoints[0] < 1 or checkpoints[-1] > epochs:
        raise ValueError('Checkpoint epochs must lie in the actual training plan')
    data_path, output_dir = Path(data_path), Path(output_dir)
    actual_data_sha = file_sha256(data_path)
    if expected_data_sha256 and actual_data_sha != expected_data_sha256:
        raise ValueError('Source data SHA-256 does not match the frozen dataset')
    metadata = _jsonable(metadata or {})
    signature = dict(architecture=ARCHITECTURE, training_count=int(train_count), epochs=int(epochs),
                     checkpoint_epochs=list(checkpoints), batch_topologies=4,
                     last_batch_policy='retain_remaining_distinct_topologies_no_drop_no_padding',
                     updates_per_epoch=updates_per_epoch(train_count),
                     epoch_loss_weighting='actual_topology_frequency_case_count',
                     initial_seed=int(initial_seed), sampler_seed=int(sampler_seed),
                     learning_rate=float(learning_rate), minimum_learning_rate=float(minimum_learning_rate),
                     validation_start=int(validation_start), validation_count=int(validation_count),
                     data_sha256=actual_data_sha, loss_name=LOSS_NAME, device=str(device),
                     metadata=metadata)
    if resume_path is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise ValueError('A resume must retain its original run directory and caches')

    # 数据身份检查及调用方的运行环境预检不属于训练墙钟。每次正式规模在新进程
    # 和独立目录创建自己的缓存，缓存构建/读取及训练尺度归约均从这里开始计时。
    started = time.perf_counter()
    dataset = load_dataset(data_path, output_dir, train_count, validation_start,
                           validation_count, expected_data_sha256=actual_data_sha)
    for mask, ids in zip(dataset['masks'], dataset['rows']):
        if not np.array_equal(np.asarray(active_rows(mask), dtype=np.int64), ids):
            raise ValueError('H5 active rows differ from the supplied electromagnetic interface')
    arrays, initial_rng_state = initial_state(initial_seed)
    net = make_model(torch, arrays, device)
    torch.set_num_threads(threads)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, betas=(.9, .999),
                                 eps=1e-8, weight_decay=1e-5, foreach=False, fused=False)
    rng = np.random.default_rng(sampler_seed)
    group_digest = hashlib.sha256()
    epoch_history, resource_history, checkpoint_history, validation_history = [], [], [], []
    start_epoch, total_updates, cost_offset, pause_count = 0, 0, 0.0, 0
    terminal_resource_notice = None
    if resume_path is not None:
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('Resume plan, data or architecture does not match the saved trajectory')
        net.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch, total_updates = int(state['epoch']), int(state['total_updates'])
        if start_epoch >= epochs:
            raise ValueError('The saved trajectory already completed its planned epochs')
        if total_updates != start_epoch * updates_per_epoch(train_count):
            raise ValueError('Saved epoch/update count is inconsistent')
        # hashlib 无可移植的状态序列化。仅重放几何置换恢复摘要，不读取任何标签。
        for _ in range(start_epoch):
            groups = epoch_groups(train_count, rng)
            update_group_digest(group_digest, groups)
        if group_digest.hexdigest() != state['topology_group_sha256'] or _jsonable(rng.bit_generator.state) != state['sampler_rng_state']:
            raise ValueError('Saved sampler state differs from exact epoch replay')
        rng.bit_generator.state = state['sampler_rng_state']
        torch.set_rng_state(state['torch_cpu_rng_state'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(state['torch_cuda_rng_states'])
        epoch_history = state['epoch_history']
        resource_history = state['resource_history']
        checkpoint_history = state['checkpoint_history']
        validation_history = state['validation_history']
        pause_count = int(state['pause_count'])
        cost_offset = float(state['active_wall_seconds'])
        # 已落盘的单次 cost stamp 比 state 中较早的临时时间精确。恢复时优先它。
        stamp_path = output_dir / f"cost_epoch_{start_epoch}.json"
        if stamp_path.exists():
            stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
            if int(stamp['epoch']) != start_epoch:
                raise ValueError('Resume cost stamp has inconsistent epoch')
            cost_offset = float(stamp['cumulative_active_wall_seconds'])

    def elapsed():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        return cost_offset + time.perf_counter() - started

    def summary(status, epoch, reason=None):
        return dict(status=status, architecture=ARCHITECTURE, parameter_count=parameter_count(),
                    completed_epochs=epoch, planned_epochs=epochs, total_updates=total_updates,
                    training_count=train_count, updates_per_epoch=updates_per_epoch(train_count),
                    last_batch_topologies=((train_count - 1) % 4) + 1,
                    validation_topologies=validation_count, validation_conditions=validation_count * 3,
                    output_scale=float(dataset['scale']), normalization='train-only global RMS',
                    loss_name=LOSS_NAME, checkpoint_epochs=list(checkpoints),
                    topology_group_sha256=group_digest.hexdigest(), pause_count=pause_count,
                    pause_reason=reason, metadata=metadata, signature=signature,
                    terminal_resource_notice=terminal_resource_notice,
                    cache_reused=bool(dataset['cache_reused']),
                    demod_cache_bytes=dataset['demod_cache_bytes'],
                    validation_reference_cache_bytes=dataset['raw_validation_cache_bytes'],
                    cost_scope=COST_SCOPE, checkpoints=checkpoint_history,
                    cost_stamp_file=f'cost_epoch_{epoch}.json',
                    environment=dict(torch=str(torch.__version__), numpy=str(np.__version__),
                                     device=str(device), threads=threads,
                                     deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                                     cuda_version=str(torch.version.cuda)))

    def save_checkpoint(epoch, status, reason=None, run_validation=True):
        nonlocal validation_history, checkpoint_history
        # 正常计划检查点有完整验证；资源暂停只存当前轨迹，不临时新增一轮
        # 300-case 验证，也不把暂停位置变成新的模型选择候选。
        if run_validation:
            records, statistics = _validate(torch, net, dataset, active_rows, device, epoch)
            validation_history.extend(records)
        filename = f'checkpoint_{epoch}.npz' if run_validation else f'paused_checkpoint_{epoch}.npz'
        model_path, state_path = output_dir / filename, output_dir / f'complete_state_{epoch}.pt'
        model_arrays = export_model(net, dataset['scale'], N=dataset['N'], M=dataset['M'], Q=dataset['Q'])
        model_arrays.update(training_count=np.asarray(train_count), training_epochs=np.asarray(epochs),
                            saved_epoch=np.asarray(epoch), loss_name=np.asarray(LOSS_NAME),
                            training_protocol_sha256=np.asarray(str(metadata.get('protocol_sha256', ''))))
        np.savez_compressed(model_path, **model_arrays)
        _sync_file(model_path)
        resources = _resources(torch, device, resource_sampler)
        if run_validation:
            cp = dict(epoch=epoch, training_count=train_count, total_updates=total_updates,
                      model_file=model_path.name, state_file=state_path.name,
                      cost_stamp_file=f'cost_epoch_{epoch}.json', **statistics, **resources)
            checkpoint_history.append(cp)
        state = dict(schema='stationary-cnn-complete-state-v1', signature=signature, metadata=metadata,
                     model_state=_cpu_tree(torch, net.state_dict()), optimizer_state=_cpu_tree(torch, optimizer.state_dict()),
                     epoch=epoch, total_updates=total_updates, initial_rng_state=_jsonable(initial_rng_state),
                     sampler_rng_state=_jsonable(rng.bit_generator.state), torch_cpu_rng_state=torch.get_rng_state(),
                     torch_cuda_rng_states=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                     topology_group_sha256=group_digest.hexdigest(), epoch_history=epoch_history,
                     resource_history=resource_history, checkpoint_history=checkpoint_history,
                     validation_history=validation_history, active_wall_seconds=elapsed(), pause_count=pause_count)
        torch.save(state, state_path)
        _sync_file(state_path)
        _write_csv(output_dir / 'validation_conditions.csv', validation_history,
                   ('training_count', 'epoch', 'topology_index', 'frequency_GHz', 'relative_current_error'))
        _write_csv(output_dir / 'training_history.csv', epoch_history,
                   ('epoch', 'total_updates', 'learning_rate', 'matrix_relative_squared_loss', 'maximum_gradient_norm'))
        resource_fields = sorted(set().union(*(row.keys() for row in resource_history)))
        _write_csv(output_dir / 'resource_history.csv', resource_history, resource_fields)
        cp_fields = sorted(set().union(*(row.keys() for row in checkpoint_history)))
        _write_csv(output_dir / 'checkpoint_summary.csv', checkpoint_history, cp_fields)
        report = summary(status, epoch, reason)
        _write_json(output_dir / 'training_summary.json', report)
        # 唯一成本边界在上述全部常规文件已经同步后。e/f 使用同一 stamp。
        # stamp 自身的写入不能包含在它所记录的时间中，故单独明示排除。
        boundary_resources = _resources(torch, device, resource_sampler)
        stamp = dict(epoch=epoch, cumulative_active_wall_seconds=elapsed(),
                     cost_scope=COST_SCOPE, regular_outputs_synchronized=True,
                     **boundary_resources)
        _write_json(output_dir / f'cost_epoch_{epoch}.json', stamp)
        return report, stamp

    net.train()
    for epoch in range(start_epoch + 1, epochs + 1):
        lr = (minimum_learning_rate + (learning_rate - minimum_learning_rate) *
              (.5 + .5 * math.cos(math.pi * (epoch - 1) / max(1, epochs - 1))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        groups = epoch_groups(train_count, rng)
        update_group_digest(group_digest, groups)
        epoch_loss, epoch_case_counts, maximum_gradient_norm = [], [], 0.0
        for epoch_update, group in enumerate(groups, 1):
            arrays = batch_arrays(dataset, group)
            batch = [torch.from_numpy(np.ascontiguousarray(value)).to(device) for value in arrays]
            optimizer.zero_grad(set_to_none=True)
            predicted = net(batch[0], batch[1], batch[2])
            loss = loss_torch(torch, predicted, batch[3], batch[4])
            if not bool(torch.isfinite(loss).item()):
                raise ArithmeticError('Nonfinite full-matrix training loss')
            loss.backward()
            # 梯度裁剪采用逐参数 float64 范数平方累加，保持两类训练监督的
            # 同一数值约定。Torch 默认 float32 范数或 foreach epsilon 会改轨迹。
            norm_squared = math.fsum(float(parameter.grad.detach().double().square().sum().item())
                                     for parameter in net.parameters() if parameter.grad is not None)
            norm = math.sqrt(norm_squared)
            if not math.isfinite(norm):
                raise ArithmeticError('Nonfinite full-matrix gradient norm')
            if norm > 5.:
                with torch.no_grad():
                    for parameter in net.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(5. / (norm + 1e-12))
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            optimizer.step()
            epoch_loss.append(float(loss.detach().item()))
            epoch_case_counts.append(3 * len(group))
            total_updates += 1
            del batch, predicted, loss, arrays
            if after_update is not None:
                after_update(epoch, epoch_update, total_updates)
        epoch_history.append(dict(epoch=epoch, total_updates=total_updates, learning_rate=lr,
            matrix_relative_squared_loss=case_weighted_epoch_loss(epoch_loss, epoch_case_counts),
            maximum_gradient_norm=maximum_gradient_norm))
        resources = _resources(torch, device, resource_sampler)
        resource_history.append(dict(epoch=epoch, total_updates=total_updates, **resources))
        if epoch == epochs:
            # 训练终点始终完成唯一终点验证与保存。资源回调的提示仍保留，
            # 不能把已经完成的 epoch 标成无法续训的 paused 状态。
            terminal_resource_notice = (after_epoch(epoch, total_updates, resources)
                                        if after_epoch is not None else None)
            report, stamp = save_checkpoint(epoch, 'complete')
        else:
            if epoch in checkpoints:
                report, stamp = save_checkpoint(epoch, 'running')
                resources = _resources(torch, device, resource_sampler)
            pause_reason = after_epoch(epoch, total_updates, resources) if after_epoch is not None else None
            if pause_reason is not None:
                pause_count += 1
                report, stamp = save_checkpoint(epoch, 'paused', str(pause_reason), run_validation=False)
                return dict(report, cost_stamp=stamp)
    return dict(report, cost_stamp=stamp)

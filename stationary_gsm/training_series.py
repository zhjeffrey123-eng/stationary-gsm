"""六个训练规模各用新进程，固定验证图案用于选择规模。"""
from pathlib import Path
import json
import math
import subprocess
import sys

SIZES=(100,300,600,1000,2000,5000)
EPOCHS=(10,20,40,60,80)


def train_series(labels_path,target):
    """串行新进程，保持四线程、64单元及相同80轮预算。

    每个子进程独立计算真实OS累计峰值；等待前一规模结束再启动下一规模。
    标签生成、父进程调度和等待时间不计入子进程训练wallclock。
    """
    target=Path(target)
    target.mkdir(parents=True,exist_ok=False)
    for n in SIZES:
        subprocess.run([sys.executable,'-m','stationary_gsm','train-profile',
                        '--data',str(Path(labels_path).resolve()),
                        '--out',str((target/f'training_N{n}').resolve()),
                        '--training-count',str(n),'--hidden','64','--epochs','80',
                        '--batches','128','--batch','4096','--seed','120026',
                        '--subset-seed','20261011','--validation-interval','5',
                        '--checkpoint-epochs','10','20','40','60','80','--threads','4'],
                       check=True)
    return select_validation(target)


def select_validation(target):
    """六规模中选择epoch10至80验证电流中位数下降最多的规模。

    平局容差1e-12，取较小规模。只读取运行中新生成的验证摘要，
    不读取测试集，也不以训练loss或单端口平方误差替代整矩阵电流中位数。
    """
    target=Path(target)
    destination=target/'validation_selection.json'
    if destination.exists():raise FileExistsError('验证选择已冻结，不覆盖已有结果')
    candidates=[]
    common_validation=None
    for n in SIZES:
        run=target/f'training_N{n}'
        summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
        protocol=json.loads((run/'training_protocol.json').read_text(encoding='utf-8'))
        if summary['status']!='training_complete' or summary['epochs']!=80 or summary['test_evaluated_during_training']:
            raise ValueError('六个规模必须均完成固定80轮且不含测试评价')
        if protocol['architecture']!=[71,64,64,2] or protocol['checkpoint_epochs']!=list(EPOCHS):
            raise ValueError('模型结构和检查点不符合固定六规模协议')
        for key,value in {'training_count':n,'terminal_epoch':80,'batches_per_epoch':128,
                          'coefficients_per_batch':4096,'train_seed':120026,
                          'subset_seed':20261011,'validation_interval':5}.items():
            if protocol[key]!=value:raise ValueError('各规模必须保持相同固定训练预算和seed')
        validation=protocol['validation_zero_based_indices']
        if validation!=list(range(5000,5100)) or (common_validation is not None and validation!=common_validation):
            raise ValueError('必须使用同一固定100验证图案')
        common_validation=validation
        records=summary['validation_checkpoint_records']
        if [r['epoch'] for r in records]!=list(EPOCHS) or any(r['n_conditions']!=300 for r in records):
            raise ValueError('各检查点必须包含100验证图案乘3频率，共300条件')
        first=float(records[0]['current_median_relative'])
        last=float(records[-1]['current_median_relative'])
        if not math.isfinite(first) or not math.isfinite(last):
            raise ArithmeticError('验证端点非有限，停止选择')
        candidates.append(dict(training_count=n,validation_median_epoch10=first,
                               validation_median_epoch80=last,validation_median_drop=first-last))
    maximum=max(r['validation_median_drop'] for r in candidates)
    tied=[r['training_count'] for r in candidates if maximum-r['validation_median_drop']<=1e-12]
    selected=min(tied)
    manifest=dict(status='frozen_from_validation_only',selected_training_count=selected,
        training_counts=list(SIZES),checkpoint_epochs=list(EPOCHS),terminal_epoch=80,
        validation_conditions_per_model=300,selection_tie_tolerance=1e-12,
        selection_metric='median relative Frobenius current error at epoch10 minus epoch80',
        selection_rule='maximum positive or negative drop; ties within1e-12 choose smaller training_count',
        candidates=candidates,selection_used_test=False)
    destination.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest

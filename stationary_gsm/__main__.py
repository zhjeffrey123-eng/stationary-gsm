"""训练、标签重建与新模型提取入口；输出目录必须尚不存在。"""
import argparse
import json
import os
from pathlib import Path
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='PEC pixel-sheet training and stationary GSM extraction')
    sub=parser.add_subparsers(dest='command',required=True)
    profiled=sub.add_parser('train-profile',help='磁盘缓存标签、固定子集和检查点，记录Windows实际进程资源')
    profiled.add_argument('--data',type=Path,default=ROOT/'data/training_dataset_5000.h5')
    profiled.add_argument('--out',type=Path,required=True)
    profiled.add_argument('--training-count',type=int,default=600)
    profiled.add_argument('--hidden',type=int,choices=[64,128],default=64)
    profiled.add_argument('--epochs',type=int,default=80)
    profiled.add_argument('--batches',type=int,default=128)
    profiled.add_argument('--batch',type=int,default=4096)
    profiled.add_argument('--seed',type=int,default=120026)
    profiled.add_argument('--subset-seed',type=int,default=20261011)
    profiled.add_argument('--validation-interval',type=int,default=5)
    profiled.add_argument('--checkpoint-epochs',type=int,nargs='*',default=[10,20,40,60,80])
    profiled.add_argument('--threads',type=int,choices=[4],default=4)
    series=sub.add_parser('train-series',help='100/300/600/1000/2000/5000各用新进程串行训练，再冻结验证选择')
    series.add_argument('--data',type=Path,default=ROOT/'data/training_dataset_5000.h5')
    series.add_argument('--out',type=Path,required=True)
    series.add_argument('--threads',type=int,choices=[4],default=4)
    select=sub.add_parser('select-validation',help='仅由六次完整训练生成的固定100验证图案摘要选择规模')
    select.add_argument('--runs',type=Path,required=True)
    select.add_argument('--threads',type=int,choices=[4],default=4)
    labels=sub.add_parser('generate-labels',help='用训练模板里的4400已知几何现场重建标签，复制原600训练/100验证')
    labels.add_argument('--template',type=Path,default=ROOT/'data/training_dataset_5000.h5')
    labels.add_argument('--out',type=Path,required=True)
    labels.add_argument('--threads',type=int,choices=[4],default=4)
    example=sub.add_parser('demo',help='用自己训练的模型和现场EFIE参考解演示同电流三算子')
    example.add_argument('--model',type=Path,required=True)
    example.add_argument('--out',type=Path,required=True)
    example.add_argument('--frequency',type=float,default=7.)
    example.add_argument('--tolerance',type=float,default=.1)
    example.add_argument('--maximum-iterations',type=int,default=64)
    example.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error('threads必须为正整数')
    # 必须先设环境，再导入NumPy/SciPy/H5模块；threadpool_limits同时约束已载入的库。
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[name]=str(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.command=='train-profile':
            from .profile import train_profile
            result=train_profile(args.data,args.out,training_count=args.training_count,hidden=args.hidden,
                epochs=args.epochs,batches=args.batches,batch=args.batch,seed=args.seed,
                subset_seed=args.subset_seed,validation_interval=args.validation_interval,
                checkpoint_epochs=tuple(args.checkpoint_epochs))
        elif args.command=='train-series':
            from .training_series import train_series
            result=train_series(args.data,args.out)
        elif args.command=='select-validation':
            from .training_series import select_validation
            result=select_validation(args.runs)
        elif args.command=='generate-labels':
            from .label_generation import generate_labels
            result=generate_labels(args.template,args.out)
        else:
            from .example import demo
            result=demo(args.model,args.out,args.frequency,args.tolerance,args.maximum_iterations)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

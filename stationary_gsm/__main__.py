"""训练及新模型提取示例；所有生成文件写到尚不存在的新目录。"""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .data import ROOT

def main():
    parser=argparse.ArgumentParser(description='PEC pixel-sheet current training and stationary GSM extraction')
    sub=parser.add_subparsers(dest='command',required=True)
    training=sub.add_parser('train',help='仅用训练split=1更新网络，验证split=2选择模型')
    training.add_argument('--data',type=Path,default=ROOT/'data/training.h5')
    training.add_argument('--out',type=Path,required=True)
    training.add_argument('--hidden',type=int,choices=[64,128],default=128)
    training.add_argument('--epochs',type=int,default=80)
    training.add_argument('--batches',type=int,default=128)
    training.add_argument('--batch',type=int,default=4096)
    training.add_argument('--seed',type=int,default=120026)
    training.add_argument('--validation-interval',type=int,default=5)
    training.add_argument('--checkpoint-epochs',type=int,nargs='*',default=[10])
    training.add_argument('--threads',type=int,default=4)
    example=sub.add_parser('demo',help='用自己训练的模型及现场EFIE参考解演示同电流三算子')
    example.add_argument('--model',type=Path,required=True)
    example.add_argument('--out',type=Path,required=True)
    example.add_argument('--frequency',type=float,default=7.)
    example.add_argument('--tolerance',type=float,default=.1)
    example.add_argument('--maximum-iterations',type=int,default=64)
    example.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1: parser.error('threads必须为正整数')
    with threadpool_limits(limits=args.threads):
        if args.command=='train':
            from .network import train
            result=train(args.data,args.out,epochs=args.epochs,batches=args.batches,
                         batch=args.batch,seed=args.seed,validation_interval=args.validation_interval,
                         hidden=args.hidden,checkpoint_epochs=tuple(args.checkpoint_epochs))
        else:
            from .example import demo
            result=demo(args.model,args.out,args.frequency,args.tolerance,args.maximum_iterations)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()

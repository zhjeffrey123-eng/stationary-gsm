"""命令行入口。所有复算结果写到用户指定的新目录。"""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .data import ROOT,PAPER


def main():
    parser = argparse.ArgumentParser(description='Periodic PEC metasurface stationary GSM extraction')
    parser.add_argument('command',choices=['verify','demo','audit','train','labels','novel-labels',
        'evaluate','stack','fine','fields','figures'])
    parser.add_argument('--data-root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--network-only',action='store_true')
    parser.add_argument('--model',type=Path,default=None)
    parser.add_argument('--N',type=int,nargs='+',default=[48])
    parser.add_argument('--frequencies',type=float,nargs='+',default=[7.])
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,40.])
    parser.add_argument('--ratios',type=int,nargs='+',default=[8])
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--batches',type=int,default=128)
    parser.add_argument('--batch',type=int,default=4096)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    if args.threads<1 or args.count<1:
        parser.error('threads 和 count 必须为正整数')
    with threadpool_limits(limits=args.threads):
        if args.command in ('verify','demo','audit'):
            from .validation import statistics,demo,regression
            result = {'verify':lambda:statistics(args.data_root),'demo':demo,
                      'audit':lambda:regression(args.data_root)}[args.command]()
            if args.out:
                args.out.parent.mkdir(parents=True,exist_ok=True)
                if args.out.exists():
                    parser.error('输出报告已存在，请使用新路径')
                args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            if args.out is None:
                parser.error('复算需要 --out 指定尚不存在的新目录')
            from . import workflows as w
            if args.command=='train':
                from .network import train
                result = train(args.data_root/'output/pec8x8_trial/dataset.h5',args.out,
                    epochs=args.epochs,batches=args.batches,batch=args.batch)
            elif args.command in ('labels','novel-labels'):
                result = w.make_labels(args.out,args.data_root,args.command=='novel-labels',args.count)
            elif args.command=='evaluate':
                result = w.evaluate(args.out,args.data_root,args.count,not args.network_only,args.model)
            elif args.command=='stack':
                result = w.stack_validation(args.out,args.data_root,args.full)
            elif args.command=='fine':
                result = w.fine_validation(args.out,args.data_root,args.N,args.frequencies,args.angles,args.ratios)
            elif args.command=='fields':
                result = w.field_maps(args.out,args.data_root)
            elif args.command=='figures':
                from .figures import render
                result = render(args.data_root,args.out)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':
    main()

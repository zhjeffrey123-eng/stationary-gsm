"""从用户提供的电流文件提取GSM，或运行一个独立的算子使用示例。"""
import argparse, json
from pathlib import Path
from threadpoolctl import threadpool_limits

def main():
    parser=argparse.ArgumentParser(description='Periodic PEC current-to-GSM extraction')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('extract',help='读取自己的NPZ电流与几何，执行同电流三方法提取')
    run.add_argument('--input',type=Path,required=True)
    run.add_argument('--out',type=Path,required=True)
    demo=commands.add_parser('demo',help='现场生成单层算子示例，不读取训练或测试数据库')
    demo.add_argument('--out',type=Path,required=True)
    demo.add_argument('--frequency',type=float,default=7.)
    demo.add_argument('--N',type=int,default=24)
    demo.add_argument('--Q',type=int,default=192)
    demo.add_argument('--current-error',type=float,default=.02)
    args=parser.parse_args()
    from .example import from_currents, demo as demonstration
    with threadpool_limits(limits=4):
        result=(from_currents(args.input,args.out) if args.command=='extract' else
                demonstration(args.out,args.frequency,args.N,args.Q,args.current_error))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()

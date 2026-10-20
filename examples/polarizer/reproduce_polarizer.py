"""Recompute reference responses or the identical-current readout comparison."""
from pathlib import Path
import argparse,json,csv
from dataclasses import replace
from . import polarizer_model as c
import numpy as np
from .polarizer_currents import first_true_gmres
def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["reference","matched"],default="reference")
    parser.add_argument("--geometry",type=Path,default=Path(__file__).resolve().parents[2]/"data"/"polarizer"/"polarizer_geometry.json")
    parser.add_argument("--frequencies",help="Comma-separated physical GHz; default41points10-11")
    parser.add_argument("--N",type=int)
    parser.add_argument("--output",type=Path,default=Path("polarizer_recomputed"))
    args=parser.parse_args(argv);geo=json.loads(args.geometry.read_text())
    masks=np.asarray(geo["masks"]);shifts=np.asarray(geo["origin_shifts_reference_m"]);gaps=np.asarray(geo["gaps_m"])
    freq=np.array([float(x)for x in args.frequencies.split(",")])if args.frequencies else np.array(geo["frequency_GHz"])
    N=args.N or 48;args.output.mkdir(parents=True,exist_ok=False)
    rows=[];raw={"frequency_GHz":freq}
    for fi,f in enumerate(freq):
        if args.mode=="reference":
            S,d,uv=c.coupled(masks,shifts,gaps,float(f),N);outputs={"coupled":S};raw[f"f{fi}_S_lab"]=S;raw[f"f{fi}_S_uv"]=uv
        else:
            sheets={m:[]for m in["exact","projection","reciprocal","stationary"]};cache={}
            for mask,shift in zip(masks,shifts):
                key=(mask.tobytes(),shift.tobytes())
                if key not in cache:
                    a=c.ph.sheet(mask,float(f)*c.SCALE,N,8*N,1,0,0);p=c.translated_ports(float(f),N,1,shift);a=replace(a,p=p,m=p)
                    B=p["B"][a.ids];X=c.solve(c.ph.block(a.plus,a.ids,a.ids),B,check_finite=False);Y,diag=first_true_gmres(a)
                    cache[key]={"exact":c.ph.assemble_gsm(p["C"][:,a.ids]@X),**{m:c.extraction.extract(a,Y,method=m)[0]for m in["projection","reciprocal","stationary"]}}
                for m in sheets:sheets[m].append(cache[key][m])
            pairs={m:c.cascade(ss,np.arange(5),gaps,float(f),N,1)for m,ss in sheets.items()};outputs={m:s[1]for m,s in pairs.items()}
            for m,(full,S)in pairs.items():raw[f"f{fi}_{m}_full36"]=full;raw[f"f{fi}_{m}_lab"]=S
        for method,S in outputs.items():
            metrics=c.metrics(S)
            for pol in["x","y"]:
                row={"frequency_GHz":float(f),"method":method,"input":pol,**metrics[pol]}
                if args.mode=="matched":
                    row["complex4_GSM_error_matched"]=float(np.linalg.norm(S-outputs["exact"])/np.linalg.norm(outputs["exact"]))
                    row["complex36_GSM_error_matched"]=float(np.linalg.norm(pairs[method][0]-pairs["exact"][0])/np.linalg.norm(pairs["exact"][0]))
                rows.append(row)
        print(float(f),"GHz completed",flush=True)
    with(args.output/"response.csv").open("w",newline="",encoding="utf-8")as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(args.output/"matrices.npz",**raw)
if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()

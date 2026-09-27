from __future__ import annotations
import argparse, ast, json, re
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

DEFAULT_THRESHOLDS=[0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,0.98]
S1='source1_entity_id'; MID='matched_entity_id'; TRUTH='matched_entity_ids'; PROB='probability'

def norm(v):
    if v is None or pd.isna(v): return ''
    return str(v).strip()

def parse_ids(v):
    if v is None or pd.isna(v): return []
    s=str(v).strip()
    if not s or s.lower() in {'nan','none','null','[]','()'}: return []
    parsed=None
    if s.startswith(('[','(','{')):
        for parser in (json.loads, ast.literal_eval):
            try:
                x=parser(s)
                if isinstance(x,dict): x=list(x.values())
                if isinstance(x,(list,tuple,set)): parsed=x; break
            except (ValueError,SyntaxError,TypeError,json.JSONDecodeError): pass
    items=parsed if parsed is not None else re.split(r'[,;|]',s)
    out=[]; seen=set()
    for x in items:
        x=norm(x)
        if x and x not in seen: out.append(x); seen.add(x)
    return out

def load_ground_truth(path):
    df=pd.read_csv(path,sep='\t',dtype=str)
    req={S1,TRUTH}; miss=req-set(df.columns)
    if miss: raise ValueError(f'Ground truth missing columns: {sorted(miss)}')
    out={}
    for s1,ids in df[[S1,TRUTH]].itertuples(index=False,name=None):
        s1=norm(s1)
        if s1: out.setdefault(s1,set()).update(parse_ids(ids))
    return out

def load_scored(path):
    df=pd.read_csv(path,sep='\t',dtype=str)
    req={S1,MID,PROB}; miss=req-set(df.columns)
    if miss: raise ValueError(f'Scored candidates missing columns: {sorted(miss)}')
    df=df[[S1,MID,PROB]].copy(); df[S1]=df[S1].map(norm); df[MID]=df[MID].map(norm)
    df[PROB]=pd.to_numeric(df[PROB],errors='coerce')
    if df[PROB].isna().any(): raise ValueError('Found non-numeric probability values.')
    if ((df[PROB]<0)|(df[PROB]>1)).any(): raise ValueError('Probabilities must be in [0,1].')
    return df

def load_candidate_set(path):
    df=pd.read_csv(path,sep='\t',dtype=str)
    miss={S1,MID}-set(df.columns)
    if miss: raise ValueError(f'Candidate set missing columns: {sorted(miss)}')
    return {(norm(a),norm(b)) for a,b in df[[S1,MID]].itertuples(index=False,name=None) if norm(a) and norm(b)}

def sanity_check(scored,candidate_pairs=None):
    dup=int(scored.duplicated([S1,MID]).sum())
    invalid=sum(not re.match(r'^(S2|S3)-',x) for x in scored[MID])
    outside=0 if candidate_pairs is None else sum((a,b) not in candidate_pairs for a,b in scored[[S1,MID]].itertuples(index=False,name=None))
    return {'duplicate_predicted_ids':dup,'invalid_s2_s3_ids':invalid,'predictions_outside_candidate_set':outside}

def f05(p,r):
    d=.25*p+r
    return 0.0 if d==0 else (1.25*p*r)/d

def evaluate_threshold(scored,truth,threshold):
    sel=scored[scored[PROB]>=threshold]
    pred={}
    for a,b in sel[[S1,MID]].itertuples(index=False,name=None): pred.setdefault(a,set()).add(b)
    universe=set(truth)|set(pred); rows=[]
    for s1 in sorted(universe):
        pset=pred.get(s1,set()); tset=truth.get(s1,set()); tp=len(pset&tset)
        p=tp/len(pset) if pset else 0.0; r=tp/len(tset) if tset else 0.0
        rows.append((s1,len(pset),len(tset),tp,p,r,f05(p,r)))
    d=pd.DataFrame(rows,columns=[S1,'predicted_count','ground_truth_count','true_positive_count','precision','recall','f0.5'])
    n=len(universe); wp=sum(bool(pred.get(s,set())) for s in universe)
    return {'threshold':threshold,'macro_precision':d.precision.mean() if n else 0.0,'macro_recall':d.recall.mean() if n else 0.0,'macro_f0.5':d['f0.5'].mean() if n else 0.0,'S1_count':n,'S1_with_prediction':wp,'S1_with_zero_prediction':n-wp},d

def evaluate_thresholds(scored_path,ground_truth_path,candidate_path=None,thresholds=DEFAULT_THRESHOLDS):
    scored=load_scored(scored_path); truth=load_ground_truth(ground_truth_path)
    cps=load_candidate_set(candidate_path) if candidate_path else None
    checks=sanity_check(scored,cps)
    if any(checks.values()): raise ValueError(f'Sanity checks failed: {checks}')
    return pd.DataFrame([evaluate_threshold(scored,truth,float(t))[0] for t in thresholds]),checks

def self_test():
    truth={'S1-1':{'S2-1','S3-1'},'S1-2':{'S2-2'},'S1-3':set()}
    scored=pd.DataFrame([('S1-1','S2-1',.90),('S1-1','S3-1',.80),('S1-2','S2-2',.40)],columns=[S1,MID,PROB])
    a,_=evaluate_threshold(scored,truth,.50); assert a['S1_count']==3 and a['S1_with_prediction']==1 and a['S1_with_zero_prediction']==2
    assert evaluate_threshold(scored,truth,.30)[0]['macro_f0.5']>a['macro_f0.5']
    dup=pd.concat([scored,scored.iloc[[0]]],ignore_index=True); assert sanity_check(dup) ['duplicate_predicted_ids']==1
    bad=scored.copy(); bad.loc[0,MID]='BAD-1'; assert sanity_check(bad)['invalid_s2_s3_ids']==1
    assert sanity_check(scored,{('S1-1','S2-1'),('S1-1','S3-1')})['predictions_outside_candidate_set']==1
    print('PASS: Phase 3 evaluator self-test')
    print('PASS: zero-prediction / zero-ground-truth / multiple-ground-truth S1 handling')
    print('PASS: duplicate, invalid-ID, and outside-candidate-set sanity checks')
    print('PASS: threshold evaluation and macro F0.5')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scored'); ap.add_argument('--ground-truth',default='dataset/train/train_ground_truth.tsv'); ap.add_argument('--candidate-set'); ap.add_argument('--output',default='experiments/phase3_results.csv'); ap.add_argument('--thresholds',nargs='+',type=float,default=DEFAULT_THRESHOLDS); ap.add_argument('--self-test',action='store_true'); args=ap.parse_args()
    if args.self_test: self_test(); return
    if not args.scored: ap.error('--scored is required unless --self-test is used')
    result,checks=evaluate_thresholds(args.scored,args.ground_truth,args.candidate_set,args.thresholds); Path(args.output).parent.mkdir(parents=True,exist_ok=True); result.to_csv(args.output,index=False); print(result.to_string(index=False)); print(checks)
if __name__=='__main__': main()

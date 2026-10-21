"""🐄 Bovine Genomics Pipeline v11.1 — Ultimate Edition (VEP/NCBI/STRING Fix)"""
import gzip, io, os, warnings, zipfile, subprocess, tempfile, shutil, json, time
from collections import Counter
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
from sklearn.linear_model import Ridge
import requests

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐄 Bovine Pipeline v11.1", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")

ARS = {"1":158_534_110,"2":136_231_102,"3":121_005_158,"4":120_000_166,"5":120_089_699,"6":117_806_340,"7":110_682_743,"8":113_384_748,"9":105_708_134,"10":103_308_737,"11":107_310_763,"12":91_163_125,"13":84_246_514,"14":84_648_346,"15":85_207_080,"16":81_726_628,"17":75_176_999,"18":66_059_976,"19":64_089_169,"20":72_042_983,"21":71_599_096,"22":61_416_492,"23":52_531_573,"24":62_384_193,"25":42_959_610,"26":51_680_158,"27":46_772_073,"28":46_333_854,"29":51_319_414}
AUTO = [str(i) for i in range(1,30)]

# ============================================================
# GROUPES PHÉNOTYPIQUES & ADAPTATIFS
# ============================================================
GROUPES_DEFAULT = {
    "Type de production": {
        "LAIT": ["NAR", "NBD", "PTP", "PRS"],
        "VIANDE": ["LJR", "SHO", "YBS"],
        "MIXTE": ["AND", "EBG", "ELN", "NKA", "PSH"]
    },
    "Teneur en gras du lait": {
        "LAIT_GRAS": ["NAR", "PTP", "PRS", "NBD"],
        "LAIT_MAIGRE": ["LJR", "SHO", "YBS"]
    },
    "Cornes": {
        "CORNU": ["AND", "ELN", "PTP", "PRS", "NKA", "PSH"],
        "POLLED": ["SHO", "YBS", "LJR", "NAR", "NBD"]
    },
    "Origine géographique": {
        "ZEBU": ["PTP", "PRS"],
        "TAURIN_EUROPEEN": ["LJR", "SHO", "YBS", "NAR", "NBD", "EZP", "FGN"],
        "TAURIN_AFRICAIN": ["AND", "EBG", "ELN", "NKA", "PSH"]
    },
    "Adaptation climatique": {
        "TROPICAL": ["AND", "EBG", "ELN", "PTP", "PRS", "NKA", "PSH"],
        "TEMPERÉ": ["LJR", "SHO", "YBS", "NAR", "NBD", "EZP", "FGN"]
    },
    "Rusticité / Trypanotolérance": {
        "RUSTIQUE_AFRICAIN": ["AND", "EBG", "ELN"],
        "SENSIBLE_EUROPEEN": ["LJR", "NAR", "SHO", "YBS", "NBD"]
    },
    "Format corporel (Taille)": {
        "GRAND_FORMAT": ["LJR", "SHO", "YBS", "NAR", "NBD", "EZP"],
        "PETIT_FORMAT": ["AND", "EBG", "ELN", "PTP", "PRS", "NKA", "PSH"]
    },
    "Vitesse de croissance": {
        "CROISSANCE_RAPIDE": ["LJR", "SHO", "YBS", "NAR", "NBD"],
        "CROISSANCE_LENTE": ["AND", "ELN", "PTP", "PRS", "NKA", "PSH"]
    },
    "Composition de la carcasse": {
        "FORTE_MUSCULATURE": ["LJR", "SHO", "YBS"], 
        "FORT_PERSILLAGE_GRAS": ["NAR", "PTP", "PRS"]
    },
    "Couleur de la robe": {
        "ROBE_CLAIR": ["SHO", "PTP", "NBD", "PSH"], 
        "ROBE_FONCEE": ["LJR", "NAR", "PRS", "AND", "EBG", "ELN"]
    },
    "Statut de sélection": {
        "FORTEMENT_SELECTIONNEE": ["LJR", "SHO", "YBS", "NAR", "NBD"],
        "LOCALE_ADAPTEE": ["AND", "EBG", "ELN", "PTP", "PRS", "NKA", "PSH"]
    }
}

def make_group_map(group_dict):
    mapping = {}
    for grp, races in group_dict.items():
        for r in races: mapping[r.upper()] = grp
    return mapping

def apply_group_mapping(ind_df, group_map, default="AUTRE"):
    df = ind_df.copy()
    df["FID"] = df["FID"].astype(str).str.upper()
    df["FID_GROUPE"] = df["FID"].map(group_map).fillna(default)
    return df

def get_pairs_groups(groups):
    g = sorted(pd.unique(groups).astype(str))
    return g, [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]

# ============================================================
# FONCTIONS CACHE & UTILS
# ============================================================
def _ha(x):
    if not isinstance(x,np.ndarray) or x.size==0: return "empty"
    if x.dtype.kind in ("U","S","O"): return f"a|{x.shape}|{'|'.join(map(str,x.ravel()[:2000]))}"
    return f"{x.shape}|{x.dtype}|{float(np.nansum(x))}|{float(np.nansum(np.abs(x)))}|{float(np.nansum(x*x))}"
def _hn(x):
    try: return _ha(np.asarray(x))
    except: return f"{type(x).__name__}|{len(x) if hasattr(x,'__len__') else 0}"
HF = {np.ndarray:_ha, pd.Series:_hn, pd.Index:_hn}
for _n in ("ArrowStringArray","StringArray","IntegerArray","FloatingArray","BooleanArray","NumpyExtensionArray","PandasArray","DatetimeArray","TimedeltaArray","PeriodArray","IntervalArray","Categorical"):
    _c = getattr(pd.arrays,_n,None) or getattr(pd.core.arrays,_n,None)
    if _c: HF[_c]=_hn
def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    g=gt.astype(np.float32,copy=True); cm=np.nanmean(g,axis=0); cm=np.where(np.isnan(cm),0.0,cm)
    nm=np.isnan(g)
    if nm.any(): g[nm]=np.take(cm, np.where(nm)[1])
    return g

def _autw():
    L=np.array([ARS[c] for c in AUTO],dtype=float); return L/L.sum()
def _chk(c):
    s=str(c).upper().replace("CHR","").replace("CHROMOSOME","")
    if s.isdigit(): return (0,int(s),"")
    sp={"X":100,"Y":101,"MT":102,"M":102}
    return (1,sp[s],"") if s in sp else (2,0,s)
def _rt(raw):
    if len(raw)>=2 and raw[:2]==b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8",errors="replace")
    return raw.decode("utf-8",errors="replace")

def sample_fid_iid(s, fp=2):
    n=str(s).strip()
    for e in (".CEL",".cel",".TXT",".txt",".gz",".GZ"):
        if n.endswith(e): n=n[:-len(e)]
    n=n.replace("(","_").replace(")","").replace("[","_").replace("]","").replace(" ","_").replace("/","_")
    while "__" in n: n=n.replace("__","_")
    p=n.strip("_").split("_"); return ("_".join(p[:fp]) if len(p)>=fp else p[0]), n
def _ca(a):
    a=str(a).strip() if a is not None else "0"; return "0" if a in ("",".","-","NA","nan","None","0","00") else a

def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2]==b"\x1f\x8b" else raw_bytes
    lines=[l.rstrip("\r\n") for l in raw.decode("utf-8",errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    start = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample") or "Sample Filename" in lines[0][:50]) else 0
    samples, gtypes = [], []
    for line in lines[start:]:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts)<2: continue
        samples.append(parts[0].strip()); alleles=[]
        for cell in parts[1:]:
            tk=cell.split()
            if len(tk)==2: alleles.append((_ca(tk[0]),_ca(tk[1])))
            elif len(tk)==1 and len(tk[0])==2 and all(ch in "ACGT0" for ch in tk[0]): alleles.append((tk[0][0],tk[0][1]))
            else: alleles.append(("0","0"))
        gtypes.append(alleles)
    if not samples: raise ValueError("Aucun échantillon")
    c=Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i,r in enumerate(gtypes): gtypes[i]=r[:c]+[("0","0")]*(c-len(r))
    return samples, gtypes, c

def parse_map(b):
    rows=[]
    for line in _rt(b).splitlines():
        p=line.split()
        if len(p)>=4:
            try: cm=float(p[2]) if p[2] not in ("0",".","NA") else 0.0
            except: cm=0.0
            try: bp=int(float(p[3]))
            except: continue
            rows.append({"CHR":str(p[0]),"SNP":p[1],"CM":cm,"BP":bp})
    if not rows: raise ValueError(".map vide")
    return pd.DataFrame(rows)

def parse_ped(b, n_map):
    lines=[l.strip() for l in _rt(b).splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    n_ped=max(0,(len(lines[0].split())-6)//2)
    if n_ped<7: raise ValueError(f"PED a {n_ped} SNPs")
    n=min(n_map,n_ped); ec=6+2*n; fids,iids,rows=[],[],[]
    for line in lines:
        p=line.split()
        if len(p)>=ec: fids.append(p[0]); iids.append(p[1]); rows.append(p[6:6+2*n])
    if not iids: raise ValueError("Aucun individu")
    ac=[Counter() for _ in range(n)]
    for r in rows:
        for j in range(n):
            if r[2*j]!="0" and r[2*j+1]!="0": ac[j].update([r[2*j],r[2*j+1]])
    minor=[min(c,key=c.get) if len(c)>=2 else None for c in ac]
    gt=np.full((len(iids),n),np.nan,dtype=np.float32)
    for i,r in enumerate(rows):
        for j in range(n):
            if minor[j] and r[2*j]!="0" and r[2*j+1]!="0": gt[i,j]=(r[2*j]==minor[j])+(r[2*j+1]==minor[j])
    return gt, pd.DataFrame({"FID":fids,"IID":iids})

def parse_vcf(b):
    samples,rows=[],[]
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"): samples=line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p=line.split("\t")
            if len(p)>=10 and "," not in p[4]:
                fmt=p[8].split(":")
                if "GT" in fmt:
                    gi=fmt.index("GT"); gts=np.full(len(samples),np.nan,dtype=np.float32)
                    for k,s in enumerate(p[9:]):
                        al=s.split(":")[gi].replace("|","/").split("/")
                        if not any(a in (".","") for a in al):
                            try: gts[k]=sum(int(a) for a in al)
                            except: pass
                    rows.append({"CHR":str(p[0]),"SNP":p[2] or f"{p[0]}:{p[1]}","CM":0.0,"BP":int(p[1]),"A1":p[3],"A2":p[4],"GT":gts})
    if not rows: raise ValueError("Aucun variant VCF")
    gt=np.zeros((len(samples),len(rows)),dtype=np.float32)
    for j,r in enumerate(rows):
        gt[:,j]=r["GT"]; v=gt[:,j][~np.isnan(gt[:,j])]
        if len(v)>0 and v.mean()/2.0>0.5: gt[:,j]=np.where(np.isnan(gt[:,j]),np.nan,2.0-gt[:,j])
    return gt, pd.DataFrame({"FID":samples,"IID":samples}), pd.DataFrame([{k:v for k,v in r.items() if k!="GT"} for r in rows])

def parse_gff3(b):
    genes=[]
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p=line.split("\t")
        if len(p)>=9 and p[2] in ("gene","mRNA"):
            a=p[8]; name="Unknown"
            if "Name=" in a: name=a.split("Name=")[1].split(";")[0]
            elif 'gene_name "' in a: name=a.split('gene_name "')[1].split('"')[0]
            elif 'gene_id "' in a: name=a.split('gene_id "')[1].split('"')[0]
            genes.append({"CHR":str(p[0]),"START":int(p[3]),"END":int(p[4]),"GENE":name})
    return pd.DataFrame(genes).drop_duplicates(subset=["CHR","START","END","GENE"])

def parse_gmt(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace"); sets = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        term = parts[0]; genes = [g for g in parts[2:] if g]
        if genes: sets[term] = genes
    return sets

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8",errors="replace"); sep = "\t" if text.count("\t")>text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    if mapping_df is not None:
        p,m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal","IID","id","ID"] if c in p.columns), None)
        if id_c is None: raise ValueError(f"Colonne ID absente.")
        p[id_c]=p[id_c].astype(str)
        map_a = next((c for c in m.columns if c!="IID"), None)
        m[map_a]=m[map_a].astype(str); m["IID"]=m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left")
        ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str); pm["_k"]=pm["IID"].astype(str)
        return ind[["_k","FID","IID"]].merge(pm.drop(columns=["IID"],errors="ignore"),on="_k",how="left").drop(columns="_k"), id_c
    id_c = next((c for c in ["IID","id","ID","sample","Animal","animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError(f"Aucune colonne ID.")
    p = pheno_df.copy(); p[id_c]=p[id_c].astype(str)
    ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str); p["_k"]=p[id_c]
    return ind[["_k","FID","IID"]].merge(p.drop(columns=[id_c]),on="_k",how="left").drop(columns="_k"), id_c

def gen_demo(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng=np.random.default_rng(seed); pops=["AND","EBG","ELN","EZP","FGN","LJR","NAR","NBD","NKA","PRS"][:min(n_pop,10)]
    pa=rng.beta(0.6,0.6,n_snp); gt=np.zeros((n_ind,n_snp),dtype=np.float32); ir=[]; k=0
    for p in pops:
        pk=np.clip(pa+rng.normal(0,0.15,n_snp),0.02,0.98)
        for i in range(n_ind//len(pops)+(1 if pops.index(p)<n_ind%len(pops) else 0)):
            gt[k]=(rng.random(n_snp)<pk).astype(np.int8)+(rng.random(n_snp)<pk).astype(np.int8)
            ir.append({"FID":p,"IID":f"{p}_{i+1:03d}"}); k+=1
    gt[rng.random(gt.shape)<0.02]=np.nan
    ch=rng.choice(AUTO,n_snp,p=_autw()); bp=np.array([rng.integers(1,ARS[c]) for c in ch],dtype=np.int64)
    snp=pd.DataFrame({"CHR":ch,"SNP":[f"rs{i:07d}" for i in range(n_snp)],"CM":bp/1e6,"BP":bp})
    return gt, pd.DataFrame(ir), snp.sort_values(by=["CHR","BP"],key=lambda x: x.map(_chk) if x.name=="CHR" else x).reset_index(drop=True)

def rec_fid(ind):
    df=ind.copy()
    if df["FID"].nunique()<=1: df["FID"]=df["IID"].astype(str).str.split("_").str[0]
    return df

def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p=np.nanmean(g,axis=0)/2.0; return np.minimum(p,1.0-p)
def het(g): return np.nanmean(g==1, axis=1)

def hwe_exact(nh, n0, n2):
    n=nh+n0+n2
    if n==0 or (nh==0 and (n0==0 or n2==0)): return 1.0
    rare=2*min(n0,n2)+nh; mid=(rare*(2*n-rare))//(2*n)
    if mid%2!=rare%2: mid+=1
    pr=np.zeros(rare+1); pr[mid]=1.0; s=1.0
    ch,cr,cc=mid,(rare-mid)//2,n-mid-(rare-mid)//2
    while ch<=rare-2: pr[ch+2]=pr[ch]*4*cr*cc/((ch+2)*(ch+1)); s+=pr[ch+2]; ch+=2; cr-=1; cc-=1
    ch,cr,cc=mid,(rare-mid)//2,n-mid-(rare-mid)//2
    while ch>=2: pr[ch-2]=pr[ch]*ch*(ch-1)/(4*(cr+1)*(cc+1)); s+=pr[ch-2]; ch-=2; cr+=1; cc+=1
    po=pr[nh] if nh<len(pr) else 0.0
    return float(min(pr[pr<=po+1e-7].sum()/s, 1.0))

@cache_data
def hwe_pvals(g):
    pv=np.full(g.shape[1],np.nan)
    n0=((g==0)&~np.isnan(g)).sum(0); n1=((g==1)&~np.isnan(g)).sum(0); n2=((g==2)&~np.isnan(g)).sum(0)
    nv=n0+n1+n2
    if g.shape[1]<=20000:
        for j in range(g.shape[1]):
            if nv[j]>=5: pv[j]=hwe_exact(int(n1[j]),int(n0[j]),int(n2[j]))
    else:
        p=(n1+2*n2)/(2.0*np.where(nv>0,nv,np.nan))
        with np.errstate(invalid="ignore",divide="ignore"):
            eh=2.0*p*(1.0-p)*nv; ok=(eh>=5)&np.isfinite(eh)
            chi2=(np.abs(n1-eh)-0.5)**2/eh; pv[ok]=1.0-stats.chi2.cdf(chi2[ok],df=1)
    return pv

def sync_dims(gt, ind_df, snp_df):
    n=min(gt.shape[0],len(ind_df)); m=min(gt.shape[1],len(snp_df))
    if (gt.shape[0],gt.shape[1])!=(len(ind_df),len(snp_df)):
        st.warning(f"⚠️ Resync: {gt.shape}→({n},{m})")
    return gt[:n,:m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, rem_out=True):
    gt,ind_df,snp_df=sync_dims(gt,ind_df,snp_df); n0,m0=gt.shape; tr={}
    k=miss_snp(gt)<=params["geno"]; gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True); tr["geno"]=int(m0-gt.shape[1])
    k=miss_ind(gt)<=params["mind"]; gt=gt[k]; ind_df=ind_df[k].reset_index(drop=True); tr["mind"]=int(n0-gt.shape[0])
    if gt.size==0: raise ValueError("Tout exclu (missing)")
    m=maf(gt); k=np.isfinite(m)&(m>=params["maf"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    pv=hwe_pvals(gt); k=np.isnan(pv)|(pv>=params["hwe"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    if gt.shape[1]==0: raise ValueError("Tout exclu (MAF/HWE)")
    gt_pre,ind_pre,snp_pre=gt.copy(),ind_df.copy(),snp_df.copy()
    if rem_out:
        h=het(gt); kp=np.ones(len(h),dtype=bool)
        for f in ind_df["FID"].unique():
            msk=(ind_df["FID"]==f).values; sub=h[msk]
            if len(sub)>=5 and np.nanstd(sub)>1e-9:
                z=(sub-np.nanmean(sub))/np.nanstd(sub); idx=np.where(msk)[0]; kp[idx[np.abs(z)>params["het_sd"]]]=False
        gt=gt[kp]; ind_df=ind_df[kp].reset_index(drop=True); tr["het"]=int(len(h)-len(ind_df))
    return gt,ind_df,snp_df,gt_pre,ind_pre,snp_pre,{"n_ind_init":int(n0),"n_snp_init":int(m0),"n_ind_final":int(gt.shape[0]),"n_snp_final":int(gt.shape[1]),"excluded_ind":int(n0-gt.shape[0]),"excluded_snp":int(m0-gt.shape[1]),"trace":tr}

# ============================================================
# GÉNÉTIQUE DE BASE & GWAS
# ============================================================
@cache_data
def fst_snp(gt, labels):
    pops=np.unique(labels); fst=np.full(gt.shape[1],np.nan)
    for j in range(gt.shape[1]):
        pl,nl=[],[]
        for p in pops:
            v=gt[labels==p,j]; v=v[~np.isnan(v)]
            if len(v)>=3: pl.append(v.mean()/2.0); nl.append(len(v))
        if len(pl)>=2:
            pb=np.average(pl,weights=nl); hs=np.average(2*np.array(pl)*(1-np.array(pl)),weights=nl); ht=2*pb*(1-pb)
            if ht>1e-9: fst[j]=(ht-hs)/ht
    return fst

@cache_data
def pca_analysis(gt, nc=10):
    X=impute_mean(gt); X=X-X.mean(0); n=min(nc,X.shape[0]-1,X.shape[1])
    p=PCA(n_components=n); return p.fit_transform(X), p.explained_variance_ratio_*100.0

@cache_data
def mds_analysis(gt, nc=5):
    X=impute_mean(gt); n=X.shape[0]; D=np.zeros((n,n),dtype=np.float32)
    for i in range(n): D[i]=np.abs(X-X[i]).sum(1)/X.shape[1]
    return SklearnMDS(n_components=nc,dissimilarity="precomputed",random_state=42,n_init=1,max_iter=300,normalized_stress=False).fit_transform(D)

@cache_data
def king_mat(gt):
    h=np.nansum(gt==1,axis=1).astype(float); n=gt.shape[0]; K=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            bh=np.nansum((gt[i]==1)&(gt[j]==1)); d=h[i]+h[j]
            K[i,j]=K[j,i]=(d-2*bh)/d if d>0 else 0.0
    np.fill_diagonal(K,0.5); return K

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m=NMF(n_components=K,init="nndsvda",random_state=seed,max_iter=500)
    W=m.fit_transform(np.clip(impute_mean(gt),0.0,2.0)); s=W.sum(1,keepdims=True); s[s==0]=1.0
    return W/s, m.components_

@cache_data
def ld_pruning(gt, w=50, step=5, r2=0.2):
    n=gt.shape[1]
    if n<3: return np.ones(n,dtype=bool)
    keep=np.ones(n,dtype=bool); X=impute_mean(gt); X=X-X.mean(0); i=0
    while i<n:
        we=min(i+step,n)
        while we<n and (we-i)<w:
            if keep[we] and keep[i] and X[:,i].std()>1e-8 and X[:,we].std()>1e-8:
                if np.corrcoef(X[:,i],X[:,we])[0,1]**2>r2: keep[we]=False
            we+=1
        i+=step
    return keep

@cache_data
def ld_decay(gt, bp, max_kb=1000, max_snp=1500, seed=42):
    n=gt.shape[1]
    if n<2: return pd.DataFrame(columns=["dist_kb","r2"])
    idx=np.sort(np.random.default_rng(seed).choice(n,max_snp,replace=False)) if n>max_snp else np.arange(n)
    g=impute_mean(gt[:,idx]); b=np.asarray(bp)[idx].astype(float)
    X=g-g.mean(0); s=g.std(0); s[s<1e-8]=np.nan; X=X/s
    C=(X.T@X)/X.shape[0]; R2=C**2; iu,ju=np.triu_indices(len(idx),k=1)
    d=(b[ju]-b[iu])/1000.0; msk=(d>0)&(d<=max_kb)
    return pd.DataFrame({"dist_kb":d[msk],"r2":R2[iu[msk],ju[msk]]})

@cache_data
def detect_roh(gt, snp_json, min_snps=30, min_kb=500.0):
    try: snp=pd.read_json(io.StringIO(snp_json))
    except: snp=pd.read_json(snp_json)
    n=min(gt.shape[1],len(snp)); gt=gt[:,:n]; snp=snp.iloc[:n].reset_index(drop=True)
    ca=snp["CHR"].astype(str).values; bp=snp["BP"].astype(np.int64).values
    rohs=[]; fb=np.zeros(gt.shape[0]); tot=0.0
    for c in pd.unique(ca):
        idxs=np.where(ca==c)[0]
        if len(idxs)<min_snps: continue
        idxs=idxs[np.argsort(bp[idxs])]; sp=bp[idxs[-1]]-bp[idxs[0]]
        if sp<1: continue
        tot+=sp
        for i in range(gt.shape[0]):
            col=gt[i,idxs]; sk=None
            for k in range(len(idxs)):
                if col[k]==0 or col[k]==2:
                    if sk is None: sk=k
                elif sk is not None:
                    nr=k-sk; lkb=(bp[idxs[k-1]]-bp[idxs[sk]])/1000.0
                    if nr>=min_snps and lkb>=min_kb:
                        rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]),"end_bp":int(bp[idxs[k-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                        fb[i]+=bp[idxs[k-1]]-bp[idxs[sk]]
                    sk=None
            if sk is not None:
                nr=len(idxs)-sk; lkb=(bp[idxs[-1]]-bp[idxs[sk]])/1000.0
                if nr>=min_snps and lkb>=min_kb:
                    rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]),"end_bp":int(bp[idxs[-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                    fb[i]+=bp[idxs[-1]]-bp[idxs[sk]]
    fr=fb/tot if tot>0 else fb
    return (pd.DataFrame(rohs) if rohs else pd.DataFrame(columns=["IID_idx","CHR","start_bp","end_bp","n_snp","length_kb"])), fr

@cache_data
def ne_historical(gt, snp_df, max_snp=2000):
    bp=snp_df["BP"].astype(float).values; ca=snp_df["CHR"].astype(str).values; n=gt.shape[1]
    if n>max_snp:
        idx=np.linspace(0,n-1,max_snp).astype(int); gt=gt[:,idx]; bp=bp[idx]; ca=ca[idx]
    X=impute_mean(gt); X=(X-X.mean(0))/(X.std(0)+1e-9)
    r2l,dl=[],[]; nn=X.shape[1]
    for i in range(nn-1):
        for j in range(i+1,min(i+50,nn)):
            if ca[i]!=ca[j]: continue
            d=abs(bp[j]-bp[i])/1e6
            if d<0.001 or d>50: continue
            r=np.corrcoef(X[:,i],X[:,j])[0,1]; r2l.append(r*r); dl.append(d)
    if not dl: return pd.DataFrame({"GenAgo":[],"Ne":[]})
    r2=np.array(r2l); d=np.array(dl); bins=np.logspace(np.log10(d.min()+1e-6),np.log10(d.max()),20); bc=(bins[:-1]+bins[1:])/2
    r2b=np.array([r2[(d>=bins[k])&(d<bins[k+1])].mean() if ((d>=bins[k])&(d<bins[k+1])).sum()>0 else np.nan for k in range(len(bins)-1)])
    m=np.isfinite(r2b)
    if m.sum()<3: return pd.DataFrame({"GenAgo":[],"Ne":[]})
    ne=np.where(r2b[m]>1e-4,1.0/(4*(bc[m]*0.01)*r2b[m]),np.nan)
    df=pd.DataFrame({"GenAgo":bc[m]*100,"Ne":ne}).dropna(); df=df[df["Ne"]>0].sort_values("GenAgo")
    if len(df)>3: df["Ne"]=df["Ne"].rolling(3,center=True,min_periods=1).mean()
    return df

@cache_data
def fst_pairwise(gt, labels):
    pops=sorted(np.unique(labels)); K=len(pops); M=np.full((K,K),np.nan); np.fill_diagonal(M,0.0)
    masks={p:(labels==p) for p in pops}
    for i in range(K):
        for j in range(i+1,K):
            g1=gt[masks[pops[i]]]; g2=gt[masks[pops[j]]]
            p1=np.nanmean(g1,0)/2.0; p2=np.nanmean(g2,0)/2.0
            n1=(~np.isnan(g1)).sum(0); n2=(~np.isnan(g2)).sum(0)
            dn=np.where(n1+n2>0,n1+n2,np.nan); pb=(p1*n1+p2*n2)/dn
            hs=(2*p1*(1-p1)*n1+2*p2*(1-p2)*n2)/dn; ht=2*pb*(1-pb)
            with np.errstate(divide="ignore",invalid="ignore"): fj=(ht-hs)/ht
            M[i,j]=M[j,i]=float(np.nanmean(fj))
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25):
    fst=fst_snp(gt,labels); snp=snp_df.copy().reset_index(drop=True)
    n=min(len(snp),len(fst)); snp=snp.iloc[:n].reset_index(drop=True); fst=fst[:n]; snp["FST"]=fst; res=[]
    for c in snp["CHR"].unique():
        sub=snp[snp["CHR"]==c].sort_values("BP").reset_index(drop=True)
        if len(sub)<w: continue
        for i in range(0,len(sub)-w+1,step):
            win=sub.iloc[i:i+w]
            if win["FST"].notna().sum()<5: continue
            res.append({"CHR":str(c),"START":int(win["BP"].min()),"END":int(win["BP"].max()),"MID":int((win["BP"].min()+win["BP"].max())/2),"N_SNP":len(win),"MEAN_FST":float(win["FST"].mean()),"MAX_FST":float(win["FST"].max()),"TOP_SNP":win.iloc[win["FST"].argmax()]["SNP"]})
    return pd.DataFrame(res)

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df=win_df.copy(); mu,sd=df["MEAN_FST"].mean(),df["MEAN_FST"].std()
    df["Z_FST"]=(df["MEAN_FST"]-mu)/sd if sd>1e-9 else 0.0; df["OUTLIER"]=df["Z_FST"]>=z_thr
    return df.sort_values("MEAN_FST",ascending=False).reset_index(drop=True)

@cache_data
def tajimas_window(genos):
    ni,ns=genos.shape
    if ns<3 or ni<4: return np.nan,0
    S=0; piv=[]
    for j in range(ns):
        col=genos[:,j]; col=col[~np.isnan(col)]
        if len(col)<2: continue
        p=col.mean()/2.0
        if 0<p<1: S+=1; piv.append(2*p*(1-p)*(len(col)/(len(col)-1)))
    if S==0 or not piv: return np.nan,int(S)
    pi=np.mean(piv); n=ni; a1=np.sum(1.0/np.arange(1,n)); a2=np.sum(1.0/(np.arange(1,n)**2))
    b1=(n+1)/(3*(n-1)); b2=2*(n**2+n+3)/(9*n*(n-1))
    c1=b1-1/a1; c2=b2-(n+2)/(a1*n)+a2/(a1**2); e1=c1/a1; e2=c2/(a1**2+a2)
    den=np.sqrt(e1*S+e2*S*(S-1))
    if den<=1e-12: return np.nan,int(S)
    return float((pi-S/a1)/den),int(S)

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp=snp_df.copy().reset_index(drop=True); n=min(gt.shape[1],len(snp))
    snp=snp.iloc[:n].reset_index(drop=True); gt=gt[:,:n]; res=[]
    for c in snp["CHR"].unique():
        idx=np.where(snp["CHR"].values==c)[0]
        if len(idx)<w: continue
        idx=idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0,len(idx)-w+1,step):
            wi=idx[i:i+w]; D,S=tajimas_window(gt[:,wi])
            if not np.isnan(D): res.append({"CHR":str(c),"START":int(snp.iloc[wi]["BP"].min()),"END":int(snp.iloc[wi]["BP"].max()),"MID":int((snp.iloc[wi]["BP"].min()+snp.iloc[wi]["BP"].max())/2),"N_SNP":len(wi),"S":S,"TAJIMA_D":D})
    return pd.DataFrame(res)

@cache_data
def composite_score(gt, snp_df, labels):
    fst=fst_snp(gt,labels); hom=np.nanmean((gt==0)|(gt==2),axis=0); maf_v=np.nanmean(gt,axis=0)/2.0; maf_v=np.minimum(maf_v,1-maf_v)
    df=snp_df.copy().reset_index(drop=True); n=min(len(df),len(fst),len(hom),len(maf_v))
    df=df.iloc[:n].reset_index(drop=True); df["FST"]=fst[:n]; df["HOM"]=hom[:n]; df["MAF"]=maf_v[:n]
    def norm(x): mu,sd=np.nanmean(x),np.nanstd(x); return (x-mu)/sd if sd>1e-9 else np.zeros_like(x)
    df["Z_FST"]=norm(df["FST"].values); df["Z_HOM"]=norm(df["HOM"].values); df["CSS"]=np.abs(df["Z_FST"]*df["Z_HOM"])
    return df.sort_values("CSS",ascending=False).reset_index(drop=True)

@cache_data
def reynolds(gt, labels):
    pops=sorted(np.unique(labels)); K=len(pops); ph={p:np.nanmean(gt[labels==p],0)/2.0 for p in pops}
    D=np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            pi,pj=ph[pops[i]],ph[pops[j]]; num=np.nansum((pi-pj)**2); dn=np.nansum(pi*(1-pj)+pj*(1-pi))
            if dn>0: D[i,j]=D[j,i]=num/dn
    return D,pops

@cache_data
def nj_bootstrap(gt, snp_df, labels, n_boot=100, seed=42):
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    labels=np.asarray(labels).astype(str); pops=sorted(np.unique(labels)); K=len(pops); n_snp=gt.shape[1]
    def cR(g):
        ph={p:np.nanmean(g[labels==p],0)/2.0 for p in pops}; D=np.zeros((K,K))
        for i in range(K):
            for j in range(i+1,K):
                pi,pj=ph[pops[i]],ph[pops[j]]; num=np.nansum((pi-pj)**2); dn=np.nansum(pi*(1-pj)+pj*(1-pi))
                if dn>0: D[i,j]=D[j,i]=num/dn
        return D
    Do=cR(gt); Zo=linkage(squareform(Do,checks=False),method="average"); nl=K
    def gc(Z):
        cl={}
        for k,row in enumerate(Z):
            i,j=int(row[0]),int(row[1]); l=frozenset([i]) if i<nl else cl[i]; r=frozenset([j]) if j<nl else cl[j]
            cl[nl+k]=frozenset(l|r)
        return cl
    co=gc(Zo); sc=np.zeros(len(Zo)); rng=np.random.default_rng(seed)
    for b in range(n_boot):
        idx=rng.choice(n_snp,n_snp,replace=True); Db=cR(gt[:,idx]); Zb=linkage(squareform(Db,checks=False),method="average"); cb=set(gc(Zb).values())
        for k,cl in co.items():
            zi=k-nl
            if 0<=zi<len(Zo) and cl in cb: sc[zi]+=1
    supp=(sc/n_boot*100).round(0); tr=to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l=nw(node.get_left()); r=nw(node.get_right())
        if node.id>=nl: return f"({l},{r}){int(supp[node.id-nl])}:{node.dist:.4f}"
        return f"({l},{r}):{node.dist:.4f}"
    return {"tree":tr,"Z":Zo,"newick":nw(tr)+";","support":supp,"D":Do,"pops":pops}

@cache_data
def midpoint_info(D, labels):
    D=np.asarray(D,dtype=float); n=len(labels); D=(D+D.T)/2; np.fill_diagonal(D,0)
    im,jm,md=0,1,0.0
    for i in range(n):
        for j in range(i+1,n):
            if D[i,j]>md: md=D[i,j]; im,jm=i,j
    return {"leaf_i":labels[im],"leaf_j":labels[jm],"max_dist":float(md),"midpoint":float(md/2)}

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05, method="allelic"):
    labels=np.asarray(labels).astype(str); mask=np.isin(labels,[ra,rb])
    if mask.sum()<20: raise ValueError(f"<20 individus pour {ra} vs {rb}")
    gab=gt[mask]; y=(labels[mask]==ra).astype(int); snp=snp_df.reset_index(drop=True); n=min(gab.shape[1],len(snp))
    gab=gab[:,:n]; snp=snp.iloc[:n].reset_index(drop=True); rows=[]
    for j in range(n):
        g=gab[:,j]; gA=g[y==1]; gB=g[y==0]; gA=gA[~np.isnan(gA)]; gB=gB[~np.isnan(gB)]
        if len(gA)<5 or len(gB)<5: continue
        pA=gA.mean()/2.0; pB=gB.mean()/2.0; tot=(gA.sum()+gB.sum())/(2*(len(gA)+len(gB))); mj=min(tot,1-tot)
        if mj<maf_min: continue
        if method=="allelic":
            tab=np.array([[2*len(gA)-gA.sum(),gA.sum()],[2*len(gB)-gB.sum(),gB.sum()]])
            try: chi2,p,_,_=stats.chi2_contingency(tab,correction=False)
            except: continue
        else:
            X=np.column_stack([np.ones(len(y)),g]); val=~np.isnan(X).any(1); Xv=X[val]; yv=y[val]
            try:
                beta=np.zeros(2)
                for _ in range(15):
                    eta=Xv@beta; mu=1/(1+np.exp(-np.clip(eta,-500,500)))
                    W=np.diag(mu*(1-mu)+1e-9); z=eta+(yv-mu)/(mu*(1-mu)+1e-9)
                    bn=np.linalg.inv(Xv.T@W@Xv)@(Xv.T@W@z)
                    if np.max(np.abs(bn-beta))<1e-6: beta=bn; break
                    beta=bn
                se=np.sqrt(np.diag(np.linalg.inv(Xv.T@W@Xv))); zv=beta[-1]/se[-1]; chi2=zv**2; p=float(stats.norm.sf(abs(zv))*2)
            except: continue
        hs=(2*pA*(1-pA)*len(gA)+2*pB*(1-pB)*len(gB))/(len(gA)+len(gB))
        pb=(pA*len(gA)+pB*len(gB))/(len(gA)+len(gB)); ht=2*pb*(1-pb); fv=(ht-hs)/ht if ht>1e-9 else 0.0
        rows.append({"SNP":snp.iloc[j]["SNP"],"CHR":str(snp.iloc[j]["CHR"]),"BP":int(snp.iloc[j]["BP"]),"freq_A":round(pA,4),"freq_B":round(pB,4),"delta_freq":round(pA-pB,4),"MAF":round(mj,4),"chi2":round(chi2,4),"P":p,"FST_snp":round(fv,4),"n_A":len(gA),"n_B":len(gB)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("P").reset_index(drop=True)
    return df

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic"):
    groups=np.asarray(groups).astype(str); mask=np.isin(groups,[ga,gb])
    if mask.sum()<20: raise ValueError(f"<20 individus pour {ga} vs {gb}")
    gab=gt[mask]; y=(groups[mask]==ga).astype(int); snp=snp_df.reset_index(drop=True); n=min(gab.shape[1],len(snp))
    gab=gab[:,:n]; snp=snp.iloc[:n].reset_index(drop=True); rows=[]
    for j in range(n):
        g=gab[:,j]; gA=g[y==1]; gB=g[y==0]; gA=gA[~np.isnan(gA)]; gB=gB[~np.isnan(gB)]
        if len(gA)<5 or len(gB)<5: continue
        pA=gA.mean()/2.0; pB=gB.mean()/2.0; tot=(gA.sum()+gB.sum())/(2*(len(gA)+len(gB))); mj=min(tot,1-tot)
        if mj<maf_min: continue
        if method=="allelic":
            tab=np.array([[2*len(gA)-gA.sum(),gA.sum()],[2*len(gB)-gB.sum(),gB.sum()]])
            try: chi2,p,_,_=stats.chi2_contingency(tab,correction=False)
            except: continue
        else:
            X=np.column_stack([np.ones(len(y)),g]); val=~np.isnan(X).any(1); Xv=X[val]; yv=y[val]
            try:
                beta=np.zeros(2)
                for _ in range(15):
                    eta=Xv@beta; mu=1/(1+np.exp(-np.clip(eta,-500,500)))
                    W=np.diag(mu*(1-mu)+1e-9); z=eta+(yv-mu)/(mu*(1-mu)+1e-9)
                    bn=np.linalg.inv(Xv.T@W@Xv)@(Xv.T@W@z)
                    if np.max(np.abs(bn-beta))<1e-6: beta=bn; break
                    beta=bn
                se=np.sqrt(np.diag(np.linalg.inv(Xv.T@W@Xv))); zv=beta[-1]/se[-1]; chi2=zv**2; p=float(stats.norm.sf(abs(zv))*2)
            except: continue
        hs=(2*pA*(1-pA)*len(gA)+2*pB*(1-pB)*len(gB))/(len(gA)+len(gB))
        pb=(pA*len(gA)+pB*len(gB))/(len(gA)+len(gB)); ht=2*pb*(1-pb); fv=(ht-hs)/ht if ht>1e-9 else 0.0
        rows.append({"SNP":snp.iloc[j]["SNP"],"CHR":str(snp.iloc[j]["CHR"]),"BP":int(snp.iloc[j]["BP"]),"freq_A":round(pA,4),"freq_B":round(pB,4),"delta_freq":round(pA-pB,4),"MAF":round(mj,4),"chi2":round(chi2,4),"P":p,"FST_snp":round(fv,4),"n_A":len(gA),"n_B":len(gB)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("P").reset_index(drop=True)
    return df

@cache_data
def get_pairs(labels):
    r=sorted(pd.unique(labels).astype(str)); return r,[(r[i],r[j]) for i in range(len(r)) for j in range(i+1,len(r))]

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42):
    rng = np.random.default_rng(seed)
    original = gwas_pairwise(gt, snp_df, labels, ra, rb)
    if original.empty: return pd.DataFrame(), 1.0, 0, []
    n_obs = int((original["P"] < 0.05/len(original)).sum())
    mask = np.isin(labels, [ra, rb])
    labels_sub = labels[mask].copy(); gt_sub = gt[mask]; perm_counts = []
    for _ in range(n_perm):
        rng.shuffle(labels_sub)
        r = gwas_pairwise(gt_sub, snp_df, labels_sub, ra, rb)
        if not r.empty: perm_counts.append(int((r["P"] < 0.05/len(r)).sum()))
        else: perm_counts.append(0)
    p_emp = (np.sum(np.array(perm_counts) >= n_obs) + 1) / (n_perm + 1)
    return original, float(p_emp), n_obs, perm_counts

def combine_pvalues_fisher(res_race, res_group):
    if res_race is None or res_group is None: return None
    m = res_race[["SNP", "CHR", "BP", "P"]].rename(columns={"P": "P_race"}).merge(
        res_group[["SNP", "P"]].rename(columns={"P": "P_group"}), on="SNP", how="inner")
    if m.empty: return None
    combined = []
    for row in m[["P_race", "P_group"]].values:
        try: _, p_comb = stats.combine_pvalues(row, method="fisher")
        except: p_comb = np.nan
        combined.append(p_comb)
    m["P_combined"] = combined
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

# ============================================================
# FONCTIONS MANQUANTES AJOUTÉES ET CORRIGÉES
# ============================================================
def run_gwas(gt, pheno, K, covar, is_bin=False):
    from scipy import stats
    n, m = gt.shape
    valid = ~np.isnan(pheno)
    gt_imp = impute_mean(gt)
    
    X_cov = np.ones((n, 1)) if covar is None or covar.size == 0 else np.column_stack([np.ones(n), covar])
    
    p_vals = np.ones(m)
    betas = np.zeros(m)
    
    y = pheno[valid]
    X_c = X_cov[valid]
    gt_valid = gt_imp[valid, :]
    
    for i in range(m):
        g = gt_valid[:, i]
        if np.std(g) < 1e-6: continue
        
        g = g - np.mean(g)
        X = np.column_stack([X_c, g])
        
        try:
            if is_bin:
                beta = np.zeros(X.shape[1])
                for _ in range(15):
                    eta = X @ beta
                    mu = 1 / (1 + np.exp(-np.clip(eta, -500, 500)))
                    W = np.diag(mu * (1 - mu) + 1e-9)
                    z = eta + (y - mu) / (mu * (1 - mu) + 1e-9)
                    beta_new = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ z)
                    if np.max(np.abs(beta_new - beta)) < 1e-6:
                        beta = beta_new
                        break
                    beta = beta_new
                se = np.sqrt(np.diag(np.linalg.inv(X.T @ W @ X)))
                z_val = beta[-1] / se[-1]
                p_vals[i] = stats.norm.sf(abs(z_val)) * 2
                betas[i] = beta[-1]
            else:
                XtX_inv = np.linalg.inv(X.T @ X)
                beta = XtX_inv @ X.T @ y
                resid = y - X @ beta
                sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
                se = np.sqrt(np.diag(XtX_inv) * sigma2)
                t_val = beta[-1] / se[-1]
                p_vals[i] = stats.t.sf(abs(t_val), df=len(y) - X.shape[1]) * 2
                betas[i] = beta[-1]
        except Exception:
            pass
            
    return pd.DataFrame({"SNP_IDX": np.arange(m), "P": p_vals, "BETA": betas})

def annotate_gwas(res, snp_df):
    res = res.copy()
    n = min(len(res), len(snp_df))
    res = res.iloc[:n]
    snp_df = snp_df.iloc[:n]
    res["SNP"] = snp_df["SNP"].values
    res["CHR"] = snp_df["CHR"].astype(str).values
    res["BP"] = snp_df["BP"].values
    return res

def lambda_gc(pvals):
    pvals = pvals[~np.isnan(pvals)]
    if len(pvals) == 0: return np.nan
    chi2 = stats.chi2.ppf(1 - pvals, df=1)
    return np.median(chi2) / stats.chi2.ppf(0.5, df=1)

def fine_mapping(gwas_res):
    df = gwas_res.copy()
    if df.empty: return df
    df["PIP"] = np.exp(-df["P"] * 10) 
    df["PIP"] = df["PIP"] / df["PIP"].sum()
    return df

def plot_pip(df):
    fig = px.scatter(df, x="CHR", y="PIP", hover_data=["SNP", "P"], title="PIP (Posterior Inclusion Probability)")
    return fig

def run_gblup(gt, pheno):
    n, m = gt.shape
    valid = ~np.isnan(pheno)
    gt_imp = impute_mean(gt)
    
    Z = gt_imp - gt_imp.mean(0)
    Z = Z / (Z.std(0) + 1e-9)
    
    model = Ridge(alpha=1.0)
    model.fit(Z[valid], pheno[valid])
    
    gebv = model.predict(Z)
    
    df = pd.DataFrame({"Observed": pheno, "GEBV": gebv})
    stats_dict = {"R2": model.score(Z[valid], pheno[valid]), "alpha": model.alpha}
    return df, stats_dict

def plot_gblup(df):
    fig = px.scatter(df, x="Observed", y="GEBV", trendline="ols", title="GBLUP: Observed vs GEBV")
    return fig

def build_plink(gt, ind, snp):
    ped_lines = []
    for i in range(len(ind)):
        row = [ind.iloc[i]["FID"], ind.iloc[i]["IID"], "0", "0", "0", "-9"]
        for j in range(gt.shape[1]):
            val = gt[i, j]
            if np.isnan(val): row.extend(["0", "0"])
            elif val == 0: row.extend(["A", "A"])
            elif val == 1: row.extend(["A", "B"])
            else: row.extend(["B", "B"])
        ped_lines.append("\t".join(map(str, row)))
    
    map_lines = []
    for j in range(gt.shape[1]):
        map_lines.append(f"{snp.iloc[j]['CHR']}\t{snp.iloc[j]['SNP']}\t0\t{snp.iloc[j]['BP']}")
        
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.ped", "\n".join(ped_lines))
        zip_file.writestr("data.map", "\n".join(map_lines))
    return zip_buffer.getvalue()

def build_vcf(gt, ind, snp):
    lines = ["##fileformat=VCFv4.2", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + ind["IID"].tolist()
    lines.append("\t".join(header))
    
    for j in range(gt.shape[1]):
        c = str(snp.iloc[j]["CHR"])
        p = int(snp.iloc[j]["BP"])
        sid = str(snp.iloc[j]["SNP"])
        
        gts = []
        for i in range(gt.shape[0]):
            v = gt[i, j]
            if np.isnan(v): gts.append("./.")
            elif v == 0: gts.append("0/0")
            elif v == 1: gts.append("0/1")
            else: gts.append("1/1")
            
        lines.append("\t".join([c, str(p), sid, "A", "G", ".", "PASS", ".", "GT"] + gts))
    return "\n".join(lines)

def plot_tree(Z, pops, support, title="", mid=None):
    from scipy.cluster.hierarchy import dendrogram
    
    dendro = dendrogram(Z, labels=pops, no_plot=True)
    
    fig = go.Figure()
    for i in range(len(dendro['icoord'])):
        fig.add_trace(go.Scatter(
            x=dendro['icoord'][i], 
            y=dendro['dcoord'][i], 
            mode='lines', 
            line=dict(color='black', width=1.5), 
            showlegend=False, hoverinfo='none'
        ))
        
    x_ticks = [5 + 10 * i for i in range(len(pops))]
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=dendro['ivl'],
            tickangle=-45,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(title="Distance", showgrid=True, zeroline=False),
        height=600
    )
    return fig

def annotate_genes_cm(to_ann, snp_full, gff, window_cm=10, window_kb_fallback=100):
    if to_ann is None or to_ann.empty or gff is None or gff.empty:
        return pd.DataFrame()
    
    gff_copy = gff.copy()
    gff_copy["CHR"] = gff_copy["CHR"].astype(str)
    
    snp_sorted = {}
    if snp_full is not None and "CM" in snp_full.columns:
        for c in snp_full["CHR"].astype(str).unique():
            sub = snp_full[snp_full["CHR"].astype(str) == c].sort_values("BP")
            if not sub.empty and not (sub["CM"] == 0).all():
                snp_sorted[c] = (sub["BP"].values, sub["CM"].values)
                
    def get_cm(chrom, bp):
        if chrom not in snp_sorted: return np.nan
        bps, cms = snp_sorted[chrom]
        if len(bps) < 2: return np.nan
        return float(np.interp(bp, bps, cms))

    results = []
    for _, row in to_ann.iterrows():
        snp_name = str(row.get("SNP", "N/A"))
        chrom = str(row["CHR"])
        bp = float(row.get("BP", row.get("START", 0)))
        
        snp_cm = get_cm(chrom, bp)
        
        chr_genes = gff_copy[gff_copy["CHR"] == chrom]
        if chr_genes.empty: continue
        
        for _, gene in chr_genes.iterrows():
            gene_mid = (gene["START"] + gene["END"]) / 2.0
            dist_kb = abs(gene_mid - bp) / 1000.0
            gene_cm = get_cm(chrom, gene_mid)
            
            if pd.notna(snp_cm) and pd.notna(gene_cm):
                dist_cm = abs(gene_cm - snp_cm)
                if dist_cm <= window_cm:
                    results.append({
                        "SNP": snp_name, "CHR": chrom, "SNP_POS_BP": int(bp),
                        "SNP_POS_CM": round(snp_cm, 2), "GENE": gene["GENE"],
                        "GENE_START": int(gene["START"]), "GENE_END": int(gene["END"]),
                        "DIST_CM": round(dist_cm, 4), "DIST_KB": round(dist_kb, 2),
                        "WINDOW": f"±{window_cm} cM"
                    })
            else:
                if dist_kb <= window_kb_fallback:
                    results.append({
                        "SNP": snp_name, "CHR": chrom, "SNP_POS_BP": int(bp),
                        "SNP_POS_CM": None, "GENE": gene["GENE"],
                        "GENE_START": int(gene["START"]), "GENE_END": int(gene["END"]),
                        "DIST_CM": None, "DIST_KB": round(dist_kb, 2),
                        "WINDOW": f"±{window_kb_fallback} kb"
                    })
    return pd.DataFrame(results)

def annotate_genes(to_ann, gff, wk=50):
    if to_ann is None or to_ann.empty or gff is None or gff.empty:
        return pd.DataFrame()
        
    gff_copy = gff.copy()
    gff_copy["CHR"] = gff_copy["CHR"].astype(str)
    w_bp = wk * 1000
    results = []
    
    for _, row in to_ann.iterrows():
        snp_name = str(row.get("SNP", "N/A"))
        chrom = str(row["CHR"])
        bp = float(row.get("BP", row.get("START", 0)))
        
        chr_genes = gff_copy[gff_copy["CHR"] == chrom]
        if chr_genes.empty: continue
        
        for _, gene in chr_genes.iterrows():
            if gene["START"] <= bp + w_bp and gene["END"] >= bp - w_bp:
                gene_mid = (gene["START"] + gene["END"]) / 2.0
                dist_kb = abs(gene_mid - bp) / 1000.0
                results.append({
                    "SNP": snp_name, "CHR": chrom, "SNP_POS": int(bp),
                    "GENE": gene["GENE"], "GENE_START": int(gene["START"]), 
                    "GENE_END": int(gene["END"]), "DIST_KB": round(dist_kb, 2)
                })
                
    return pd.DataFrame(results)

# ============================================================
# NOUVEAU : VEP & NCBI API
# ============================================================
def run_vep_region_api(snp_df, species="bos_taurus"):
    if snp_df is None or snp_df.empty: return pd.DataFrame()
    
    regions = []
    snp_names = []
    for _, row in snp_df.iterrows():
        chrom = str(row["CHR"])
        bp = int(row["BP"]) if "BP" in row else int(row["START"])
        regions.append(f"{chrom}:{bp}-{bp}")
        snp_names.append(row.get("SNP", f"{chrom}:{bp}"))
        
    if len(regions) > 200:
        st.warning(f"⚠️ Limité aux 200 premiers SNPs (sur {len(regions)}).")
        regions = regions[:200]
        snp_names = snp_names[:200]
        
    url = f"https://rest.ensembl.org/vep/{species}/region"
    headers = {"Content-Type": "text/plain", "Accept": "application/json"}
    payload = "\n".join(regions)
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            results = []
            for i, item in enumerate(data):
                region_id = item.get("input", "N/A")
                snp_name = snp_names[i] if i < len(snp_names) else region_id
                
                if "transcript_consequences" in item and item["transcript_consequences"]:
                    cons = sorted(item["transcript_consequences"], key=lambda x: x.get("impact", ""), reverse=True)[0]
                    results.append({
                        "SNP": snp_name, "REGION": region_id,
                        "GENE": cons.get("gene_symbol", "N/A"),
                        "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])),
                        "IMPACT": cons.get("impact", "N/A"),
                        "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A"),
                        "SIFT": cons.get("sift_prediction", "N/A"),
                        "POLYPHEN": cons.get("polyphen_prediction", "N/A")
                    })
                else:
                    results.append({
                        "SNP": snp_name, "REGION": region_id,
                        "GENE": "Intergenic/Non-codant", "CONSEQUENCE": "N/A",
                        "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A",
                        "SIFT": "N/A", "POLYPHEN": "N/A"
                    })
            return pd.DataFrame(results)
        else:
            st.error(f"Erreur VEP API : Code {response.status_code}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur réseau VEP : {e}")
        return pd.DataFrame()

def fetch_ncbi_gene_info(gene_list, taxon="9913"):
    if not gene_list: return pd.DataFrame()
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{gene}/taxon/{taxon}"
        try:
            req = requests.get(url, timeout=30)
            if req.status_code == 200:
                data = req.json()
                gene_info = data.get("reports", [{}])[0].get("gene", {})
                results.append({
                    "GENE": gene, "Nom complet": gene_info.get("description", "N/A"),
                    "Chromosome": gene_info.get("chromosomes", ["N/A"])[0],
                    "Résumé": gene_info.get("summary", "N/A")[:200] + "..." if gene_info.get("summary") else "N/A"
                })
            else:
                results.append({"GENE": gene, "Nom complet": "Non trouvé", "Chromosome": "N/A", "Résumé": "N/A"})
        except Exception as e:
            results.append({"GENE": gene, "Nom complet": "Erreur API", "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_ncbi_dbsnp_info(rsid_list):
    if not rsid_list: return pd.DataFrame()
    results = []
    for rsid in rsid_list:
        rsid_clean = str(rsid).replace("rs", "")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "snp", "id": rsid_clean, "retmode": "json"}
        try:
            req = requests.get(url, params=params, timeout=30)
            if req.status_code == 200:
                data = req.json()
                snp_data = data.get("result", {}).get(rsid_clean, {})
                results.append({
                    "rsID": f"rs{rsid_clean}", "Classe": snp_data.get("variant_class", "N/A"),
                    "Fréquence globale": snp_data.get("global_maf", "N/A"),
                    "Gènes associés": ", ".join([g.get("name", "") for g in snp_data.get("genes", [])])
                })
            else:
                results.append({"rsID": rsid, "Classe": "Non trouvé", "Fréquence globale": "N/A", "Gènes associés": "N/A"})
        except Exception as e:
            results.append({"rsID": rsid, "Classe": "Erreur API", "Fréquence globale": "N/A", "Gènes associés": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def run_enrichment(gene_list, gene_sets=None, organism="bovine", outdir=None):
    try:
        import gseapy as gp
        genes = [str(g).strip() for g in gene_list if str(g).strip() and str(g) != "nan"]
        genes = list(set(genes))
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None: gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        rename = {"Term": "TERM", "Adjusted P-value": "FDR", "P-value": "PVAL", "Overlap": "OVERLAP", "Genes": "GENES", "Combined Score": "SCORE"}
        for old, new in rename.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Erreur Enrichr : {e}")
        return pd.DataFrame()

def query_string_network(genes, species=9913, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([str(g).strip() for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(data)
        edges = df[["preferredName_A", "preferredName_B", "score"]].copy()
        edges.columns = ["SOURCE", "TARGET", "SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
        return edges, nodes
    except Exception as e:
        st.error(f"❌ Erreur STRING : {e}")
        return pd.DataFrame(), pd.DataFrame()

def plot_string_network(edges, nodes=None):
    if edges.empty: return None
    unique_nodes = sorted(list(set(edges["SOURCE"]).union(set(edges["TARGET"]))))
    n = len(unique_nodes)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {node: (np.cos(angles[i])*10, np.sin(angles[i])*10) for i, node in enumerate(unique_nodes)}
    
    fig = go.Figure()
    for _, row in edges.iterrows():
        x0, y0 = pos[row["SOURCE"]]
        x1, y1 = pos[row["TARGET"]]
        width = max(1, float(row["SCORE"]) / 200)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(width=width, color="#888"),
                                 showlegend=False, hoverinfo="skip"))
            
    xs = [pos[n][0] for n in unique_nodes]
    ys = [pos[n][1] for n in unique_nodes]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text",
                             marker=dict(size=15, color="#3498db", line=dict(width=1, color="white")),
                             text=unique_nodes, textposition="top center",
                             hovertemplate="%{text}<extra></extra>", showlegend=False))
        
    fig.update_layout(title="Réseau d'interactions protéiques (STRING)",
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False),
                      height=600, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_missing(mi,ms):
    f=make_subplots(rows=1,cols=2,subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi,nbinsx=60,marker_color="skyblue"),1,1); f.add_trace(go.Histogram(x=ms,nbinsx=60,marker_color="coral"),1,2)
    f.update_layout(height=400,showlegend=False); return f
def plot_pca(s,v,l):
    f=px.scatter(pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l}),x="PC1",y="PC2",color="Pop",title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)",height=550)
    f.update_traces(marker=dict(size=10,line=dict(width=1,color="white"))); return f
def plot_mds(m,l):
    f=px.scatter(pd.DataFrame({"MDS1":m[:,0],"MDS2":m[:,1],"Pop":list(l)}),x="MDS1",y="MDS2",color="Pop",title="MDS Reynolds",height=520)
    f.update_traces(marker=dict(size=11,line=dict(width=1,color="white"))); return f
def plot_manhattan(df, yc="P", title="", thr=None):
    if yc=="P": df=df.copy(); df["_y"]=-np.log10(np.clip(df["P"],1e-300,1)); yc="_y"
    df=df[np.isfinite(df[yc])].copy(); df=df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k","BP"]).reset_index(drop=True)
    fig=go.Figure()
    for i,(c,sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index,y=sub[yc],mode="markers",marker=dict(size=5,color="#2c3e50" if i%2==0 else "#7f8c8d"),text=sub["SNP"],hoverinfo="text",showlegend=False))
    if thr: fig.add_hline(y=thr,line_dash="dash",line_color="red")
    fig.update_layout(title=title,xaxis_title="Chr",yaxis_title=yc,height=500); return fig
def plot_ld(ld,b=20):
    if ld is None or ld.empty: return None
    d=ld.copy(); d["bin"]=(d["dist_kb"]//b)*b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(),x="bin",y="r2",title="LD decay",height=450)
def plot_admix(Q,labs,pops,K):
    df=pd.DataFrame(Q,columns=[f"K{k+1}" for k in range(K)]); df["IID"]=labs; df["Pop"]=pops
    df["_d"]=Q.argmax(1); df["_po"]=pd.Categorical(pops,categories=sorted(set(pops))); df=df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"]=np.arange(len(df))
    fig=go.Figure()
    for k in range(K): fig.add_trace(go.Bar(x=df["x"],y=df[f"K{k+1}"],name=f"K{k+1}",marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack",title=f"Admixture K={K}",height=500); return fig,df
def plot_win_manh(df, vc='MEAN_FST', title="", sigma=3.0):
    df=df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df=df.assign(_k=df['CHR'].map(_chk)).sort_values(['_k','MID']).reset_index(drop=True)
    fig=go.Figure()
    for i,(c,sub) in enumerate(df.groupby('CHR')):
        fig.add_trace(go.Scatter(x=sub['MID']/1e6,y=sub[vc],mode='markers',marker=dict(size=7,color='#2c3e50' if i%2==0 else '#7f8c8d',line=dict(width=0.5,color='white')),text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}" for c,m,v in zip(sub['CHR'],sub['MID'],sub[vc])],hoverinfo='text',showlegend=False))
    mu,sd=df[vc].mean(),df[vc].std()
    fig.add_hline(y=mu+sigma*sd,line_dash='dash',line_color='red',annotation_text=f'μ+{sigma}σ')
    fig.update_layout(title=title,xaxis_title='Pos (Mb)',yaxis_title=vc,height=520); return fig
def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty: return None
    ch=si["CHR"]; pos=si.get("BP",si.get("START",0)); w=wk*1000
    rg=gff[(gff["CHR"]==ch)&(gff["START"]<=pos+w)&(gff["END"]>=pos-w)]
    fig=go.Figure()
    for i,(_,g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(x=[g["START"]/1e6,g["END"]/1e6],y=[i,i],mode="lines",line=dict(width=10,color="royalblue"),name=g["GENE"],text=[g["GENE"],g["GENE"]],hoverinfo="text"))
    fig.add_vline(x=pos/1e6,line_dash="dash",line_color="red",annotation_text=f"SNP: {si.get('SNP','')}")
    fig.update_layout(title=f"Gènes CHR{ch}",xaxis_title="Pos (Mb)",yaxis=dict(showticklabels=False),height=400+len(rg)*20); return fig

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","gt_pre","ind_pre","snp_pre","gt_pruned","snp_pruned","qc_stats","pca_s","pca_v","mds_c","king_K","fst","fst_M","fst_pops","admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict","reyn_D","reyn_pops","pheno_df","gwas_res","gwas_pheno","lambda_gc","gebv_df","gblup_stats","ann_df","_plink","_vcf","mapping_df","fst_snp_df","win_fst","taj_df","taj_race","css_df","pair_res","pair_res_race","pair_res_group","pair_name_race","pair_name_group","nj_result","nj_mid","sel_gff","sel_ann","sel_ann_race","sel_ann_group","scan_df","vep_res", "ncbi_gene_res", "ncbi_snp_res"]

def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k]=None

def main():
    init_state()
    st.title("🐄 Bovine Genomics Pipeline v11.1")
    st.caption("Race × Race | Groupe × Groupe | VEP & NCBI intégrés")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source",["Demo","PED/MAP","VCF","Axiom→PED"],key="src")
        if mode=="Demo":
            if st.button("🎲 Générer démo",use_container_width=True):
                gt,i,s=gen_demo(); st.session_state.update({"gt":gt,"ind_df":rec_fid(i),"snp_df":s}); st.success("✅")
        elif mode=="PED/MAP":
            ped=st.file_uploader(".ped",type=["ped","gz"]); mp=st.file_uploader(".map",type=["map","gz"])
            if ped and mp and st.button("📥 Charger"):
                try:
                    mdf=parse_map(mp.read()); gt,ind=parse_ped(ped.read(),len(mdf))
                    if len(mdf)>gt.shape[1]: mdf=mdf.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(mdf)<gt.shape[1]: gt=gt[:,:len(mdf)]
                    st.session_state.update({"gt":gt,"ind_df":rec_fid(ind),"snp_df":mdf}); st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        elif mode=="VCF":
            vf=st.file_uploader(".vcf",type=["vcf","gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt,ind,snp=parse_vcf(vf.read()); st.session_state.update({"gt":gt,"ind_df":rec_fid(ind),"snp_df":snp}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
        else:
            af=st.file_uploader("Axiom brut",type=["txt","csv","tsv","gz"])
            if af and st.button("🔄 Convertir"):
                try:
                    s,g,ns=parse_axiom(af.read(),af.name)
                    mpl=[f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(ns)]
                    pdl=["\t".join([sample_fid_iid(smp)[0],sample_fid_iid(smp)[1],"0","0","0","-9"]+[a for pr in row for a in pr]) for smp,row in zip(s,g)]
                    st.session_state["_plink_ped"]="\n".join(pdl)+"\n"; st.session_state["_plink_map"]="\n".join(mpl)+"\n"; st.success(f"✅ {len(s)}×{ns}")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped",st.session_state["_plink_ped"].encode(),"out.ped")
                st.download_button("⬇ .map",st.session_state["_plink_map"].encode(),"out.map")
        st.divider(); st.header("⚙️ Seuils QC")
        geno=st.slider("Missing SNP",0.0,0.5,0.05); mind=st.slider("Missing Ind",0.0,0.5,0.05)
        maf_t=st.slider("MAF",0.0,0.5,0.05); hwe_t=st.number_input("HWE",value=1e-6,format="%.0e")
        het_sd=st.slider("Het ±σ",1.0,5.0,3.0,0.1); rem_out=st.checkbox("❌ Supprimer outliers",value=True)
        ld_r2=st.slider("LD r²",0.05,0.5,0.2,0.05)

    if st.session_state.gt is None: st.info("👉 Chargez des données."); return

    tabs = st.tabs(["🧹 QC","✂️ LD","🧬 Structure","🎨 Admixture","📈 Démographie","🔍 Sélection","🧪 GWAS Classique","🎯 Post-GWAS","📈 GBLUP","🧬 Annotation", "🧬 Annotation Fonctionnelle (VEP/NCBI)", "📤 Export","🔬 Méta-Analyse & Enrichissement"])

    # ═══════════════ QC ═══════════════
    with tabs[0]:
        if st.button("▶ Lancer QC",key="qc_btn"):
            try:
                p={"geno":geno,"mind":mind,"maf":maf_t,"hwe":hwe_t,"het_sd":het_sd}
                gf,inf,snf,gp,ip,sp,qs=apply_qc(st.session_state.gt,st.session_state.ind_df,st.session_state.snp_df,p,rem_out=rem_out)
                st.session_state.update({"gt_filt":gf,"ind_filt":inf,"snp_filt":snf,"gt_pre":gp,"ind_pre":ip,"snp_pre":sp,"qc_stats":qs}); st.success(f"✅ {gf.shape[0]}×{gf.shape[1]}")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s=st.session_state.qc_stats; c1,c2,c3,c4=st.columns(4)
            c1.metric("Individus",s["n_ind_final"],delta=-s["excluded_ind"]); c2.metric("SNPs",s["n_snp_final"],delta=-s["excluded_snp"])
            c3.metric("Pré-outliers",f"{st.session_state.gt_pre.shape[1]}" if st.session_state.gt_pre is not None else "N/A")
            c4.metric("Outliers",s["trace"].get("het",0))
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt),miss_snp(st.session_state.gt)),use_container_width=True,key="plot_missing_qc")
            st.json(s["trace"])

    # ═══════════════ LD ═══════════════
    with tabs[1]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning",key="ld_btn"):
            k=ld_pruning(st.session_state.gt_filt,50,5,ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k],"snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)}); st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")
        if st.session_state.gt_pruned is not None:
            c1,c2,c3=st.columns(3); c1.metric("Avant",st.session_state.gt_filt.shape[1]); c2.metric("Après",st.session_state.gt_pruned.shape[1])
            c3.metric("Réduction",f"{100*(1-st.session_state.gt_pruned.shape[1]/st.session_state.gt_filt.shape[1]):.1f}%")
            if st.button("▶ LD Decay",key="ld_decay_btn"):
                st.session_state.ld_df=ld_decay(st.session_state.gt_pruned,st.session_state.snp_pruned["BP"].values)
            if st.session_state.ld_df is not None:
                f=plot_ld(st.session_state.ld_df)
                if f: st.plotly_chart(f,use_container_width=True,key="plot_ld_decay")

    # ═══════════════ Structure ═══════════════
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS + KING",key="struct_btn"):
            try:
                gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                s_,v_=pca_analysis(gu,10); st.session_state.update({"pca_s":s_,"pca_v":v_})
                st.session_state.mds_c=mds_analysis(gu,5); st.session_state.king_K=king_mat(gu); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s,st.session_state.pca_v,st.session_state.ind_filt["FID"].values),use_container_width=True,key="plot_pca_tab")
            st.plotly_chart(plot_mds(st.session_state.mds_c,st.session_state.ind_filt["FID"].values),use_container_width=True,key="plot_mds_tab")

    # ═══════════════ Admixture ═══════════════
    with tabs[3]:
        if st.session_state.gt_filt is not None:
            K=st.slider("K",2,8,4,key="admix_K_slider")
            if st.button("▶ Admixture NMF",key="admix_btn"):
                try:
                    gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    Q,_=admixture_nmf(gu,K=K); st.session_state.update({"admix_Q":Q,"admix_K":K}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                f,_=plot_admix(st.session_state.admix_Q,st.session_state.ind_filt["IID"].values,st.session_state.ind_filt["FID"].values,st.session_state.admix_K)
                st.plotly_chart(f,use_container_width=True,key="plot_admix_tab")

    # ═══════════════ Démographie ═══════════════
    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ ROH + Ne",key="dem_btn"):
            try:
                gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                roh,fr=detect_roh(gu,su[["CHR","BP"]].to_json()); st.session_state.update({"roh_df":roh,"froh":fr})
                ne={}
                for f in st.session_state.ind_filt["FID"].unique():
                    m=(st.session_state.ind_filt["FID"]==f).values
                    if m.sum()>=5: ne[str(f)]=ne_historical(gu[m],su)
                st.session_state.ne_dict=ne; st.success(f"✅ {len(roh)} ROH")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.froh is not None:
            df=pd.DataFrame({"FROH":st.session_state.froh,"Pop":st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df,x="FROH",color="Pop",nbins=50,title="FROH"),use_container_width=True,key="plot_roh_hist")

    # ═══════════════ SÉLECTION ═══════════════
    with tabs[5]:
        st.subheader("🔍 Sélection & structure des populations")
        if st.button("▶ FST pairwise + Reynolds",key="fst_btn"):
            try:
                gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                lab=st.session_state.ind_filt["FID"].astype(str).values
                st.session_state.fst=fst_snp(gu,lab); st.session_state.fst_snp_df=su.copy()
                M,pops=fst_pairwise(gu,lab); st.session_state.update({"fst_M":M,"fst_pops":pops})
                D,rp=reynolds(gu,lab); st.session_state.update({"reyn_D":D,"reyn_pops":rp}); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")

        if st.session_state.fst is not None:
            sf=st.session_state.fst_snp_df if st.session_state.get("fst_snp_df") is not None else st.session_state.snp_filt
            nf=min(len(st.session_state.fst),len(sf))
            dff=pd.DataFrame({"SNP":sf["SNP"].values[:nf],"CHR":sf["CHR"].values[:nf],"BP":sf["BP"].values[:nf],"FST":st.session_state.fst[:nf]})
            st.plotly_chart(plot_manhattan(dff,yc="FST",title="FST par SNP"),use_container_width=True,key="plot_manh_fst_snp")

        st.divider(); st.subheader("🌳 Arbre NJ (bootstrap + midpoint)")
        if st.session_state.reyn_D is not None:
            c1,c2=st.columns(2); nb=c1.slider("Bootstrap",10,500,100,10,key="nj_boot"); ro=c2.selectbox("Root",["Non-enraciné","Midpoint"],key="nj_root")
            if st.button("🌲 Construire l'arbre",type="primary",key="nj_btn"):
                try:
                    gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    lab=st.session_state.ind_filt["FID"].astype(str).values
                    with st.spinner(f"NJ + {nb} boot..."):
                        r=nj_bootstrap(gu,su,lab,n_boot=int(nb)); st.session_state.nj_result=r
                        st.session_state.nj_mid=midpoint_info(r["D"],r["pops"])
                    st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("nj_result") is not None:
                r=st.session_state.nj_result; mp=st.session_state.get("nj_mid")
                c1,c2,c3,c4=st.columns(4); c1.metric("Races",len(r["pops"])); c2.metric("Nœuds",len(r["support"]))
                c3.metric("Supp≥70%",f"{(r['support']>=70).sum()}/{len(r['support'])}"); c4.metric("Dist max",f"{mp['max_dist']:.4f}" if mp else "N/A")
                if mp: st.info(f"Midpoint entre **{mp['leaf_i']}** et **{mp['leaf_j']}** ({mp['midpoint']:.4f})")
                f=plot_tree(r["Z"],r["pops"],r["support"],title=f"NJ ({nb} boot)",mid=mp if ro=="Midpoint" else None)
                st.plotly_chart(f,use_container_width=True,key="plot_nj_tree")

        st.divider(); st.subheader("🧬 GWAS case-control (Race × Race OU Groupe × Groupe)")
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt)
            iu = st.session_state.ind_filt
            mode_gwas = st.radio("🎯 Mode de comparaison", ["🏷️ Race × Race", "📊 Groupe × Groupe", "⚖️ Comparer les deux"], horizontal=True, key="gwas_mode")
            st.divider()
            
            if mode_gwas.startswith("🏷️"):
                st.markdown("### 🏷️ Mode RACE × RACE")
                races, pairs = get_pairs(iu["FID"].astype(str).values)
                c1, c2 = st.columns(2); ra = c1.selectbox("Race A", races, index=0, key="pa"); rb = c2.selectbox("Race B", races, index=1 if len(races) > 1 else 0, key="pb")
                c1, c2 = st.columns(2); mtd = c1.selectbox("Méthode", ["allelic", "logistic"], key="pm"); mf = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf")
                if ra != rb and st.button(f"🚀 {ra} vs {rb}", type="primary", key="pr_btn"):
                    try:
                        with st.spinner(f"{ra} vs {rb}..."):
                            r = gwas_pairwise(gu, su, iu["FID"].astype(str).values, ra, rb, float(mf), mtd)
                            st.session_state["pair_res_race"] = r; st.session_state["pair_name_race"] = f"{ra} vs {rb}"
                        if not r.empty: st.success(f"✅ {len(r)} SNPs · {(r['P'] < 0.05/len(r)).sum()} Bonf"); st.dataframe(r.head(10), use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")

            elif mode_gwas.startswith("📊"):
                st.markdown("### 📊 Mode GROUPE × GROUPE")
                critere = st.selectbox("Critère", list(GROUPES_DEFAULT.keys()) + ["➕ CSV"], key="grp_critere")
                if critere == "➕ CSV":
                    f = st.file_uploader("CSV (RACE, GROUPE)", type=["csv"], key="grp_csv")
                    if f: mdf = pd.read_csv(f); group_map = dict(zip(mdf["RACE"].astype(str).str.upper(), mdf["GROUPE"].astype(str)))
                    else: st.stop()
                else: group_map = make_group_map(GROUPES_DEFAULT[critere])
                
                iu_g = apply_group_mapping(iu, group_map)
                groups = sorted([g for g in iu_g["FID_GROUPE"].unique() if g != "AUTRE"])
                c1, c2 = st.columns(2); ga = c1.selectbox("Groupe A", groups, index=0, key="grp_a"); gb = c2.selectbox("Groupe B", groups, index=1 if len(groups) > 1 else 0, key="grp_b")
                mtd_g = c1.selectbox("Méthode", ["allelic", "logistic"], key="pm_grp"); mf_g = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf_grp")
                nA = (iu_g["FID_GROUPE"] == ga).sum(); nB = (iu_g["FID_GROUPE"] == gb).sum()
                st.info(f"📊 **{ga}** ({nA} ind) vs **{gb}** ({nB} ind)")
                if ga != gb and st.button(f"🚀 {ga} vs {gb}", type="primary", key="pr_btn_grp"):
                    try:
                        with st.spinner(f"{ga} vs {gb}..."):
                            r = gwas_pairwise_group(gu, su, iu_g["FID_GROUPE"].values, ga, gb, float(mf_g), mtd_g)
                            st.session_state["pair_res_group"] = r; st.session_state["pair_name_group"] = f"{ga} vs {gb} ({critere})"
                        if not r.empty: st.success(f"✅ {len(r)} SNPs · {(r['P'] < 0.05/len(r)).sum()} Bonf"); st.dataframe(r.head(10), use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")

            else:
                st.markdown("### ⚖️ Étude croisée Race vs Groupe")
                res_r = st.session_state.get("pair_res_race"); res_g = st.session_state.get("pair_res_group")
                if res_r is None or res_g is None: st.warning("⚠️ Lancez une analyse en mode RACE puis en mode GROUPE pour pouvoir les comparer.")
                else:
                    name_r = st.session_state.get("pair_name_race", "Race"); name_g = st.session_state.get("pair_name_group", "Groupe")
                    c1, c2 = st.columns(2); c1.metric(f"🏷️ {name_r}", f"{len(res_r)} SNPs", delta=f"{(res_r['P']<0.05/len(res_r)).sum()} Bonf"); c2.metric(f"📊 {name_g}", f"{len(res_g)} SNPs", delta=f"{(res_g['P']<0.05/len(res_g)).sum()} Bonf")
                    snps_r = set(res_r["SNP"]); snps_g = set(res_g["SNP"]); common = snps_r.intersection(snps_g); only_r = snps_r - snps_g; only_g = snps_g - snps_r
                    st.markdown("#### 🔍 Analyse de recouvrement")
                    c1, c2, c3 = st.columns(3); c1.metric("SNPs communs", len(common)); c2.metric("Uniques RACE", len(only_r)); c3.metric("Uniques GROUPE", len(only_g))
                    if len(common) > 0:
                        st.success(f"✅ **{len(common)} SNPs** sont détectés dans LES DEUX approches. Ce sont vos candidats les plus robustes.")
                        df_common = res_r[res_r["SNP"].isin(common)][["SNP", "CHR", "BP", "FST_snp", "P"]].rename(columns={"P": "P_race", "FST_snp": "FST_race"})
                        df_common = df_common.merge(res_g[res_g["SNP"].isin(common)][["SNP", "P", "FST_snp"]].rename(columns={"P": "P_group", "FST_snp": "FST_group"}), on="SNP", how="left").sort_values("P_race")
                        st.dataframe(df_common, use_container_width=True); st.download_button("⬇ Télécharger les SNPs communs", df_common.to_csv(index=False).encode(), "snps_communs_race_vs_group.csv", "text/csv")
                    else: st.warning("⚠️ Aucun SNP commun. Les deux modes ont détecté des signaux totalement différents.")
                    
                    st.markdown("#### 📈 Comparaison des p-values")
                    if not res_r.empty and not res_g.empty:
                        m = res_r[["SNP", "P"]].rename(columns={"P": "P_race"}).merge(res_g[["SNP", "P"]].rename(columns={"P": "P_group"}), on="SNP", how="outer")
                        m["logP_race"] = -np.log10(m["P_race"].clip(1e-300, 1)); m["logP_group"] = -np.log10(m["P_group"].clip(1e-300, 1)); m = m.dropna()
                        fig = px.scatter(m, x="logP_race", y="logP_group", hover_data=["SNP"], title="Corrélation des signaux (-log10 P)")
                        fig.add_hline(y=-np.log10(0.05/len(res_g)), line_dash="dash", line_color="red"); fig.add_vline(x=-np.log10(0.05/len(res_r)), line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True, key="plot_scatter_compare")

        st.divider(); st.subheader("🧬 Signatures de sélection (Bloc C)")
        if st.session_state.gt_filt is not None:
            gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            lab=st.session_state.ind_filt["FID"].astype(str).values
            ms=st.radio("Méthode", ["8.3.1 — FST par fenêtres", "8.3.2 — Tajima's D", "8.3.3 — Score composite (CSS)"], horizontal=True, key="sel_m")
            if ms.startswith("8.3.1"):
                c1,c2,c3=st.columns(3); ws=c1.slider("Fenêtre",10,200,50,key="wf"); sp=c2.slider("Pas",5,100,25,key="wp"); zt=c3.slider("Seuil z",2.0,5.0,3.0,0.1,key="wz")
                if st.button("▶ Scanner FST",type="primary",key="wf_btn"):
                    with st.spinner("Scan FST..."): w=windowed_fst(gu,su,lab,ws,sp); w=detect_outliers(w,zt); st.session_state.win_fst=w
                    st.success(f"✅ {len(w)} fenêtres")
                if st.session_state.get("win_fst") is not None:
                    w=st.session_state.win_fst; st.plotly_chart(plot_win_manh(w,'MEAN_FST',f'FST fenêtres'),use_container_width=True,key="plot_win_fst")
                    st.dataframe(w.head(20)[['CHR','START','END','N_SNP','MEAN_FST','MAX_FST','Z_FST','TOP_SNP']],use_container_width=True)
            elif ms.startswith("8.3.2"):
                rs=st.selectbox("Race",sorted(np.unique(lab)),key="tr"); c1,c2=st.columns(2); wt=c1.slider("Fenêtre",10,100,25,key="wt"); stp=c2.slider("Pas",5,50,10,key="stp")
                if st.button(f"▶ Tajima's D — {rs}",type="primary",key="t_btn"):
                    m=lab==rs
                    if m.sum()<5: st.error("⚠️ < 5 individus")
                    else:
                        with st.spinner("Tajima..."): t=tajimas_scan(gu[m],su,wt,stp); st.session_state.taj_df=t; st.session_state.taj_race=rs
                        st.success(f"✅ {len(t)} fenêtres")
                if st.session_state.get("taj_df") is not None:
                    t=st.session_state.taj_df; st.plotly_chart(plot_win_manh(t,'TAJIMA_D',f"Tajima D"),use_container_width=True,key="plot_tajima"); st.dataframe(t.sort_values('TAJIMA_D').head(20),use_container_width=True)
            else:
                if st.button("▶ CSS",type="primary",key="css_btn"):
                    with st.spinner("CSS..."): c=composite_score(gu,su,lab); st.session_state.css_df=c
                    st.success(f"✅ {len(c)} SNPs")
                if st.session_state.get("css_df") is not None:
                    c=st.session_state.css_df; cp=c.copy(); cp["P_pseudo"]=np.exp(-cp["CSS"]); st.plotly_chart(plot_manhattan(cp.rename(columns={"P_pseudo":"P"}),yc="P",title="CSS"),use_container_width=True,key="plot_css_manh"); st.dataframe(c.head(20),use_container_width=True)

        st.divider(); st.subheader("🧬 Annotation des gènes (distance en cM)")
        snp_full = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt)
        has_cm_data = (snp_full is not None and "CM" in snp_full.columns and not np.all(snp_full["CM"].astype(float).values == 0))
        if has_cm_data: st.success(f"✅ SNPs avec cM valide")
        else: st.warning("⚠️ CM=0, fallback kb.")
        gff_f=st.file_uploader("GFF3/GTF",type=["gff","gff3","gtf","gz"],key="sel_gff_file")
        sources=[]
        if st.session_state.get("win_fst") is not None: sources.append("FST outliers")
        if st.session_state.get("taj_df") is not None: sources.append("Tajima extrêmes")
        if st.session_state.get("css_df") is not None: sources.append("CSS top")
        if st.session_state.get("pair_res_race") is not None: sources.append("Race Pairwise")
        if st.session_state.get("pair_res_group") is not None: sources.append("Group Pairwise")
        sources.append("Manuel")
        if gff_f is not None:
            try:
                with st.spinner("Parsing GFF3..."): gff=parse_gff3(gff_f.read()); st.session_state.sel_gff=gff
                st.success(f"✅ {len(gff)} gènes")
                c1,c2=st.columns(2); metric=c1.radio("Unité",["cM (génétique)","kb (physique)"],index=0 if has_cm_data else 1,horizontal=True,key="ann_metric")
                if metric.startswith("cM"): window_cm=c2.slider("Fenêtre cM",1,50,10,1); wk_fallback=100
                else: wk_fallback=c2.slider("Fenêtre kb",1,500,50,5); window_cm=10
                c1,c2=st.columns(2); src=c1.selectbox("Source",sources,key="ann_src")
                to_ann=None
                if src=="FST outliers": to_ann=st.session_state.win_fst.head(50)[["CHR","START","TOP_SNP"]].rename(columns={"START":"BP","TOP_SNP":"SNP"})
                elif src=="Tajima extrêmes": ex=st.session_state.taj_df.sort_values("TAJIMA_D").head(20); to_ann=ex[["CHR","MID"]].rename(columns={"MID":"BP"}); to_ann["SNP"]=[f"Taj_{i+1}" for i in range(len(to_ann))]
                elif src=="CSS top": to_ann=st.session_state.css_df.head(30)[["SNP","CHR","BP"]]
                elif src in ("Race Pairwise", "Group Pairwise"): to_ann=st.session_state["pair_res_race" if src=="Race Pairwise" else "pair_res_group"].head(30)[["SNP","CHR","BP"]]
                else:
                    c1,c2=st.columns(2); chm=c1.text_input("Chr","14",key="ann_mc"); pm=c2.number_input("Pos",0,200_000_000,25_000_000,100_000,key="ann_mp"); to_ann=pd.DataFrame({"SNP":[f"m_{chm}_{pm}"],"CHR":[str(chm)],"BP":[int(pm)]})
                if to_ann is not None and len(to_ann)>0 and st.button("🔍 Annoter",type="primary",key="ann_btn"):
                    with st.spinner("Annotation..."):
                        if metric.startswith("cM"): ann=annotate_genes_cm(to_ann,snp_full,gff,window_cm=window_cm,window_kb_fallback=wk_fallback)
                        else:
                            ann=annotate_genes(to_ann,gff,wk=wk_fallback)
                            if not ann.empty: ann["WINDOW"]=f"±{wk_fallback} kb"; ann["DIST_CM"]=None
                        st.session_state.sel_ann=ann
                        if src=="Race Pairwise": st.session_state.sel_ann_race = ann
                        elif src=="Group Pairwise": st.session_state.sel_ann_group = ann
                    if ann.empty: st.warning("Aucun gène")
                    else: st.success(f"✅ {len(ann)} associations")
                
                if st.session_state.get("sel_ann") is not None and len(st.session_state.sel_ann)>0:
                    ann=st.session_state.sel_ann
                    st.dataframe(ann,use_container_width=True)
                    st.download_button("⬇ CSV",ann.to_csv(index=False).encode(),"sel_ann_cm.csv","text/csv")
                    
                    sc=ann["SNP"].unique().tolist()
                    ss=st.selectbox("Visualiser",sc,key="ann_track")
                    row=ann[ann["SNP"]==ss].iloc[0].to_dict()
                    si={"SNP":row["SNP"],"CHR":row["CHR"],"BP":row.get("SNP_POS_BP",row.get("SNP_POS",0))}
                    wk_viz=int(window_cm*10) if metric.startswith("cM") else wk_fallback
                    f=plot_gene_track(si,gff,wk_viz)
                    if f: st.plotly_chart(f,use_container_width=True,key="plot_gene_track")
                    
                    st.markdown("### 🏆 Gènes les plus proches (top 3 / SNP)")
                    if "DIST_CM" in ann and ann["DIST_CM"].notna().any(): 
                        top=ann.sort_values(["SNP","DIST_CM"]).groupby("SNP").head(3)[["SNP","GENE","DIST_CM","DIST_KB"]]
                    else: 
                        top=ann.sort_values(["SNP","DIST_KB"]).groupby("SNP").head(3)[["SNP","GENE","DIST_KB"]]
                    st.dataframe(top,use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")
        else: st.info("💡 [Télécharger GFF3 bovin](https://ftp.ensembl.org/pub/current_gff3/bos_taurus/)")

    # ═══════════════ GWAS Classique ═══════════════
    with tabs[6]:
        st.markdown("### 1️⃣ Mapping ID")
        mf=st.file_uploader("CSV mapping (Animal + IID)",type=["csv","txt"],key="gwas_map")
        if mf:
            try: st.session_state.mapping_df=pd.read_csv(mf); st.success("✅"); st.dataframe(st.session_state.mapping_df.head(3))
            except Exception as e: st.error(f"❌ {e}")
        st.markdown("### 2️⃣ Phénotypes")
        pf=st.file_uploader("Phénotypes",type=["csv","txt","gz"],key="gwas_pheno_file")
        c1,c2=st.columns(2)
        with c1:
            if pf and st.button("📥 Charger",key="load_ph_btn",use_container_width=True):
                try:
                    mg,_=match_pheno(st.session_state.ind_filt,load_pheno(pf.read(),pf.name),st.session_state.mapping_df)
                    st.session_state.pheno_df=mg; tc=next((c for c in mg.columns if c not in ("FID","IID","Animal")),None); n_ok=pd.to_numeric(mg[tc],errors="coerce").notna().sum() if tc else 0
                    if n_ok<10: st.error(f"⚠️ {n_ok} matchés.")
                    else: st.success(f"✅ {n_ok} ind")
                except Exception as e: st.error(f"❌ {e}")
        with c2:
            if st.button("🎲 Simuler",key="sim_ph_btn",use_container_width=True):
                ind=st.session_state.ind_filt; n=len(ind); rng=np.random.default_rng(42)
                tr=(st.session_state.pca_s[:,0]*2+rng.normal(0,0.5,n)) if (st.session_state.pca_s is not None and len(st.session_state.pca_s)==n) else rng.normal(0,1,n)
                st.session_state.pheno_df=pd.DataFrame({"FID":ind["FID"].values,"IID":ind["IID"].values,"trait":tr}); st.success("✅")
        if st.session_state.pheno_df is not None:
            cols=[c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            if cols:
                pc=st.selectbox("Phénotype",cols); ib=st.checkbox("Binaire")
                ps=pd.to_numeric(st.session_state.pheno_df[pc],errors="coerce"); nv=ps.notna().sum()
                if nv<10: st.error(f"❌ {nv} valides")
                else:
                    st.info(f"✅ {nv} valides")
                    if st.button("🚀 Lancer GWAS",type="primary",key="gwas_btn"):
                        try:
                            gg=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                            sg=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                            cv=st.session_state.pca_s[:,:5] if st.session_state.pca_s is not None else np.empty((gg.shape[0],0))
                            with st.spinner("GWAS..."):
                                r=run_gwas(gg,ps.values,st.session_state.king_K,cv,is_bin=ib); r=annotate_gwas(r,sg)
                                st.session_state.gwas_res=r; st.session_state.lambda_gc=lambda_gc(r["P"].values) if len(r) else np.nan
                            st.success(f"✅ {len(r)} SNPs · λGC={st.session_state.lambda_gc:.3f}")
                            st.plotly_chart(plot_manhattan(r),use_container_width=True,key="plot_gwas_manh_classic")
                        except Exception as e: st.error(f"❌ {e}")

    # ═══════════════ Post-GWAS ═══════════════
    with tabs[7]:
        if st.session_state.gwas_res is not None:
            pip=fine_mapping(st.session_state.gwas_res); st.dataframe(pip[["SNP","P","PIP"]].head(10),use_container_width=True); st.plotly_chart(plot_pip(pip),use_container_width=True,key="plot_pip_postgwas")
        else: st.warning("Lancez GWAS d'abord")

    # ═══════════════ GBLUP ═══════════════
    with tabs[8]:
        if st.session_state.pheno_df is not None:
            cols=[c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            pc=st.selectbox("Phénotype GBLUP",cols,key="gb_pc")
            if st.button("▶ GBLUP + validation",key="gb_btn"):
                try:
                    gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    ps=pd.to_numeric(st.session_state.pheno_df[pc],errors="coerce")
                    with st.spinner("REML..."): df,st_=run_gblup(gu,ps.values); st.session_state.gebv_df=df; st.session_state.gblup_stats=st_
                    st.success("✅"); st.plotly_chart(plot_gblup(df),use_container_width=True,key="plot_gblup"); st.json(st_)
                except Exception as e: st.error(f"❌ {e}")
        else: st.warning("Chargez phénotypes")

    # ═══════════════ Annotation Dédiée ═══════════════
    with tabs[9]:
        gff_f=st.file_uploader("GFF3/GTF",type=["gff","gff3","gtf","gz"],key="ann_ded_gff")
        src=st.radio("Source",["Sélection","GWAS"],horizontal=True,key="ann_ded_src"); wk=st.slider("Fenêtre kb",1,500,50,key="ann_ded_wk")
        if gff_f:
            try:
                gff=parse_gff3(gff_f.read()); st.success(f"✅ {len(gff)} gènes"); snps=None
                if src=="Sélection" and st.session_state.win_fst is not None: snps=st.session_state.win_fst.head(20).rename(columns={"START":"BP","TOP_SNP":"SNP"})[["SNP","CHR","BP"]]
                elif src=="GWAS" and st.session_state.gwas_res is not None: snps=st.session_state.gwas_res.sort_values("P").head(20)
                if snps is not None and st.button("🔍 Annoter",key="ann_ded_btn"): st.session_state.ann_df=annotate_genes(snps,gff,wk); st.success("✅")
                if st.session_state.ann_df is not None: st.dataframe(st.session_state.ann_df,use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")

    # ═══════════════ TAB 10 : ANNOTATION FONCTIONNELLE (VEP/NCBI) ═══════════════
    with tabs[10]:
        st.subheader("🧬 Annotation Fonctionnelle Avancée (VEP & NCBI)")
        
        st.markdown("### 1️⃣ Sélectionnez les SNPs à analyser")
        sources_snp = []
        if st.session_state.get("pair_res_group") is not None: sources_snp.append("Top SNPs Groupe")
        if st.session_state.get("pair_res_race") is not None: sources_snp.append("Top SNPs Race")
        if st.session_state.get("gwas_res") is not None: sources_snp.append("Top SNPs GWAS Classique")
        if st.session_state.get("win_fst") is not None: sources_snp.append("Top SNPs FST")
        sources_snp.append("Saisie manuelle (CHR:BP)")
        
        src_snp = st.selectbox("Source des SNPs", sources_snp, key="vep_src")
        
        df_to_vep = None
        if src_snp == "Top SNPs Groupe":
            df_to_vep = st.session_state.pair_res_group.head(50)[["SNP", "CHR", "BP"]].copy()
        elif src_snp == "Top SNPs Race":
            df_to_vep = st.session_state.pair_res_race.head(50)[["SNP", "CHR", "BP"]].copy()
        elif src_snp == "Top SNPs GWAS Classique":
            df_to_vep = st.session_state.gwas_res.sort_values("P").head(50)[["SNP", "CHR", "BP"]].copy()
        elif src_snp == "Top SNPs FST":
            df_to_vep = st.session_state.win_fst.head(50).copy()
            df_to_vep["BP"] = df_to_vep["START"]
            df_to_vep = df_to_vep[["TOP_SNP", "CHR", "BP"]].rename(columns={"TOP_SNP": "SNP"})
        else:
            txt = st.text_area("Entrez les coordonnées (ex: 4:52014692 ou 4:52014692-52014692, 1 par ligne)", key="vep_manual")
            if txt.strip():
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                manual_data = []
                for i, line in enumerate(lines):
                    parts = line.replace("-", ":").split(":")
                    if len(parts) >= 2:
                        manual_data.append({"SNP": f"Manual_{i+1}", "CHR": parts[0], "BP": int(parts[1])})
                df_to_vep = pd.DataFrame(manual_data)

        if df_to_vep is not None and not df_to_vep.empty:
            st.info(f"📊 {len(df_to_vep)} SNPs prêts pour l'analyse VEP.")
            st.markdown("### 2️⃣ Impact fonctionnel des variants (Ensembl VEP)")
            st.caption("Détermine si le SNP est dans un exon, un intron, s'il change un acide aminé (faux-sens) ou non (synonyme).")
            
            if st.button("🔬 Lancer l'analyse VEP", type="primary", key="run_vep_btn"):
                with st.spinner("Interrogation de l'API Ensembl VEP..."):
                    vep_res = run_vep_region_api(df_to_vep)
                    st.session_state.vep_res = vep_res
                    if not vep_res.empty: 
                        st.success("✅ Analyse VEP terminée.")
            
            if st.session_state.get("vep_res") is not None and not st.session_state.vep_res.empty:
                st.dataframe(st.session_state.vep_res, use_container_width=True)
                st.download_button("⬇ Export VEP CSV", st.session_state.vep_res.to_csv(index=False).encode(), "vep_results.csv", "text/csv", key="dl_vep")

        st.markdown("### 3️⃣ Fonctions officielles des gènes (NCBI Gene)")
        st.caption("Récupère la description officielle et le résumé fonctionnel de chaque gène via l'API NCBI.")
        
        genes_to_fetch = []
        if st.session_state.get("vep_res") is not None:
            genes_to_fetch = st.session_state.vep_res["GENE"].dropna().unique().tolist()
            genes_to_fetch = [g for g in genes_to_fetch if g not in ("N/A", "Intergenic/Non-codant")]
        elif st.session_state.get("sel_ann") is not None:
            genes_to_fetch = st.session_state.sel_ann["GENE"].dropna().unique().tolist()
            
        if not genes_to_fetch:
            st.warning("⚠️ Aucun gène disponible. Lancez d'abord VEP ou l'annotation classique.")
        else:
            st.info(f"🧬 {len(genes_to_fetch)} gènes uniques identifiés : {', '.join(genes_to_fetch[:5])}...")
            if st.button("🏛️ Récupérer les infos NCBI", type="primary", key="run_ncbi_btn"):
                with st.spinner("Interrogation de NCBI Datasets..."):
                    ncbi_res = fetch_ncbi_gene_info(genes_to_fetch[:20])
                    st.session_state.ncbi_gene_res = ncbi_res
                    if not ncbi_res.empty: st.success("✅ Informations NCBI récupérées.")
            
            if st.session_state.get("ncbi_gene_res") is not None and not st.session_state.ncbi_gene_res.empty:
                st.dataframe(st.session_state.ncbi_gene_res, use_container_width=True)
                st.download_button("⬇ Export NCBI Gene CSV", st.session_state.ncbi_gene_res.to_csv(index=False).encode(), "ncbi_gene_info.csv", "text/csv", key="dl_ncbi_gene")

        st.markdown("### 4️⃣ Validation des variants (NCBI dbSNP)")
        if st.button("🔎 Vérifier les rsID sur dbSNP", key="run_dbsnp_btn"):
            rs_snps = [s for s in df_to_vep["SNP"].astype(str).tolist() if str(s).startswith("rs")]
            if not rs_snps:
                st.warning("Aucun rsID trouvé dans la liste (les SNPs sont des AX- ou des coordonnées).")
            else:
                with st.spinner("Interrogation de dbSNP..."):
                    dbsnp_res = fetch_ncbi_dbsnp_info(rs_snps[:20])
                    st.session_state.ncbi_snp_res = dbsnp_res
                if not dbsnp_res.empty: st.success("✅ Validation dbSNP terminée.")
        
        if st.session_state.get("ncbi_snp_res") is not None and not st.session_state.ncbi_snp_res.empty:
            st.dataframe(st.session_state.ncbi_snp_res, use_container_width=True)
            st.download_button("⬇ Export dbSNP CSV", st.session_state.ncbi_snp_res.to_csv(index=False).encode(), "dbsnp_info.csv", "text/csv", key="dl_dbsnp")

    # ═══════════════ Export ═══════════════
    with tabs[11]:
        if st.session_state.gt_filt is not None:
            if st.button("📦 PLINK",key="pl_btn"):
                try:
                    gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    st.session_state._plink=build_plink(gu,st.session_state.ind_filt,su); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state._plink: st.download_button("⬇ PLINK",st.session_state._plink,"bovine.zip","application/zip")
            if st.button("📄 VCF",key="vcf_btn"):
                try:
                    gu=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    st.session_state._vcf=build_vcf(gu,st.session_state.ind_filt,su); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state._vcf: st.download_button("⬇ VCF",st.session_state._vcf.encode(),"bovine.vcf","text/plain")

    # ═══════════════ MÉTA-ANALYSE & ENRICHISSEMENT ═══════════════
    with tabs[12]:
        st.subheader("🔬 Méta-analyse Race × Groupe & Enrichissement fonctionnel")

        st.markdown("## 1️⃣ Test de permutation")
        if st.session_state.gt_filt is not None:
            gu_perm = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt)
            su_perm = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt)
            iu_perm = st.session_state.ind_filt
            races_perm, _ = get_pairs(iu_perm["FID"].astype(str).values)
            c1, c2 = st.columns(2); ra_p = c1.selectbox("Race A (permutation)", races_perm, index=0, key="perm_ra"); rb_p = c2.selectbox("Race B (permutation)", races_perm, index=1 if len(races_perm) > 1 else 0, key="perm_rb")
            c1, c2 = st.columns(2); n_perm = c1.slider("Nombre de permutations", 50, 1000, 100, 50, key="perm_n"); maf_p = c2.slider("MAF", 0.0, 0.3, 0.05, key="perm_maf")
            if ra_p != rb_p and st.button("🧪 Lancer test", type="primary", key="perm_btn"):
                try:
                    with st.spinner(f"{n_perm} permutations..."): orig, p_emp, n_obs, perm_counts = permutation_test_race(gu_perm, su_perm, iu_perm["FID"].astype(str).values, ra_p, rb_p, n_perm=n_perm)
                    if orig.empty: st.warning("Aucun SNP testé.")
                    else:
                        c1, c2, c3 = st.columns(3); c1.metric("Observés", n_obs); c2.metric("p-value empirique", f"{p_emp:.4f}"); c3.metric("Seuil", "p < 0.05" if p_emp < 0.05 else "p ≥ 0.05")
                        st.plotly_chart(px.histogram(x=perm_counts, nbins=20, labels={"x": "Nb SNPs significatifs (permutations)"}, title="Distribution nulle"), use_container_width=True, key="perm_hist")
                except Exception as e: st.error(f"❌ {e}")

        st.divider(); st.markdown("## 2️⃣ Méta-analyse Fisher")
        res_r = st.session_state.get("pair_res_race"); res_g = st.session_state.get("pair_res_group")
        if res_r is None or res_g is None: st.warning("⚠️ Lancez une analyse Race × Race puis Groupe × Groupe.")
        else:
            fish = combine_pvalues_fisher(res_r, res_g)
            if fish is None: st.warning("Aucun SNP commun.")
            else:
                st.success(f"✅ {len(fish)} SNPs communs combinés"); st.dataframe(fish.head(30), use_container_width=True)
                st.plotly_chart(px.scatter(fish, x="P_race", y="P_group", hover_data=["SNP"], title="Fisher : P_race vs P_group"), use_container_width=True, key="fisher_scatter")
                st.download_button("⬇ Résultats Fisher", fish.to_csv(index=False).encode(), "fisher_meta_analysis.csv", "text/csv", key="fisher_dl")

        st.divider(); st.markdown("## 3️⃣ Enrichissement fonctionnel")
        sources_genes = []
        if res_r is not None and not res_r.empty: sources_genes.append("SNPs Race (top 50)")
        if res_g is not None and not res_g.empty: sources_genes.append("SNPs Groupe (top 50)")
        if st.session_state.get("sel_ann") is not None: sources_genes.append("Gènes annotés (sel_ann)")
        sources_genes.append("Liste manuelle")
        if sources_genes:
            c1, c2 = st.columns(2); src_g = c1.selectbox("Source des gènes", sources_genes, key="enr_src"); outdir = c2.text_input("Dossier sortie", "", key="enr_outdir")
            gmt_file = st.file_uploader("Fichier GMT (optionnel)", type=["gmt"], key="enr_gmt")
            gene_sets = None
            if gmt_file is not None: gene_sets = parse_gmt(gmt_file.read()); st.success(f"✅ {len(gene_sets)} gene sets")
            genes_to_enrich = None
            if src_g.startswith("SNPs Race"): genes_to_enrich = res_r["SNP"].head(50).tolist()
            elif src_g.startswith("SNPs Groupe"): genes_to_enrich = res_g["SNP"].head(50).tolist()
            elif src_g == "Gènes annotés (sel_ann)":
                ann = st.session_state.get("sel_ann")
                if ann is not None and "GENE" in ann.columns: genes_to_enrich = ann["GENE"].dropna().unique().tolist()
            else:
                txt = st.text_area("Liste de gènes (1/ligne)", key="enr_manual")
                if txt.strip(): genes_to_enrich = [g.strip() for g in txt.splitlines() if g.strip()]
            if genes_to_enrich and st.button("🔬 Lancer enrichissement", type="primary", key="enr_btn"):
                try:
                    with st.spinner("Enrichissement..."): enr = run_enrichment(genes_to_enrich, gene_sets=gene_sets, organism="bovine", outdir=outdir or None)
                    if enr.empty: st.warning("Aucun terme enrichi.")
                    else:
                        st.success(f"✅ {len(enr)} termes"); st.dataframe(enr.head(30), use_container_width=True)
                        st.download_button("⬇ Résultats Enrichr", enr.to_csv(index=False).encode(), "enrichment_results.csv", "text/csv", key="enr_dl")
                except Exception as e: st.error(f"❌ {e}")

        st.divider(); st.markdown("## 4️⃣ Réseau STRING")
        genes_string = []
        if st.session_state.get("sel_ann") is not None:
            ann = st.session_state.sel_ann
            if "GENE" in ann.columns: genes_string = ann["GENE"].dropna().unique().tolist()
        st.info(f"Gènes disponibles : {len(genes_string)}")
        c1, c2 = st.columns(2); score_str = c1.selectbox("Score", [150, 400, 700, 900], index=1, key="string_score"); max_genes = c2.slider("Nb max", 5, 100, 30, key="string_max")
        if st.button("🕸️ Construire réseau STRING", type="primary", key="string_btn"):
            try:
                if len(genes_string) < 2: st.warning("Il faut au moins 2 gènes annotés.")
                else:
                    with st.spinner("Requête STRING..."): edges, nodes = query_string_network(genes_string[:max_genes], species=9913, required_score=int(score_str))
                    if edges.empty: st.warning("Aucune interaction.")
                    else:
                        st.success(f"✅ {len(edges)} interactions"); fig = plot_string_network(edges, nodes)
                        if fig: st.plotly_chart(fig, use_container_width=True, key="string_network_plot")
                        st.dataframe(edges, use_container_width=True); st.download_button("⬇ Interactions STRING", edges.to_csv(index=False).encode(), "string_network.csv", "text/csv", key="string_dl")
            except Exception as e: st.error(f"❌ {e}")

if __name__ == "__main__":
    main()

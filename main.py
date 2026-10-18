"""
🐄 Bovine Genomics Pipeline v8.3 — Lab-Grade + Signatures de Sélection
"""
# §1 ── Imports & Config
import gzip, io, os, warnings, zipfile, subprocess, tempfile, shutil
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="🐄 Bovine Pipeline v8.3", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")

ARS_UCD12_LENGTHS = {"1":158_534_110,"2":136_231_102,"3":121_005_158,"4":120_000_166,"5":120_089_699,"6":117_806_340,"7":110_682_743,"8":113_384_748,"9":105_708_134,"10":103_308_737,"11":107_310_763,"12":91_163_125,"13":84_246_514,"14":84_648_346,"15":85_207_080,"16":81_726_628,"17":75_176_999,"18":66_059_976,"19":64_089_169,"20":72_042_983,"21":71_599_096,"22":61_416_492,"23":52_531_573,"24":62_384_193,"25":42_959_610,"26":51_680_158,"27":46_772_073,"28":46_333_854,"29":51_319_414}
BOVINE_AUTOSOMES = [str(i) for i in range(1,30)]
HWE_EXACT_MAX_SNP = 20_000

# §2 ── Cache Streamlit
def _hash_arr(x):
    if not isinstance(x, np.ndarray) or x.size == 0: return "empty"
    if x.dtype.kind in ("U","S","O"): return f"arr|{x.shape}|{'|'.join(map(str,x.ravel()[:2000]))}"
    return f"{x.shape}|{x.dtype}|{float(np.nansum(x))}|{float(np.nansum(np.abs(x)))}|{float(np.nansum(x*x))}"

def _hash_any(x):
    try: return _hash_arr(np.asarray(x))
    except: return f"{type(x).__name__}|{len(x) if hasattr(x,'__len__') else 0}"

HASH_FUNCS = {np.ndarray:_hash_arr, pd.Series:_hash_any, pd.Index:_hash_any}
for _n in ("ArrowStringArray","StringArray","IntegerArray","FloatingArray","BooleanArray","NumpyExtensionArray","PandasArray","DatetimeArray","TimedeltaArray","PeriodArray","IntervalArray","Categorical"):
    _c = getattr(pd.arrays,_n,None) or getattr(pd.core.arrays,_n,None)
    if _c: HASH_FUNCS[_c]=_hash_any

def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    gt2 = gt.astype(np.float32, copy=True)
    cm = np.nanmean(gt2, axis=0); cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(gt2)
    if nm.any(): gt2[nm] = np.take(cm, np.where(nm)[1])
    return gt2

def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES], dtype=float)
    return lens / lens.sum()

def _chr_sort_key(c):
    s = str(c).upper().replace("CHR","").replace("CHROMOSOME","")
    if s.isdigit(): return (0,int(s),"")
    sp = {"X":100,"Y":101,"MT":102,"M":102}
    return (1,sp[s],"") if s in sp else (2,0,s)

def _read_text(raw):
    if len(raw)>=2 and raw[:2]==b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8",errors="replace")
    return raw.decode("utf-8",errors="replace")

# §3 ── I/O Parsing
def sample_to_fid_iid(s, fp=2):
    n = str(s).strip()
    for ext in (".CEL",".cel",".TXT",".txt",".gz",".GZ"):
        if n.endswith(ext): n=n[:-len(ext)]
    n = n.replace("(","_").replace(")","").replace("[","_").replace("]","").replace(" ","_").replace("/","_")
    while "__" in n: n=n.replace("__","_")
    p = n.strip("_").split("_")
    return ("_".join(p[:fp]) if len(p)>=fp else p[0]), n

def _clean_allele(a):
    a = str(a).strip() if a is not None else "0"
    return "0" if a in ("",".","-","NA","nan","None","0","00") else a

def parse_axiom(raw_bytes, filename=""):
    raw = gzip.decompress(raw_bytes) if filename.endswith(".gz") or raw_bytes[:2]==b"\x1f\x8b" else raw_bytes
    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8",errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    start = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample") or "Sample Filename" in lines[0][:50]) else 0
    samples, gtypes = [], []
    for line in lines[start:]:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts)<2: continue
        samples.append(parts[0].strip()); alleles=[]
        for cell in parts[1:]:
            tk = cell.split()
            if len(tk)==2: alleles.append((_clean_allele(tk[0]),_clean_allele(tk[1])))
            elif len(tk)==1 and len(tk[0])==2 and all(ch in "ACGT0" for ch in tk[0]): alleles.append((tk[0][0],tk[0][1]))
            else: alleles.append(("0","0"))
        gtypes.append(alleles)
    if not samples: raise ValueError("Aucun échantillon")
    common = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i,r in enumerate(gtypes): gtypes[i]=r[:common]+[("0","0")]*(common-len(r))
    return samples, gtypes, common

def parse_map(map_bytes):
    rows=[]
    for line in _read_text(map_bytes).splitlines():
        p=line.split()
        if len(p)>=4:
            try: cm=float(p[2]) if p[2] not in ("0",".","NA") else 0.0
            except: cm=0.0
            try: bp=int(float(p[3]))
            except: continue
            rows.append({"CHR":str(p[0]),"SNP":p[1],"CM":cm,"BP":bp})
    if not rows: raise ValueError(".map vide")
    return pd.DataFrame(rows)

def parse_ped(ped_bytes, n_snp_map):
    lines = [l.strip() for l in _read_text(ped_bytes).splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    n_snp_ped = max(0, (len(lines[0].split())-6)//2)
    if n_snp_ped < 7: raise ValueError(f"PED a {n_snp_ped} SNPs, format invalide")
    n_snp = min(n_snp_map, n_snp_ped); exp_cols = 6+2*n_snp
    fids, iids, rows = [], [], []
    for line in lines:
        p=line.split()
        if len(p)>=exp_cols:
            fids.append(p[0]); iids.append(p[1]); rows.append(p[6:6+2*n_snp])
    if not iids: raise ValueError("Aucun individu chargé")
    ac = [Counter() for _ in range(n_snp)]
    for r in rows:
        for j in range(n_snp):
            if r[2*j]!="0" and r[2*j+1]!="0": ac[j].update([r[2*j],r[2*j+1]])
    minor = [min(c,key=c.get) if len(c)>=2 else None for c in ac]
    gt = np.full((len(iids),n_snp), np.nan, dtype=np.float32)
    for i,r in enumerate(rows):
        for j in range(n_snp):
            if minor[j] and r[2*j]!="0" and r[2*j+1]!="0": gt[i,j] = (r[2*j]==minor[j])+(r[2*j+1]==minor[j])
    return gt, pd.DataFrame({"FID":fids,"IID":iids})

def parse_vcf(vcf_bytes):
    samples, rows = [], []
    for line in _read_text(vcf_bytes).splitlines():
        if line.startswith("#CHROM"): samples = line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p = line.split("\t")
            if len(p)>=10 and "," not in p[4]:
                fmt = p[8].split(":")
                if "GT" in fmt:
                    gi = fmt.index("GT"); gts = np.full(len(samples), np.nan, dtype=np.float32)
                    for k,s in enumerate(p[9:]):
                        al = s.split(":")[gi].replace("|","/").split("/")
                        if not any(a in (".","") for a in al):
                            try: gts[k]=sum(int(a) for a in al)
                            except: pass
                    rows.append({"CHR":str(p[0]),"SNP":p[2] or f"{p[0]}:{p[1]}","CM":0.0,"BP":int(p[1]),"A1":p[3],"A2":p[4],"GT":gts})
    if not rows: raise ValueError("Aucun variant VCF")
    gt = np.zeros((len(samples),len(rows)), dtype=np.float32)
    for j,r in enumerate(rows):
        gt[:,j]=r["GT"]; v=gt[:,j][~np.isnan(gt[:,j])]
        if len(v)>0 and v.mean()/2.0>0.5: gt[:,j]=np.where(np.isnan(gt[:,j]),np.nan,2.0-gt[:,j])
    return gt, pd.DataFrame({"FID":samples,"IID":samples}), pd.DataFrame([{k:v for k,v in r.items() if k!="GT"} for r in rows])

def parse_gff3(gff_bytes):
    genes=[]
    for line in _read_text(gff_bytes).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p=line.split("\t")
        if len(p)>=9 and p[2] in ("gene","mRNA"):
            attrs=p[8]; name="Unknown"
            if "Name=" in attrs: name=attrs.split("Name=")[1].split(";")[0]
            elif 'gene_name "' in attrs: name=attrs.split('gene_name "')[1].split('"')[0]
            elif 'gene_id "' in attrs: name=attrs.split('gene_id "')[1].split('"')[0]
            genes.append({"CHR":str(p[0]),"START":int(p[3]),"END":int(p[4]),"GENE":name})
    return pd.DataFrame(genes).drop_duplicates(subset=["CHR","START","END","GENE"])

def load_pheno(file_bytes, filename=""):
    raw = gzip.decompress(file_bytes) if filename.endswith(".gz") else file_bytes
    text = raw.decode("utf-8",errors="replace")
    sep = "\t" if text.count("\t")>text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_col = next((c for c in ["Animal","IID","id","ID"] if c in p.columns), None)
        if id_col is None: raise ValueError(f"Colonne ID absente. Dispo: {list(p.columns)}")
        p[id_col]=p[id_col].astype(str)
        map_animal = next((c for c in m.columns if c!="IID"), None)
        if map_animal is None or "IID" not in m.columns: raise ValueError("CSV mapping doit avoir 'IID' + nom animal")
        m[map_animal]=m[map_animal].astype(str); m["IID"]=m["IID"].astype(str)
        p_merged = p.merge(m, left_on=id_col, right_on=map_animal, how="left")
        ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str)
        p_merged["_k"]=p_merged["IID"].astype(str)
        return ind[["_k","FID","IID"]].merge(p_merged.drop(columns=["IID"],errors="ignore"),on="_k",how="left").drop(columns="_k"), id_col
    id_col = next((c for c in ["IID","id","ID","sample","Animal","animal"] if c in pheno_df.columns), None)
    if id_col is None: raise ValueError(f"Aucune colonne ID. Dispo: {list(pheno_df.columns)}")
    p = pheno_df.copy(); p[id_col]=p[id_col].astype(str)
    ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str); p["_k"]=p[id_col]
    return ind[["_k","FID","IID"]].merge(p.drop(columns=[id_col]),on="_k",how="left").drop(columns="_k"), id_col

def generate_demo(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng=np.random.default_rng(seed)
    pops=["AND","EBG","ELN","EZP","FGN","LJR","NAR","NBD","NKA","PRS"][:min(n_pop,10)]
    p_anc=rng.beta(0.6,0.6,n_snp); gt=np.zeros((n_ind,n_snp),dtype=np.float32); ind_rows=[]; k=0
    for p in pops:
        p_k=np.clip(p_anc+rng.normal(0,0.15,n_snp),0.02,0.98)
        for i in range(n_ind//len(pops)+(1 if pops.index(p)<n_ind%len(pops) else 0)):
            gt[k]=(rng.random(n_snp)<p_k).astype(np.int8)+(rng.random(n_snp)<p_k).astype(np.int8)
            ind_rows.append({"FID":p,"IID":f"{p}_{i+1:03d}"}); k+=1
    gt[rng.random(gt.shape)<0.02]=np.nan
    chr_names=rng.choice(BOVINE_AUTOSOMES,n_snp,p=_autosome_weights())
    bps=np.array([rng.integers(1,ARS_UCD12_LENGTHS[c]) for c in chr_names],dtype=np.int64)
    snp_df=pd.DataFrame({"CHR":chr_names,"SNP":[f"rs{i:07d}" for i in range(n_snp)],"CM":bps/1_000_000,"BP":bps})
    return gt, pd.DataFrame(ind_rows), snp_df.sort_values(by=["CHR","BP"],key=lambda x: x.map(_chr_sort_key) if x.name=="CHR" else x).reset_index(drop=True)

def reconstruct_fid(ind_df):
    df = ind_df.copy()
    if df["FID"].nunique()<=1: df["FID"]=df["IID"].astype(str).str.split("_").str[0]
    return df

# §4 ── Métriques QC
def miss_ind(gt): return np.isnan(gt).mean(axis=1)
def miss_snp(gt): return np.isnan(gt).mean(axis=0)
def maf(gt): p=np.nanmean(gt,axis=0)/2.0; return np.minimum(p,1.0-p)
def hetero(gt): return np.nanmean(gt==1, axis=1)

def hwe_exact(n_het, n_h0, n_h2):
    n = n_het+n_h0+n_h2
    if n==0 or (n_het==0 and (n_h0==0 or n_h2==0)): return 1.0
    rare = 2*min(n_h0,n_h2)+n_het; mid = (rare*(2*n-rare))//(2*n)
    if mid%2!=rare%2: mid+=1
    probs=np.zeros(rare+1); probs[mid]=1.0; s=1.0
    ch,cr,cc = mid, (rare-mid)//2, n-mid-(rare-mid)//2
    while ch<=rare-2:
        probs[ch+2]=probs[ch]*4*cr*cc/((ch+2)*(ch+1)); s+=probs[ch+2]; ch+=2; cr-=1; cc-=1
    ch,cr,cc = mid, (rare-mid)//2, n-mid-(rare-mid)//2
    while ch>=2:
        probs[ch-2]=probs[ch]*ch*(ch-1)/(4*(cr+1)*(cc+1)); s+=probs[ch-2]; ch-=2; cr+=1; cc+=1
    p_obs = probs[n_het] if n_het<len(probs) else 0.0
    return float(min(probs[probs<=p_obs+1e-7].sum()/s, 1.0))

@cache_data
def hwe_pvals(gt):
    pv = np.full(gt.shape[1], np.nan)
    n0 = ((gt==0)&~np.isnan(gt)).sum(0); n1 = ((gt==1)&~np.isnan(gt)).sum(0); n2 = ((gt==2)&~np.isnan(gt)).sum(0)
    nv = n0+n1+n2
    if gt.shape[1] <= HWE_EXACT_MAX_SNP:
        for j in range(gt.shape[1]):
            if nv[j]>=5: pv[j]=hwe_exact(int(n1[j]),int(n0[j]),int(n2[j]))
    else:
        p=(n1+2*n2)/(2.0*np.where(nv>0,nv,np.nan))
        with np.errstate(invalid="ignore",divide="ignore"):
            eh = 2.0*p*(1.0-p)*nv; ok = (eh>=5)&np.isfinite(eh)
            chi2 = (np.abs(n1-eh)-0.5)**2/eh; pv[ok] = 1.0-stats.chi2.cdf(chi2[ok],df=1)
    return pv

# §5 ── QC Filtres
def apply_qc(gt, ind_df, snp_df, params, remove_outliers=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df)
    n0, m0 = gt.shape; tr = {}
    k = miss_snp(gt)<=params["geno"]; gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True); tr["geno"]=int(m0-gt.shape[1])
    k = miss_ind(gt)<=params["mind"]; gt=gt[k]; ind_df=ind_df[k].reset_index(drop=True); tr["mind"]=int(n0-gt.shape[0])
    if gt.size==0: raise ValueError("Tout exclu (missingness)")
    m=maf(gt); k=np.isfinite(m)&(m>=params["maf"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    pv=hwe_pvals(gt); k=np.isnan(pv)|(pv>=params["hwe"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    if gt.shape[1]==0: raise ValueError("Tout exclu (MAF/HWE)")
    gt_pre, ind_pre, snp_pre = gt.copy(), ind_df.copy(), snp_df.copy()
    if remove_outliers:
        h = hetero(gt); keep=np.ones(len(h),dtype=bool)
        for f in ind_df["FID"].unique():
            msk=(ind_df["FID"]==f).values; sub=h[msk]
            if len(sub)>=5 and np.nanstd(sub)>1e-9:
                z=(sub-np.nanmean(sub))/np.nanstd(sub); idx=np.where(msk)[0]
                keep[idx[np.abs(z)>params["het_sd"]]]=False
        gt=gt[keep]; ind_df=ind_df[keep].reset_index(drop=True); tr["het"]=int(len(h)-len(ind_df))
    return gt, ind_df, snp_df, gt_pre, ind_pre, snp_pre, {"n_ind_init":int(n0),"n_snp_init":int(m0),"n_ind_final":int(gt.shape[0]),"n_snp_final":int(gt.shape[1]),"excluded_ind":int(n0-gt.shape[0]),"excluded_snp":int(m0-gt.shape[1]),"trace":tr}

def sync_dims(gt, ind_df, snp_df):
    n=min(gt.shape[0],len(ind_df)); m=min(gt.shape[1],len(snp_df))
    if (gt.shape[0],gt.shape[1])!=(len(ind_df),len(snp_df)):
        st.warning(f"⚠️ Désynchronisation corrigée : {gt.shape}→({n},{m})")
    return gt[:n,:m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

# §6 ── Génétique des populations
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
def pca_analysis(gt, ncomp=10):
    X=impute_mean(gt); X=X-X.mean(0); nc=min(ncomp,X.shape[0]-1,X.shape[1])
    p=PCA(n_components=nc); return p.fit_transform(X), p.explained_variance_ratio_*100.0

@cache_data
def mds_analysis(gt, ncomp=5):
    X=impute_mean(gt); n=X.shape[0]; D=np.zeros((n,n),dtype=np.float32)
    for i in range(n): D[i]=np.abs(X-X[i]).sum(1)/X.shape[1]
    return SklearnMDS(n_components=ncomp,dissimilarity="precomputed",random_state=42,n_init=1,max_iter=300,normalized_stress=False).fit_transform(D)

@cache_data
def king_mat(gt):
    het=np.nansum(gt==1,axis=1).astype(float); n=gt.shape[0]; K=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            bh=np.nansum((gt[i]==1)&(gt[j]==1)); d=het[i]+het[j]
            K[i,j]=K[j,i]=(d-2*bh)/d if d>0 else 0.0
    np.fill_diagonal(K,0.5); return K

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m=NMF(n_components=K,init="nndsvda",random_state=seed,max_iter=500)
    W=m.fit_transform(np.clip(impute_mean(gt),0.0,2.0))
    s=W.sum(1,keepdims=True); s[s==0]=1.0
    return W/s, m.components_

@cache_data
def admixture_cv(gt, K_range=(2,6), n_reps=3, seed=42):
    rng=np.random.default_rng(seed); X=np.clip(impute_mean(gt),0.0,2.0)
    n,m=X.shape; hold=rng.random((n,m))<0.05; Xt=X.copy(); Xt[hold]=0.0; res={}
    for K in range(K_range[0],K_range[1]):
        errs=[]
        for r in range(n_reps):
            mod=NMF(n_components=K,init="nndsvda",random_state=seed+r,max_iter=300)
            W=mod.fit_transform(Xt); H=mod.components_
            errs.append(np.mean((X[hold]-(W@H)[hold])**2))
        res[K]={"mean":float(np.mean(errs)),"std":float(np.std(errs))}
    return res

@cache_data
def fst_pairwise(gt, labels):
    pops=sorted(np.unique(labels)); K=len(pops); M=np.full((K,K),np.nan); np.fill_diagonal(M,0.0)
    masks={p:(labels==p) for p in pops}
    for i in range(K):
        for j in range(i+1,K):
            g1=gt[masks[pops[i]]]; g2=gt[masks[pops[j]]]
            p1=np.nanmean(g1,0)/2.0; p2=np.nanmean(g2,0)/2.0
            n1=(~np.isnan(g1)).sum(0); n2=(~np.isnan(g2)).sum(0)
            dn=np.where(n1+n2>0,n1+n2,np.nan)
            pb=(p1*n1+p2*n2)/dn; hs=(2*p1*(1-p1)*n1+2*p2*(1-p2)*n2)/dn; ht=2*pb*(1-pb)
            with np.errstate(divide="ignore",invalid="ignore"): fj=(ht-hs)/ht
            M[i,j]=M[j,i]=float(np.nanmean(fj))
    return M, pops

# §7 ── Démographie
@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n=gt.shape[1]
    if n<3: return np.ones(n,dtype=bool)
    keep=np.ones(n,dtype=bool); X=impute_mean(gt); X=X-X.mean(0); i=0
    while i<n:
        we=min(i+step,n)
        while we<n and (we-i)<window:
            if keep[we] and keep[i] and X[:,i].std()>1e-8 and X[:,we].std()>1e-8:
                if np.corrcoef(X[:,i],X[:,we])[0,1]**2>r2_thr: keep[we]=False
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
    C=(X.T@X)/X.shape[0]; R2=C**2
    iu,ju=np.triu_indices(len(idx),k=1); d=(b[ju]-b[iu])/1000.0; m=(d>0)&(d<=max_kb)
    return pd.DataFrame({"dist_kb":d[m],"r2":R2[iu[m],ju[m]]})

@cache_data
def detect_roh(gt, snp_json, min_snps=30, min_kb=500.0):
    try: snp_df=pd.read_json(io.StringIO(snp_json))
    except: snp_df=pd.read_json(snp_json)
    n=min(gt.shape[1],len(snp_df)); gt=gt[:,:n]; snp_df=snp_df.iloc[:n].reset_index(drop=True)
    chr_arr=snp_df["CHR"].astype(str).values; bp=snp_df["BP"].astype(np.int64).values
    rohs=[]; froh_bp=np.zeros(gt.shape[0]); gen_total=0.0
    for chrom in pd.unique(chr_arr):
        idxs=np.where(chr_arr==chrom)[0]
        if len(idxs)<min_snps: continue
        idxs=idxs[np.argsort(bp[idxs])]; span=bp[idxs[-1]]-bp[idxs[0]]
        if span<1: continue
        gen_total+=span
        for i in range(gt.shape[0]):
            col=gt[i,idxs]; st_k=None
            for k in range(len(idxs)):
                if col[k]==0 or col[k]==2:
                    if st_k is None: st_k=k
                else:
                    if st_k is not None:
                        nr=k-st_k; lkb=(bp[idxs[k-1]]-bp[idxs[st_k]])/1000.0
                        if nr>=min_snps and lkb>=min_kb:
                            rohs.append({"IID_idx":i,"CHR":chrom,"start_bp":int(bp[idxs[st_k]]),"end_bp":int(bp[idxs[k-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                            froh_bp[i]+=bp[idxs[k-1]]-bp[idxs[st_k]]
                        st_k=None
            if st_k is not None:
                nr=len(idxs)-st_k; lkb=(bp[idxs[-1]]-bp[idxs[st_k]])/1000.0
                if nr>=min_snps and lkb>=min_kb:
                    rohs.append({"IID_idx":i,"CHR":chrom,"start_bp":int(bp[idxs[st_k]]),"end_bp":int(bp[idxs[-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                    froh_bp[i]+=bp[idxs[-1]]-bp[idxs[st_k]]
    froh=froh_bp/gen_total if gen_total>0 else froh_bp
    return pd.DataFrame(rohs) if rohs else pd.DataFrame(columns=["IID_idx","CHR","start_bp","end_bp","n_snp","length_kb"]), froh

@cache_data
def ne_historical(gt, snp_df, max_snp=2000):
    bp=snp_df["BP"].astype(float).values; ca=snp_df["CHR"].astype(str).values
    n=gt.shape[1]
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
    r2=np.array(r2l); d=np.array(dl)
    bins=np.logspace(np.log10(d.min()+1e-6),np.log10(d.max()),20); bc=(bins[:-1]+bins[1:])/2
    r2b=np.array([r2[(d>=bins[k])&(d<bins[k+1])].mean() if ((d>=bins[k])&(d<bins[k+1])).sum()>0 else np.nan for k in range(len(bins)-1)])
    m=np.isfinite(r2b)
    if m.sum()<3: return pd.DataFrame({"GenAgo":[],"Ne":[]})
    ne=np.where(r2b[m]>1e-4,1.0/(4*(bc[m]*0.01)*r2b[m]),np.nan)
    df=pd.DataFrame({"GenAgo":bc[m]*100,"Ne":ne}).dropna()
    df=df[df["Ne"]>0].sort_values("GenAgo")
    if len(df)>3: df["Ne"]=df["Ne"].rolling(3,center=True,min_periods=1).mean()
    return df

# §8 ── Sélection
@cache_data
def windowed_scan(gt, snp_df, labels, window=50):
    n=min(gt.shape[1],len(snp_df)); gt=gt[:,:n]; snp_df=snp_df.iloc[:n].reset_index(drop=True)
    fst=fst_snp(gt,labels); hom=np.nanmean((gt==0)|(gt==2),axis=0)
    df=snp_df.copy(); df["FST"]=fst; df["HOM"]=hom; df["SCORE"]=df["FST"].fillna(0)*df["HOM"]
    res=[]
    for c in df["CHR"].unique():
        sub=df[df["CHR"]==c].sort_values("BP")
        if len(sub)<window: continue
        for i in range(0,len(sub)-window,window//2):
            w=sub.iloc[i:i+window]
            res.append({"CHR":c,"START":int(w["BP"].min()),"END":int(w["BP"].max()),"MEAN_SCORE":float(w["SCORE"].mean()),"SNP":w.iloc[w["SCORE"].argmax()]["SNP"]})
    return pd.DataFrame(res)

@cache_data
def reynolds(gt, labels):
    pops=sorted(np.unique(labels)); K=len(pops)
    ph={p:np.nanmean(gt[labels==p],0)/2.0 for p in pops}; D=np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            pi,pj=ph[pops[i]],ph[pops[j]]
            num=np.nansum((pi-pj)**2); dn=np.nansum(pi*(1-pj)+pj*(1-pi))
            if dn>0: D[i,j]=D[j,i]=num/dn
    return D, pops

def nj_newick(D, labels):
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        Z=linkage(squareform(D,checks=False),method="average"); tree=to_tree(Z)
        def rec(n): return f"{labels[n.id]}:{n.dist:.4f}" if n.is_leaf() else f"({rec(n.get_left())},{rec(n.get_right())}):{n.dist:.4f}"
        return rec(tree)
    except Exception as e: return f"Err: {e}"

# §8bis ── SIGNATURES DE SÉLECTION
@cache_data
def windowed_fst_scan(gt, snp_df, labels, n_snp_window=50, step=25):
    fst = fst_snp(gt, labels); snp = snp_df.copy().reset_index(drop=True)
    n = min(len(snp), len(fst)); snp = snp.iloc[:n].reset_index(drop=True); fst = fst[:n]
    snp['FST'] = fst; results = []
    for chrom in snp['CHR'].unique():
        sub = snp[snp['CHR']==chrom].sort_values('BP').reset_index(drop=True)
        if len(sub) < n_snp_window: continue
        for i in range(0, len(sub)-n_snp_window+1, step):
            win = sub.iloc[i:i+n_snp_window]
            if win['FST'].notna().sum() < 5: continue
            results.append({'CHR':str(chrom),'START':int(win['BP'].min()),'END':int(win['BP'].max()),
                'MID':int((win['BP'].min()+win['BP'].max())/2),'N_SNP':len(win),
                'MEAN_FST':float(win['FST'].mean()),'MAX_FST':float(win['FST'].max()),
                'P95_FST':float(win['FST'].quantile(0.95)),'TOP_SNP':win.iloc[win['FST'].argmax()]['SNP']})
    return pd.DataFrame(results)

@cache_data
def detect_outlier_windows(win_df, z_threshold=3.0):
    df = win_df.copy()
    mu, sd = df['MEAN_FST'].mean(), df['MEAN_FST'].std()
    df['Z_FST'] = (df['MEAN_FST']-mu)/sd if sd>1e-9 else 0.0
    df['OUTLIER'] = df['Z_FST'] >= z_threshold
    return df.sort_values('MEAN_FST', ascending=False).reset_index(drop=True)

@cache_data
def tajimas_d_window(genos, window_size=None):
    n_ind, n_snp = genos.shape
    if n_snp<3 or n_ind<4: return np.nan, 0
    S=0; pi_values=[]
    for j in range(n_snp):
        col = genos[:,j]; col = col[~np.isnan(col)]
        if len(col)<2: continue
        p = col.mean()/2.0
        if 0<p<1:
            S+=1; pi_values.append(2*p*(1-p)*(len(col)/(len(col)-1)))
    if S==0 or not pi_values: return np.nan, int(S)
    pi = np.mean(pi_values); n = n_ind
    a1 = np.sum(1.0/np.arange(1,n)); a2 = np.sum(1.0/(np.arange(1,n)**2))
    b1 = (n+1)/(3*(n-1)); b2 = 2*(n**2+n+3)/(9*n*(n-1))
    c1 = b1 - 1/a1; c2 = b2 - (n+2)/(a1*n) + a2/(a1**2)
    e1 = c1/a1; e2 = c2/(a1**2+a2)
    denom = np.sqrt(e1*S + e2*S*(S-1))
    if denom<=1e-12: return np.nan, int(S)
    return float((pi - S/a1)/denom), int(S)

@cache_data
def tajimas_d_scan(gt, snp_df, window_size=25, step=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; results = []
    for chrom in snp['CHR'].unique():
        idx = np.where(snp['CHR'].values==chrom)[0]
        if len(idx)<window_size: continue
        idx = idx[np.argsort(snp['BP'].values[idx])]
        for i in range(0, len(idx)-window_size+1, step):
            win_idx = idx[i:i+window_size]; D, S = tajimas_d_window(gt[:, win_idx])
            if not np.isnan(D):
                results.append({'CHR':str(chrom),'START':int(snp.iloc[win_idx]['BP'].min()),
                    'END':int(snp.iloc[win_idx]['BP'].max()),
                    'MID':int((snp.iloc[win_idx]['BP'].min()+snp.iloc[win_idx]['BP'].max())/2),
                    'N_SNP':len(win_idx),'S':S,'TAJIMA_D':D})
    return pd.DataFrame(results)

@cache_data
def composite_selection_score(gt, snp_df, labels):
    fst = fst_snp(gt, labels); hom = np.nanmean((gt==0)|(gt==2), axis=0)
    maf_v = np.nanmean(gt, axis=0)/2.0; maf_v = np.minimum(maf_v, 1-maf_v)
    df = snp_df.copy().reset_index(drop=True); n = min(len(df), len(fst), len(hom), len(maf_v))
    df = df.iloc[:n].reset_index(drop=True)
    df['FST']=fst[:n]; df['HOM']=hom[:n]; df['MAF']=maf_v[:n]
    def norm(x):
        mu,sd = np.nanmean(x), np.nanstd(x)
        return (x-mu)/sd if sd>1e-9 else np.zeros_like(x)
    df['Z_FST']=norm(df['FST'].values); df['Z_HOM']=norm(df['HOM'].values)
    df['CSS']=np.abs(df['Z_FST']*df['Z_HOM'])
    return df.sort_values('CSS', ascending=False).reset_index(drop=True)

def plot_windowed_manhattan(win_df, value_col='MEAN_FST', title="", sigma=3.0):
    df = win_df[np.isfinite(win_df[value_col])].copy()
    if df.empty: return None
    df = df.assign(_k=df['CHR'].map(_chr_sort_key)).sort_values(['_k','MID']).reset_index(drop=True)
    fig = go.Figure()
    for i,(chrom,sub) in enumerate(df.groupby('CHR')):
        txt=[f"{c}:{int(m/1e6)}Mb<br>{value_col}={v:.4f}<br>SNP: {s}"
             for c,m,v,s in zip(sub['CHR'],sub['MID'],sub[value_col],sub.get('TOP_SNP',['—']*len(sub)))]
        fig.add_trace(go.Scatter(x=sub['MID']/1e6,y=sub[value_col],mode='markers',
            marker=dict(size=7,color='#2c3e50' if i%2==0 else '#7f8c8d',line=dict(width=0.5,color='white')),
            text=txt,hoverinfo='text',showlegend=False))
    mu,sd=df[value_col].mean(),df[value_col].std()
    fig.add_hline(y=mu+sigma*sd,line_dash='dash',line_color='red',
                  annotation_text=f'μ+{sigma}σ={mu+sigma*sd:.4f}')
    fig.update_layout(title=title,xaxis_title='Position (Mb)',yaxis_title=value_col,height=520)
    return fig

# §8ter ── GWAS PAIRWISE ENTRE RACES
@cache_data
def gwas_pairwise_races(gt, snp_df, labels, race_a, race_b, maf_min=0.05, method="allelic"):
    labels = np.asarray(labels).astype(str); mask_ab = np.isin(labels, [race_a,race_b])
    if mask_ab.sum()<20: raise ValueError(f"<20 individus pour {race_a} vs {race_b}")
    gt_ab = gt[mask_ab]; y = (labels[mask_ab]==race_a).astype(int)
    snp_ab = snp_df.reset_index(drop=True); n_snp = min(gt_ab.shape[1], len(snp_ab))
    gt_ab = gt_ab[:,:n_snp]; snp_ab = snp_ab.iloc[:n_snp].reset_index(drop=True)
    rows=[]
    for j in range(n_snp):
        g = gt_ab[:,j]; gA=g[y==1]; gB=g[y==0]
        gA=gA[~np.isnan(gA)]; gB=gB[~np.isnan(gB)]
        if len(gA)<5 or len(gB)<5: continue
        pA=gA.mean()/2.0; pB=gB.mean()/2.0
        tot=(gA.sum()+gB.sum())/(2*(len(gA)+len(gB))); maf_v=min(tot,1-tot)
        if maf_v<maf_min: continue
        if method=="allelic":
            table=np.array([[2*len(gA)-gA.sum(),gA.sum()],[2*len(gB)-gB.sum(),gB.sum()]])
            try: chi2,p,_,_ = stats.chi2_contingency(table, correction=False)
            except: continue
        else:
            X=np.column_stack([np.ones(len(y)),g]); valid=~np.isnan(X).any(1)
            X_v=X[valid]; y_v=y[valid]
            try:
                beta=np.zeros(2)
                for _ in range(15):
                    eta=X_v@beta; mu=1/(1+np.exp(-np.clip(eta,-500,500)))
                    W=np.diag(mu*(1-mu)+1e-9); z=eta+(y_v-mu)/(mu*(1-mu)+1e-9)
                    bn=np.linalg.inv(X_v.T@W@X_v)@(X_v.T@W@z)
                    if np.max(np.abs(bn-beta))<1e-6: beta=bn; break
                    beta=bn
                se=np.sqrt(np.diag(np.linalg.inv(X_v.T@W@X_v)))
                z_val=beta[-1]/se[-1]; chi2=z_val**2; p=float(stats.norm.sf(abs(z_val))*2)
            except: continue
        hs=(2*pA*(1-pA)*len(gA)+2*pB*(1-pB)*len(gB))/(len(gA)+len(gB))
        pb=(pA*len(gA)+pB*len(gB))/(len(gA)+len(gB)); ht=2*pb*(1-pb)
        fst_snp_v=(ht-hs)/ht if ht>1e-9 else 0.0
        rows.append({"SNP":snp_ab.iloc[j]["SNP"],"CHR":str(snp_ab.iloc[j]["CHR"]),
            "BP":int(snp_ab.iloc[j]["BP"]),"freq_A":round(pA,4),"freq_B":round(pB,4),
            "delta_freq":round(pA-pB,4),"MAF":round(maf_v,4),"chi2":round(chi2,4),
            "P":p,"FST_snp":round(fst_snp_v,4),"n_A":len(gA),"n_B":len(gB)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
        try:
            from statsmodels.stats.multitest import multipletests
            _, df["P_fdr"], _, _ = multipletests(df["P"].values, method="fdr_bh")
        except: df["P_fdr"]=np.nan
    return df

@cache_data
def get_race_pairs(labels):
    races = sorted(pd.unique(labels).astype(str))
    return races, [(races[i],races[j]) for i in range(len(races)) for j in range(i+1,len(races))]

# §9 ── GWAS
def _vc_delta(Y, X0, K):
    n=len(Y); d,U=np.linalg.eigh(K); d=np.maximum(d,0); Yt=U.T@Y; X0t=U.T@X0
    def nll(dl):
        if dl<=0: return 1e18
        w=1.0/np.sqrt(dl*d+1.0); Yw=Yt*w; Xw=X0t*w[:,None]
        try: beta,*_=np.linalg.lstsq(Xw,Yw,rcond=None)
        except: return 1e18
        s2=float(np.sum((Yw-Xw@beta)**2)/n)
        return 1e18 if s2<=0 else -(-0.5*(n*np.log(s2)+np.sum(np.log(dl*d+1.0))))
    grid=np.logspace(-3,3,30); lls=np.array([nll(dl) for dl in grid]); ib=int(np.argmin(lls))
    try: return max(float(minimize_scalar(nll,bounds=(grid[max(0,ib-1)],grid[min(29,ib+1)]),method="bounded").x),1e-6)
    except: return max(float(grid[ib]),1e-6)

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_binary=False):
    valid=np.isfinite(pheno)
    if valid.sum()<10: raise ValueError(f"<10 individus valides ({valid.sum()})")
    if K is None or not isinstance(K,np.ndarray) or K.shape[0]!=gt.shape[0]: K=np.eye(gt.shape[0])
    Y=pheno[valid]; gt_v=gt[valid]; K_v=K[np.ix_(valid,valid)]
    X0 = np.ones((len(Y),1)) if covs is None or covs.size==0 else np.column_stack([np.ones(len(Y)),np.asarray(covs)[valid]])
    X0=X0[:,np.std(X0,axis=0)>1e-10]
    if use_lmm and not is_binary:
        dl=_vc_delta(Y,X0,K_v); d,U=np.linalg.eigh(K_v); d=np.maximum(d,0)
        w=1.0/np.sqrt(dl*d+1.0); Yt=(U.T@Y)*w; X0t=(U.T@X0)*w[:,None]
    else: Yt,X0t=Y,X0
    out={"SNP_idx":[],"MAF":[],"BETA":[],"SE":[],"T":[],"P":[]}
    for j in range(gt_v.shape[1]):
        g=gt_v[:,j].astype(float)
        if np.isnan(g).mean()>miss_max: continue
        g=np.where(np.isnan(g),np.nanmean(g),g); p=g.mean()/2.0; mj=min(p,1-p)
        if mj<maf_min: continue
        if is_binary:
            X=np.column_stack([X0,g]); beta=np.zeros(X.shape[1])
            for _ in range(10):
                eta=X@beta; mu=1/(1+np.exp(-np.clip(eta,-500,500)))
                W=np.diag(mu*(1-mu)+1e-9); z=eta+(Y-mu)/(mu*(1-mu)+1e-9)
                try:
                    bn=np.linalg.inv(X.T@W@X)@(X.T@W@z)
                    if np.max(np.abs(bn-beta))<1e-6: break
                    beta=bn
                except: break
            try:
                se=np.sqrt(np.diag(np.linalg.inv(X.T@W@X)))
                if se[-1]>0:
                    tv=beta[-1]/se[-1]; out["SNP_idx"].append(j); out["MAF"].append(mj); out["BETA"].append(beta[-1]); out["SE"].append(se[-1]); out["T"].append(tv); out["P"].append(float(2*stats.norm.sf(abs(tv))))
            except: pass
        else:
            X=np.column_stack([X0t,(U.T@g)*w if use_lmm else g])
            try: beta,*_=np.linalg.lstsq(X,Yt,rcond=None)
            except: continue
            res=Yt-X@beta; s2=float(np.sum(res**2)/max(len(Y)-X.shape[1],1))
            if s2<=0: continue
            try: se=float(np.sqrt(s2*np.linalg.inv(X.T@X)[-1,-1]))
            except: continue
            if se<=0: continue
            tv=beta[-1]/se
            out["SNP_idx"].append(j); out["MAF"].append(mj); out["BETA"].append(beta[-1]); out["SE"].append(se); out["T"].append(float(tv)); out["P"].append(float(2*t_dist.sf(abs(tv),df=max(len(Y)-X.shape[1],1))))
    return pd.DataFrame(out)

def annotate_gwas(res, snp_df):
    if res.empty: return res
    idx=res["SNP_idx"].astype(int).values; res=res.copy()
    for c in ["SNP","CHR","BP","CM"]:
        res[c] = snp_df[c].astype(str if c in ("SNP","CHR") else float if c=="CM" else int).values[idx]
    return res[["SNP_idx","SNP","CHR","BP","CM","MAF","BETA","SE","T","P"]]

def lambda_gc(pvals):
    p=np.asarray(pvals); p=p[np.isfinite(p)&(p>0)&(p<=1)]
    if len(p)<10: return np.nan
    return float(np.median(stats.chi2.isf(p,df=1))/stats.chi2.ppf(0.5,df=1))

# §10 ── Post-GWAS
def fine_mapping(res):
    if res.empty: return res
    df=res.copy().sort_values("P"); v=0.04
    df["ABF"]=np.exp(0.5*(df["BETA"]**2/(df["SE"]**2+v))-0.5*np.log(1+v/df["SE"]**2))
    df["PIP"]=df["ABF"]/df["ABF"].sum()
    return df.sort_values("PIP",ascending=False)

# §11 ── GBLUP
def run_gblup(gt, pheno, k_folds=5):
    valid=np.isfinite(pheno)
    if valid.sum()<20: raise ValueError("<20 phénotypes")
    Y=pheno[valid]; X=impute_mean(gt[valid])
    p=X.mean(0)/2.0; Z=X-2*p
    G=(Z@Z.T)/(2*np.sum(p*(1-p)))+np.eye(len(Y))*1e-5
    I=np.eye(len(Y)); best_h2,best_ll=0.5,-np.inf
    for h2 in np.linspace(0.01,0.99,20):
        lam=(1-h2)/h2
        try:
            ll=-0.5*(np.linalg.slogdet(G+lam*I)[1]+len(Y)*np.log(Y.T@np.linalg.inv(G+lam*I)@Y))
            if ll>best_ll: best_ll,best_h2=ll,h2
        except: pass
    lam=(1-best_h2)/best_h2; GEBV=G@np.linalg.inv(G+lam*I)@Y
    idx=np.arange(len(Y)); np.random.shuffle(idx); folds=np.array_split(idx,k_folds); yp=np.zeros_like(Y)
    for te in folds:
        tr=np.setdiff1d(idx,te)
        Gtt=G[np.ix_(tr,tr)]; Gvt=G[np.ix_(te,tr)]
        yp[te]=Gvt@np.linalg.inv(Gtt+lam*np.eye(len(tr)))@Y[tr]
    corr=np.corrcoef(Y,yp)[0,1] if len(Y)>2 else 0.0
    return pd.DataFrame({"IID":np.where(valid)[0],"Phenotype":Y,"GEBV":GEBV}), {"h2":best_h2,"corr":corr,"r2":corr**2,"rmse":float(np.sqrt(np.mean((Y-yp)**2)))}

# §12 ── Annotation
def annotate_genes(snp_df, gff_df, window_kb=50):
    w=window_kb*1000; res=[]
    for _,s in snp_df.iterrows():
        pos=s.get("BP",s.get("START",0)); ch=str(s["CHR"])
        msk=(gff_df["CHR"]==ch)&(gff_df["START"]<=pos+w)&(gff_df["END"]>=pos-w)
        ng=gff_df[msk]
        if not ng.empty:
            for _,g in ng.iterrows():
                res.append({"SNP":s.get("SNP",f"{ch}:{pos}"),"CHR":ch,"SNP_POS":pos,"GENE":g["GENE"],"GENE_START":g["START"],"GENE_END":g["END"],"DIST_KB":round(abs(g["START"]-pos)/1000.0,2)})
    return pd.DataFrame(res)

# §13 ── Export
def build_plink(gt, ind_df, snp_df, prefix="bovine"):
    bio=BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as zf:
        n,m=gt.shape; nb=(n+3)//4; buf=bytearray([0x6C,0x1B,0x01])
        for j in range(m):
            bits=np.zeros(n,dtype=np.uint8); bits[np.isnan(gt[:,j])]=0b01
            bits[gt[:,j]==0]=0b11; bits[gt[:,j]==1]=0b10; bits[gt[:,j]==2]=0b00
            packed=np.zeros(nb,dtype=np.uint8)
            for k in range(n): packed[k//4]|=(bits[k]&0x03)<<(2*(k%4))
            buf.extend(packed.tobytes())
        zf.writestr(f"{prefix}.bed",bytes(buf))
        zf.writestr(f"{prefix}.bim","\n".join(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t{int(r['BP'])}\tA\tG" for _,r in snp_df.iterrows())+"\n")
        zf.writestr(f"{prefix}.fam","\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _,r in ind_df.iterrows())+"\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Bovine"):
    h=["##fileformat=VCFv4.2",f"##source=BovinePipeline-{proj}",'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(ind_df["IID"].astype(str))]
    b=[]
    for j in range(gt.shape[1]):
        r=snp_df.iloc[j]
        gts=["./." if np.isnan(gt[i,j]) else "0/0" if gt[i,j]==0 else "0/1" if gt[i,j]==1 else "1/1" for i in range(gt.shape[0])]
        b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\tA\tG\t.\tPASS\t.\tGT\t"+"\t".join(gts))
    return "\n".join(h+b)+"\n"

# §14 ── Visualisations
def plot_missing(mi,ms):
    f=make_subplots(rows=1,cols=2,subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi,nbinsx=60,marker_color="skyblue"),1,1)
    f.add_trace(go.Histogram(x=ms,nbinsx=60,marker_color="coral"),1,2)
    f.update_layout(height=400,showlegend=False); return f

def plot_pca(s,v,l):
    df=pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l})
    f=px.scatter(df,x="PC1",y="PC2",color="Pop",title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)",height=550)
    f.update_traces(marker=dict(size=10,line=dict(width=1,color="white"))); return f

def plot_mds(m,l):
    df=pd.DataFrame({"MDS1":m[:,0],"MDS2":m[:,1],"Pop":list(l)})
    f=px.scatter(df,x="MDS1",y="MDS2",color="Pop",title="MDS Reynolds",height=520)
    f.update_traces(marker=dict(size=11,line=dict(width=1,color="white"))); return f

def plot_manhattan(df, ycol="P", title="Manhattan", thr=None):
    if ycol=="P": df=df.copy(); df["_y"]=-np.log10(np.clip(df["P"],1e-300,1)); ycol="_y"
    df=df[np.isfinite(df[ycol])].copy(); df=df.assign(_k=df["CHR"].map(_chr_sort_key)).sort_values(["_k","BP"]).reset_index(drop=True)
    fig=go.Figure()
    for i,(c,sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index,y=sub[ycol],mode="markers",
            marker=dict(size=5,color="#2c3e50" if i%2==0 else "#7f8c8d"),
            text=sub["SNP"],hoverinfo="text",showlegend=False))
    if thr: fig.add_hline(y=thr,line_dash="dash",line_color="red")
    fig.update_layout(title=title,xaxis_title="Chromosome",yaxis_title=ycol,height=500); return fig

def plot_ld(ld,b=20):
    if ld is None or ld.empty: return None
    d=ld.copy(); d["bin"]=(d["dist_kb"]//b)*b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(),x="bin",y="r2",title="LD decay",height=450)

def plot_admix(Q,labs,pops,K):
    df=pd.DataFrame(Q,columns=[f"K{k+1}" for k in range(K)]); df["IID"]=labs; df["Pop"]=pops
    df["_d"]=Q.argmax(1); df["_po"]=pd.Categorical(pops,categories=sorted(set(pops)))
    df=df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"]=np.arange(len(df))
    fig=go.Figure()
    for k in range(K): fig.add_trace(go.Bar(x=df["x"],y=df[f"K{k+1}"],name=f"K{k+1}",marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack",title=f"Admixture K={K}",height=500); return fig,df

def plot_gblup(df):
    fig=px.scatter(df,x="Phenotype",y="GEBV",title="GBLUP: Phénotype vs GEBV")
    fig.update_traces(marker=dict(size=10,color="royalblue"))
    if len(df)>1:
        try:
            z=np.polyfit(df["Phenotype"],df["GEBV"],1); p=np.poly1d(z)
            xr=np.linspace(df["Phenotype"].min(),df["Phenotype"].max(),100)
            fig.add_trace(go.Scatter(x=xr,y=p(xr),mode="lines",name="OLS",line=dict(color="red",dash="dash")))
        except: pass
    return fig

def plot_pip(df): return px.bar(df.head(20),x="SNP",y="PIP",title="Fine-Mapping PIP",color="PIP",color_continuous_scale="Viridis")

def plot_gene_track(snp_info, gff, wk=50):
    if gff is None or gff.empty: return None
    ch=snp_info["CHR"]; pos=snp_info.get("BP",snp_info.get("START",0)); w=wk*1000
    rg=gff[(gff["CHR"]==ch)&(gff["START"]<=pos+w)&(gff["END"]>=pos-w)]
    fig=go.Figure()
    for i,(_,g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(x=[g["START"]/1e6,g["END"]/1e6],y=[i,i],mode="lines",
            line=dict(width=10,color="royalblue"),name=g["GENE"],text=[g["GENE"],g["GENE"]],hoverinfo="text"))
    fig.add_vline(x=pos/1e6,line_dash="dash",line_color="red",annotation_text=f"SNP: {snp_info.get('SNP','')}")
    fig.update_layout(title=f"Gènes autour CHR{ch}",xaxis_title="Position (Mb)",yaxis=dict(showticklabels=False),height=400+len(rg)*20); return fig

# §15 ── Pont R
R_SCRIPTS = {"qqman": '''suppressMessages({library(qqman);library(data.table)})
args<-commandArgs(TRUE); df<-fread(args[1]); out<-args[2]
df<-df[!is.na(df$P)&df$P>0,]
png(paste0(out,"_manh.png"),2400,1400,res=200)
manhattan(df,chr="CHR",bp="BP",snp="SNP",p="P",suggestiveline=-log10(1e-5),genomewideline=-log10(5e-8),col=c("grey30","grey60"),chrlabs=as.character(1:29))
dev.off()
png(paste0(out,"_qq.png"),1400,1400,res=200); qq(df$P); dev.off()
cat("OK")'''}

def r_available(): return shutil.which("Rscript") is not None

def run_r(script_key, args_list, timeout=300):
    if not r_available(): return "R non installé", False
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",suffix=".R",delete=False) as f:
            f.write(R_SCRIPTS[script_key]); tmp=f.name
        r=subprocess.run(["Rscript",tmp]+[str(a) for a in args_list],capture_output=True,text=True,timeout=timeout)
        return (r.stdout+r.stderr, r.returncode==0)
    except Exception as e: return str(e), False
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

def r_qqman(res_df, prefix):
    if res_df.empty: return None, "GWAS vide"
    try:
        tmp_csv=tempfile.mktemp(suffix=".csv"); res_df[["SNP","CHR","BP","P"]].to_csv(tmp_csv,index=False)
        pc=prefix.replace(" ","_"); out, ok = run_r("qqman",[tmp_csv,pc])
        if os.path.exists(tmp_csv): os.unlink(tmp_csv)
        if ok and os.path.exists(pc+"_manh.png"): return (pc+"_manh.png", pc+"_qq.png"), out
        return None, out
    except Exception as e: return None, str(e)

# §16 ── UI Streamlit
_STATE_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","gt_pre","ind_pre","snp_pre",
    "gt_pruned","snp_pruned","qc_stats","pca_s","pca_v","mds_c","king_K","fst","fst_M","fst_pops",
    "admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict","reyn_D","reyn_pops",
    "sel_df","scan_df","pheno_df","gwas_res","gwas_pheno","lambda_gc","gebv_df","gblup_stats",
    "annotation_df","_plink","_vcf","mapping_df","fst_snp_df",
    "win_fst","taj_df","taj_race_cur","css_df","pair_res"]

def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state: st.session_state[k]=None

def main():
    init_state()
    st.title("🐄 Bovine Genomics Pipeline v8.3 — Signatures de Sélection")
    st.caption(f"R {'✅' if r_available() else '⚠️'} · 16 sections · 11 onglets")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source", ["Demo","PED/MAP","VCF","Axiom→PED"], key="src")
        if mode=="Demo":
            if st.button("🎲 Générer démo", use_container_width=True):
                gt, ind, snp = generate_demo(); st.session_state.update({"gt":gt,"ind_df":reconstruct_fid(ind),"snp_df":snp}); st.success("✅")
        elif mode=="PED/MAP":
            ped=st.file_uploader(".ped",type=["ped","gz"]); mp=st.file_uploader(".map",type=["map","gz"])
            if ped and mp and st.button("📥 Charger"):
                try:
                    mdf=parse_map(mp.read()); gt,ind=parse_ped(ped.read(),len(mdf))
                    if len(mdf)>gt.shape[1]: mdf=mdf.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(mdf)<gt.shape[1]: gt=gt[:,:len(mdf)]
                    st.session_state.update({"gt":gt,"ind_df":reconstruct_fid(ind),"snp_df":mdf}); st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        elif mode=="VCF":
            vf=st.file_uploader(".vcf",type=["vcf","gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt,ind,snp=parse_vcf(vf.read()); st.session_state.update({"gt":gt,"ind_df":reconstruct_fid(ind),"snp_df":snp}); st.success("✅ VCF")
                except Exception as e: st.error(f"❌ {e}")
        else:
            af=st.file_uploader("Axiom brut",type=["txt","csv","tsv","gz"])
            if af and st.button("🔄 Convertir"):
                try:
                    s,g,nsnp=parse_axiom(af.read(),af.name)
                    mp_lines=[f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(nsnp)]
                    ped_lines=["\t".join([sample_to_fid_iid(smp)[0],sample_to_fid_iid(smp)[1],"0","0","0","-9"]+[a for pr in row for a in pr]) for smp,row in zip(s,g)]
                    st.session_state["_plink_ped_text"]="\n".join(ped_lines)+"\n"
                    st.session_state["_plink_map_text"]="\n".join(mp_lines)+"\n"
                    st.success(f"✅ {len(s)}×{nsnp}")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped_text"):
                st.download_button("⬇ .ped",st.session_state["_plink_ped_text"].encode(),"out.ped")
                st.download_button("⬇ .map",st.session_state["_plink_map_text"].encode(),"out.map")
        st.divider(); st.header("⚙️ Seuils QC")
        geno=st.slider("Missing SNP",0.0,0.5,0.05); mind=st.slider("Missing Ind",0.0,0.5,0.05)
        maf_t=st.slider("MAF",0.0,0.5,0.05); hwe_t=st.number_input("HWE",value=1e-6,format="%.0e")
        het_sd=st.slider("Het ±σ",1.0,5.0,3.0,0.1); rem_out=st.checkbox("❌ Supprimer outliers",value=True)
        ld_r2=st.slider("LD r²",0.05,0.5,0.2,0.05)

    if st.session_state.gt is None: st.info("👉 Chargez des données."); return

    tabs = st.tabs(["🧹 QC","✂️ LD","🧬 Structure","🎨 Admixture","📈 Démographie","🔍 Sélection","🧪 GWAS","🎯 Post-GWAS","📈 GBLUP","🧬 Annotation","📤 Export"])

    # ── QC ──
    with tabs[0]:
        if st.button("▶ Lancer QC", key="qc_btn"):
            try:
                p={"geno":geno,"mind":mind,"maf":maf_t,"hwe":hwe_t,"het_sd":het_sd}
                gt_f,ind_f,snp_f,gt_pre,ind_pre,snp_pre,qs=apply_qc(st.session_state.gt,st.session_state.ind_df,st.session_state.snp_df,p,remove_outliers=rem_out)
                st.session_state.update({"gt_filt":gt_f,"ind_filt":ind_f,"snp_filt":snp_f,"gt_pre":gt_pre,"ind_pre":ind_pre,"snp_pre":snp_pre,"qc_stats":qs})
                st.success(f"✅ QC: {gt_f.shape[0]} ind × {gt_f.shape[1]} snps")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s=st.session_state.qc_stats
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Individus",s["n_ind_final"],delta=-s["excluded_ind"]); c2.metric("SNPs",s["n_snp_final"],delta=-s["excluded_snp"])
            c3.metric("Pré-outliers",f"{st.session_state.gt_pre.shape[1]} SNP" if st.session_state.gt_pre is not None else "N/A")
            c4.metric("Outliers exclus",s["trace"].get("het",0))
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt),miss_snp(st.session_state.gt)),use_container_width=True)
            st.json(s["trace"])

    # ── LD ──
    with tabs[1]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning",key="ld_btn"):
            k=ld_pruning(st.session_state.gt_filt,window=50,step=5,r2_thr=ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k],"snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs indépendants")
        if st.session_state.gt_pruned is not None:
            if st.button("▶ LD Decay",key="ld_decay_btn"):
                st.session_state.ld_df=ld_decay(st.session_state.gt_pruned,st.session_state.snp_pruned["BP"].values)
            if st.session_state.ld_df is not None:
                fig=plot_ld(st.session_state.ld_df)
                if fig: st.plotly_chart(fig,use_container_width=True)

    # ── Structure ──
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS + KING",key="struct_btn"):
            try:
                gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                s_,v_=pca_analysis(gt_use,10); st.session_state.update({"pca_s":s_,"pca_v":v_})
                st.session_state.mds_c=mds_analysis(gt_use,5); st.session_state.king_K=king_mat(gt_use); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s,st.session_state.pca_v,st.session_state.ind_filt["FID"].values),use_container_width=True)
            st.plotly_chart(plot_mds(st.session_state.mds_c,st.session_state.ind_filt["FID"].values),use_container_width=True)

    # ── Admixture ──
    with tabs[3]:
        if st.session_state.gt_filt is not None:
            K=st.slider("K",2,8,4,key="admix_K_slider")
            if st.button("▶ Admixture NMF",key="admix_btn"):
                try:
                    gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    Q,_=admixture_nmf(gt_use,K=K); st.session_state.update({"admix_Q":Q,"admix_K":K}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                fig,_=plot_admix(st.session_state.admix_Q,st.session_state.ind_filt["IID"].values,st.session_state.ind_filt["FID"].values,st.session_state.admix_K)
                st.plotly_chart(fig,use_container_width=True)

    # ── Démographie ──
    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ ROH + Ne",key="dem_btn"):
            try:
                gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                snp_use=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                roh,froh=detect_roh(gt_use,snp_use[["CHR","BP"]].to_json()); st.session_state.update({"roh_df":roh,"froh":froh})
                ne={}
                for f in st.session_state.ind_filt["FID"].unique():
                    m=(st.session_state.ind_filt["FID"]==f).values
                    if m.sum()>=5: ne[str(f)]=ne_historical(gt_use[m],snp_use)
                st.session_state.ne_dict=ne; st.success(f"✅ {len(roh)} ROH")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.froh is not None:
            df=pd.DataFrame({"FROH":st.session_state.froh,"Pop":st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df,x="FROH",color="Pop",nbins=50,title="Distribution FROH"),use_container_width=True)

        # ── Sélection (avec annotation intégrée) ──
    with tabs[5]:
        if st.button("▶ FST pairwise + Reynolds", key="fst_btn"):
            try:
                gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                snp_use=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                labels=st.session_state.ind_filt["FID"].astype(str).values
                st.session_state.fst=fst_snp(gt_use,labels); st.session_state.fst_snp_df=snp_use.copy()
                M,pops=fst_pairwise(gt_use,labels); st.session_state.update({"fst_M":M,"fst_pops":pops})
                D,rp=reynolds(gt_use,labels); st.session_state.update({"reyn_D":D,"reyn_pops":rp}); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")

        if st.session_state.fst is not None:
            snp_fst = st.session_state.fst_snp_df if st.session_state.get("fst_snp_df") is not None else st.session_state.snp_filt
            n_fst = min(len(st.session_state.fst), len(snp_fst))
            df_fst = pd.DataFrame({"SNP":snp_fst["SNP"].values[:n_fst],"CHR":snp_fst["CHR"].values[:n_fst],
                                   "BP":snp_fst["BP"].values[:n_fst],"FST":st.session_state.fst[:n_fst]})
            st.plotly_chart(plot_manhattan(df_fst, ycol="FST", title="FST par SNP"), use_container_width=True)

        # ── ARBRE PHYLOGÉNÉTIQUE AVANCÉ ──
        st.divider()
        st.subheader("🌳 Arbre phylogénétique (NJ + Bootstrap + Midpoint)")

        if st.session_state.reyn_D is not None:
            c1, c2 = st.columns(2)
            n_boot = c1.slider("Bootstrap réplicats", 10, 500, 100, 10, key="nj_boot")
            root_opt = c2.selectbox("Enracinement", ["Non-enraciné", "Midpoint rooted"], key="nj_root")

            if st.button("🌲 Construire l'arbre avec bootstrap", type="primary", key="nj_tree_btn"):
                try:
                    gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    snp_use=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    labels_arr=st.session_state.ind_filt["FID"].astype(str).values
                    with st.spinner(f"NJ + {n_boot} bootstraps..."):
                        result = nj_tree_with_bootstrap(gt_use, snp_use, labels_arr, n_boot=int(n_boot))
                        st.session_state.nj_result = result
                        mp = midpoint_root_info(result["D"], result["pops"])
                        st.session_state.nj_midpoint = mp
                    st.success(f"✅ Arbre construit ({n_boot} réplicats)")
                except Exception as e: st.error(f"❌ {e}")

            if st.session_state.get("nj_result") is not None:
                res = st.session_state.nj_result
                mp = st.session_state.get("nj_midpoint")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Races", len(res["pops"]))
                c2.metric("Nœuds", len(res["support"]))
                c3.metric("Support ≥ 70%", f"{(res['support']>=70).sum()}/{len(res['support'])}")
                c4.metric("Distance max", f"{mp['max_dist']:.4f}" if mp else "N/A")
                if mp:
                    st.info(f"**Midpoint** entre **{mp['leaf_i']}** et **{mp['leaf_j']}** (distance {mp['max_dist']:.4f}, milieu à {mp['midpoint']:.4f})")
                fig_tree = plot_tree_plotly(
                    res["Z"], res["pops"], res["support"],
                    title=f"Arbre NJ — {len(res['pops'])} races (bootstrap, n={n_boot})",
                    midpoint_info=mp if root_opt == "Midpoint rooted" else None
                )
                st.plotly_chart(fig_tree, use_container_width=True)
                st.caption("🟢 Support ≥ 70% · 🟠 50–70% · 🔴 < 50% · Ligne violette = midpoint")
                with st.expander("📄 Newick avec support (iTOL/FigTree compatible)"):
                    st.code(res["newick"], language="text")
                st.download_button("⬇ Télécharger Newick (.nwk)",
                                   res["newick"].encode(),
                                   "bovine_tree_bootstrap.nwk",
                                   "text/plain")
                with st.expander("📊 Détail des supports bootstrap"):
                    supp_df = pd.DataFrame({
                        "Nœud": [f"N{i+1}" for i in range(len(res["support"]))],
                        "Support (%)": res["support"].astype(int),
                        "Distance": res["Z"][:,2].round(4)
                    })
                    st.dataframe(supp_df, use_container_width=True)

        # ── SIGNATURES DE SÉLECTION ──
        st.divider()
        st.subheader("🧬 Signatures de sélection (3 méthodes)")
        if st.session_state.gt_filt is not None:
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            snp_use = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            labels = st.session_state.ind_filt["FID"].astype(str).values
            method_sel = st.radio("Méthode :", ["FST par fenêtres","Tajima's D (intra-race)","Score composite FST×HOM"], horizontal=True, key="sel_method")

            if method_sel.startswith("FST"):
                c1,c2=st.columns(2); ws=c1.slider("Fenêtre (SNPs)",10,200,50,key="win_size"); sp=c2.slider("Pas",5,100,25,key="win_step")
                z_thr=st.slider("Seuil outlier (z)",2.0,5.0,3.0,0.1,key="z_thr")
                if st.button("▶ Scanner FST fenêtres", type="primary", key="win_fst_btn"):
                    with st.spinner("Scan FST..."):
                        win = windowed_fst_scan(gt_use, snp_use, labels, n_snp_window=ws, step=sp)
                        win = detect_outlier_windows(win, z_threshold=z_thr); st.session_state.win_fst=win
                    st.success(f"✅ {len(win)} fenêtres · {win['OUTLIER'].sum()} outliers")
                if st.session_state.get("win_fst") is not None:
                    win = st.session_state.win_fst
                    st.plotly_chart(plot_windowed_manhattan(win,'MEAN_FST','FST par fenêtres'),use_container_width=True)
                    st.dataframe(win.head(20)[['CHR','START','END','N_SNP','MEAN_FST','MAX_FST','Z_FST','TOP_SNP']],use_container_width=True)
                    st.download_button("⬇ Export fenêtres (CSV)",win.to_csv(index=False).encode(),"fst_windows.csv","text/csv")

            elif method_sel.startswith("Tajima"):
                race_sel = st.selectbox("Race", sorted(np.unique(labels)), key="taj_race")
                c1,c2=st.columns(2); ws_t=c1.slider("Fenêtre",10,100,25,key="taj_win"); sp_t=c2.slider("Pas",5,50,10,key="taj_step")
                if st.button(f"▶ Tajima's D — {race_sel}", type="primary", key="taj_btn"):
                    mask=labels==race_sel
                    if mask.sum()<5: st.error(f"⚠️ {mask.sum()} individus seulement")
                    else:
                        with st.spinner(f"Tajima's D..."):
                            taj=tajimas_d_scan(gt_use[mask],snp_use,window_size=ws_t,step=sp_t)
                            st.session_state.taj_df=taj; st.session_state.taj_race_cur=race_sel
                        st.success(f"✅ {len(taj)} fenêtres")
                if st.session_state.get("taj_df") is not None and len(st.session_state.taj_df)>0:
                    taj=st.session_state.taj_df
                    c1,c2,c3=st.columns(3); c1.metric("Fenêtres",len(taj)); c2.metric("D<-2 (balayage)",(taj['TAJIMA_D']<-2).sum()); c3.metric("D>+2 (struct.)",(taj['TAJIMA_D']>2).sum())
                    st.plotly_chart(plot_windowed_manhattan(taj,'TAJIMA_D',f"Tajima's D — {st.session_state.taj_race_cur}"),use_container_width=True)
                    st.dataframe(taj.sort_values('TAJIMA_D').head(20),use_container_width=True)
                    st.download_button("⬇ Export Tajima's D (CSV)",taj.to_csv(index=False).encode(),f"tajima_{st.session_state.taj_race_cur}.csv","text/csv")

            else:
                if st.button("▶ Score composite (FST×HOM)", type="primary", key="css_btn"):
                    with st.spinner("Score composite..."):
                        css=composite_selection_score(gt_use,snp_use,labels); st.session_state.css_df=css
                    st.success(f"✅ {len(css)} SNPs")
                if st.session_state.get("css_df") is not None:
                    css=st.session_state.css_df
                    st.plotly_chart(plot_manhattan(css.rename(columns={"CSS":"P"}).assign(P=lambda d: np.exp(-d["CSS"])),ycol="P",title="Score composite"),use_container_width=True)
                    st.dataframe(css.head(20)[['SNP','CHR','BP','FST','HOM','MAF','Z_FST','Z_HOM','CSS']],use_container_width=True)

        # ── GWAS PAIRWISE ENTRE RACES ──
        st.divider()
        st.subheader("🧬 GWAS entre races (case-control)")
        if st.session_state.gt_filt is not None:
            gt_use = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            snp_use = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            ind_use = st.session_state.ind_filt
            races, pairs = get_race_pairs(ind_use["FID"].astype(str).values)
            st.write(f"**{len(races)} races** → **{len(pairs)} paires**")
            c1,c2=st.columns(2); ra=c1.selectbox("Race A (case=1)",races,index=0,key="pair_a"); rb=c2.selectbox("Race B (ctrl=0)",races,index=1 if len(races)>1 else 0,key="pair_b")
            c1,c2=st.columns(2); method=c1.selectbox("Méthode",["allelic (chi²)","logistic (IRLS)"],key="pair_method"); maf_min=c2.slider("MAF min",0.0,0.3,0.05,key="pair_maf")
            mk="allelic" if method.startswith("allelic") else "logistic"
            if ra==rb: st.warning("⚠️ Choisissez deux races différentes")
            elif st.button(f"🚀 {ra} vs {rb}", type="primary", key="pair_btn"):
                try:
                    with st.spinner(f"{ra} vs {rb}..."):
                        res = gwas_pairwise_races(gt_use, snp_use, ind_use["FID"].astype(str).values, ra, rb, maf_min=float(maf_min), method=mk)
                        st.session_state.pair_res=res
                    if res.empty: st.warning("Aucun SNP")
                    else:
                        n_sig=(res["P"]<0.05/len(res)).sum(); n_fdr=(res["P_fdr"]<0.05).sum() if "P_fdr" in res else 0
                        c1,c2,c3,c4=st.columns(4); c1.metric("SNPs",len(res)); c2.metric("Bonferroni",n_sig); c3.metric("FDR<0.05",n_fdr); c4.metric("FST moy",f"{res['FST_snp'].mean():.4f}")
                        st.plotly_chart(plot_manhattan(res,ycol="P",title=f"{ra} vs {rb}"),use_container_width=True)
                        st.dataframe(res.head(20)[["SNP","CHR","BP","freq_A","freq_B","delta_freq","MAF","chi2","P","FST_snp"]],use_container_width=True)
                        st.plotly_chart(px.histogram(res,x="delta_freq",nbins=60,title=f"Δ fréquence {ra} vs {rb}"),use_container_width=True)
                        st.download_button(f"⬇ {ra} vs {rb} (CSV)",res.to_csv(index=False).encode(),f"pair_{ra}_{rb}.csv","text/csv")
                except Exception as e: st.error(f"❌ {e}")

        # ── 🆕 ANNOTATION GFF3 (dans Sélection) ──
        st.divider()
        st.subheader("🧬 Annotation génomique des régions candidates")
        st.caption("Associez les SNPs issus des analyses de sélection aux gènes du génome bovin (ARS-UCD1.2).")

        gff_f = st.file_uploader(
            "Fichier d'annotation GFF3/GTF bovin (.gff3, .gtf, .gz)",
            type=["gff","gff3","gtf","gz"], key="sel_gff_file"
        )

        # Sources disponibles
        sources = []
        if st.session_state.get("win_fst") is not None and len(st.session_state.win_fst) > 0:
            sources.append("FST fenêtres (outliers)")
        if st.session_state.get("taj_df") is not None and len(st.session_state.taj_df) > 0:
            sources.append("Tajima's D (fenêtres extrêmes)")
        if st.session_state.get("css_df") is not None and len(st.session_state.css_df) > 0:
            sources.append("Score composite (top SNPs)")
        if st.session_state.get("pair_res") is not None and len(st.session_state.pair_res) > 0:
            sources.append("GWAS pairwise (top différenciés)")
        sources.append("Recherche manuelle par position")

        if gff_f is not None:
            try:
                with st.spinner("Chargement du GFF3..."):
                    gff_df = parse_gff3(gff_f.read())
                st.success(f"✅ {len(gff_df)} gènes chargés ({gff_df['CHR'].nunique()} chromosomes)")
                st.session_state.sel_gff = gff_df

                c1, c2 = st.columns(2)
                source_sel = c1.selectbox("Source des SNPs à annoter", sources, key="sel_ann_src")
                window_kb = c2.slider("Fenêtre autour du SNP (kb)", 1, 500, 50, key="sel_ann_win")

                snps_to_annotate = None

                if source_sel == "FST fenêtres (outliers)":
                    win = st.session_state.win_fst
                    outliers = win[win.get("OUTLIER", False) == True] if "OUTLIER" in win.columns else win.head(20)
                    if len(outliers) == 0: outliers = win.head(20)
                    st.info(f"🎯 {len(outliers)} fenêtres outliers à annoter")
                    snps_to_annotate = outliers.head(50)[["CHR","START","TOP_SNP"]].copy()
                    snps_to_annotate = snps_to_annotate.rename(columns={"START":"BP","TOP_SNP":"SNP"})

                elif source_sel == "Tajima's D (fenêtres extrêmes)":
                    taj = st.session_state.taj_df
                    n_extreme = st.slider("N fenêtres extrêmes (les plus négatives)", 5, 100, 20, key="sel_ann_taj_n")
                    extremes = taj.sort_values("TAJIMA_D").head(n_extreme)
                    st.info(f"🎯 {len(extremes)} fenêtres avec D le plus négatif")
                    snps_to_annotate = extremes[["CHR","MID"]].copy().rename(columns={"MID":"BP"})
                    snps_to_annotate["SNP"] = ["Tajima_" + str(i+1) for i in range(len(snps_to_annotate))]

                elif source_sel == "Score composite (top SNPs)":
                    css = st.session_state.css_df
                    n_top = st.slider("N top SNPs", 10, 200, 30, key="sel_ann_css_n")
                    top_css = css.head(n_top)[["SNP","CHR","BP"]].copy()
                    st.info(f"🎯 {len(top_css)} top SNPs du score composite")
                    snps_to_annotate = top_css

                elif source_sel == "GWAS pairwise (top différenciés)":
                    pair = st.session_state.pair_res
                    n_top = st.slider("N top SNPs différenciés", 10, 200, 30, key="sel_ann_pair_n")
                    top_pair = pair.head(n_top)[["SNP","CHR","BP"]].copy()
                    st.info(f"🎯 {len(top_pair)} top SNPs de la comparaison")
                    snps_to_annotate = top_pair

                else:  # Recherche manuelle
                    c1, c2 = st.columns(2)
                    chrom_manual = c1.text_input("Chromosome (ex: 14)", "14", key="sel_man_chr")
                    pos_manual = c2.number_input("Position (bp)", 0, 200_000_000, 25_000_000, 100000, key="sel_man_pos")
                    snps_to_annotate = pd.DataFrame({"SNP": [f"manual_{chrom_manual}_{pos_manual}"],
                                                      "CHR": [str(chrom_manual)],
                                                      "BP": [int(pos_manual)]})

                if snps_to_annotate is not None and len(snps_to_annotate) > 0:
                    if st.button("🔍 Lancer l'annotation", type="primary", key="sel_ann_btn"):
                        with st.spinner("Recherche des gènes..."):
                            try:
                                ann = annotate_genes(snps_to_annotate, gff_df, window_kb=window_kb)
                                st.session_state.sel_annotation = ann
                                if ann.empty:
                                    st.warning(f"⚠️ Aucun gène trouvé dans ±{window_kb} kb. Essayez une fenêtre plus large.")
                                else:
                                    n_snps = ann["SNP"].nunique()
                                    n_genes = ann["GENE"].nunique()
                                    st.success(f"✅ {len(ann)} associations · {n_snps} SNPs · {n_genes} gènes uniques")
                            except Exception as e: st.error(f"❌ {e}")

                # Affichage des résultats
                if st.session_state.get("sel_annotation") is not None and len(st.session_state.sel_annotation) > 0:
                    ann = st.session_state.sel_annotation
                    st.markdown("### 📋 Gènes identifiés")
                    st.dataframe(ann, use_container_width=True)

                    # Stats
                    c1, c2, c3 = st.columns(3)
                    c1.metric("SNPs annotés", ann["SNP"].nunique())
                    c2.metric("Gènes", ann["GENE"].nunique())
                    c3.metric("Fenêtre ±", f"{window_kb} kb")

                    # Export
                    st.download_button("⬇ Export annotation (CSV)",
                                       ann.to_csv(index=False).encode(),
                                       "selection_annotation.csv", "text/csv")

                    # Gene track pour un SNP sélectionné
                    st.markdown("### 🎨 Visualisation de la région")
                    snp_choices = ann["SNP"].unique().tolist()
                    sel_snp = st.selectbox("Choisir un SNP", snp_choices, key="sel_ann_track_snp")
                    row = ann[ann["SNP"] == sel_snp].iloc[0].to_dict()
                    snp_info = {"SNP": row["SNP"], "CHR": row["CHR"], "BP": row["SNP_POS"]}
                    fig_track = plot_gene_track(snp_info, gff_df, wk=window_kb)
                    if fig_track:
                        st.plotly_chart(fig_track, use_container_width=True)

                    # Top gènes par fréquence
                    st.markdown("### 🏆 Gènes les plus fréquemment touchés")
                    gene_counts = ann["GENE"].value_counts().reset_index()
                    gene_counts.columns = ["GENE", "N_SNPs"]
                    st.dataframe(gene_counts.head(20), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur parsing GFF3 : {e}")
        else:
            st.info("💡 **Télécharger le GFF3 bovin :** [Ensembl ARS-UCD1.2](https://ftp.ensembl.org/pub/current_gff3/bos_taurus/)")

    # ── GWAS ──
    with tabs[6]:
        st.markdown("### 1️⃣ Mapping ID (optionnel)")
        mf=st.file_uploader("CSV mapping (Animal + IID)",type=["csv","txt"])
        if mf:
            try: st.session_state.mapping_df=pd.read_csv(mf); st.success("✅ Mapping"); st.dataframe(st.session_state.mapping_df.head(3))
            except Exception as e: st.error(f"❌ {e}")
        st.markdown("### 2️⃣ Phénotypes")
        pf=st.file_uploader("Phénotypes (CSV/TSV)",type=["csv","txt","gz"])
        c1,c2=st.columns(2)
        with c1:
            if pf and st.button("📥 Charger phénotypes",key="load_pheno_btn",use_container_width=True):
                try:
                    merged,_=match_pheno(st.session_state.ind_filt,load_pheno(pf.read(),pf.name),st.session_state.mapping_df)
                    st.session_state.pheno_df=merged
                    tc=next((c for c in merged.columns if c not in ("FID","IID","Animal")),None)
                    n_ok=pd.to_numeric(merged[tc],errors="coerce").notna().sum() if tc else 0
                    if n_ok<10: st.error(f"⚠️ {n_ok} matchés. Uploader mapping ID.")
                    else: st.success(f"✅ {n_ok} individus associés")
                except Exception as e: st.error(f"❌ {e}")
        with c2:
            if st.button("🎲 Simuler phénotype",key="sim_pheno_btn",use_container_width=True):
                ind=st.session_state.ind_filt; n=len(ind); rng=np.random.default_rng(42)
                trait=(st.session_state.pca_s[:,0]*2.0+rng.normal(0,0.5,n)) if (st.session_state.pca_s is not None and len(st.session_state.pca_s)==n) else rng.normal(0,1,n)
                st.session_state.pheno_df=pd.DataFrame({"FID":ind["FID"].values,"IID":ind["IID"].values,"trait":trait}); st.success("✅ trait_simu créé")
        if st.session_state.pheno_df is not None:
            cols=[c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            if cols:
                pcol=st.selectbox("Phénotype",cols); is_bin=st.checkbox("Binaire (0/1)")
                use_r=st.checkbox("Utiliser R qqman (publication)",value=False)
                ps=pd.to_numeric(st.session_state.pheno_df[pcol],errors="coerce"); nv=ps.notna().sum()
                if nv<10: st.error(f"❌ {nv} valides")
                else:
                    st.info(f"✅ {nv} valides")
                    if st.button("🚀 Lancer GWAS",type="primary",key="gwas_btn"):
                        try:
                            gt_gwas=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                            snp_gwas=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                            covs=st.session_state.pca_s[:,:5] if st.session_state.pca_s is not None else np.empty((gt_gwas.shape[0],0))
                            with st.spinner("GWAS..."):
                                res=run_gwas(gt_gwas,ps.values,st.session_state.king_K,covs,is_binary=is_bin); res=annotate_gwas(res,snp_gwas)
                                st.session_state.gwas_res=res; st.session_state.lambda_gc=lambda_gc(res["P"].values) if len(res) else np.nan
                            st.success(f"✅ {len(res)} SNPs · λGC={st.session_state.lambda_gc:.3f}")
                            st.plotly_chart(plot_manhattan(res),use_container_width=True)
                            if use_r and not res.empty:
                                pngs,msg=r_qqman(res,"bovine_gwas")
                                if pngs: st.image(pngs[0],caption="R Manhattan"); st.image(pngs[1],caption="R QQ")
                                else: st.warning(f"R: {msg}")
                        except Exception as e: st.error(f"❌ {e}")

    # ── Post-GWAS ──
    with tabs[7]:
        if st.session_state.gwas_res is not None:
            pip=fine_mapping(st.session_state.gwas_res); st.dataframe(pip[["SNP","P","PIP"]].head(10),use_container_width=True)
            st.plotly_chart(plot_pip(pip),use_container_width=True)
        else: st.warning("Lancez GWAS d'abord")

    # ── GBLUP ──
    with tabs[8]:
        if st.session_state.pheno_df is not None:
            cols=[c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            pcol=st.selectbox("Phénotype",cols,key="gb_pcol")
            if st.button("▶ GBLUP + validation",key="gb_btn"):
                try:
                    gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    ps=pd.to_numeric(st.session_state.pheno_df[pcol],errors="coerce")
                    with st.spinner("REML..."):
                        df,stats=run_gblup(gt_use,ps.values); st.session_state.gebv_df=df; st.session_state.gblup_stats=stats
                    st.success("✅"); st.plotly_chart(plot_gblup(df),use_container_width=True); st.json(stats)
                except Exception as e: st.error(f"❌ {e}")
        else: st.warning("Chargez phénotypes")

    # ── Annotation ──
    with tabs[9]:
        gff_f=st.file_uploader("GFF3/GTF",type=["gff","gff3","gtf","gz"]); src=st.radio("Source",["Sélection","GWAS"],horizontal=True); wk=st.slider("Fenêtre (kb)",1,500,50)
        if gff_f:
            try:
                gff=parse_gff3(gff_f.read()); st.success(f"✅ {len(gff)} gènes")
                snps=None
                if src=="Sélection" and st.session_state.scan_df is not None: snps=st.session_state.scan_df.rename(columns={"START":"BP"}).head(20)
                elif src=="GWAS" and st.session_state.gwas_res is not None: snps=st.session_state.gwas_res.sort_values("P").head(20)
                if snps is not None and st.button("🔍 Annoter",key="ann_btn"):
                    st.session_state.annotation_df=annotate_genes(snps,gff,wk); st.success("✅")
                if st.session_state.annotation_df is not None:
                    st.dataframe(st.session_state.annotation_df,use_container_width=True)
                    snp_sel=st.selectbox("Visualiser",st.session_state.annotation_df["SNP"].unique())
                    row=snps[snps["SNP"]==snp_sel].iloc[0].to_dict() if snps is not None and snp_sel in snps["SNP"].values else {"SNP":snp_sel,"CHR":st.session_state.annotation_df[st.session_state.annotation_df["SNP"]==snp_sel]["CHR"].iloc[0],"BP":st.session_state.annotation_df[st.session_state.annotation_df["SNP"]==snp_sel]["SNP_POS"].iloc[0]}
                    fig=plot_gene_track(row,gff,wk)
                    if fig: st.plotly_chart(fig,use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")

    # ── Export ──
    with tabs[10]:
        if st.session_state.gt_filt is not None:
            if st.button("📦 Générer PLINK",key="plink_btn"):
                try:
                    gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    snp_use=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    st.session_state._plink=build_plink(gt_use,st.session_state.ind_filt,snp_use); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state._plink: st.download_button("⬇ PLINK .zip",st.session_state._plink,"bovine.zip","application/zip")
            if st.button("📄 Générer VCF",key="vcf_btn"):
                try:
                    gt_use=st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    snp_use=st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    st.session_state._vcf=build_vcf(gt_use,st.session_state.ind_filt,snp_use); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state._vcf: st.download_button("⬇ VCF",st.session_state._vcf.encode(),"bovine.vcf","text/plain")

if __name__ == "__main__":
    main()

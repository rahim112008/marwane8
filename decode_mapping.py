"""
Décode le pattern des IIDs et propose plusieurs mappings candidats
pour les 200 individus du phénotype.
"""
import gzip
import re
import pandas as pd
from collections import Counter

PED_FILE = "votre_fichier.ped"
PHENO_FILE = "Phenotype_AFC_Yield.txt"
OUT_PREFIX = "mapping_candidate"

# ═══════════════════════════════════════════════════════════
def read_ped_iids(path):
    opener = gzip.open if path.endswith(".gz") else open
    iids, seen = [], set()
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 6 and p[1] not in seen:
                seen.add(p[1])
                iids.append((p[0], p[1]))
    return iids

def read_pheno_ids(path):
    sep = "\t" if "\t" in open(path).readline() else ","
    df = pd.read_csv(path, sep=sep)
    col = next((c for c in ["Animal","IID","id","ID"] if c in df.columns), None)
    return [str(x) for x in df[col].astype(str)], df

def decompose_iid(iid):
    """Décompose un IID en ses composants."""
    s = iid.replace(".CEL","").replace(".cel","")
    s = s.replace("(","").replace(")","")
    parts = s.split("_")
    return parts

def main():
    print("="*70)
    print("  Décodeur de mapping — Bovine Pipeline")
    print("="*70)
    
    iids = read_ped_iids(PED_FILE)
    pheno_ids, df_ph = read_pheno_ids(PHENO_FILE)
    
    print(f"\n📊 {len(iids)} IIDs uniques dans le .ped")
    print(f"📊 {len(pheno_ids)} animaux dans le phénotype")
    
    # Analyse du pattern
    print("\n" + "="*70)
    print("  ANALYSE DU PATTERN")
    print("="*70)
    
    parts_list = [decompose_iid(iid) for _, iid in iids]
    max_parts = max(len(p) for p in parts_list)
    
    print(f"\nNombre max de composants par IID : {max_parts}")
    print(f"\nExemples de décomposition :")
    for parts in parts_list[:5]:
        print(f"  {' | '.join(parts)}")
    
    # Pour chaque position, lister les valeurs uniques
    print(f"\nValeurs uniques par position :")
    for pos in range(max_parts):
        values = [p[pos] for p in parts_list if len(p) > pos]
        unique = Counter(values)
        n_unique = len(unique)
        sample = list(unique.keys())[:5]
        print(f"  Position {pos:2d} : {n_unique:4d} valeurs uniques — ex: {sample}")
    
    # Trouver la position "numérique" qui varie le plus (candidat animal ID)
    print("\n" + "="*70)
    print("  POSITIONS CANDIDATES POUR L'ID ANIMAL")
    print("="*70)
    
    candidates = []
    for pos in range(max_parts):
        values = [p[pos] for p in parts_list if len(p) > pos]
        n_unique = len(set(values))
        # Un bon candidat a beaucoup de valeurs uniques (proche du nb d'individus)
        if n_unique > len(iids) * 0.5:
            # Extraire les chiffres
            digits = [re.findall(r"\d+", v)[0] for v in values if re.findall(r"\d+", v)]
            if digits:
                unique_digits = sorted(set(int(d) for d in digits))
                candidates.append({
                    "position": pos,
                    "n_unique": n_unique,
                    "min": min(unique_digits),
                    "max": max(unique_digits),
                    "sample": values[:5]
                })
    
    for c in candidates:
        print(f"  Position {c['position']:2d} : {c['n_unique']} valeurs, "
              f"plage {c['min']}–{c['max']}")
    
    # Générer les mappings candidats
    print("\n" + "="*70)
    print("  GÉNÉRATION DE MAPPINGS CANDIDATS")
    print("="*70)
    
    pheno_sorted = sorted(pheno_ids, key=lambda x: int(x) if x.isdigit() else 0)
    
    # Candidat 1 : mapping séquentiel (IID dans l'ordre du .ped → animal dans l'ordre)
    if len(iids) >= len(pheno_sorted):
        mapping1 = pd.DataFrame({
            "Animal": pheno_sorted[:len(iids)],
            "IID": [iid for _, iid in iids[:len(pheno_sorted)]],
            "FID": [fid for fid, _ in iids[:len(pheno_sorted)]]
        })
        f1 = f"{OUT_PREFIX}_1_sequential.csv"
        mapping1.to_csv(f1, index=False)
        print(f"\n📁 Candidat 1 (séquentiel) : {f1}")
        print(f"   Hypothèse : ordre du .ped = ordre des IDs du phénotype")
        print(mapping1.head(3).to_string(index=False))
    
    # Candidat 2 : extraction du numéro dans le nom
    for c in candidates:
        pos = c["position"]
        rows = []
        for fid, iid in iids:
            parts = decompose_iid(iid)
            if len(parts) > pos:
                digits = re.findall(r"\d+", parts[pos])
                if digits:
                    # Tenter d'extraire un numéro 3-chiffres ou proche
                    num = digits[0]
                    # Si on a "05", tester 105 et 5
                    candidates_nums = [num, f"1{num.zfill(2)}", num.lstrip("0")]
                    for test in candidates_nums:
                        if test in pheno_ids:
                            rows.append({"Animal": test, "IID": iid, "FID": fid})
                            break
        
        if rows:
            mapping = pd.DataFrame(rows).drop_duplicates(subset=["IID"])
            fn = f"{OUT_PREFIX}_2_pos{pos}.csv"
            mapping.to_csv(fn, index=False)
            print(f"\n📁 Candidat 2 (position {pos}) : {fn} — {len(mapping)} matchés")
            print(mapping.head(3).to_string(index=False))
    
    # Candidat 3 : matching sur les digits du FID+num
    print("\n" + "="*70)
    print("  DIAGNOSTIC FINAL")
    print("="*70)
    print(f"\nPhenotype IDs : {len(pheno_sorted)} — ex: {pheno_sorted[:5]} ... {pheno_sorted[-3:]}")
    print(f"\nIIDs exemples :")
    for fid, iid in iids[:5]:
        print(f"  [{fid}] {iid}")
    print(f"\nIIDs derniers :")
    for fid, iid in iids[-3:]:
        print(f"  [{fid}] {iid}")
    
    print("\n⚠️  IMPORTANT :")
    print("  Vérifiez que les mappings candidats sont corrects")
    print("  en comparant les 5 premières lignes avec vos données réelles.")
    print("  Si aucun candidat ne marche, demandez la clé au laboratoire.")
    print("\n💡 Prochaine étape :")
    print("  1. Ouvrez les fichiers mapping_candidate_*.csv")
    print("  2. Vérifiez que chaque IID correspond bien au bon animal")
    print("  3. Renommez le bon fichier en 'mapping.csv'")
    print("  4. Uploadez-le dans l'app (onglet GWAS → Mapping ID)")

if __name__ == "__main__":
    main()

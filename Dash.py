"""
Dashboard interactif Streamlit — Rapprochement P&L
Dashboard unifié : PTF | Dérivées | TFR / TFR TITRE | Évolution quotidienne
"""

import base64
import io
import os
import re
import unicodedata
from typing import Optional


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Rapprochement P&L",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_logo_path() -> Optional[str]:
    """Cherche automatiquement un logo placé dans le même dossier que l'application."""
    preferred_names = [
        "logo.png", "logo.jpg", "logo.jpeg", "logo.webp",
        "LOGO.png", "LOGO.jpg", "LOGO.jpeg", "LOGO.webp",
    ]
    for name in preferred_names:
        path = os.path.join(BASE_DIR, name)
        if os.path.isfile(path):
            return path

    for name in os.listdir(BASE_DIR):
        lower = name.lower()
        if lower.startswith("logo") and lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return os.path.join(BASE_DIR, name)
    return None


LOGO_PATH = find_logo_path()

PRIMARY_RED = "#C00000"
DARK = "#12172B"
PALETTE = ["#1565C0", "#C00000", "#26A69A", "#F4B400", "#7A1FA2", "#43A047",
           "#EF6C00", "#546E7A", "#00838F", "#AD1457"]

CUSTOM_CSS = f"""
<style>
    .main-header {{
        background: linear-gradient(90deg, #ffffff 0%, #f7f7f9 100%);
        padding: 18px 30px;
        border-radius: 12px;
        border-bottom: 4px solid {PRIMARY_RED};
        margin-bottom: 22px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 50px;
    }}
    .main-header h1 {{
        color: {PRIMARY_RED};
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
        text-align: center;
    }}
    .main-header .logo-topright {{
        position: absolute;
        top: 12px;
        right: 22px;
        height: 48px;
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fb 100%);
        border: 1px solid #e3e5ea;
        border-left: 4px solid {PRIMARY_RED};
        border-radius: 12px;
        padding: 16px 14px;
        box-shadow: 0 3px 12px rgba(18, 23, 43, 0.06);
    }}
    div[data-testid="stMetricLabel"] {{
        color: #5d6473;
        font-weight: 700;
    }}
    div[data-testid="stMetricValue"] {{
        color: {DARK};
        font-weight: 800;
    }}
    .module-header {{
        background: #ffffff;
        border: 1px solid #e7e8ec;
        border-left: 5px solid {PRIMARY_RED};
        border-radius: 12px;
        padding: 15px 18px;
        margin: 4px 0 18px 0;
        box-shadow: 0 3px 12px rgba(18, 23, 43, 0.05);
    }}
    .module-header h2 {{
        margin: 0 0 4px 0;
        color: {DARK};
        font-size: 1.35rem;
    }}
    .module-header p {{
        margin: 0;
        color: #687080;
        font-size: 0.94rem;
    }}
    .file-status {{
        background: #f7f8fa;
        border-radius: 8px;
        padding: 7px 9px;
        margin-top: 5px;
        font-size: 0.82rem;
        color: #545b69;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f2f2f5;
        border-radius: 8px 8px 0 0;
        padding: 10px 22px;
        font-weight: 700;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_RED} !important;
        color: white !important;
    }}
    /* Stabilise la largeur des tableaux pendant les reruns Streamlit. */
    div[data-testid="stDataFrame"] {{
        width: 100%;
        overflow: hidden;
    }}
    div[data-testid="stDataFrame"] > div {{
        width: 100%;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _logo_html(path: Optional[str], height_px: int = 48) -> str:
    """Encode le logo en base64 pour l'intégrer directement dans l'en-tête."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    mime_ext = "jpeg" if ext in ("jpg", "jpeg") else ext
    return (
        f'<img class="logo-topright" '
        f'style="height:{height_px}px" '
        f'src="data:image/{mime_ext};base64,{b64}" />'
    )


logo_tag = _logo_html(LOGO_PATH)

header_html = (
    '<div class="main-header">'
    + logo_tag
    + '<h1>Rapprochement P&amp;L</h1>'
    + '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)
# BUILD: PTF_SWAP_FIX_V9


# ----------------------------------------------------------------------
# OUTILS GÉNÉRIQUES
# ----------------------------------------------------------------------
def _norm(txt) -> str:
    """Normalise un texte : minuscule, sans accents, sans espaces/underscores superflus."""
    if txt is None:
        return ""
    txt = str(txt).strip().lower()
    if txt in ("nan", "none"):
        return ""
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"[^a-z0-9]+", " ", txt).strip()
    return txt


def find_column(df: pd.DataFrame, aliases: list) -> str:
    """Retrouve la vraie colonne d'un dataframe à partir d'une liste d'alias normalisés."""
    norm_map = {_norm(c): c for c in df.columns}
    for alias in aliases:
        a = _norm(alias)
        if a in norm_map:
            return norm_map[a]
    for alias in aliases:
        a = _norm(alias)
        for key, orig in norm_map.items():
            if key and (a in key or key in a):
                return orig
    return None


def to_number(series: pd.Series) -> pd.Series:
    """Convertit une colonne (parfois formatée à la française) en numérique."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def fmt_money(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{x:,.2f}".replace(",", " ").replace(".", ",")
    except (TypeError, ValueError):
        return str(x)


def color_for(label: str, mapping: dict) -> str:
    if label not in mapping:
        mapping[label] = PALETTE[len(mapping) % len(PALETTE)]
    return mapping[label]


def _to_buffer(uploaded_file) -> io.BytesIO:
    """Copie le contenu d'un fichier chargé (ou d'un chemin) dans un buffer réutilisable."""
    if hasattr(uploaded_file, "getvalue"):
        return io.BytesIO(uploaded_file.getvalue())
    with open(uploaded_file, "rb") as f:
        data = f.read()
    return io.BytesIO(data)


def _excel_cell_display_value(cell):
    """Conserve notamment les zéros à gauche définis par un format Excel 000..."""
    value = cell.value
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fmt = str(cell.number_format or "").split(";")[0].strip()
        # Exemple : valeur 123 avec format Excel 000000 => texte 000123
        if re.fullmatch(r"0+", fmt) and float(value).is_integer():
            return f"{int(value):0{len(fmt)}d}"
    return value


def _read_excel_preserving_display(uploaded_file, sheet_index: int = 0) -> pd.DataFrame:
    """Lit la feuille demandée (1re par défaut) et préserve les identifiants formatés avec des zéros à gauche."""
    filename = str(getattr(uploaded_file, "name", uploaded_file)).lower()
    if filename.endswith(".xlsx"):
        try:
            from openpyxl import load_workbook

            wb = load_workbook(_to_buffer(uploaded_file), data_only=True, read_only=False)
            ws = wb.worksheets[sheet_index]
            rows = [
                [_excel_cell_display_value(cell) for cell in row]
                for row in ws.iter_rows()
            ]
            # Texte fusionné (ex. commentaire sur 2 lignes) : repris sur chaque ligne
            # de la fusion pour rester aligné avec la ligne de données.
            for rng in ws.merged_cells.ranges:
                top = rows[rng.min_row - 1][rng.min_col - 1] if rng.min_row - 1 < len(rows) else None
                if isinstance(top, str) and top.strip():
                    for r in range(rng.min_row - 1, min(rng.max_row, len(rows))):
                        for c in range(rng.min_col - 1, rng.max_col):
                            if c < len(rows[r]) and rows[r][c] is None:
                                rows[r][c] = top
            return pd.DataFrame(rows)
        except Exception:
            # Repli sûr sur pandas si le classeur contient une structure atypique.
            pass

    return pd.read_excel(_to_buffer(uploaded_file), sheet_name=sheet_index, header=None, dtype=object)


def series_as_text(value) -> str:
    """Force un numéro de série en texte sans ajouter de .0 indésirable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sheet_names(uploaded_file) -> list:
    """Liste des feuilles d'un classeur (chemin)."""
    try:
        from openpyxl import load_workbook

        wb = load_workbook(_to_buffer(uploaded_file), read_only=True)
        names = list(wb.sheetnames)
        wb.close()
        return names
    except Exception:
        return list(pd.ExcelFile(_to_buffer(uploaded_file)).sheet_names)


def _detect_header(raw: pd.DataFrame, keywords: list, max_scan: int):
    """Retourne (ligne d'en-tête, score) pour une feuille brute."""
    header_row = 0
    best_score = 0
    for i in range(min(max_scan, len(raw))):
        row_vals = [_norm(v) for v in raw.iloc[i].tolist()]
        score = sum(1 for kw in keywords if any(kw in val for val in row_vals if val))
        if score > best_score:
            best_score = score
            header_row = i
        if score >= max(3, len(keywords) - 1):
            header_row = i
            best_score = score
            break
    return header_row, best_score


def smart_read_excel(
    uploaded_file,
    keywords: list,
    max_scan: int = 15,
    sheet_index=None,
    required: list = None,
) -> pd.DataFrame:
    """
    Lit un fichier Excel en détectant automatiquement la FEUILLE (n'importe
    laquelle) puis la ligne d'en-tête (utile quand il y a des titres /
    bannières au-dessus du vrai tableau).
    - sheet_index=None : toutes les feuilles sont analysées, la plus
      pertinente (meilleur score de mots-clés) est retenue.
    - required : mots-clés qui doivent obligatoirement figurer dans l'en-tête.
    """
    n_sheets = len(_sheet_names(uploaded_file)) if sheet_index is None else 1
    candidates = range(n_sheets) if sheet_index is None else [sheet_index]

    best = None  # (score, nb_lignes, raw, header_row)
    for idx in candidates:
        try:
            raw_i = _read_excel_preserving_display(uploaded_file, idx)
        except Exception:
            continue
        if raw_i.empty:
            continue
        hdr_i, score_i = _detect_header(raw_i, keywords, max_scan)
        if required:
            hdr_vals = [_norm(v) for v in raw_i.iloc[hdr_i].tolist()]
            if not all(any(r in v for v in hdr_vals if v) for r in required):
                score_i = -1
        key = (score_i, len(raw_i))
        if best is None or key > best[:2]:
            best = (score_i, len(raw_i), raw_i, hdr_i)

    if best is None:
        raise ValueError("Aucune feuille lisible trouvée dans le classeur.")
    _, _, raw, header_row = best

    # La ligne détectée devient l'en-tête ; les données commencent juste après.
    columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = columns
    df = df.reset_index(drop=True)
    # Supprime les colonnes totalement vides.
    df = df.dropna(axis=1, how="all")
    return df


COMMENT_COL = "COMMENTAIRE"
COMMENT_LABEL = "Commentaire"
COMMENT_HEADERS = {
    "commentaire", "commentaires", "comment", "comments",
    "observation", "observations", "remarque", "remarques",
}


def extract_comments(df_raw: pd.DataFrame, used_cols: list) -> pd.Series:
    """
    Récupère les commentaires saisis dans les colonnes SANS en-tête situées
    à côté du tableau (ex. « Bonds forwards, cessions non comptabilisées par
    Murex »). Retourne une Série alignée sur l'index de df_raw.
    """
    used = {c for c in used_cols if c is not None}
    parts = []
    for i, name in enumerate(df_raw.columns):
        # Colonne « Commentaire » (ou sans en-tête) située à côté du tableau.
        if name in used or _norm(name) not in ({""} | COMMENT_HEADERS):
            continue
        col = df_raw.iloc[:, i]

        def _txt(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            if isinstance(v, (int, float)):
                return ""
            t = str(v).strip()
            return "" if t.lower() in ("nan", "none", "nat") else t

        parts.append(col.map(_txt))
    if not parts:
        return pd.Series("", index=df_raw.index)
    out = parts[0]
    for extra in parts[1:]:
        out = (out + " " + extra).str.strip()
    return out.fillna("")


def has_comments(df: pd.DataFrame) -> bool:
    return COMMENT_COL in df.columns and df[COMMENT_COL].astype(str).str.strip().ne("").any()


def render_module_header(title: str, description: str) -> None:
    """Affiche un en-tête cohérent pour chaque module."""
    st.markdown(
        f'<div class="module-header"><h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def uploaded_file_status(uploaded_file) -> None:
    """Affiche le nom et la taille du fichier chargé de manière discrète."""
    if uploaded_file is None:
        st.markdown('<div class="file-status">Aucun fichier chargé</div>', unsafe_allow_html=True)
        return
    size_kb = getattr(uploaded_file, "size", 0) / 1024
    st.markdown(
        f'<div class="file-status">✓ {uploaded_file.name} · {size_kb:,.1f} Ko</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# SIDEBAR — CHARGEMENT DES FICHIERS
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ENTREPRISE")
    if LOGO_PATH:
        with open(LOGO_PATH, "rb") as logo_file:
            logo_b64 = base64.b64encode(logo_file.read()).decode("utf-8")
        logo_ext = os.path.splitext(LOGO_PATH)[1].lower().lstrip(".")
        logo_mime = "jpeg" if logo_ext in ("jpg", "jpeg") else logo_ext
        st.markdown(
            '<div style="text-align:center; padding:8px 0 14px 0;">'
            f'<img src="data:image/{logo_mime};base64,{logo_b64}" alt="Logo entreprise" '
            'style="max-width:150px; width:100%; height:auto; object-fit:contain;">'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Logo introuvable dans le dossier de l’application.")

    st.divider()
    st.caption("Rapprochement P&L — tableau de pilotage")
    st.caption("Version unifiée · 2026-07-29")

    st.divider()
    st.markdown("### 📂 Fichiers")

    # ------------------------------------------------------------------
    # Stockage partagé côté serveur : tous les visiteurs voient les mêmes
    # fichiers ; seul l'administrateur peut les charger ou les supprimer.
    # ------------------------------------------------------------------
    FILE_SLOTS = {
        "ptf": "Récapitulatif (PTF)",
        "der": "Dérivées",
        "tfrd": "TFR Dérivées",
        "tfrt": "TFR Titre",
        "evo": "Récapitulatif Général (Évolution)",
    }

    class SharedFile:
        """Fichier conservé en mémoire sur le serveur (compatible avec le reste du code)."""
        def __init__(self, name: str, data: bytes):
            self.name = name
            self.data = data
            self.size = len(data)

        def getvalue(self) -> bytes:
            return self.data

    @st.cache_resource
    def get_shared_store() -> dict:
        return {k: None for k in FILE_SLOTS}

    def _rerun():
        """Compatible avec les anciennes et nouvelles versions de Streamlit."""
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    store = get_shared_store()

    try:
        admin_password = st.secrets.get("ADMIN_PASSWORD", None)
    except Exception:
        admin_password = None

    with st.expander("🔒 Administrateur"):
        typed_pwd = st.text_input("Mot de passe", type="password", key="admin_pwd")
    is_admin = bool(admin_password) and typed_pwd == admin_password

    if is_admin:
        st.success("Mode administrateur actif")
        for slot, label in FILE_SLOTS.items():
            ver_key = f"ver_{slot}"
            st.session_state.setdefault(ver_key, 0)
            up = st.file_uploader(label, type=["xlsx"], key=f"up_{slot}_{st.session_state[ver_key]}")
            if up is not None:
                store[slot] = SharedFile(up.name, up.getvalue())
                st.session_state[ver_key] += 1
                _rerun()
            current = store[slot]
            if current is not None:
                c1, c2 = st.columns([4, 1])
                c1.caption(f"✓ {current.name} · {current.size / 1024:,.1f} Ko")
                if c2.button("🗑", key=f"del_{slot}", help=f"Supprimer {label}"):
                    store[slot] = None
                    _rerun()
    else:
        for slot, label in FILE_SLOTS.items():
            current = store[slot]
            if current is None:
                st.caption(f"✗ {label} : non disponible")
            else:
                st.caption(f"✓ {label} : {current.name}")

    ptf_file = store["ptf"]
    dérivées_file = store["der"]
    tfr_dérivées_file = store["tfrd"]
    tfr_titre_file = store["tfrt"]
    evolution_file = store["evo"]

# ----------------------------------------------------------------------
# CHARGEMENT / NETTOYAGE PTF
# ----------------------------------------------------------------------
def load_ptf(file) -> pd.DataFrame:
    df_raw = smart_read_excel(file, ["portfolio", "compta", "gestion", "ecart"])

    col_ptf = find_column(df_raw, ["portfolio", "portefeuille"])
    col_compta = find_column(df_raw, ["compta", "comptabilite", "comptabilité"])
    col_gestion = find_column(df_raw, ["gestion"])
    col_ecart = find_column(df_raw, ["ecart", "écart"])

    if col_ptf is None or col_compta is None or col_gestion is None:
        raise ValueError(
            "Colonnes introuvables (PORTFOLIO / COMPTA / GESTION). "
            f"Colonnes détectées dans le fichier : {list(df_raw.columns)}"
        )

    out = pd.DataFrame()
    out["PORTFOLIO"] = df_raw[col_ptf].ffill().astype(str).str.strip()
    out["COMPTA"] = to_number(df_raw[col_compta])
    out["GESTION"] = to_number(df_raw[col_gestion])
    if col_ecart is not None:
        out["ECART"] = to_number(df_raw[col_ecart])
    else:
        out["ECART"] = out["GESTION"] - out["COMPTA"]
    out[COMMENT_COL] = extract_comments(
        df_raw, [col_ptf, col_compta, col_gestion, col_ecart]
    )

    out = out.dropna(subset=["PORTFOLIO"])
    out = out[(out["PORTFOLIO"].str.len() > 0) & (out["PORTFOLIO"].str.lower() != "nan")]
    out = out.dropna(subset=["COMPTA", "GESTION"], how="all")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# CHARGEMENT / NETTOYAGE SWAP / OPTIONS
# ----------------------------------------------------------------------
def canon_type(value: str) -> str:
   n = _norm(value)
   if not n:
       return "N/A"
   if "swap" in n and "hors" in n and "murex" in n:
       return "SWAP HORS MUREX"
   if "swap" in n and "murex" in n:
       return "SWAP MUREX"
   if "option" in n and "change" in n:
       return "OPTIONS DE CHANGE"
   if "trs" in n and "gold" in n:
       return "TRS GOLD"
   if "trs" in n and "etf" in n:
       return "TRS ETF"
   if "future" in n and (
      "matiere" in n
      or "masi" in n
):
      return "FUTURES MATIERES PREMIERES ET MASI 20"
   if "forward" in n:
      return "Forwards"
   return str(value).strip().upper()


def load_swap(file) -> pd.DataFrame:
    df_raw = smart_read_excel(file, ["date", "type", "murex", "mo", "ecart", "serie"], required=["murex"])

    col_date = find_column(df_raw, ["date"])
    col_type = find_column(df_raw, ["type"])
    col_numserie = find_column(df_raw, ["num serie", "numero serie", "n serie"])
    col_libelle = find_column(df_raw, ["libelle serie", "libelle"])
    col_murex = find_column(df_raw, ["murex"])
    col_mo = find_column(df_raw, ["mo"])
    col_ecart = find_column(df_raw, ["ecart", "écart"])

    missing = []
    if col_murex is None:
        missing.append("MUREX")
    if col_mo is None:
        missing.append("MO")
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier SWAP : {', '.join(missing)}. "
            f"Colonnes détectées : {list(df_raw.columns)}"
        )

    out = pd.DataFrame()
    if col_date is not None:
        raw_dates = df_raw[col_date].ffill()
        out["DATE"] = pd.to_datetime(raw_dates, errors="coerce", dayfirst=True).dt.normalize()
    else:
        out["DATE"] = pd.NaT

    out["TYPE"] = (
        df_raw[col_type].ffill().map(canon_type)
        if col_type is not None else "N/A"
    )
    out["NUM_SERIE"] = (
        df_raw[col_numserie].map(series_as_text)
        if col_numserie is not None else ""
    )
    out["LIBELLE"] = df_raw[col_libelle].astype(str).str.strip() if col_libelle is not None else ""
    out["MUREX"] = to_number(df_raw[col_murex])
    out["MO"] = to_number(df_raw[col_mo])
    if col_ecart is not None:
        out["ECART"] = to_number(df_raw[col_ecart])
    else:
        out["ECART"] = out["MUREX"] - out["MO"]
    out[COMMENT_COL] = extract_comments(
        df_raw,
        [col_date, col_type, col_numserie, col_libelle, col_murex, col_mo, col_ecart],
    )
    # Si le commentaire est écrit sur une autre ligne du même bloc (même type),
    # il est rattaché à l'unique ligne de données de ce bloc.
    _has_val = out["MUREX"].notna() | out["MO"].notna()
    _run = (out["TYPE"] != out["TYPE"].shift()).cumsum()
    for _, _idx in out.groupby(_run).groups.items():
        _data = [i for i in _idx if _has_val[i]]
        if len(_data) == 1 and not str(out.at[_data[0], COMMENT_COL]).strip():
            _txt = [out.at[i, COMMENT_COL] for i in _idx if str(out.at[i, COMMENT_COL]).strip()]
            if _txt:
                out.at[_data[0], COMMENT_COL] = " ".join(dict.fromkeys(_txt))

    out = out.dropna(subset=["MUREX", "MO"], how="all")
    out["TYPE"] = out["TYPE"].map(canon_type)
    out = out[out["TYPE"].astype(str).str.len() > 0]
    return out.reset_index(drop=True)



# ----------------------------------------------------------------------
# CHARGEMENT TFR / TFR TITRE — CONSERVATION DES CELLULES VIDES
# ----------------------------------------------------------------------
def load_tfr_table(file) -> pd.DataFrame:
    """
    Lit un fichier TFR sans recopier les valeurs des lignes précédentes.

    Les cellules vides internes restent vides, notamment dans PCEC,
    Num série, Libellé série et Produits. Seules les lignes entièrement
    vides situées après la fin réelle du tableau sont retirées.
    """
    df_raw = smart_read_excel(
        file,
        ["code ligne", "libelle", "pcec", "num serie", "produits", "solde final"],
        max_scan=20,
    )

    col_code = find_column(df_raw, ["code ligne"])
    col_libelle = find_column(df_raw, ["libelle", "libellé"])
    col_pcec = find_column(df_raw, ["pcec"])
    col_num_serie = find_column(
        df_raw,
        ["num serie", "numéro série", "numero serie", "n serie"],
    )
    col_libelle_serie = find_column(
        df_raw,
        ["libelle serie", "libellé série"],
    )
    col_produits = find_column(
        df_raw,
        ["produits", "produit", "prduits"],
    )
    col_solde = find_column(
        df_raw,
        ["solde final", "solde final apres impact", "solde final après impact"],
    )

    required = {
        "Code ligne": col_code,
        "LIBELLE": col_libelle,
        "PCEC": col_pcec,
        "Num série": col_num_serie,
        "LIBELLE série": col_libelle_serie,
        "Produits": col_produits,
        "Solde final": col_solde,
    }
    missing = [label for label, column in required.items() if column is None]
    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
            + f". Colonnes détectées : {list(df_raw.columns)}"
        )

    # Sélection stricte et ordre identique à Excel. AUCUN ffill().
    display = pd.DataFrame(
        {
            "Code ligne": df_raw[col_code],
            "LIBELLE": df_raw[col_libelle],
            "PCEC": df_raw[col_pcec],
            "Num série": df_raw[col_num_serie],
            "LIBELLE série": df_raw[col_libelle_serie],
            "Produits": df_raw[col_produits],
            "Solde final": df_raw[col_solde],
        }
    )

    # Conserver les identifiants et leurs zéros à gauche comme du texte.
    for column in ["Code ligne", "PCEC", "Num série"]:
        display[column] = display[column].map(series_as_text)

    # Les autres textes vides restent réellement vides dans Streamlit.
    for column in ["LIBELLE", "LIBELLE série", "Produits"]:
        display[column] = display[column].map(
            lambda value: ""
            if value is None or (isinstance(value, float) and pd.isna(value))
            else str(value).strip()
        )

    display["Solde final"] = to_number(display["Solde final"])

    # Retirer uniquement les lignes entièrement vides à la fin du fichier.
    # Les lignes vides au milieu du tableau sont conservées exactement.
    non_empty = display.apply(
        lambda row: any(
            not (
                value is None
                or (isinstance(value, float) and pd.isna(value))
                or (isinstance(value, str) and value.strip() == "")
            )
            for value in row
        ),
        axis=1,
    )
    if non_empty.any():
        last_used_index = non_empty[non_empty].index[-1]
        display = display.loc[:last_used_index]
    else:
        display = display.iloc[0:0]

    return display.reset_index(drop=True)

# ----------------------------------------------------------------------
# CHARGEMENT / PRÉPARATION — ÉVOLUTION QUOTIDIENNE
# ----------------------------------------------------------------------
def load_evolution_quotidienne(file):
    """
    Lit le fichier d'évolution quotidienne selon la structure du code source fourni :
    - ligne 1 : dates ;
    - ligne 2 : COMPTA / GESTION / ECART ;
    - colonne 1 à partir de la ligne 3 : noms des portefeuilles.
    """
    raw = None
    for _i in range(len(_sheet_names(file))):
        _cand = pd.read_excel(_to_buffer(file), sheet_name=_i, header=None)
        if _cand.shape[0] >= 2 and any(
            str(v).strip().upper() in {"COMPTA", "GESTION", "ECART"}
            for v in _cand.iloc[1, 1:].tolist()
        ):
            raw = _cand
            break
    if raw is None:
        raw = pd.read_excel(_to_buffer(file), header=None)

    if raw.shape[0] < 3 or raw.shape[1] < 4:
        raise ValueError(
            "Le fichier doit contenir au minimum deux lignes d'en-tête "
            "et une ligne de portefeuille."
        )

    dates_row = raw.iloc[0, 1:]
    labels_row = raw.iloc[1, 1:]

    columns = []
    current_date = None
    for date_value, label_value in zip(dates_row, labels_row):
        label = str(label_value).strip().upper() if pd.notna(label_value) else ""

        if pd.notna(date_value) and str(date_value).strip().upper() not in {
            "COMPTA", "GESTION", "ECART", ""
        }:
            try:
                current_date = pd.to_datetime(date_value).strftime("%d/%m/%Y")
            except Exception:
                current_date = str(date_value).strip()

        columns.append((current_date, label))

    valid_types = {"COMPTA", "GESTION", "ECART"}
    if not any(label in valid_types for _, label in columns):
        raise ValueError(
            "La deuxième ligne doit contenir les libellés COMPTA, GESTION et ECART."
        )

    portfolio_series = raw.iloc[2:, 0]
    keep_rows = portfolio_series.notna() & portfolio_series.astype(str).str.strip().ne("")
    portfolio_names = portfolio_series.loc[keep_rows].astype(str).str.strip().tolist()
    values = raw.iloc[2:, 1:].loc[keep_rows].values

    if not portfolio_names:
        raise ValueError("Aucun portefeuille n'a été trouvé dans le fichier.")

    multi_index = pd.MultiIndex.from_tuples(columns, names=["Date", "Type"])
    data = pd.DataFrame(values, index=portfolio_names, columns=multi_index)
    data = data.apply(pd.to_numeric, errors="coerce")

    available_types = set(data.columns.get_level_values("Type"))
    missing = valid_types - available_types
    if missing:
        raise ValueError("Types manquants : " + ", ".join(sorted(missing)))

    df_compta = data.xs("COMPTA", axis=1, level="Type")
    df_gestion = data.xs("GESTION", axis=1, level="Type")
    df_ecart = data.xs("ECART", axis=1, level="Type")

    common_dates = [
        date for date in df_compta.columns
        if date in df_gestion.columns and date in df_ecart.columns and date is not None
    ]
    if not common_dates:
        raise ValueError("Aucune date commune exploitable n'a été trouvée.")

    df_compta = df_compta.loc[:, common_dates]
    df_gestion = df_gestion.loc[:, common_dates]
    df_ecart = df_ecart.loc[:, common_dates]

    return portfolio_names, common_dates, df_compta, df_gestion, df_ecart


EVOLUTION_POSITIVE = "#2ecc71"
EVOLUTION_NEGATIVE = "#e74c3c"
EVOLUTION_COMPTA = "#1f77b4"
EVOLUTION_GESTION = "#f1c40f"
EVOLUTION_PTF_COLORS = [
    "#1f77b4", "#2ecc71", "#9b59b6",
    "#e67e22", "#1abc9c", "#ff6b6b",
    "#f1c40f", "#3498db", "#e84393",
]


def evolution_ptf_color(index: int) -> str:
    return EVOLUTION_PTF_COLORS[index % len(EVOLUTION_PTF_COLORS)]


def evolution_fmt_num(value, _position=None) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}Md"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def evolution_bar_colors(values):
    return [
        EVOLUTION_POSITIVE if float(value) >= 0 else EVOLUTION_NEGATIVE
        for value in values
    ]


def evolution_set_date_axis(axis, dates, rotation=40):
    axis.set_xticks(range(len(dates)))
    axis.set_xticklabels(dates, rotation=rotation, ha="right", fontsize=8)


def evolution_figure_compta_gestion(portfolios, dates, df_compta, df_gestion):
    fig, axes = plt.subplots(
        len(portfolios), 1,
        figsize=(20, max(5, 5 * len(portfolios))),
        sharex=False,
    )
    if len(portfolios) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Évolution COMPTA vs GESTION",
        fontsize=16, fontweight="bold", color="#222222", y=1.01,
    )
    x_pos = list(range(len(dates)))

    for index, (portfolio, axis) in enumerate(zip(portfolios, axes)):
        compta = df_compta.loc[portfolio].to_numpy(dtype=float)
        gestion = df_gestion.loc[portfolio].to_numpy(dtype=float)
        axis.set_facecolor("white")
        axis.plot(x_pos, compta, color=EVOLUTION_COMPTA, lw=2.5, marker="o", ms=5, label="COMPTA")
        axis.plot(x_pos, gestion, color=EVOLUTION_GESTION, lw=2.5, marker="s", ms=5, label="GESTION", linestyle="--")
        axis.fill_between(x_pos, compta, gestion, alpha=0.12, color="#999999")
        axis.set_title(portfolio, fontsize=11, color=evolution_ptf_color(index), pad=6)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(evolution_fmt_num))
        axis.legend(loc="upper right", fontsize=9, facecolor="white", edgecolor="#d0d0d0")
        evolution_set_date_axis(axis, dates)
        axis.axhline(0, color="#777777", lw=0.8)
        axis.grid(alpha=0.5, color="#dddddd", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    return fig


def evolution_figure_ecarts(portfolios, dates, df_ecart):
    fig, axes = plt.subplots(
        len(portfolios), 1,
        figsize=(20, max(5, 5 * len(portfolios))),
        sharex=False,
    )
    if len(portfolios) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Évolution de l'ÉCART (COMPTA - GESTION)",
        fontsize=16, fontweight="bold", color="#222222", y=1.01,
    )
    x_pos = list(range(len(dates)))

    for index, (portfolio, axis) in enumerate(zip(portfolios, axes)):
        ecart = df_ecart.loc[portfolio].to_numpy(dtype=float)
        colors = evolution_bar_colors(ecart)
        axis.set_facecolor("white")
        axis.bar(x_pos, ecart, color=colors, edgecolor="#ffffff", width=0.6)
        axis.plot(x_pos, ecart, color="#222222", lw=1, alpha=0.45, marker="o", ms=3)

        finite = np.abs(ecart[np.isfinite(ecart)])
        max_abs = finite.max() if finite.size and finite.max() > 0 else 1
        for j, (value, color) in enumerate(zip(ecart, colors)):
            if not np.isfinite(value):
                continue
            offset = max_abs * 0.04 if value >= 0 else -max_abs * 0.10
            axis.text(j, value + offset, evolution_fmt_num(value), ha="center", fontsize=6.5, color=color, fontweight="bold")

        axis.set_title(portfolio, fontsize=11, color=evolution_ptf_color(index), pad=6)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(evolution_fmt_num))
        axis.axhline(0, color="#777777", lw=1)
        evolution_set_date_axis(axis, dates)
        axis.grid(axis="y", alpha=0.5, color="#dddddd", linestyle="--", linewidth=0.6)
        axis.legend(
            handles=[
                mpatches.Patch(color=EVOLUTION_POSITIVE, label="Écart positif"),
                mpatches.Patch(color=EVOLUTION_NEGATIVE, label="Écart négatif"),
            ],
            loc="upper right", fontsize=9, facecolor="white", edgecolor="#d0d0d0",
        )

    fig.tight_layout()
    return fig


def evolution_figure_comparaison(portfolios, dates, df_compta, df_gestion, df_ecart):
    last_date = dates[-1]
    compta = df_compta[last_date].to_numpy(dtype=float)
    gestion = df_gestion[last_date].to_numpy(dtype=float)
    ecart = df_ecart[last_date].to_numpy(dtype=float)
    labels = [str(name)[:35] for name in portfolios]
    x = np.arange(len(labels))
    width = 0.28

    fig, axis = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")
    bars_compta = axis.bar(x - width, compta, width, label="COMPTA", color=EVOLUTION_COMPTA, edgecolor="#ffffff")
    bars_gestion = axis.bar(x, gestion, width, label="GESTION", color=EVOLUTION_GESTION, edgecolor="#ffffff")
    ecart_colors = evolution_bar_colors(ecart)
    for xi, (value, color) in enumerate(zip(ecart, ecart_colors)):
        axis.bar(x[xi] + width, value, width, color=color, edgecolor="#ffffff")

    for bars, values in ((bars_compta, compta), (bars_gestion, gestion)):
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + abs(value) * 0.02,
                evolution_fmt_num(value),
                ha="center", fontsize=8, color="#222222", fontweight="bold",
            )

    finite = np.abs(ecart[np.isfinite(ecart)])
    max_abs = finite.max() if finite.size and finite.max() > 0 else 1
    for xi, (value, color) in enumerate(zip(ecart, ecart_colors)):
        if not np.isfinite(value):
            continue
        y_text = value + max_abs * 0.12 if value >= 0 else max_abs * 0.08
        axis.text(x[xi] + width, y_text, evolution_fmt_num(value), ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    axis.set_title(f"Comparaison des portefeuilles au {last_date}", fontsize=14, fontweight="bold", color="#222222", pad=12)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=9, color="#222222", rotation=10, ha="right")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(evolution_fmt_num))
    axis.axhline(0, color="#777777", lw=0.8)
    axis.margins(y=0.18)
    axis.legend(
        handles=[
            mpatches.Patch(color=EVOLUTION_COMPTA, label="COMPTA"),
            mpatches.Patch(color=EVOLUTION_GESTION, label="GESTION"),
            mpatches.Patch(color=EVOLUTION_POSITIVE, label="ÉCART positif"),
            mpatches.Patch(color=EVOLUTION_NEGATIVE, label="ÉCART négatif"),
        ],
        fontsize=9, facecolor="white", edgecolor="#d0d0d0",
    )
    axis.grid(axis="y", alpha=0.5, color="#dddddd", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    return fig


def evolution_figure_ecart_cumule(portfolios, dates, df_ecart):
    fig, axis = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")
    x_pos = list(range(len(dates)))

    for index, portfolio in enumerate(portfolios):
        cumulative = df_ecart.loc[portfolio].to_numpy(dtype=float).cumsum()
        color = evolution_ptf_color(index)
        axis.plot(x_pos, cumulative, color=color, lw=2.5, marker="o", ms=5, label=str(portfolio)[:35])
        axis.fill_between(x_pos, cumulative, alpha=0.08, color=color)

    axis.set_title("Écart cumulé par portefeuille", fontsize=14, fontweight="bold", color="#222222", pad=12)
    evolution_set_date_axis(axis, dates)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(evolution_fmt_num))
    axis.axhline(0, color="#777777", lw=1, linestyle="--")
    axis.legend(fontsize=9, facecolor="white", edgecolor="#d0d0d0")
    axis.grid(alpha=0.5, color="#dddddd", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    return fig


def show_load_error(label: str, file, error: Exception):
    st.error(f"Erreur de lecture du fichier {label} : {error}")
    with st.expander("Aperçu brut du fichier (pour diagnostic)"):
        try:
            raw_preview = pd.read_excel(_to_buffer(file), header=None).head(10)
            st.dataframe(raw_preview, use_container_width=True)
        except Exception as e2:
            st.write(f"Impossible d'afficher un aperçu : {e2}")


# ----------------------------------------------------------------------
# STYLING DES TABLEAUX
# ----------------------------------------------------------------------
def style_ptf_table(df: pd.DataFrame):
    # La colonne de commentaires n'apparaît que si le fichier en contient.
    if COMMENT_COL in df.columns and not has_comments(df):
        df = df.drop(columns=[COMMENT_COL])
    color_map = {}
    row_colors = [color_for(p, color_map) for p in df["PORTFOLIO"]]

    def highlight_rows(row):
        idx = df.index.get_loc(row.name)
        styles = [f"background-color: {row_colors[idx]}; color: white; font-weight:600"] * len(row)
        # Commentaire : coloré comme la ligne s'il existe, sinon cellule neutre (comme dans Excel).
        if COMMENT_COL in row.index and not str(row[COMMENT_COL]).strip():
            styles[list(row.index).index(COMMENT_COL)] = ""
        return styles

    styler = (
        df.style.apply(highlight_rows, axis=1)
        .format({"COMPTA": fmt_money, "GESTION": fmt_money, "ECART": fmt_money})
        .set_properties(**{"text-align": "center"})
    )
    return styler


def format_swap_table(df: pd.DataFrame):
    """
    Formate le tableau Dérivées sans modifier les données métier.
    La date est affichée au format JJ/MM/AAAA, sans heure.
    """
    display = df.copy()
    if COMMENT_COL in display.columns and not has_comments(display):
        display = display.drop(columns=[COMMENT_COL])

    # Affichage stable de la date : 30/06/2026 au lieu de 30/06/2026 00:00:00.
    if "DATE" in display.columns:
        parsed_dates = pd.to_datetime(display["DATE"], errors="coerce", dayfirst=True)
        display["DATE"] = parsed_dates.dt.strftime("%d/%m/%Y").fillna("")

    # Garder les numéros de série comme texte.
    if "Num série" in display.columns:
        display["Num série"] = display["Num série"].map(series_as_text)

    # Cette fonction choisit la couleur de chaque ligne.
    def colorer_ligne(ligne):
        type_produit = str(ligne["Type"]).strip()
        if type_produit == "SWAP MUREX":
            style = "background-color: #4472C4; color: white;"
        elif type_produit == "SWAP HORS MUREX":
            style = "background-color: #C00000; color: white;"
        elif type_produit == "OPTIONS DE CHANGE":
            style = "background-color: #70AD47; color: white;"
        elif type_produit == "TRS GOLD":
            style = "background-color: #FFD966; color: black;"
        elif type_produit == "TRS ETF":
            style = "background-color: #ED7D31; color: white;"
        elif type_produit == "FUTURES MATIERES PREMIERES ET MASI 20":
            style = "background-color: #7030A0; color: white;"
        elif type_produit == "Forwards":
            style = "background-color: #00B050; color: white;"
        else:
            style = "background-color: white; color: black;"

        # Appliquer la même couleur à toutes les cellules de la ligne.
        styles = [style] * len(ligne)
        # Commentaire : coloré si présent, sinon cellule neutre (comme dans Excel).
        if COMMENT_COL in ligne.index and not str(ligne[COMMENT_COL]).strip():
            styles[list(ligne.index).index(COMMENT_COL)] = ""
        return styles

    styler = (
        display.style
        .format({
            "MUREX": fmt_money,
            "MO": fmt_money,
            "Ecart": fmt_money,
        })
        .apply(colorer_ligne, axis=1)
        .set_properties(**{
            "text-align": "center",
            "white-space": "nowrap",
        })
    )
    return styler



def format_tfr_table(df: pd.DataFrame):
    """Colore uniquement les en-têtes du tableau TFR, jamais les lignes."""
    display = df.copy()

    styler = (
        display.style
        .format({"Solde final": fmt_money}, na_rep="")
        .set_properties(**{
            "background-color": "white",
            "color": "#222222",
            "text-align": "left",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#C00000"),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("text-align", "center"),
                ],
            }
        ])
    )
    return styler


TFR_COLUMN_CONFIG = {
    "Code ligne": st.column_config.TextColumn("Code ligne"),
    "LIBELLE": st.column_config.TextColumn("LIBELLE"),
    "PCEC": st.column_config.TextColumn("PCEC"),
    "Num série": st.column_config.TextColumn("Num série"),
    "LIBELLE série": st.column_config.TextColumn("LIBELLE série"),
    "Produits": st.column_config.TextColumn("Produits"),
    "Solde final": st.column_config.NumberColumn("Solde final", format="%.2f"),
}


PTF_COLUMN_CONFIG = {
    "PORTFOLIO": st.column_config.TextColumn("PORTFOLIO", width="medium"),
    "COMPTA": st.column_config.NumberColumn("COMPTA", format="%.2f", width="medium"),
    "GESTION": st.column_config.NumberColumn("GESTION", format="%.2f", width="medium"),
    "ECART": st.column_config.NumberColumn("ECART", format="%.2f", width="medium"),
    COMMENT_COL: st.column_config.TextColumn(COMMENT_LABEL, width="large"),
}


SWAP_COLUMN_CONFIG = {
    "DATE": st.column_config.TextColumn("DATE", width="small"),
    "Type": st.column_config.TextColumn("Type", width="large"),
    "MUREX": st.column_config.NumberColumn("MUREX", format="%.2f", width="medium"),
    "MO": st.column_config.NumberColumn("MO", format="%.2f", width="medium"),
    "Ecart": st.column_config.NumberColumn("Ecart", format="%.2f", width="medium"),
    "Num série": st.column_config.TextColumn("Num série", width="medium"),
    COMMENT_COL: st.column_config.TextColumn(COMMENT_LABEL, width="large"),
}


# ----------------------------------------------------------------------
# ONGLETS PRINCIPAUX
# ----------------------------------------------------------------------
tab_ptf, tab_swap, tab_tfr, tab_evolution = st.tabs(
    ["PTF", "Dérivées", "TFR Dérivées / TFR TITRE", "Évolution quotidienne"]
)

# ======================================================================
# FENÊTRE 1 — PTF
# ======================================================================
with tab_ptf:
    render_module_header("PTF", "Rapprochement Compta-Gestion.")
    if ptf_file is None:
        st.info("Fichier PTF (Récapitulatif) non disponible : l'administrateur doit le charger.")
    else:
        df_ptf = None
        try:
            df_ptf = load_ptf(ptf_file)
        except Exception as e:
            show_load_error("PTF", ptf_file, e)

        if df_ptf is not None and not df_ptf.empty:
            st.caption(f"Fichier : {ptf_file.name} · {len(df_ptf)} ligne(s) détectée(s)")
            df_ptf_totals = (
                df_ptf.groupby("PORTFOLIO", as_index=False)[["COMPTA", "GESTION", "ECART"]]
                .sum(min_count=1)
                .sort_values("PORTFOLIO")
                .reset_index(drop=True)
            )
            portfolios = df_ptf_totals["PORTFOLIO"].tolist()
            ptf_placeholder = "-- Sélectionner un portefeuille --"
            selected_ptf = st.selectbox(
                "Portefeuille analysé",
                options=[ptf_placeholder] + portfolios,
                # Nouvelle clé pour ne pas réutiliser un ancien état Streamlit.
                key="select_ptf_dynamic_totals_v10",
            )

            st.divider()

            # Les cartes utilisent exactement la ligne agrégée du PTF sélectionné
            # dans le tableau des totaux. Elles sont donc recalculées à chaque choix.
            ptf_is_unselected = selected_ptf == ptf_placeholder
            df_ptf_scope = (
                df_ptf.copy()
                if ptf_is_unselected
                else df_ptf.loc[df_ptf["PORTFOLIO"].eq(selected_ptf)].copy()
            )

            if ptf_is_unselected:
                total_compta = total_gestion = total_ecart = None
            else:
                selected_totals = (
                    df_ptf_totals.loc[
                        df_ptf_totals["PORTFOLIO"].eq(selected_ptf),
                        ["COMPTA", "GESTION", "ECART"],
                    ]
                    .iloc[0]
                )
                total_compta = selected_totals["COMPTA"]
                total_gestion = selected_totals["GESTION"]
                total_ecart = selected_totals["ECART"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total COMPTA", "" if ptf_is_unselected else fmt_money(total_compta))
            c2.metric("Total GESTION", "" if ptf_is_unselected else fmt_money(total_gestion))
            c3.metric("Écart total", "" if ptf_is_unselected else fmt_money(total_ecart))

            st.divider()

            if ptf_is_unselected:
                st.markdown("#### Tableau complet des portefeuilles")
                st.dataframe(
                    style_ptf_table(df_ptf),
                    use_container_width=True,
                    height=340,
                    hide_index=True,
                    column_config=PTF_COLUMN_CONFIG,
                )

                st.markdown("#### Comparaison Compta vs Gestion par portefeuille")
                fig_bar = go.Figure()
                fig_bar.add_bar(name="COMPTA", x=df_ptf["PORTFOLIO"], y=df_ptf["COMPTA"],
                                 marker_color="#1976D2")
                fig_bar.add_bar(name="GESTION", x=df_ptf["PORTFOLIO"], y=df_ptf["GESTION"],
                                 marker_color="#43A047")
                fig_bar.update_layout(barmode="group", height=430,
                                       xaxis_tickangle=-25, legend_title_text="")
                st.plotly_chart(fig_bar, use_container_width=True, key="ptf_global_compta_gestion")

                st.markdown("#### Écarts par portefeuille")
                fig_ecart = px.bar(
                    df_ptf, x="PORTFOLIO", y="ECART", color="ECART",
                    color_continuous_scale=["#2E7D32", "#FFC107", "#C00000"],
                    text_auto=".2s",
                )
                fig_ecart.update_layout(height=420, xaxis_tickangle=-25)
                st.plotly_chart(fig_ecart, use_container_width=True, key="ptf_global_ecarts")

            else:
                st.markdown(f"#### Détail — {selected_ptf}")

                fig = go.Figure()
                fig.add_bar(name="COMPTA", x=["COMPTA"], y=[total_compta], marker_color="#1976D2")
                fig.add_bar(name="GESTION", x=["GESTION"], y=[total_gestion], marker_color="#43A047")
                fig.add_bar(name="ÉCART", x=["ÉCART"], y=[total_ecart], marker_color="#C00000")
                fig.update_layout(
                    height=420, showlegend=False,
                    title=f"Compta / Gestion / Écart — {selected_ptf}"
                )
                st.plotly_chart(fig, use_container_width=True, key="ptf_selected_comparison")

                st.markdown("##### Ligne(s) du tableau source")
                st.dataframe(
                    style_ptf_table(df_ptf_scope),
                    use_container_width=True,
                    height=340,
                    hide_index=True,
                    column_config=PTF_COLUMN_CONFIG,
                )
        elif df_ptf is not None:
            st.warning("Le fichier chargé ne contient aucune donnée exploitable.")


# ======================================================================
# FENÊTRE 2 — SWAP / OPTIONS DE CHANGE
# ======================================================================
with tab_swap:
    render_module_header("Dérivées", "Analyse des rapprochements MUREX / MO par type de produit dérivé.")
    if dérivées_file is None:
        st.info("Fichier Dérivées non disponible : l'administrateur doit le charger.")
    else:
        df_swap = None
        try:
            df_swap = load_swap(dérivées_file)
        except Exception as e:
            show_load_error("Dérivées", dérivées_file, e)

        if df_swap is not None and not df_swap.empty:
            st.caption(f"Fichier : {dérivées_file.name} · {len(df_swap)} ligne(s) détectée(s)")
            # Les seules valeurs autorisées à l'écran sont SWAP et OPTIONS DE CHANGE.
            # Toute variante ECHANGE / ÉCHANGE / ÉCHANGER / EXCHANGE est normalisée en SWAP.
            df_swap["TYPE"] = df_swap["TYPE"].map(canon_type)
            present_types = set(
                df_swap["TYPE"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            types_dispo = [
                label for label in (
                    "SWAP MUREX",
                    "SWAP HORS MUREX",
                    "OPTIONS DE CHANGE",
                    "TRS GOLD",
                    "TRS ETF",
                    "FUTURES MATIERES PREMIERES ET MASI 20",
                    "Forwards",
                )
                if label.upper() in {t.upper() for t in present_types}
            ]
            type_placeholder = "-- Sélectionner un type de Dérivées --"
            selected_type = st.selectbox(
                "Type de dérivé analysé",
                options=[type_placeholder] + types_dispo,
                key="select_type_v9",
            )

            type_is_unselected = selected_type == type_placeholder
            if type_is_unselected:
                df_scope = df_swap.copy()
            else:
                df_scope = df_swap[df_swap["TYPE"] == selected_type].copy()

            st.divider()

            # Les cartes restent vides avant le choix du type.
            c1, c2, c3 = st.columns(3)
            if type_is_unselected:
                c1.metric("Total MUREX", "")
                c2.metric("Total MO", "")
                c3.metric("Écart total", "")
            else:
                c1.metric("Total MUREX", fmt_money(df_scope["MUREX"].sum(min_count=1)))
                c2.metric("Total MO", fmt_money(df_scope["MO"].sum(min_count=1)))
                c3.metric("Écart total", fmt_money(df_scope["ECART"].sum(min_count=1)))

            st.divider()

            display_cols = ["DATE", "TYPE", "MUREX", "MO", "ECART"]
            if COMMENT_COL in df_swap.columns:
                display_cols.append(COMMENT_COL)
            rename_cols = {
                "TYPE": "Type","ECART": "Ecart",
            }

            # ---------- Cas : rien sélectionné → vue globale ----------
            if type_is_unselected:
                st.markdown("#### Tableau complet")
                st.dataframe(
                    format_swap_table(df_swap[display_cols].rename(columns=rename_cols)),
                    use_container_width=True,
                    height=340,
                    hide_index=True,
                    column_config=SWAP_COLUMN_CONFIG,
                )

                st.markdown("#### Répartition Murex / MO par type")
                agg = df_swap.groupby("TYPE")[["MUREX", "MO"]].sum().reset_index()
                fig_type = go.Figure()
                fig_type.add_bar(name="MUREX", x=agg["TYPE"], y=agg["MUREX"], marker_color="#C00000")
                fig_type.add_bar(name="MO", x=agg["TYPE"], y=agg["MO"], marker_color="#1976D2")
                fig_type.update_layout(barmode="group", height=420)
                st.plotly_chart(fig_type, use_container_width=True, key="swap_global_murex_mo")

                st.markdown("#### Écart total par type")
                agg_ecart = df_swap.groupby("TYPE")["ECART"].sum().reset_index()
                fig_ecart = px.bar(
                    agg_ecart, x="TYPE", y="ECART", color="ECART",
                    color_continuous_scale=["#2E7D32", "#FFC107", "#C00000"],
                    text_auto=".2s",
                )
                fig_ecart.update_layout(height=420)
                st.plotly_chart(fig_ecart, use_container_width=True, key="swap_global_ecarts")

            # ---------- Cas : type choisi ----------
            else:
                st.markdown(f"#### Détail — {selected_type}")

                if df_scope.empty:
                    st.warning("Aucune donnée pour cette sélection.")
                else:
                    st.dataframe(
                        format_swap_table(df_scope[display_cols].rename(columns=rename_cols)),
                        use_container_width=True,
                        height=340,
                        hide_index=True,
                        column_config=SWAP_COLUMN_CONFIG,
                    )

                    # Le graphique utilise les sommes de toutes les lignes du type choisi.
                    total_murex = df_scope["MUREX"].sum(min_count=1)
                    total_mo = df_scope["MO"].sum(min_count=1)
                    total_ecart = df_scope["ECART"].sum(min_count=1)

                    fig = go.Figure()
                    fig.add_bar(
                        name="MUREX",
                        x=[selected_type],
                        y=[total_murex],
                        marker_color="#C00000",
                        text=[fmt_money(total_murex)],
                        textposition="outside",
                    )
                    fig.add_bar(
                        name="MO",
                        x=[selected_type],
                        y=[total_mo],
                        marker_color="#1976D2",
                        text=[fmt_money(total_mo)],
                        textposition="outside",
                    )
                    fig.update_layout(
                        barmode="group",
                        height=420,
                        title=f"Totaux MUREX et MO — {selected_type}",
                        xaxis_title="",
                        yaxis_title="Montant",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="swap_selected_comparison")

                    st.caption(f"Écart total : {fmt_money(total_ecart)}")
        elif df_swap is not None:
            st.warning("Le fichier chargé ne contient aucune donnée exploitable.")


# ======================================================================
# FENÊTRE 3 — TFR / TFR TITRE
# ======================================================================
with tab_tfr:
    render_module_header("TFR Dérivées / TFR TITRE", "Simulation TFR.")
    source_placeholder = "-- Sélectionner TFR Dérivées ou TFR TITRE --"
    selected_tfr_source = st.selectbox(
        "Source TFR à afficher",
        options=[source_placeholder, "TFR Dérivées", "TFR TITRE"],
        key="select_tfr_source_v1",
    )

    if selected_tfr_source == source_placeholder:
        st.info("Sélectionnez TFR Dérivées ou TFR TITRE dans la liste ci-dessus.")

    else:
        selected_file = tfr_dérivées_file if selected_tfr_source == "TFR Dérivées" else tfr_titre_file

        if selected_file is None:
            st.info(f"Fichier {selected_tfr_source} non disponible : l'administrateur doit le charger.")
        else:
            df_tfr = None
            try:
                df_tfr = load_tfr_table(selected_file)
            except Exception as e:
                show_load_error(selected_tfr_source, selected_file, e)

            if df_tfr is not None and not df_tfr.empty:
                st.caption(f"Fichier : {selected_file.name} · {len(df_tfr)} ligne(s) détectée(s)")
                st.markdown(f"#### Tableau {selected_tfr_source}")
                st.dataframe(
                    format_tfr_table(df_tfr),
                    use_container_width=True,
                    height=560,
                    hide_index=True,
                    column_config=TFR_COLUMN_CONFIG,
                )
            elif df_tfr is not None:
                st.warning(
                    f"Le fichier {selected_tfr_source} ne contient aucune donnée exploitable."
                )

# ======================================================================
# FENÊTRE 4 — ÉVOLUTION QUOTIDIENNE
# ======================================================================
with tab_evolution:
    render_module_header("Évolution quotidienne", "Suivi quotidien du P&L.")
    if evolution_file is None:
        st.info("Fichier Récapitulatif Général non disponible : l'administrateur doit le charger.")
    else:
        evolution_data = None
        try:
            evolution_data = load_evolution_quotidienne(evolution_file)
        except Exception as error:
            show_load_error("Évolution quotidienne", evolution_file, error)

        if evolution_data is not None:
            portfolios, dates, df_compta_evo, df_gestion_evo, df_ecart_evo = evolution_data

            st.caption(
                f"Fichier : {evolution_file.name} · {len(portfolios)} portefeuille(s) · "
                f"Période : {dates[0]} au {dates[-1]} ({len(dates)} date(s))"
            )

            selected_portfolio = st.selectbox(
                "Portefeuille analysé",
                options=portfolios,
                key="evolution_portefeuille_selection",
            )

            selected_portfolios = [selected_portfolio]
            selected_compta = df_compta_evo.loc[selected_portfolios]
            selected_gestion = df_gestion_evo.loc[selected_portfolios]
            selected_ecart = df_ecart_evo.loc[selected_portfolios]

            last_date = dates[-1]
            latest_compta = selected_compta.loc[selected_portfolio, last_date]
            latest_gestion = selected_gestion.loc[selected_portfolio, last_date]
            latest_ecart = selected_ecart.loc[selected_portfolio, last_date]
            k1, k2, k3 = st.columns(3)
            k1.metric(f"COMPTA · {last_date}", fmt_money(latest_compta))
            k2.metric(f"GESTION · {last_date}", fmt_money(latest_gestion))
            k3.metric(f"Écart · {last_date}", fmt_money(latest_ecart))

            st.markdown(f"### Évolution COMPTA vs GESTION — {selected_portfolio}")
            figure_1 = evolution_figure_compta_gestion(
                selected_portfolios,
                dates,
                selected_compta,
                selected_gestion,
            )
            st.pyplot(figure_1, use_container_width=True)
            plt.close(figure_1)

            st.divider()
            st.markdown(f"### Évolution de l’ÉCART — {selected_portfolio}")
            figure_2 = evolution_figure_ecarts(
                selected_portfolios,
                dates,
                selected_ecart,
            )
            st.pyplot(figure_2, use_container_width=True)
            plt.close(figure_2)

            st.divider()
            st.markdown(f"### Comparaison au {last_date} — {selected_portfolio}")
            figure_3 = evolution_figure_comparaison(
                selected_portfolios,
                dates,
                selected_compta,
                selected_gestion,
                selected_ecart,
            )
            st.pyplot(figure_3, use_container_width=True)
            plt.close(figure_3)


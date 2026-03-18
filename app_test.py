# app.py
import json
import re
import uuid
from copy import deepcopy
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from create_index import (
    compute_level1_indices,
    compute_level2_indices,
    compute_level3_indices,
)



# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CFG_PATH = BASE_DIR / "index_config_D2_D3.json"



# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_cfg() -> dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Helpers generici
# -----------------------------
def make_fake_user_id(name: str) -> str:
    clean = (name or "").strip().lower()
    if not clean:
        return ""
    return f"sim_{uuid.uuid5(uuid.NAMESPACE_DNS, clean)}"


def percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value):
        return float("nan")
    return 100.0 * (s.le(value).sum() / len(s))


def radar_compare_plot(
    labels,
    values_a,
    values_b,
    label_a="Comune A",
    label_b="Comune B",
    title="Radar plot",
):
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    angles += angles[:1]
    a = np.r_[a, a[0]]
    b = np.r_[b, b[0]]

    fig = plt.figure(figsize=(7.5, 7.5), dpi=120)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.plot(angles, a, linewidth=2, label=label_a)
    ax.fill(angles, a, alpha=0.15)

    ax.plot(angles, b, linewidth=2, label=label_b)
    ax.fill(angles, b, alpha=0.10)

    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    return fig


# -----------------------------
# Store comuni simulati
# -----------------------------
def ensure_simulated_store():
    if "simulated_municipalities" not in st.session_state:
        st.session_state["simulated_municipalities"] = pd.DataFrame()


def upsert_simulated_municipality(result_row: pd.DataFrame):
    store = st.session_state["simulated_municipalities"].copy()
    uid = result_row["quesito.user_id"].iloc[0]

    if not store.empty and "quesito.user_id" in store.columns:
        store = store[store["quesito.user_id"] != uid].copy()

    store = pd.concat([store, result_row], ignore_index=True)
    st.session_state["simulated_municipalities"] = store


def delete_simulated_municipality(uid: str):
    store = st.session_state["simulated_municipalities"].copy()
    if not store.empty and "quesito.user_id" in store.columns:
        store = store[store["quesito.user_id"] != uid].copy()
    st.session_state["simulated_municipalities"] = store


def clear_simulated_store():
    st.session_state["simulated_municipalities"] = pd.DataFrame()


# -----------------------------
# Weight state management
# -----------------------------
def make_default_weights(cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {"level2": {}, "level3": {}}
    for lvl in ("level2", "level3"):
        indices = cfg.get(lvl, {}).get("indices", {})
        for idx_name, idx_cfg in indices.items():
            comps = idx_cfg.get("components", {})
            out[lvl][idx_name] = {k: float(v) for k, v in comps.items()}
    return out


def ensure_weight_state(default_weights: dict[str, dict[str, dict[str, float]]]):
    if "weights_override" not in st.session_state:
        st.session_state["weights_override"] = {"level2": {}, "level3": {}}
    if "weights_default" not in st.session_state:
        st.session_state["weights_default"] = default_weights


def reset_weights_to_default():
    st.session_state["weights_override"] = {"level2": {}, "level3": {}}


def get_effective_components(level: str, index_name: str, cfg: dict[str, Any]) -> dict[str, float]:
    ovr = st.session_state["weights_override"].get(level, {}).get(index_name)
    if ovr and isinstance(ovr, dict) and len(ovr) > 0:
        return {k: float(v) for k, v in ovr.items()}
    return {k: float(v) for k, v in cfg[level]["indices"][index_name].get("components", {}).items()}


def apply_overrides_to_config(cfg: dict[str, Any]) -> dict[str, Any]:
    config_app = deepcopy(cfg)
    for lvl in ("level2", "level3"):
        for idx_name, ovr in st.session_state["weights_override"].get(lvl, {}).items():
            if (
                idx_name in config_app.get(lvl, {}).get("indices", {})
                and isinstance(ovr, dict)
                and len(ovr) > 0
            ):
                config_app[lvl]["indices"][idx_name]["components"] = {
                    k: float(v) for k, v in ovr.items()
                }
    return config_app


def override_weights_ui(
    cfg: dict[str, Any],
    level_key: str,
    target_indices: list[str],
    label_prefix: str,
):
    if not target_indices:
        return

    for idx in target_indices:
        base_components = (
            cfg.get(level_key, {}).get("indices", {}).get(idx, {}).get("components", {})
        )
        if not base_components:
            continue

        effective = get_effective_components(level_key, idx, cfg)

        with st.sidebar.expander(f"{label_prefix}: {idx}", expanded=(len(target_indices) == 1)):
            st.caption(
                "Pesi grezzi: se `normalize_weights=true` verranno normalizzati dalla funzione."
            )
            weights_new = {}
            for comp, w_default in base_components.items():
                w_eff = float(effective.get(comp, w_default))
                weights_new[comp] = st.slider(
                    f"{idx.replace('_', ' ')} · {comp.replace('_', ' ')}",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(w_eff),
                    step=0.05,
                    key=f"{level_key}:{idx}:{comp}",
                )

            st.session_state["weights_override"][level_key][idx] = weights_new


# -----------------------------
# Questionario dinamico
# -----------------------------
def get_question_text(idx_cfg: dict[str, Any]) -> str:
    return idx_cfg.get("question") or idx_cfg.get("domanda") or "Domanda"


def get_domanda_id(idx_cfg: dict[str, Any]) -> int:
    vals = idx_cfg.get("filters", {}).get("domanda_id", [])
    if not vals:
        raise ValueError("domanda_id non trovato nella config")
    return int(vals[0])


def get_question_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Costruisce una lista di domande da mostrare una sola volta.
    La chiave di deduplica è principalmente la domanda_id.
    """
    question_specs: list[dict[str, Any]] = []
    seen_domanda_ids = set()

    for _, idx_cfg in config["level1"]["indices"].items():
        domanda_id = get_domanda_id(idx_cfg)
        if domanda_id in seen_domanda_ids:
            continue
        seen_domanda_ids.add(domanda_id)

        scoring = idx_cfg.get("scoring", {})
        score_type = scoring.get("score_type", "value_map")
        response_col = scoring.get("response_col", "risposta_valore")
        row_id_col = scoring.get("row_id_col")
        selection_mode = bool(scoring.get("selection_mode", False))

        spec: dict[str, Any] = {
            "question_key": f"q_{domanda_id}",
            "domanda_id": domanda_id,
            "question_text": get_question_text(idx_cfg),
            "description": idx_cfg.get("description", ""),
            "score_type": score_type,
            "response_col": response_col,
            "row_id_col": row_id_col,
            "selection_mode": selection_mode,
            "weights": scoring.get("weights", {}),
            "value_map": scoring.get("value_map", {}),
            "row_value_maps": scoring.get("row_value_maps", {}),
            "integration_weight_matrix": scoring.get("integration_weight_matrix", {}),
            "non_integrated_value": scoring.get("non_integrated_value"),
            "none_value": scoring.get("none_value"),
            "min_value": float(scoring.get("min_value", 0)),
            "max_value": float(scoring.get("max_value", 100)),
        }

        # selection_mode=true -> domanda multi-selezione "piatta"
        if selection_mode:
            spec["row_labels"] = []
            spec["options"] = list(scoring.get("weights", {}).keys())

        # multi_select_rows_weight_matrix
        elif score_type == "multi_select_rows_weight_matrix":
            row_labels = list(scoring.get("integration_weight_matrix", {}).keys())
            spec["row_labels"] = row_labels
            options_by_row = {}
            for row_label, row_map in scoring.get("integration_weight_matrix", {}).items():
                opts = list(row_map.keys())
                non_val = scoring.get("non_integrated_value")
                if non_val:
                    opts.append(non_val)
                options_by_row[row_label] = list(dict.fromkeys(opts))
            spec["options_by_row"] = options_by_row

        # value_map_by_row
        elif score_type == "value_map_by_row":
            row_value_maps = scoring.get("row_value_maps", {})
            spec["row_labels"] = list(row_value_maps.keys())
            spec["options_by_row"] = {
                row_label: list(vmap.keys()) for row_label, vmap in row_value_maps.items()
            }

        # righe + value_map classico / percentage / percentage_binary
        elif row_id_col and scoring.get("weights"):
            spec["row_labels"] = list(scoring.get("weights", {}).keys())
            spec["options"] = list(scoring.get("value_map", {}).keys())

        # domanda singola
        else:
            spec["row_labels"] = []
            spec["options"] = list(scoring.get("value_map", {}).keys())

        question_specs.append(spec)

    question_specs = sort_question_specs(question_specs)
    return question_specs


def extract_question_number(question_text: str) -> tuple:
    """
    Estrae il prefisso numerico iniziale dal testo domanda.
    Esempi:
    '7.5 ...'  -> (7, 5)
    '9.2 ...'  -> (9, 2)
    '11.3 ...' -> (11, 3)
    Se non trova nulla, manda in fondo.
    """
    if not isinstance(question_text, str):
        return (9999, 9999)

    m = re.match(r"^\s*(\d+)(?:\.(\d+))?", question_text.strip())
    if not m:
        return (9999, 9999)

    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) is not None else 0
    return (major, minor)


def sort_question_specs(question_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        question_specs,
        key=lambda x: (
            extract_question_number(x.get("question_text", "")),
            x.get("domanda_id", 999999),
        ),
    )


def render_questionnaire(question_specs: list[dict[str, Any]]):
    st.subheader("Questionario per il calcolo degli indici")
    st.caption(
        "Le domande sono generate direttamente dalla configurazione degli indici di livello 1."
    )

    for q in question_specs:
        question_key = q["question_key"]
        question_text = q["question_text"]
        description = q["description"]
        score_type = q["score_type"]
        row_labels = q.get("row_labels", [])
        selection_mode = q.get("selection_mode", False)

        with st.expander(question_text, expanded=False):
            if description:
                st.caption(description)

            # 1) selection_mode=true
            if selection_mode:
                opts = q.get("options", [])
                none_value = q.get("none_value")

                selected = st.multiselect(
                    "Seleziona una o più opzioni",
                    options=opts,
                    default=[],
                    key=f"ans::{question_key}::__selection__",
                    help="Se non selezioni nulla, verrà considerata l'opzione di assenza prevista dalla configurazione, se presente.",
                )

                if none_value and none_value in selected and len(selected) > 1:
                    st.warning(
                        f"Hai selezionato '{none_value}' insieme ad altre opzioni. "
                        "Nel calcolo verrà ignorata l'opzione di assenza."
                    )
                continue

            # 2) domanda singola
            if not row_labels:
                opts = q.get("options", [])
                st.selectbox(
                    "Risposta",
                    options=opts,
                    key=f"ans::{question_key}::__single__",
                )
                continue

            # 3) multi_select_rows_weight_matrix
            if score_type == "multi_select_rows_weight_matrix":
                non_val = q.get("non_integrated_value")
                options_by_row = q.get("options_by_row", {})

                for row_label in row_labels:
                    selected = st.multiselect(
                        row_label,
                        options=options_by_row.get(row_label, []),
                        default=[],
                        key=f"ans::{question_key}::{row_label}",
                        help="Puoi selezionare più integrazioni. Se lasci vuoto, sarà considerato il valore di non integrazione.",
                    )

                    if non_val and non_val in selected and len(selected) > 1:
                        st.warning(
                            f"Per '{row_label}': hai selezionato '{non_val}' insieme ad altre opzioni. "
                            "Nel calcolo verrà ignorato il valore di non integrazione."
                        )
                continue

            # 4) value_map_by_row
            if score_type == "value_map_by_row":
                options_by_row = q.get("options_by_row", {})
                for row_label in row_labels:
                    st.selectbox(
                        row_label,
                        options=options_by_row.get(row_label, []),
                        key=f"ans::{question_key}::{row_label}",
                    )
                continue

            # 5) percentage / percentage_binary
            if score_type in {"percentage", "percentage_binary"}:
                min_value = int(q.get("min_value", 0))
                max_value = int(q.get("max_value", 100))
                for row_label in row_labels:
                    st.slider(
                        row_label,
                        min_value=min_value,
                        max_value=max_value,
                        value=min_value,
                        step=1,
                        key=f"ans::{question_key}::{row_label}",
                    )
                continue

            # 6) value_map per riga
            opts = q.get("options", [])
            for row_label in row_labels:
                st.selectbox(
                    row_label,
                    options=opts,
                    key=f"ans::{question_key}::{row_label}",
                )


def build_response_long_df(
    question_specs: list[dict[str, Any]],
    comune_user_id: str,
    comune_nome: str,
) -> pd.DataFrame:
    rows = []

    for q in question_specs:
        question_key = q["question_key"]
        domanda_id = q["domanda_id"]
        score_type = q["score_type"]
        response_col = q["response_col"]
        row_id_col = q.get("row_id_col")
        row_labels = q.get("row_labels", [])
        selection_mode = q.get("selection_mode", False)

        # 1) selection_mode=true
        if selection_mode:
            key = f"ans::{question_key}::__selection__"
            selected = st.session_state.get(key, []) or []
            none_value = q.get("none_value")

            if not selected and none_value:
                selected = [none_value]

            if none_value in selected and len(selected) > 1:
                selected = [x for x in selected if x != none_value]

            for sel in selected:
                row = {
                    "quesito.user_id": comune_user_id,
                    "quesito.username": comune_nome,
                    "domanda_id": domanda_id,
                    "risposta_voce": None,
                    "risposta_valore": np.nan,
                }
                row[response_col] = sel
                rows.append(row)
            continue

        # 2) domanda singola
        if not row_labels:
            key = f"ans::{question_key}::__single__"
            if key not in st.session_state:
                continue

            value = st.session_state.get(key)
            row = {
                "quesito.user_id": comune_user_id,
                "quesito.username": comune_nome,
                "domanda_id": domanda_id,
                "risposta_voce": None,
                "risposta_valore": np.nan,
            }
            row[response_col] = value
            rows.append(row)
            continue

        # 3) domande per riga
        for row_label in row_labels:
            key = f"ans::{question_key}::{row_label}"
            if key not in st.session_state:
                continue

            value = st.session_state.get(key)

            # multi_select_rows_weight_matrix
            if score_type == "multi_select_rows_weight_matrix":
                selected = value or []
                non_val = q.get("non_integrated_value")

                if not selected and non_val is not None:
                    selected = [non_val]

                if non_val in selected and len(selected) > 1:
                    selected = [x for x in selected if x != non_val]

                for sel in selected:
                    row = {
                        "quesito.user_id": comune_user_id,
                        "quesito.username": comune_nome,
                        "domanda_id": domanda_id,
                        "risposta_voce": row_label,
                        "risposta_valore": np.nan,
                    }
                    row[response_col] = sel
                    rows.append(row)

            # percentage / percentage_binary
            elif score_type in {"percentage", "percentage_binary"}:
                row = {
                    "quesito.user_id": comune_user_id,
                    "quesito.username": comune_nome,
                    "domanda_id": domanda_id,
                    "risposta_voce": row_label,
                    "risposta_valore": np.nan,
                }
                row[response_col] = float(value) if value is not None else np.nan
                rows.append(row)

            # value_map_by_row o value_map per riga
            else:
                row = {
                    "quesito.user_id": comune_user_id,
                    "quesito.username": comune_nome,
                    "domanda_id": domanda_id,
                    "risposta_voce": None,
                    "risposta_valore": np.nan,
                }
                if row_id_col:
                    row[row_id_col] = row_label
                row[response_col] = value
                rows.append(row)

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return pd.DataFrame(
            columns=[
                "quesito.user_id",
                "quesito.username",
                "domanda_id",
                "risposta_voce",
                "risposta_valore",
            ]
        )

    if "risposta_voce" not in df_long.columns:
        df_long["risposta_voce"] = None
    if "risposta_valore" not in df_long.columns:
        df_long["risposta_valore"] = np.nan

    return df_long


# -----------------------------
# Profilo componenti
# -----------------------------
def get_components_for_index(
    config_app: dict[str, Any], level_key: str, idx_name: str
) -> dict[str, float]:
    if level_key not in ("level2", "level3"):
        return {}
    return {
        k: float(v)
        for k, v in config_app.get(level_key, {})
        .get("indices", {})
        .get(idx_name, {})
        .get("components", {})
        .items()
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Simulatore Indici Comuni 2025", layout="wide")
st.title("Simulatore Indici Mappa dei Comuni Digitali 2025")

cfg = load_cfg()
question_specs = get_question_specs(cfg)

default_weights = make_default_weights(cfg)
ensure_weight_state(default_weights)
ensure_simulated_store()

# Sidebar - Comune simulato
st.sidebar.header("Comune simulato")

comune_nome = st.sidebar.text_input("Nome del Comune", value="")
ordine_classi = [
    "0-2500 ab.",
    "2500-5000 ab.",
    "5000-20000 ab.",
    "20000-50000 ab.",
    "50000-250000 ab.",
    ">250000 ab.",
]
classe_dimensionale = st.sidebar.selectbox("Classe dimensionale", ordine_classi, index=2)

simulated_user_id = make_fake_user_id(comune_nome) if comune_nome.strip() else ""

# Sidebar selectors indice
st.sidebar.header("Indice da visualizzare")

levels_map = {
    "Livello 1 (base)": ("level1", cfg.get("level1", {}).get("indices", {})),
    "Livello 2 (compositi)": ("level2", cfg.get("level2", {}).get("indices", {})),
    "Livello 3 (macro)": ("level3", cfg.get("level3", {}).get("indices", {})),
}
level_label = st.sidebar.radio("Livello", list(levels_map.keys()), index=1)
level_key, level_indices = levels_map[level_label]

if not level_indices:
    st.error(f"Nessun indice trovato in config per {level_key}.")
    st.stop()

idx_name = st.sidebar.selectbox("Indice da visualizzare", list(level_indices.keys()))
idx_label = idx_name.replace("_", " ")

# Sidebar pesi
st.sidebar.header("Pesi")
c_reset, c_info = st.sidebar.columns([1, 3])
with c_reset:
    if st.button(
        "Reset pesi", help="Rimuove tutti gli override e torna ai pesi del file di config."
    ):
        reset_weights_to_default()
        st.rerun()
with c_info:
    st.caption("Reset ripristina i pesi del config per tutti gli indici L2 e L3.")

mode = st.sidebar.radio(
    "Modalità",
    ["Base (solo indice selezionato)", "Avanzata (scegli più indici)"],
    index=0,
)

if mode.startswith("Base"):
    if level_key in ("level2", "level3"):
        override_weights_ui(cfg, level_key, [idx_name], "Pesi")
else:
    if level_key == "level1":
        st.sidebar.info("I pesi sono modificabili solo per livello 2 e 3.")
    else:
        selectable = list(level_indices.keys())
        selected = st.sidebar.multiselect(
            f"Seleziona indici da modificare ({level_key})",
            selectable,
            default=[idx_name],
        )
        override_weights_ui(cfg, level_key, selected, "Pesi")

config_app = apply_overrides_to_config(cfg)

with st.sidebar.expander("Archivio simulato"):
    store = st.session_state["simulated_municipalities"]
    st.caption(f"Comuni simulati salvati: **{len(store)}**")
    if st.button("Svuota archivio simulato"):
        clear_simulated_store()
        st.rerun()

# Questionario
render_questionnaire(question_specs)

# Bottone calcolo
st.divider()
if st.button("Calcola / aggiorna Comune simulato", type="primary"):
    st.session_state["run_calc"] = True

if not st.session_state.get("run_calc", False):
    st.info(
        "Inserisci nome del Comune, scegli la classe dimensionale, compila il questionario e premi il pulsante di calcolo."
    )
    st.stop()

if not comune_nome.strip():
    st.warning("Inserisci il nome del Comune simulato.")
    st.stop()

# Costruzione df_long
df_long = build_response_long_df(
    question_specs=question_specs,
    comune_user_id=simulated_user_id,
    comune_nome=comune_nome.strip(),
)

if df_long.empty:
    st.warning("Nessuna risposta disponibile: compila almeno una domanda.")
    st.stop()

# Calcolo indici
try:
    level1_df, detail_df = compute_level1_indices(df_long, config_app)
    level2_df = compute_level2_indices(level1_df, config_app)
    level3_df = compute_level3_indices(level2_df, config_app)
except Exception as e:
    st.error(f"Errore nel calcolo degli indici: {e}")
    st.stop()

# Metadati comune simulato
for dfx in (level1_df, level2_df, level3_df):
    dfx["quesito.user_id"] = simulated_user_id
    dfx["quesito.username"] = comune_nome.strip()
    dfx["classe_dimensionale_6classi"] = classe_dimensionale

# Salvataggio archivio simulato
final_df = level3_df.copy()
upsert_simulated_municipality(final_df)

# Selezione df sorgente per indice
if level_key == "level1":
    source_df = level1_df.copy()
elif level_key == "level2":
    source_df = level2_df.copy()
else:
    source_df = level3_df.copy()

if idx_name not in source_df.columns:
    st.error(f"La colonna `{idx_name}` non esiste e non è stata calcolata.")
    st.stop()

val = float(source_df[idx_name].iloc[0]) if pd.notna(source_df[idx_name].iloc[0]) else float("nan")

# Archivio simulato aggiornato
store = st.session_state["simulated_municipalities"].copy()

# Ranking simulato globale
rank = None
n_tot = 0
perc = np.nan
mean_all = np.nan
mean_class = np.nan
rank_class = None
n_class = 0

if not store.empty and idx_name in store.columns:
    svals = pd.to_numeric(store[idx_name], errors="coerce")
    current_mask = store["quesito.user_id"] == simulated_user_id

    if current_mask.any():
        current_val = pd.to_numeric(store.loc[current_mask, idx_name], errors="coerce").iloc[0]
        rank = (
            int(svals.rank(ascending=False, method="min").loc[current_mask].iloc[0])
            if pd.notna(current_val)
            else None
        )
        n_tot = int(svals.notna().sum())
        perc = percentile_rank(svals, current_val)
        mean_all = float(svals.mean()) if svals.notna().any() else np.nan

        same_class_mask = store["classe_dimensionale_6classi"] == classe_dimensionale
        same_class_vals = pd.to_numeric(store.loc[same_class_mask, idx_name], errors="coerce")
        mean_class = float(same_class_vals.mean()) if same_class_vals.notna().any() else np.nan
        n_class = int(same_class_vals.notna().sum())

        class_ranks = same_class_vals.rank(ascending=False, method="min")
        class_current_mask = same_class_mask & current_mask
        if class_current_mask.any():
            rank_class = (
                int(class_ranks.loc[class_current_mask].iloc[0]) if pd.notna(current_val) else None
            )

# Tabs
tab_indici, tab_profilo, tab_ranking, tab_debug = st.tabs(
    ["📊 Indici calcolati", "🏅 Profilo Comune", "🏆 Ranking simulato", "🧪 Debug"]
)

with tab_indici:
    st.subheader("Indice selezionato")
    c1, c2, c3 = st.columns(3)
    c1.metric("Comune", comune_nome.strip())
    c2.metric("Classe dimensionale", classe_dimensionale)
    c3.metric(idx_label, f"{val:.2f}" if pd.notna(val) else "NaN")

    st.divider()

    st.subheader("Indici di livello 1")
    st.dataframe(level1_df, use_container_width=True)

    st.subheader("Indici di livello 2")
    st.dataframe(level2_df, use_container_width=True)

    st.subheader("Indici di livello 3")
    st.dataframe(level3_df, use_container_width=True)

with tab_profilo:
    st.subheader("Profilo del Comune selezionato")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comune", comune_nome.strip())
    c2.metric("Valore indice", f"{val:.2f}" if pd.notna(val) else "NaN")
    c3.metric("Rank simulato", f"{rank} / {n_tot}" if rank is not None else "—")
    c4.metric("Percentile simulato", f"{perc:.1f}%" if pd.notna(perc) else "—")

    c5, c6, c7 = st.columns(3)
    c5.metric(
        "Media archivio simulato",
        f"{mean_all:.2f}" if pd.notna(mean_all) else "—",
        delta=(f"{(val - mean_all):+.2f}" if pd.notna(val) and pd.notna(mean_all) else None),
    )
    c6.metric("Classe dimensionale", classe_dimensionale)
    c7.metric(
        "Rank nella classe",
        f"{rank_class} / {n_class}" if rank_class is not None else "—",
    )

    if pd.notna(mean_class):
        st.metric(
            "Media classe dimensionale",
            f"{mean_class:.2f}",
            delta=(f"{(val - mean_class):+.2f}" if pd.notna(val) else None),
        )

    st.divider()

    comps = get_components_for_index(config_app, level_key, idx_name)

    if not comps:
        st.info("Per gli indici di livello 1 non ci sono componenti da mostrare.")
    else:
        comp_rows = []
        for comp_name, w in comps.items():
            if comp_name not in level3_df.columns:
                continue
            vcomp = level3_df[comp_name].iloc[0]
            vcomp = float(vcomp) if pd.notna(vcomp) else float("nan")
            comp_rows.append(
                {
                    "comp_key": comp_name,
                    "componente": comp_name.replace("_", " "),
                    "valore": vcomp,
                    "peso": float(w),
                    "contributo": (vcomp * float(w)) if pd.notna(vcomp) else float("nan"),
                }
            )

        if not comp_rows:
            st.warning("Nessuna componente trovata nei dati.")
        else:
            df_comp = pd.DataFrame(comp_rows).sort_values("peso", ascending=False)

            st.subheader("Profilo componenti (radar)")
            st.caption("Confronto tra il comune corrente e un secondo comune simulato.")

            other_options_df = store[store["quesito.user_id"] != simulated_user_id].copy()

            if other_options_df.empty:
                st.info("Crea almeno un altro comune simulato per attivare il confronto nel radar.")
            else:
                other_options_df = other_options_df[
                    ["quesito.user_id", "quesito.username"]
                ].drop_duplicates()
                other_options_df["label"] = other_options_df["quesito.username"]

                selected_other_label = st.selectbox(
                    "Confronta con",
                    options=other_options_df["label"].tolist(),
                    index=0,
                    key="radar_compare_municipality",
                )

                selected_other_uid = other_options_df.loc[
                    other_options_df["label"] == selected_other_label, "quesito.user_id"
                ].iloc[0]

                other_row = store.loc[store["quesito.user_id"] == selected_other_uid].head(1)

                vals_other = []
                for comp_key in df_comp["comp_key"].tolist():
                    if comp_key in other_row.columns:
                        vals_other.append(
                            float(pd.to_numeric(other_row[comp_key], errors="coerce").iloc[0])
                        )
                    else:
                        vals_other.append(float("nan"))

                fig = radar_compare_plot(
                    labels=df_comp["componente"].tolist(),
                    values_a=df_comp["valore"].astype(float).tolist(),
                    values_b=vals_other,
                    label_a=comune_nome.strip(),
                    label_b=selected_other_label,
                    title=f"{idx_label} — confronto componenti",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                with st.expander("Dettaglio numerico confronto", expanded=False):
                    df_out = df_comp[["componente", "valore", "peso"]].copy()
                    df_out = df_out.rename(columns={"valore": comune_nome.strip()})
                    df_out[selected_other_label] = vals_other
                    df_out["delta"] = df_out[comune_nome.strip()] - df_out[selected_other_label]
                    st.dataframe(df_out, use_container_width=True)

with tab_ranking:
    st.subheader("Ranking dei Comuni simulati")

    if store.empty:
        st.info("Non ci sono ancora Comuni simulati salvati in questa sessione.")
    else:
        if idx_name not in store.columns:
            st.warning(f"L'indice `{idx_name}` non è presente nell'archivio simulato.")
        else:
            ranking_df = store[
                ["quesito.user_id", "quesito.username", "classe_dimensionale_6classi", idx_name]
            ].copy()

            ranking_df[idx_name] = pd.to_numeric(ranking_df[idx_name], errors="coerce")
            ranking_df = ranking_df.sort_values(idx_name, ascending=False).reset_index(drop=True)
            ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)

            st.caption("Ranking complessivo")
            st.dataframe(
                ranking_df[["rank", "quesito.username", "classe_dimensionale_6classi", idx_name]],
                use_container_width=True,
            )

            st.subheader("Ranking nella stessa classe dimensionale")
            same_class = (
                ranking_df[ranking_df["classe_dimensionale_6classi"] == classe_dimensionale]
                .copy()
                .reset_index(drop=True)
            )
            same_class["rank_classe"] = np.arange(1, len(same_class) + 1)

            st.dataframe(
                same_class[
                    ["rank_classe", "quesito.username", "classe_dimensionale_6classi", idx_name]
                ],
                use_container_width=True,
            )

            st.subheader("Gestione archivio simulato")
            selectable_names = ranking_df["quesito.username"].tolist()
            if selectable_names:
                to_delete = st.selectbox(
                    "Seleziona un Comune simulato da eliminare",
                    options=selectable_names,
                    index=0,
                )
                if st.button("Elimina Comune simulato"):
                    uid_to_delete = store.loc[
                        store["quesito.username"] == to_delete, "quesito.user_id"
                    ].iloc[0]
                    delete_simulated_municipality(uid_to_delete)
                    st.rerun()

with tab_debug:
    st.subheader("Dataset long costruito dalle risposte")
    st.dataframe(df_long, use_container_width=True)

    st.subheader("Dettaglio scoring")
    st.dataframe(detail_df, use_container_width=True)

    st.subheader("Archivio simulato corrente")
    st.dataframe(store, use_container_width=True)

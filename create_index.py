from typing import Any

import numpy as np
import pandas as pd


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Filtri semplici: valore scalare => uguaglianza; lista => isin."""
    out = df.copy()
    for col, cond in filters.items():
        if col not in out.columns:
            raise ValueError(f"Colonna filtro non presente nel dataset: '{col}'")
        if isinstance(cond, list):
            out = out[out[col].isin(cond)]
        else:
            out = out[out[col] == cond]
    return out


def _prepare_value_map(scoring_cfg: dict[str, Any]) -> dict[str, float]:
    """Prepara la mappa risposta->score (matching ESATTO)."""
    if "value_map" not in scoring_cfg:
        raise ValueError("Config scoring non valida: manca 'value_map'")
    value_map = scoring_cfg["value_map"]
    if not isinstance(value_map, dict) or len(value_map) == 0:
        raise ValueError("'value_map' deve essere un dizionario non vuoto")
    return {str(k): float(v) for k, v in value_map.items()}


def _map_response_score(
    value: Any, prepared_value_map: dict[str, float], default_score: float
) -> float:
    """Mappa una risposta in punteggio usando value_map (matching esatto)."""
    key = "" if pd.isna(value) else str(value)
    return float(prepared_value_map.get(key, default_score))


def _get_max_row_score(prepared_value_map: dict[str, float], default_score: float) -> float:
    """Punteggio massimo teorico per riga (serve per normalizzazione)."""
    candidates = list(prepared_value_map.values())
    if pd.notna(default_score):
        candidates.append(float(default_score))
    return float(max(candidates)) if candidates else 0.0


def _validate_index_config(df_long: pd.DataFrame, index_name: str, idx_cfg: dict[str, Any]) -> None:
    """
    Valida la config di un indice.
    Supporta:
      - score_type='value_map'    (301, 309)
      - score_type='percentage'   (386)
      - score_type='percentage_binary' (386, ma misura disponibilità invece che livello)
    """
    required_top = ["entity_keys", "filters", "scoring"]
    for k in required_top:
        if k not in idx_cfg:
            raise ValueError(f"[{index_name}] manca la sezione '{k}'")

    entity_keys = idx_cfg["entity_keys"]
    missing_entity = [c for c in entity_keys if c not in df_long.columns]
    if missing_entity:
        raise ValueError(
            f"[{index_name}] colonne entity_keys mancanti nel dataset: {missing_entity}"
        )

    filter_cols = [c for c in idx_cfg.get("filters", {}).keys() if c not in df_long.columns]
    if filter_cols:
        raise ValueError(f"[{index_name}] colonne di filtro mancanti nel dataset: {filter_cols}")

    scoring = idx_cfg["scoring"]
    score_type = scoring.get("score_type", "value_map")  # default retrocompatibile

    if "response_col" not in scoring:
        raise ValueError(f"[{index_name}] scoring: manca 'response_col'")

    response_col = scoring["response_col"]
    if response_col not in df_long.columns:
        raise ValueError(f"[{index_name}] colonna scoring mancante nel dataset: '{response_col}'")

    if score_type == "value_map":
        if bool(scoring.get("selection_mode", False)):
            weights_cfg = scoring.get("weights", {})
            if not weights_cfg:
                raise ValueError(f"[{index_name}] selection_mode=true richiede scoring.weights")
            none_val = scoring.get("none_value", None)
            max_total = sum(
                float(w) for k, w in weights_cfg.items() if none_val is None or k != none_val
            )
            if max_total <= 0:
                raise ValueError(
                    f"[{index_name}] selection_mode=true: somma pesi (escluso none_value) deve essere > 0"
                )
        else:
            if "value_map" not in scoring:
                raise ValueError(
                    f"[{index_name}] scoring.value_map mancante (score_type='value_map')"
                )
    elif score_type in {"percentage", "percentage_binary"}:
        # per percentuali non serve value_map
        pass
    elif score_type == "multi_select_rows_weight_matrix":
        for k in ["row_id_col", "non_integrated_value", "integration_weight_matrix"]:
            if k not in scoring:
                raise ValueError(
                    f"[{index_name}] scoring.{k} mancante (score_type='multi_select_rows_weight_matrix')"
                )
        if (
            not isinstance(scoring["integration_weight_matrix"], dict)
            or len(scoring["integration_weight_matrix"]) == 0
        ):
            raise ValueError(
                f"[{index_name}] scoring.integration_weight_matrix deve essere un dict non vuoto"
            )
    elif score_type == "value_map_by_row":
        for k in ["row_id_col", "response_col", "row_value_maps"]:
            if k not in scoring:
                raise ValueError(
                    f"[{index_name}] scoring: manca '{k}' (score_type='value_map_by_row')"
                )
        if not isinstance(scoring["row_value_maps"], dict) or len(scoring["row_value_maps"]) == 0:
            raise ValueError(f"[{index_name}] scoring.row_value_maps deve essere un dict non vuoto")

        row_id_col = scoring["row_id_col"]
        if row_id_col not in df_long.columns:
            raise ValueError(
                f"[{index_name}] colonna row_id_col mancante nel dataset: '{row_id_col}'"
            )
    else:
        raise ValueError(f"[{index_name}] score_type non supportato: {score_type}")

    # row_id_col obbligatoria solo se ci sono pesi
    weights_cfg = scoring.get("weights", {})
    if weights_cfg and not bool(scoring.get("selection_mode", False)):
        if "row_id_col" not in scoring:
            raise ValueError(f"[{index_name}] 'row_id_col' è obbligatoria quando usi 'weights'")
        row_id_col = scoring["row_id_col"]
        if row_id_col not in df_long.columns:
            raise ValueError(
                f"[{index_name}] colonna row_id_col mancante nel dataset: '{row_id_col}'"
            )

    aggregation_method = idx_cfg.get("aggregation", {}).get("method", "sum")
    if aggregation_method not in {"sum", "weighted_sum", "mean", "weighted_mean"}:
        raise ValueError(f"[{index_name}] aggregation.method non supportato: {aggregation_method}")

    normalization_method = idx_cfg.get("normalization", {}).get("method", "percent_of_max_sum")
    if normalization_method not in {
        "percent_of_max_sum",
        "percent_of_max_weighted_sum",
        "identity_0_100",
        "percent_of_max_total_sum",
    }:
        raise ValueError(
            f"[{index_name}] normalization.method non supportato: {normalization_method}"
        )


def compute_indices_from_config(
    df_long: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola più indici da dataset long in un unico run.

    Supporta:
      - score_type='value_map'   (es. 301, 309)
      - score_type='percentage'  (es. 386; risposta_valore già numerica)

    Returns
    -------
    result_df : una riga per entità con indice + raw/max per ogni indice
    detail_df : dettaglio di scoring per audit/debug
    """
    if (
        "indices" not in config
        or not isinstance(config["indices"], dict)
        or len(config["indices"]) == 0
    ):
        raise ValueError("Config non valida: 'indices' mancante o vuoto")

    all_results: list[pd.DataFrame] = []
    all_details: list[pd.DataFrame] = []

    for index_name, idx_cfg in config["indices"].items():
        _validate_index_config(df_long, index_name, idx_cfg)

        entity_keys = idx_cfg["entity_keys"]
        filters = idx_cfg.get("filters", {})
        scoring = idx_cfg["scoring"]
        aggregation = idx_cfg.get("aggregation", {"method": "sum"})
        normalization = idx_cfg.get("normalization", {"method": "percent_of_max_sum", "scale": 100})

        score_type = scoring.get("score_type", "value_map")
        response_col = scoring["response_col"]
        row_id_col = scoring.get("row_id_col")
        weights_cfg = scoring.get("weights", {})

        # 1) filtro righe per indice
        dfi = _apply_filters(df_long, filters).copy()

        if dfi.empty:
            empty_cols = entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]
            all_results.append(pd.DataFrame(columns=empty_cols))
            continue

        # 2) peso (comune a tutti i tipi di indice)
        selection_mode = bool(scoring.get("selection_mode", False))

        default_weight = float(scoring.get("default_weight", 1.0))
        if weights_cfg and not selection_mode:
            if not row_id_col:
                raise ValueError(f"[{index_name}] weights presenti ma row_id_col mancante")
            if dfi[row_id_col].isna().any():
                raise ValueError(f"{row_id_col} continene nan: controllare")
            dfi["__weight"] = dfi[row_id_col].map(weights_cfg).fillna(default_weight).astype(float)
        else:
            dfi["__weight"] = 1.0

        # ==========================================================
        # A) score_type = value_map (301 / 309)
        # ==========================================================
        if score_type == "value_map":
            # --- SELECTION MODE (multi-selezione: una riga per opzione cliccata) ---
            if selection_mode:
                none_val = scoring.get("none_value", None)
                weights_cfg = scoring.get("weights", {})
                default_w = float(scoring.get("default_weight", 0.0))

                if not weights_cfg:
                    raise ValueError(f"[{index_name}] selection_mode=true richiede scoring.weights")

                # max teorico = somma pesi (escludi none_val)
                max_total = sum(
                    float(w) for k, w in weights_cfg.items() if none_val is None or k != none_val
                )
                if max_total <= 0:
                    raise ValueError(
                        f"[{index_name}] selection_mode=true: somma pesi (escluso none_value) deve essere > 0"
                    )

                # deduplica opzioni selezionate
                dfi_unique = dfi.drop_duplicates(subset=entity_keys + [response_col]).copy()

                def _opt_weight(
                    v: Any,
                    none_val: Any = none_val,
                    weights_cfg: dict[str, Any] = weights_cfg,
                    default_w: float = default_w,
                ) -> float:
                    s = "" if pd.isna(v) else str(v)
                    if none_val is not None and s == none_val:
                        return 0.0
                    return float(weights_cfg.get(s, default_w))

                dfi_unique["__weighted_score"] = dfi_unique[response_col].apply(_opt_weight)

                grouped = (
                    dfi_unique.groupby(entity_keys, dropna=False)
                    .agg(raw_score=("__weighted_score", "sum"))
                    .reset_index()
                )

                scale = float(normalization.get("scale", 100))
                grouped[index_name] = np.where(
                    max_total > 0, (grouped["raw_score"] / max_total) * scale, np.nan
                ).clip(0, scale)

                grouped[f"{index_name}__raw"] = grouped["raw_score"]
                grouped[f"{index_name}__max"] = max_total

                all_results.append(
                    grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
                )

                detail_tmp = dfi_unique.copy()
                detail_tmp["__index_name"] = index_name
                all_details.append(
                    detail_tmp[["__index_name", *entity_keys, response_col, "__weighted_score"]]
                )

                continue
            # --- FINE SELECTION MODE ---

            default_score = scoring.get("default_score", np.nan)
            prepared_value_map = _prepare_value_map(scoring)
            max_row_score = _get_max_row_score(prepared_value_map, default_score)

            # score risposta (matching esatto)
            def _score_value(
                x: Any,
                prepared_value_map: dict[str, float] = prepared_value_map,
                default_score: float = default_score,
            ) -> float:
                return _map_response_score(x, prepared_value_map, default_score)

            dfi["__row_score"] = dfi[response_col].apply(_score_value)

            # score pesato (anche con peso=1)
            dfi["__weighted_score"] = dfi["__row_score"].astype(float) * dfi["__weight"]

            # massimo teorico pesato per riga
            dfi["__row_max_score"] = float(max_row_score)
            dfi["__weighted_row_max"] = dfi["__row_max_score"] * dfi["__weight"]

            # aggregazione (sum / weighted_sum)
            agg_method = aggregation.get("method", "sum")
            if agg_method not in {"sum", "weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo aggregazione non supportato per value_map: {agg_method}"
                )

            grouped = (
                dfi.groupby(entity_keys, dropna=False)
                .agg(
                    raw_score=("__weighted_score", "sum"),
                    max_score=("__weighted_row_max", "sum"),
                    n_rows=(response_col, "size"),
                    n_scored=("__row_score", lambda s: s.notna().sum()),
                )
                .reset_index()
            )

            # normalizzazione a 100 rispetto al massimo teorico
            norm_method = normalization.get("method", "percent_of_max_sum")
            scale = float(normalization.get("scale", 100))
            if norm_method not in {"percent_of_max_sum", "percent_of_max_weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo normalizzazione non supportato per value_map: {norm_method}"
                )

            grouped[index_name] = np.where(
                grouped["max_score"] > 0,
                (grouped["raw_score"] / grouped["max_score"]) * scale,
                np.nan,
            ).clip(0, scale)

            grouped = grouped.rename(
                columns={"raw_score": f"{index_name}__raw", "max_score": f"{index_name}__max"}
            )

            result_cols = entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]
            all_results.append(grouped[result_cols])

            # dettaglio
            detail_tmp = dfi.copy()
            detail_tmp["__index_name"] = index_name
            detail_cols = [
                "__index_name",
                *entity_keys,
                *(["domanda_id"] if "domanda_id" in detail_tmp.columns else []),
                *([row_id_col] if row_id_col else []),
                response_col,
                "__row_score",
                "__weight",
                "__weighted_score",
                "__weighted_row_max",
            ]
            all_details.append(detail_tmp[detail_cols])

        # ==========================================================
        # B) score_type = percentage (386)
        # risposta_valore contiene già un numero
        # ==========================================================
        elif score_type == "percentage":
            min_value = float(scoring.get("min_value", 0))
            max_value = float(scoring.get("max_value", 100))

            exclude_zeros = bool(scoring.get("exclude_zeros", False))

            # risposta_valore è già numerica -> conversione robusta
            dfi["__row_score"] = pd.to_numeric(dfi[response_col], errors="coerce")
            dfi["__row_score"] = (
                dfi["__row_score"].clip(lower=min_value, upper=max_value).astype(float)
            )

            if exclude_zeros:
                # escludo le righe con score 0 (considerate come "non risposte" o "non disponibili") dalla media
                dfi_valid = dfi[dfi["__row_score"].notna() & (dfi["__row_score"] > 0)].copy()
            else:
                # escludo le righe non numeriche dalla media
                dfi_valid = dfi[dfi["__row_score"].notna()].copy()

            agg_method = aggregation.get("method", "weighted_mean")
            if agg_method not in {"mean", "weighted_mean"}:
                raise ValueError(
                    f"[{index_name}] Metodo aggregazione non supportato per percentage: {agg_method}"
                )

            if dfi_valid.empty:
                grouped = dfi.groupby(entity_keys, dropna=False).size().reset_index(name="n_rows")
                grouped[index_name] = np.nan
                grouped[f"{index_name}__raw"] = np.nan
                grouped[f"{index_name}__max"] = max_value
                all_results.append(
                    grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
                )

                dfi["__weighted_score"] = np.nan
                dfi["__weighted_row_max"] = np.nan
                dfi["__index_name"] = index_name
                detail_cols = [
                    "__index_name",
                    *entity_keys,
                    *(["domanda_id"] if "domanda_id" in dfi.columns else []),
                    *([row_id_col] if row_id_col else []),
                    response_col,
                    "__row_score",
                    "__weight",
                    "__weighted_score",
                    "__weighted_row_max",
                ]
                all_details.append(dfi[detail_cols])
                continue

            if agg_method == "mean":
                grouped = (
                    dfi_valid.groupby(entity_keys, dropna=False)
                    .agg(raw_score=("__row_score", "mean"), n_rows=(response_col, "size"))
                    .reset_index()
                )
            else:  # weighted_mean
                dfi_valid["__weighted_score"] = dfi_valid["__row_score"] * dfi_valid["__weight"]
                grouped = (
                    dfi_valid.groupby(entity_keys, dropna=False)
                    .agg(
                        weighted_sum=("__weighted_score", "sum"),
                        weight_sum=("__weight", "sum"),
                        n_rows=(response_col, "size"),
                    )
                    .reset_index()
                )
                grouped["raw_score"] = np.where(
                    grouped["weight_sum"] > 0,
                    grouped["weighted_sum"] / grouped["weight_sum"],
                    np.nan,
                )

            # normalizzazione identity (scala naturale 0-100)
            norm_method = normalization.get("method", "identity_0_100")
            scale = float(normalization.get("scale", 100))
            if norm_method != "identity_0_100":
                raise ValueError(
                    f"[{index_name}] Metodo normalizzazione non supportato per percentage: {norm_method}"
                )

            if (min_value, max_value) == (0.0, 100.0):
                grouped[index_name] = grouped["raw_score"]
            else:
                grouped[index_name] = np.where(
                    max_value > min_value,
                    (grouped["raw_score"] - min_value) / (max_value - min_value) * scale,
                    np.nan,
                )

            grouped[index_name] = grouped[index_name].clip(0, scale)
            grouped[f"{index_name}__raw"] = grouped["raw_score"]
            grouped[f"{index_name}__max"] = max_value

            all_results.append(
                grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
            )

            # dettaglio
            dfi["__weighted_score"] = dfi["__row_score"] * dfi["__weight"]
            dfi["__weighted_row_max"] = max_value * dfi["__weight"]
            dfi["__index_name"] = index_name
            detail_cols = [
                "__index_name",
                *entity_keys,
                *(["domanda_id"] if "domanda_id" in dfi.columns else []),
                *([row_id_col] if row_id_col else []),
                response_col,
                "__row_score",
                "__weight",
                "__weighted_score",
                "__weighted_row_max",
            ]
            all_details.append(dfi[detail_cols])

        elif score_type == "percentage_binary":
            # 386: risposta_valore è numerica (percentuale), ma l'indice misura la disponibilità:
            # > threshold => 1, altrimenti 0
            threshold = float(scoring.get("threshold_gt", 0))

            min_value = float(scoring.get("min_value", 0))
            max_value = float(scoring.get("max_value", 100))

            # conversione robusta a numerico
            dfi["__pct"] = pd.to_numeric(dfi[response_col], errors="coerce").astype(float)
            dfi["__pct"] = dfi["__pct"].clip(lower=min_value, upper=max_value)

            # binarizzazione: disponibile se pct > soglia
            dfi["__row_score"] = np.where(
                dfi["__pct"].notna() & (dfi["__pct"] > threshold), 1.0, 0.0
            )

            # score pesato
            dfi["__weighted_score"] = dfi["__row_score"] * dfi["__weight"]

            # massimo teorico per riga = 1 (poi pesato)
            dfi["__weighted_row_max"] = 1.0 * dfi["__weight"]

            # aggregazione: somma pesata
            agg_method = aggregation.get("method", "weighted_sum")
            if agg_method not in {"sum", "weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo aggregazione non supportato per percentage_binary: {agg_method}"
                )

            grouped = (
                dfi.groupby(entity_keys, dropna=False)
                .agg(
                    raw_score=("__weighted_score", "sum"),
                    max_score=("__weighted_row_max", "sum"),
                    n_rows=(response_col, "size"),
                )
                .reset_index()
            )

            # normalizzazione a 100 sul massimo teorico
            norm_method = normalization.get("method", "percent_of_max_weighted_sum")
            scale = float(normalization.get("scale", 100))
            if norm_method not in {"percent_of_max_sum", "percent_of_max_weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo normalizzazione non supportato per percentage_binary: {norm_method}"
                )

            grouped[index_name] = np.where(
                grouped["max_score"] > 0,
                (grouped["raw_score"] / grouped["max_score"]) * scale,
                np.nan,
            ).clip(0, scale)

            grouped[f"{index_name}__raw"] = grouped["raw_score"]
            grouped[f"{index_name}__max"] = grouped["max_score"]

            all_results.append(
                grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
            )

            # dettaglio audit
            detail_tmp = dfi.copy()
            detail_tmp["__index_name"] = index_name
            detail_cols = [
                "__index_name",
                *entity_keys,
                *(["domanda_id"] if "domanda_id" in detail_tmp.columns else []),
                *([row_id_col] if row_id_col else []),
                response_col,
                "__pct",
                "__row_score",
                "__weight",
                "__weighted_score",
                "__weighted_row_max",
            ]
            all_details.append(detail_tmp[detail_cols])

        elif score_type == "multi_select_rows_weight_matrix":
            non_val = scoring["non_integrated_value"]
            matrix = scoring["integration_weight_matrix"]
            default_w = float(scoring.get("default_integration_weight", 0))

            # massimo teorico complessivo = somma dei pesi (Protocollo+Ragioneria+Anagrafe) per ogni categoria
            max_total = 0.0
            for _service_cat, wmap in matrix.items():
                max_total += float(sum(float(v) for v in wmap.values()))

            # Per Comune + categoria: prendo le scelte selezionate e calcolo score categoria
            grp_keys = entity_keys + [row_id_col]

            def _service_score(
                series: pd.Series,
                service_cat: str,
                non_val: Any = non_val,
                matrix: dict[str, Any] = matrix,
                default_w: float = default_w,
            ) -> float:
                vals = set(series.dropna().astype(str).tolist())
                if not vals:
                    return 0.0
                if vals == {non_val}:
                    return 0.0
                wmap = matrix.get(service_cat, {})
                score = 0.0
                for v in vals:
                    if v == non_val:
                        continue
                    score += float(wmap.get(v, default_w))
                return score

            # calcolo punteggio per categoria (service_df)
            # Nota: per applicare service_cat alla funzione, usiamo un groupby e poi applichiamo riga per riga
            service_rows = []
            for keys, sub in dfi.groupby(grp_keys, dropna=False):
                *ent_vals, service_cat = keys
                s = _service_score(sub[response_col], service_cat)
                row = dict(zip(entity_keys, ent_vals, strict=False))
                row[row_id_col] = service_cat
                row["__service_score"] = s
                service_rows.append(row)

            service_df = pd.DataFrame(service_rows)

            # aggregazione: somma sui servizi
            agg_method = aggregation.get("method", "sum")
            if agg_method not in {"sum", "weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo aggregazione non supportato per multi_select_rows_weight_matrix: {agg_method}"
                )

            grouped = (
                service_df.groupby(entity_keys, dropna=False)
                .agg(raw_score=("__service_score", "sum"))
                .reset_index()
            )

            # normalizzazione a 100 sul massimo teorico totale
            norm_method = normalization.get("method", "percent_of_max_total_sum")
            scale = float(normalization.get("scale", 100))
            if norm_method != "percent_of_max_total_sum":
                raise ValueError(
                    f"[{index_name}] Metodo normalizzazione non supportato: {norm_method}"
                )

            grouped[index_name] = np.where(
                max_total > 0, (grouped["raw_score"] / max_total) * scale, np.nan
            ).clip(0, scale)

            grouped[f"{index_name}__raw"] = grouped["raw_score"]
            grouped[f"{index_name}__max"] = max_total

            all_results.append(
                grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
            )

            # audit: punteggio per categoria
            detail_tmp = service_df.copy()
            detail_tmp["__index_name"] = index_name
            all_details.append(
                detail_tmp[["__index_name", *entity_keys, row_id_col, "__service_score"]]
            )

        elif score_type == "value_map_by_row":
            row_id_col = scoring["row_id_col"]
            response_col = scoring["response_col"]
            default_score = float(scoring.get("default_score", 0))
            row_value_maps = scoring["row_value_maps"]

            # pesi per modalità (opzionali)
            weights_cfg = scoring.get("weights", {})
            default_weight = float(scoring.get("default_weight", 1.0))
            if weights_cfg:
                dfi["__weight"] = (
                    dfi[row_id_col].map(weights_cfg).fillna(default_weight).astype(float)
                )
            else:
                dfi["__weight"] = 1.0

            # mapping per riga: usa la mappa specifica della modalità
            def _row_score(
                r: pd.Series,
                row_id_col: str = row_id_col,
                response_col: str = response_col,
                row_value_maps: dict[str, Any] = row_value_maps,
                default_score: float = default_score,
            ) -> float:
                mode = "" if pd.isna(r[row_id_col]) else str(r[row_id_col])
                resp = "" if pd.isna(r[response_col]) else str(r[response_col])
                vm = row_value_maps.get(mode, None)
                if vm is None:
                    return default_score
                return float(vm.get(resp, default_score))

            dfi["__row_score"] = dfi.apply(_row_score, axis=1)

            # max teorico per riga = max dei valori della row_value_map della modalità
            def _row_max(
                r: pd.Series,
                row_id_col: str = row_id_col,
                row_value_maps: dict[str, Any] = row_value_maps,
                default_score: float = default_score,
            ) -> float:
                mode = "" if pd.isna(r[row_id_col]) else str(r[row_id_col])
                vm = row_value_maps.get(mode, None)
                if not vm:
                    return default_score
                return float(max(float(v) for v in vm.values()))

            dfi["__row_max_score"] = dfi.apply(_row_max, axis=1)

            # pesati
            dfi["__weighted_score"] = dfi["__row_score"] * dfi["__weight"]
            dfi["__weighted_row_max"] = dfi["__row_max_score"] * dfi["__weight"]

            # aggregazione = somma pesata
            agg_method = aggregation.get("method", "weighted_sum")
            if agg_method not in {"sum", "weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo aggregazione non supportato per value_map_by_row: {agg_method}"
                )

            grouped = (
                dfi.groupby(entity_keys, dropna=False)
                .agg(raw_score=("__weighted_score", "sum"), max_score=("__weighted_row_max", "sum"))
                .reset_index()
            )

            # normalizzazione su max teorico
            norm_method = normalization.get("method", "percent_of_max_weighted_sum")
            scale = float(normalization.get("scale", 100))
            if norm_method not in {"percent_of_max_sum", "percent_of_max_weighted_sum"}:
                raise ValueError(
                    f"[{index_name}] Metodo normalizzazione non supportato per value_map_by_row: {norm_method}"
                )

            grouped[index_name] = np.where(
                grouped["max_score"] > 0,
                (grouped["raw_score"] / grouped["max_score"]) * scale,
                np.nan,
            ).clip(0, scale)

            grouped[f"{index_name}__raw"] = grouped["raw_score"]
            grouped[f"{index_name}__max"] = grouped["max_score"]

            all_results.append(
                grouped[entity_keys + [index_name, f"{index_name}__raw", f"{index_name}__max"]]
            )

            # dettaglio (audit)
            detail_tmp = dfi.copy()
            detail_tmp["__index_name"] = index_name
            all_details.append(
                detail_tmp[
                    [
                        "__index_name",
                        *entity_keys,
                        *(["domanda_id"] if "domanda_id" in detail_tmp.columns else []),
                        row_id_col,
                        response_col,
                        "__row_score",
                        "__row_max_score",
                        "__weight",
                        "__weighted_score",
                        "__weighted_row_max",
                    ]
                ]
            )
        else:
            raise ValueError(f"[{index_name}] score_type non supportato: {score_type}")

    # merge finale di tutti gli indici
    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    result_df = all_results[0]
    for nxt in all_results[1:]:
        join_keys = [
            c
            for c in result_df.columns
            if c in nxt.columns and not c.endswith("__raw") and not c.endswith("__max")
        ]
        result_df = result_df.merge(nxt, on=join_keys, how="outer")

    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    return result_df, detail_df


def _weighted_mean_from_components(
    df: pd.DataFrame,
    components: dict[str, float],
    out_col: str,
    normalize_weights: bool = True,
    missing_policy: str = "renormalize",
) -> pd.Series:
    cols = list(components.keys())
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[{out_col}] colonna componente mancante: '{c}'")

    W = np.array([float(components[c]) for c in cols], dtype=float)
    if (W < 0).any():
        raise ValueError(f"[{out_col}] pesi negativi non ammessi")
    if W.sum() == 0:
        raise ValueError(f"[{out_col}] somma pesi = 0")
    if normalize_weights:
        W = W / W.sum()

    X = df[cols].astype(float)

    if missing_policy == "zero":
        x = X.fillna(0.0).to_numpy()
        out = (x * W).sum(axis=1)

    elif missing_policy == "drop":
        mask_any_nan = X.isna().any(axis=1)
        x = X.to_numpy()
        val = (x * W).sum(axis=1)
        out = np.where(mask_any_nan, np.nan, val)

    elif missing_policy == "renormalize":
        x = X.to_numpy()
        m = ~np.isnan(x)
        w = W.reshape(1, -1) * m
        wsum = w.sum(axis=1)
        num = np.nansum(x * w, axis=1)
        out = np.where(wsum > 0, num / wsum, np.nan)

    else:
        raise ValueError(f"[{out_col}] missing_policy non supportata: {missing_policy}")

    return pd.Series(out, index=df.index).clip(0, 100)


def compute_level1_indices(
    df_long: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola gli indici di primo livello a partire dal dataset long.
    Legge config['level1']['indices'] e usa la funzione già sviluppata compute_indices_from_config.
    """
    if "level1" not in config or "indices" not in config["level1"]:
        raise ValueError("Config: manca 'level1.indices'")

    level1_cfg = {"indices": config["level1"]["indices"]}
    return compute_indices_from_config(df_long, level1_cfg)


def compute_level2_indices(level1_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Calcola gli indici di secondo livello a partire dagli indici di primo livello.
    Legge config['level2']['indices'].
    """
    if "level2" not in config or "indices" not in config["level2"]:
        raise ValueError("Config: manca 'level2.indices'")

    df = level1_df.copy()

    for name, cfg in config["level2"]["indices"].items():
        components = cfg["components"]
        normalize_weights = bool(cfg.get("normalize_weights", True))
        missing_policy = cfg.get("missing_policy", "renormalize")
        method = cfg.get("method", "weighted_mean")
        if method != "weighted_mean":
            raise ValueError(f"[level2 {name}] method non supportato: {method}")

        df[name] = _weighted_mean_from_components(
            df=df,
            components=components,
            out_col=name,
            normalize_weights=normalize_weights,
            missing_policy=missing_policy,
        )

    return df


def compute_level3_indices(level2_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Calcola gli indici di terzo livello (macro) a partire dagli indici di secondo livello.
    Legge config['level3']['indices'].
    """
    if "level3" not in config or "indices" not in config["level3"]:
        raise ValueError("Config: manca 'level3.indices'")

    df = level2_df.copy()

    for name, cfg in config["level3"]["indices"].items():
        components = cfg["components"]
        normalize_weights = bool(cfg.get("normalize_weights", True))
        missing_policy = cfg.get("missing_policy", "renormalize")
        method = cfg.get("method", "weighted_mean")
        if method != "weighted_mean":
            raise ValueError(f"[level3 {name}] method non supportato: {method}")

        df[name] = _weighted_mean_from_components(
            df=df,
            components=components,
            out_col=name,
            normalize_weights=normalize_weights,
            missing_policy=missing_policy,
        )

    return df


# # 1) Dataset
# df_long = pd.read_excel("/Users/gabbo/Code/Work/GitHub/MappaComuniDigitali2025/scripts/risposte_df_big12.xlsx")

# # 2) Config JSON da file (301 + 309 + 386)
# with open("index_config.json", "r", encoding="utf-8") as f:
#     config = json.load(f)

# # Livello 1
# level1_df, detail_df = compute_level1_indices(df_long, config)

# # Livello 2
# level2_df = compute_level2_indices(level1_df, config)

# # Livello 3
# final_df = compute_level3_indices(level2_df, config)

# index_names = list(config["indices"].keys())
# result_df["macro_indice_media"] = result_df[index_names].mean(axis=1)
# result_df[['quesito.username','macro_indice_media'] + index_names]

# result_df.loc[:,['quesito.username','indice_utilizzo_online_servizi_386']]
# df_long.loc[(df_long['domanda_id']==308) & (df_long['quesito.username']=='Palermo'),['quesito.username','risposta_voce','risposta_valore']]
# pd.to_numeric(df_long.loc[(df_long['domanda_id']==308) & (df_long['quesito.username']=='Palermo'),'risposta_valore'], errors='coerce').mean()

# # 4) Risultati
# print(result_df.head())
# print(detail_df.head())

# # opzionale export
# # result_df.to_excel("indici_output.xlsx", index=False)
# # detail_df.to_excel("indici_dettaglio.xlsx", index=False)

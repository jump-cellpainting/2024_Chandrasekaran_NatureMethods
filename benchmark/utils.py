import os
import glob
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import kneed
import scipy
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
from copairs.map import aggregate
from copairs.compute import cosine_indexed
import copairs.compute_np as backend
import itertools
from copairs.matching import Matcher, MatcherMultilabel, dict_to_dframe
from functools import partial
from copairs.map import (
    create_matcher,
    flatten_str_list,
    get_rel_k_list,
    build_rank_list_multi,
    build_rank_lists,
    results_to_dframe,
)


def load_data(exp, plate, filetype):
    """load all data from a single experiment into a single dataframe"""
    path = os.path.join("../profiles", f"{exp}", f"{plate}", f"*_{filetype}")
    files = glob.glob(path)
    df = pd.concat(pd.read_csv(_, low_memory=False) for _ in files)
    return df


def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]


def remove_negcon_and_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=["Metadata_broad_sample"])
        .reset_index(drop=True)
    )
    return df


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = df.dropna(subset=["Metadata_broad_sample"]).reset_index(drop=True)
    return df


def concat_profiles(df1, df2):
    """Concatenate dataframes"""
    if df1.shape[0] == 0:
        df1 = df2.copy()
    else:
        frames = [df1, df2]
        df1 = pd.concat(frames, ignore_index=True, join="inner")

    return df1


def create_replicability_df(
    replicability_map_df,
    replicability_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality,
    cell,
    timepoint,
):
    _replicability_map_df = replicability_map_df
    _replicability_fr_df = replicability_fr_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_replicability_fr_df)],
    )
    _replicability_fr_df = concat_profiles(_replicability_fr_df, _fr_df)

    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _replicability_map_df = concat_profiles(_replicability_map_df, _map_df)

    _replicability_fr_df["fr"] = _replicability_fr_df["fr"].astype(float)
    _replicability_map_df["mean_average_precision"] = _replicability_map_df[
        "mean_average_precision"
    ].astype(float)

    return _replicability_map_df, _replicability_fr_df


def create_matching_df(
    matching_map_df,
    matching_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality,
    cell,
    timepoint,
):
    _matching_map_df = matching_map_df
    _matching_fr_df = matching_fr_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f"{modality}_{_cell}_{_time}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality": _modality,
            "Cell": _cell,
            "time": _time,
            "timepoint": _timepoint,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_matching_fr_df)],
    )
    _matching_fr_df = concat_profiles(_matching_fr_df, _fr_df)

    _map_df["Description"] = f"{_description}"
    _map_df["Modality"] = f"{_modality}"
    _map_df["Cell"] = f"{_cell}"
    _map_df["time"] = f"{_time}"
    _map_df["timepoint"] = f"{_timepoint}"
    _matching_map_df = concat_profiles(_matching_map_df, _map_df)

    _matching_fr_df["fr"] = _matching_fr_df["fr"].astype(float)
    _matching_map_df["mean_average_precision"] = _matching_map_df[
        "mean_average_precision"
    ].astype(float)

    return _matching_map_df, _matching_fr_df


def create_gene_compound_matching_df(
    gene_compound_matching_map_df,
    gene_compound_matching_fr_df,
    result,
    pos_sameby,
    qthreshold,
    modality_1,
    modality_2,
    cell,
    timepoint1,
    timepoint2,
):
    _gene_compound_matching_map_df = gene_compound_matching_map_df
    _gene_compound_matching_fr_df = gene_compound_matching_fr_df

    _modality_1 = modality_1
    _modality_2 = modality_2
    _cell = cell
    _timepoint_1 = timepoint1
    _timepoint_2 = timepoint2
    _time_1 = time_point(_modality_1, _timepoint_1)
    _time_2 = time_point(_modality_2, _timepoint_2)

    _description = f"{_modality_1}_{cell}_{_time_1}-{_modality_2}_{cell}_{_time_2}"

    _map_df = calculate_mAP(result, pos_sameby, qthreshold)
    _fr = calculate_fraction_retrieved(_map_df)

    _fr_df = pd.DataFrame(
        {
            "Description": _description,
            "Modality1": f"{_modality_1}_{_time_1}",
            "Modality2": f"{_modality_2}_{_time_2}",
            "Cell": _cell,
            "fr": f"{_fr:.3f}",
        },
        index=[len(_gene_compound_matching_fr_df)],
    )
    _gene_compound_matching_fr_df = concat_profiles(
        _gene_compound_matching_fr_df, _fr_df
    )

    _map_df["Description"] = f"{_description}"
    _map_df["Modality1"] = f"{_modality_1}_{_time_1}"
    _map_df["Modality2"] = f"{_modality_2}_{_time_2}"
    _map_df["Cell"] = f"{_cell}"
    _gene_compound_matching_map_df = concat_profiles(
        _gene_compound_matching_map_df, _map_df
    )

    _gene_compound_matching_fr_df["fr"] = _gene_compound_matching_fr_df["fr"].astype(
        float
    )
    _gene_compound_matching_map_df[
        "mean_average_precision"
    ] = _gene_compound_matching_map_df["mean_average_precision"].astype(float)

    return _gene_compound_matching_map_df, _gene_compound_matching_fr_df


def consensus(profiles_df, group_by_feature):
    """
    Computes the median consensus profiles.
    Parameters:
    -----------
    profiles_df: pandas.DataFrame
        dataframe of profiles
    group_by_feature: str
        Name of the column
    Returns:
    -------
    pandas.DataFrame of the same shape as `plate`
    """

    metadata_df = get_metadata(profiles_df).drop_duplicates(subset=[group_by_feature])

    feature_cols = [group_by_feature] + get_featurecols(profiles_df)
    profiles_df = (
        profiles_df[feature_cols].groupby([group_by_feature]).median().reset_index()
    )

    profiles_df = metadata_df.merge(profiles_df, on=group_by_feature)

    return profiles_df

    def cleanup(self):
        """
        Remove rows and columns that are all NaN
        """
        keep = list((self.truth_matrix.sum(axis=1) > 0))
        self.corr["keep"] = keep
        self.map1["keep"] = keep
        self.truth_matrix["keep"] = keep

        self.corr = self.corr.loc[self.corr.keep].drop(columns=["keep"])
        self.map1 = self.map1.loc[self.map1.keep].drop(columns=["keep"])
        self.truth_matrix = self.truth_matrix.loc[self.truth_matrix.keep].drop(
            columns=["keep"]
        )


def time_point(modality, time_point):
    """
    Convert time point in hr to long or short time description
    Parameters:
    -----------
    modality: str
        perturbation modality
    time_point: int
        time point in hr
    Returns:
    -------
    str of time description
    """
    if modality == "compound":
        if time_point == 24:
            time = "short"
        else:
            time = "long"
    elif modality == "orf":
        if time_point == 48:
            time = "short"
        else:
            time = "long"
    else:
        if time_point == 96:
            time = "short"
        else:
            time = "long"

    return time


def convert_pvalue(pvalue):
    """
    Convert p value format
    Parameters:
    -----------
    pvalue: float
        p value
    Returns:
    -------
    str of p value
    """
    if pvalue < 0.05:
        pvalue = "<0.05"
    else:
        pvalue = f"{pvalue:.2f}"
    return pvalue


def add_lines_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add lines to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """
    y_value = ""
    if percentile == 5:
        y_value = "fifth_percentile"
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
    fig.add_shape(
        type="line",
        x0=locations["line"][color_order.index(df_row[color_column])]["x0"],
        y0=df_row[y_value],
        x1=locations["line"][color_order.index(df_row[color_column])]["x1"],
        y1=df_row[y_value],
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
        row=row,
        col=col,
    )
    return fig


def add_text_to_violin_plots(
    fig, df_row, locations, color_order, color_column, percentile, row, col
):
    """
    Add text to the violin plots
    Parameters
    ----------
    fig: plotly figure
    df_row: row of the dataframe with the data
    locations: x locations of the lines
    color_order: order of the colors in the violin plot
    color_column: column of the dataframe with the color information
    percentile: 5 or 95
    row: row of the figure
    col: column of the figure
    Returns
    -------
    fig: plotly figure
    """

    y_value = ""
    y_percent_value = ""
    y_offset = 0
    if percentile == 5:
        y_value = "fifth_percentile"
        y_percent_value = "percent_fifth_percentile"
        y_offset = -0.08
    elif percentile == 95:
        y_value = "ninetyfifth_percentile"
        y_percent_value = "percent_ninetyfifth_percentile"
        y_offset = 0.08
    fig.add_annotation(
        x=locations["text"][color_order.index(df_row[color_column])]["x"],
        y=df_row[y_value] + y_offset,
        text=f"{df_row[y_percent_value]*100:.02f}%",
        showarrow=False,
        font=dict(
            size=16,
        ),
        row=row,
        col=col,
    )
    return fig


def calculate_mAP(result, pos_sameby, threshold):
    """
    Calculate the mean average precision
    Parameters
    ----------
    result : pandas.DataFrame of average precision values output by copairs
    pos_sameby : str of columns that define positives
    threshold : float of threshold for q-value
    Returns
    -------
    agg_result : pandas.DataFrame of mAP values grouped by pos_sameby columns
    """
    agg_result = aggregate(result, pos_sameby, threshold=0.05).rename(
        columns={"average_precision": "mean_average_precision"}
    )
    return agg_result


def calculate_fraction_retrieved(agg_result):
    """
    Calculate the fraction of labels retrieved
    Parameters
    ----------
    agg_result : pandas.DataFrame of mAP values
    Returns
    -------
    fraction_retrieved : float of fraction positive
    """
    fraction_retrieved = len(agg_result.query("above_q_threshold==True")) / len(
        agg_result
    )
    return fraction_retrieved


def compute_similarities(pairs, feats, batch_size, anti_match=False):
    dist_df = pairs[["ix1", "ix2"]].drop_duplicates().copy()
    dist_df["dist"] = cosine_indexed(feats, dist_df.values, batch_size)
    if anti_match:
        dist_df["dist"] = np.abs(dist_df["dist"])
    return pairs.merge(dist_df, on=["ix1", "ix2"])

def run_pipeline(
    meta,
    feats,
    pos_sameby,
    pos_diffby,
    neg_sameby,
    neg_diffby,
    null_size,
    anti_match=False,
    multilabel_col=None,
    batch_size=20000,
) -> pd.DataFrame:
    # Critical!, otherwise the indexing wont work
    meta = meta.reset_index(drop=True).copy()

    matcher = create_matcher(
        meta, pos_sameby, pos_diffby, neg_sameby, neg_diffby, multilabel_col
    )

    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = dict_to_dframe(dict_pairs, pos_sameby)
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = set(itertools.chain.from_iterable(dict_pairs.values()))
    neg_pairs = pd.DataFrame(neg_pairs, columns=["ix1", "ix2"])
    pos_pairs = compute_similarities(pos_pairs, feats, batch_size, anti_match)
    neg_pairs = compute_similarities(neg_pairs, feats, batch_size, anti_match)
    if multilabel_col and multilabel_col in pos_sameby:
        rel_k_list = build_rank_list_multi(pos_pairs, neg_pairs, multilabel_col)
    else:
        rel_k_list = build_rank_lists(pos_pairs, neg_pairs)
    ap_scores = rel_k_list.apply(backend.compute_ap)
    ap_scores = np.concatenate(ap_scores.values)
    null_dists = backend.compute_null_dists(rel_k_list, null_size)
    p_values = backend.compute_p_values(null_dists, ap_scores, null_size)
    result = results_to_dframe(
        meta, rel_k_list.index, p_values, ap_scores, multilabel_col
    )
    return result

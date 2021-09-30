import os
import glob
import pandas as pd
import numpy as np
import random
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import kneed
import scipy


def load_data(exp, plate, filetype):
    """load all data from a single experiment into a single dataframe"""
    path = os.path.join('../profiles',
                        f'{exp}',
                        f'{plate}',
                        f'*_{filetype}')
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


def remove_negcon_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=['Metadata_broad_sample'])
        .reset_index(drop=True)
    )
    return df


def concat_profiles(df1, df2):
    """Concatenate dataframes"""
    if df1.shape[0] == 0:
        df1 = df2.copy()
    else:
        frames = [df1, df2]
        df1 = pd.concat(frames, ignore_index=True, join="inner")

    return df1


def percent_score(null_dist, corr_dist, how):
    """
    Calculates the Percent strong or percent recall scores
    :param null_dist: Null distribution
    :param corr_dist: Correlation distribution
    :param how: "left", "right" or "both" for using the 5th percentile, 95th percentile or both thresholds
    :return: proportion of correlation distribution beyond the threshold
    """
    if how == 'right':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        return np.mean(above_threshold.astype(float))*100, perc_95
    if how == 'left':
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return np.mean(below_threshold.astype(float))*100, perc_5
    if how == 'both':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return (np.mean(above_threshold.astype(float)) + np.mean(below_threshold.astype(float)))*100, perc_95, perc_5


def corr_between_replicates(df, group_by_feature):
    """
        Correlation between replicates
        Parameters:
        -----------
        df: pd.DataFrame
        group_by_feature: Feature name to group the data frame by
        Returns:
        --------
        list-like of correlation values
     """
    replicate_corr = []
    replicate_grouped = df.groupby(group_by_feature)
    for name, group in replicate_grouped:
        group_features = get_featuredata(group)
        corr = np.corrcoef(group_features)
        if len(group_features) == 1:  # If there is only one replicate on a plate
            replicate_corr.append(np.nan)
        else:
            np.fill_diagonal(corr, np.nan)
            replicate_corr.append(np.nanmedian(corr))  # median replicate correlation
    return replicate_corr


def corr_between_non_replicates(df, n_samples, n_replicates, metadata_compound_name):
    """
        Null distribution between random "replicates".
        Parameters:
        ------------
        df: pandas.DataFrame
        n_samples: int
        n_replicates: int
        metadata_compound_name: Compound name feature
        Returns:
        --------
        list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []
    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(df))], k=n_replicates)
        sample = df.loc[compounds].copy()
        if len(sample[metadata_compound_name].unique()) == n_replicates:
            sample_features = get_featuredata(sample)
            corr = np.corrcoef(sample_features)
            np.fill_diagonal(corr, np.nan)
            null_corr.append(np.nanmedian(corr))  # median replicate correlation
    return null_corr


def correlation_between_modalities(modality_1_df, modality_2_df, modality_1, modality_2, metadata_common, metadata_perturbation):
    """
    Compute the correlation between two different modalities.
    :param modality_1_df: Profiles of the first modality
    :param modality_2_df: Profiles of the second modality
    :param modality_1: feature that identifies perturbation pairs
    :param modality_2: perturbation name feature
    :param metadata_common: perturbation name feature
    :param metadata_perturbation: perturbation name feature
    :return: list-like of correlation values
    """
    list_common_perturbation_groups = list(np.intersect1d(list(modality_1_df[metadata_common]), list(modality_2_df[metadata_common])))

    merged_df = pd.concat([modality_1_df, modality_2_df], ignore_index=False, join='inner')

    modality_1_df = merged_df.query('Metadata_modality==@modality_1')
    modality_2_df = merged_df.query('Metadata_modality==@modality_2')

    corr_modalities = []

    for group in list_common_perturbation_groups:
        modality_1_perturbation_df = modality_1_df.loc[modality_1_df[metadata_common] == group]
        modality_2_perturbation_df = modality_2_df.loc[modality_2_df[metadata_common] == group]

        for sample_1 in modality_1_perturbation_df[metadata_perturbation].unique():
            for sample_2 in modality_2_perturbation_df[metadata_perturbation].unique():
                modality_1_perturbation_sample_df = modality_1_perturbation_df.loc[modality_1_perturbation_df[metadata_perturbation] == sample_1]
                modality_2_perturbation_sample_df = modality_2_perturbation_df.loc[modality_2_perturbation_df[metadata_perturbation] == sample_2]

                modality_1_perturbation_profiles = get_featuredata(modality_1_perturbation_sample_df)
                modality_2_perturbation_profiles = get_featuredata(modality_2_perturbation_sample_df)

                corr = np.corrcoef(modality_1_perturbation_profiles, modality_2_perturbation_profiles)
                corr = corr[0:len(modality_1_perturbation_profiles), len(modality_1_perturbation_profiles):]
                corr_modalities.append(np.nanmedian(corr))  # median replicate correlation

    return corr_modalities


def null_correlation_between_modalities(modality_1_df, modality_2_df, modality_1, modality_2, metadata_common, metadata_perturbation, n_samples):
    """
    Compute the correlation between two different modalities.
    :param modality_1_df: Profiles of the first modality
    :param modality_2_df: Profiles of the second modality
    :param modality_1: "Compound", "ORF" or "CRISPR"
    :param modality_2: "Compound", "ORF" or "CRISPR"
    :param metadata_common: feature that identifies perturbation pairs
    :param metadata_perturbation: perturbation name feature
    :param n_samples: int
    :return:
    """
    list_common_perturbation_groups = list(np.intersect1d(list(modality_1_df[metadata_common]), list(modality_2_df[metadata_common])))

    merged_df = pd.concat([modality_1_df, modality_2_df], ignore_index=False, join='inner')

    modality_1_df = merged_df.query('Metadata_modality==@modality_1')
    modality_2_df = merged_df.query('Metadata_modality==@modality_2')

    null_modalities = []

    count = 0

    while count < n_samples:
        perturbations = random.choices(list_common_perturbation_groups, k=2)
        modality_1_perturbation_df = modality_1_df.loc[modality_1_df[metadata_common] == perturbations[0]]
        modality_2_perturbation_df = modality_2_df.loc[modality_2_df[metadata_common] == perturbations[1]]

        for sample_1 in modality_1_perturbation_df[metadata_perturbation].unique():
            for sample_2 in modality_2_perturbation_df[metadata_perturbation].unique():
                modality_1_perturbation_sample_df = modality_1_perturbation_df.loc[modality_1_perturbation_df[metadata_perturbation] == sample_1]
                modality_2_perturbation_sample_df = modality_2_perturbation_df.loc[modality_2_perturbation_df[metadata_perturbation] == sample_2]

                modality_1_perturbation_profiles = get_featuredata(modality_1_perturbation_sample_df)
                modality_2_perturbation_profiles = get_featuredata(modality_2_perturbation_sample_df)

                corr = np.corrcoef(modality_1_perturbation_profiles, modality_2_perturbation_profiles)
                corr = corr[0:len(modality_1_perturbation_profiles), len(modality_1_perturbation_profiles):]
                null_modalities.append(np.nanmedian(corr))  # median replicate correlation
        count += 1

    return null_modalities


class ZCA_corr(BaseEstimator, TransformerMixin):
    def __init__(self, copy=False):
        self.copy = copy

    def estimate_regularization(self, eigenvalue):
        x = [_ for _ in range(len(eigenvalue))]
        kneedle = kneed.KneeLocator(x, eigenvalue, S=1.0, curve='convex', direction='decreasing')
        reg = eigenvalue[kneedle.elbow]/10.0
        return reg # The complex part of the eigenvalue is ignored

    def fit(self, X, y=None):
        """
        Compute the mean, sphering and desphering matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, sphering and desphering
            matrices.
        """
        X = check_array(X, accept_sparse=False, copy=self.copy, ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0] - 1)
        V = np.diag(cov)
        df = pd.DataFrame(X_)
        corr = np.nan_to_num(df.corr()) # replacing nan with 0 and inf with large values
        G, T, _ = scipy.linalg.svd(corr)
        regularization = self.estimate_regularization(T.real)
        t = np.sqrt(T.clip(regularization))
        t_inv = np.diag(1.0 / t)
        v_inv = np.diag(1.0/np.sqrt(V.clip(1e-3)))
        self.sphere_ = np.dot(np.dot(np.dot(G, t_inv), G.T), v_inv)
        return self

    def transform(self, X, y=None, copy=None):
        """
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to sphere along the features axis.
        """
        check_is_fitted(self, "mean_")
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.sphere_.T)


def sphere_plate_zca_corr(plate):
    """
    sphere each plate to the DMSO negative control values
    Parameters:
    -----------
    plate: pandas.DataFrame
        dataframe of a single plate's featuredata and metadata
    Returns:
    -------
    pandas.DataFrame of the same shape as `plate`
    """
    # sphere featuredata to DMSO sphering matrix
    spherizer = ZCA_corr()
    dmso_df = plate.loc[plate.Metadata_control_type=="negcon"]
    dmso_vals = get_featuredata(dmso_df).to_numpy()
    all_vals = get_featuredata(plate).to_numpy()
    spherizer.fit(dmso_vals)
    sphered_vals = spherizer.transform(all_vals)
    # concat with metadata columns
    feature_df = pd.DataFrame(
        sphered_vals, columns=get_featurecols(plate), index=plate.index
    )
    metadata = get_metadata(plate)
    combined = pd.concat([metadata, feature_df], axis=1)
    assert combined.shape == plate.shape
    return combined
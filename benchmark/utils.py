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


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = (
        df.dropna(subset=['Metadata_broad_sample'])
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


def create_replicability_df(replicability_ap_df, replicability_map_df, precision, modality, cell, timepoint):
    _replicability_ap_df = replicability_ap_df
    _replicability_map_df = replicability_map_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f'{modality}_{_cell}_{_time}'

    _map_df = pd.DataFrame({'Description': _description,
                            'Modality': _modality,
                            'Cell': _cell,
                            'time': _time,
                            'timepoint': _timepoint,
                            'mAP': f'{precision.map:.3f}'}, index=[len(_replicability_map_df)])
    _replicability_map_df = concat_profiles(_replicability_map_df, _map_df)

    _ap_df = precision.ap_group.copy()
    _ap_df['Description'] = f'{_description}'
    _ap_df['Modality'] = f'{_modality}'
    _ap_df['Cell'] = f'{_cell}'
    _ap_df['time'] = f'{_time}'
    _ap_df['timepoint'] = f'{_timepoint}'
    _replicability_ap_df = concat_profiles(_replicability_ap_df, _ap_df)

    _replicability_map_df['mAP'] = _replicability_map_df['mAP'].astype(float)
    _replicability_ap_df['ap'] = _replicability_ap_df['ap'].astype(float)

    return _replicability_ap_df, _replicability_map_df


def create_matching_df(matching_ap_df, matching_map_df, precision, modality, cell, timepoint):
    _matching_ap_df = matching_ap_df
    _matching_map_df = matching_map_df

    _modality = modality
    _cell = cell
    _timepoint = timepoint
    _time = time_point(_modality, _timepoint)

    _description = f'{modality}_{_cell}_{_time}'

    _map_df = pd.DataFrame({'Description': _description,
                            'Modality': _modality,
                            'Cell': _cell,
                            'time': _time,
                            'timepoint': _timepoint,
                            'mAP': f'{precision.map:.3f}'}, index=[len(_matching_map_df)])
    _matching_map_df = concat_profiles(_matching_map_df, _map_df)

    _ap_df = precision.ap_group.copy()
    _ap_df['Description'] = f'{_description}'
    _ap_df['Modality'] = f'{_modality}'
    _ap_df['Cell'] = f'{_cell}'
    _ap_df['time'] = f'{_time}'
    _ap_df['timepoint'] = f'{_timepoint}'
    _matching_ap_df = concat_profiles(_matching_ap_df, _ap_df)

    _matching_map_df['mAP'] = _matching_map_df['mAP'].astype(float)
    _matching_ap_df['ap'] = _matching_ap_df['ap'].astype(float)

    return _matching_ap_df, _matching_map_df


def create_gene_compound_matching_df(gene_compound_matching_ap_df, gene_compound_matching_map_df, precision, modality_1, modality_2, cell, timepoint1, timepoint2):
    _gene_compound_matching_ap_df = gene_compound_matching_ap_df
    _gene_compound_matching_map_df = gene_compound_matching_map_df

    _modality_1 = modality_1
    _modality_2 = modality_2
    _cell = cell
    _timepoint_1 = timepoint1
    _timepoint_2 = timepoint2
    _time_1 = time_point(_modality_1, _timepoint_1)
    _time_2 = time_point(_modality_2, _timepoint_2)

    _description = f'{_modality_1}_{cell}_{_time_1}-{_modality_2}_{cell}_{_time_2}'

    _map_df = pd.DataFrame({'Description': _description,
                            'Modality1': f'{_modality_1}_{_time_1}',
                            'Modality2': f'{_modality_2}_{_time_2}',
                            'Cell': _cell,
                            'mAP': f'{precision.map:.3f}'}, index=[len(_gene_compound_matching_map_df)])
    _gene_compound_matching_map_df = concat_profiles(_gene_compound_matching_map_df, _map_df)

    _ap_df = precision.ap_group.copy()
    _ap_df['Description'] = f'{_description}'
    _ap_df['Modality1'] = f'{_modality_1}_{_time_1}'
    _ap_df['Modality2'] = f'{_modality_2}_{_time_2}'
    _ap_df['Cell'] = f'{_cell}'
    _gene_compound_matching_ap_df = concat_profiles(_gene_compound_matching_ap_df, _ap_df)

    _gene_compound_matching_map_df['mAP'] = _gene_compound_matching_map_df['mAP'].astype(float)
    _gene_compound_matching_ap_df['ap'] = _gene_compound_matching_ap_df['ap'].astype(float)

    return _gene_compound_matching_ap_df, _gene_compound_matching_map_df


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

    metadata_df = (
        get_metadata(profiles_df)
            .drop_duplicates(subset=[group_by_feature])
    )

    feature_cols = [group_by_feature] + get_featurecols(profiles_df)
    profiles_df = profiles_df[feature_cols].groupby([group_by_feature]).median().reset_index()

    profiles_df = (
        metadata_df.merge(profiles_df, on=group_by_feature)
    )

    return profiles_df


class PrecisionScores(object):
    """
    Calculate the precision scores for information retrieval.
    """
    def __init__(self, profile1, profile2, group_by_feature, mode, identify_perturbation_feature, within=False, anti_correlation=False, against_negcon=False):
        """
        Parameters:
        -----------
        profile1: pandas.DataFrame
            dataframe of profiles
        profile2: pandas.DataFrame
            dataframe of profiles
        group_by_feature: str
            Name of the feature to group by
        mode: str
            Whether compute replicability or matching
        identity_perturbation_feature: str
            Name of the feature that identifies perturbations
        within: bool, default: False
            Whether profile1 and profile2 are the same dataframe or not.
        anti_correlation: book, default: False
            Whether both anti-correlation and correlation are used in the calculation
        against_negcon: bool, default:  False
            Whether to calculate precision scores by challenging negcon.
        """
        self.sample_id_feature = 'Metadata_sample_id'
        self.control_type_feature = 'Metadata_control_type'
        self.feature = group_by_feature
        self.mode = mode
        self.identify_perturbation_feature = identify_perturbation_feature
        self.within = within
        self.anti_correlation = anti_correlation
        self.against_negcon = against_negcon

        self.profile1 = self.process_profiles(profile1)
        self.profile2 = self.process_profiles(profile2)

        if self.mode == "replicability":
            self.map1 = self.profile1[[self.feature, self.sample_id_feature, self.control_type_feature]].copy()
            self.map2 = self.profile2[[self.feature, self.sample_id_feature, self.control_type_feature]].copy()
        elif self.mode == "matching":
            self.map1 = self.profile1[[self.identify_perturbation_feature, self.feature, self.sample_id_feature, self.control_type_feature]].copy()
            self.map2 = self.profile2[[self.identify_perturbation_feature, self.feature, self.sample_id_feature, self.control_type_feature]].copy()

        self.corr = self.compute_correlation()
        self.truth_matrix = self.create_truth_matrix()
        self.cleanup()

        self.ap_sample = self.calculate_average_precision_per_sample()
        self.ap_group = self.calculate_average_precision_score_per_group(self.ap_sample)
        self.map = self.calculate_mean_average_precision_score(self.ap_group)

    def process_profiles(self, _profile):
        """
        Add sample id column to profiles.
        Parameters:
        -----------
        _profile: pandas.DataFrame
            dataframe of profiles
        Returns:
        -------
        pandas.DataFrame which includes the sample id column
        """

        _profile = _profile.reset_index(drop=True)
        _feature_df = get_featuredata(_profile)
        if self.mode == "replicability":
            _metadata_df = _profile[[self.feature, self.control_type_feature]]
        elif self.mode == "matching":
            _metadata_df = _profile[[self.identify_perturbation_feature, self.feature, self.control_type_feature]]
        width = int(np.log10(len(_profile)))+1
        _perturbation_id_df = pd.DataFrame({self.sample_id_feature: [f'sample_{i:0{width}}' for i in range(len(_metadata_df))]})
        _metadata_df = pd.concat([_metadata_df, _perturbation_id_df], axis=1)
        _profile = pd.concat([_metadata_df, _feature_df], axis=1)
        return _profile

    def compute_correlation(self):
        """
        Compute correlation.
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values.
        """

        _profile1 = get_featuredata(self.profile1)
        _profile2 = get_featuredata(self.profile2)
        _sample_names_1 = list(self.profile1[self.sample_id_feature])
        _sample_names_2 = list(self.profile2[self.sample_id_feature])
        _corr = cosine_similarity(_profile1, _profile2)
        if self.anti_correlation:
            _corr = np.abs(_corr)
        _corr_df = pd.DataFrame(_corr, columns=_sample_names_2, index=_sample_names_1)
        _corr_df = self.process_self_correlation(_corr_df)
        _corr_df = self.process_negcon(_corr_df)
        return _corr_df

    def create_truth_matrix(self):
        """
        Compute truth matrix.
        Returns:
        -------
        pandas.DataFrame of binary truth values.
        """

        _truth_matrix = self.corr.unstack().reset_index()
        _truth_matrix = _truth_matrix.merge(self.map2, left_on='level_0', right_on=self.sample_id_feature, how='left').drop([self.sample_id_feature,0], axis=1)
        _truth_matrix = _truth_matrix.merge(self.map1, left_on='level_1', right_on=self.sample_id_feature, how='left').drop([self.sample_id_feature], axis=1)
        _truth_matrix['value'] = [len(np.intersect1d(x[0].split('|'), x[1].split('|'))) > 0 for x in zip(_truth_matrix[f'{self.feature}_x'], _truth_matrix[f'{self.feature}_y'])]
        if self.within and self.mode == "replicability":
            _truth_matrix['value'] = np.where(_truth_matrix['level_0'] == _truth_matrix['level_1'], 0, _truth_matrix['value'])
        elif self.within and self.mode == "matching":
            _truth_matrix['value'] = np.where(_truth_matrix[f'{self.identify_perturbation_feature}_x'] == _truth_matrix[f'{self.identify_perturbation_feature}_y'], 0,
                                              _truth_matrix['value'])

        _truth_matrix = _truth_matrix.pivot('level_1', 'level_0', 'value').reset_index().set_index('level_1')
        _truth_matrix.index.name = None
        _truth_matrix = _truth_matrix.rename_axis(None, axis=1)
        return _truth_matrix

    def calculate_average_precision_per_sample(self):
        """
        Compute average precision score per sample.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """

        _score = []
        for _sample in self.corr.index:
            _y_true, _y_pred = self.filter_nan(self.truth_matrix.loc[_sample].values, self.corr.loc[_sample].values)
            _score.append(average_precision_score(_y_true, _y_pred))

        _ap_sample_df = self.map1.copy()
        _ap_sample_df['ap'] = _score
        if self.against_negcon:
            _ap_sample_df = _ap_sample_df.query(f'{self.control_type_feature}!="negcon"').drop(columns=[self.control_type_feature]).reset_index(drop=True)
        else:
            _ap_sample_df = _ap_sample_df.drop(columns=[self.control_type_feature]).reset_index(drop=True)

        # compute corrected average precision
        random_baseline_ap = _y_true.sum()/len(_y_true)
        _ap_sample_df['ap'] -= random_baseline_ap

        return _ap_sample_df


    def calculate_average_precision_score_per_group(self, precision_score):
        """
        Compute average precision score per sample group.
        Returns:
        -------
        pandas.DataFrame of average precision values.
        """

        _precision_group_df = precision_score.groupby(self.feature).apply(lambda x: np.mean(x)).reset_index()
        return _precision_group_df

    @staticmethod
    def calculate_mean_average_precision_score(precision_score):
        """
        Compute mean average precision score.
        Returns:
        -------
        mean average precision score.
        """

        return precision_score.mean().values[0]


    def process_negcon(self, _corr_df):
        """
        Keep or remove negcon
        Parameters:
        -----------
        _corr_df: pandas.DataFrame
            pairwise correlation dataframe
        Returns:
        -------
        pandas.DataFrame of pairwise correlation values
        """
        _corr_df = _corr_df.unstack().reset_index()
        _corr_df['filter'] = 1
        _corr_df = _corr_df.merge(self.map2, left_on='level_0', right_on=self.sample_id_feature, how='left').drop([self.sample_id_feature], axis=1)
        _corr_df = _corr_df.merge(self.map1, left_on='level_1', right_on=self.sample_id_feature, how='left').drop([self.sample_id_feature], axis=1)

        if self.against_negcon:
            _corr_df['filter'] = np.where(_corr_df[f'{self.feature}_x'] != _corr_df[f'{self.feature}_y'], 0, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_x'] == "negcon", 1, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_y'] == "negcon", 0, _corr_df['filter'])
        else:
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_x'] == "negcon", 0, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_y'] == "negcon", 0, _corr_df['filter'])

        _corr_df = _corr_df.query('filter==1').reset_index(drop=True)

        if self.mode == "replicability":
            self.map1 = (
                _corr_df[['level_1', f'{self.feature}_y', f'{self.control_type_feature}_y']].copy()
                .rename(columns={'level_1': self.sample_id_feature,
                                 f'{self.feature}_y': self.feature,
                                 f'{self.control_type_feature}_y': self.control_type_feature})
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[['level_0', f'{self.feature}_x', f'{self.control_type_feature}_x']].copy()
                .rename(columns={'level_0': self.sample_id_feature,
                                 f'{self.feature}_x': self.feature,
                                 f'{self.control_type_feature}_x': self.control_type_feature})
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
        elif self.mode == "matching":
            self.map1 = (
                _corr_df[['level_1', f'{self.identify_perturbation_feature}_y', f'{self.feature}_y', f'{self.control_type_feature}_y']].copy()
                .rename(columns={'level_1': self.sample_id_feature,
                                 f'{self.feature}_y': self.feature,
                                 f'{self.control_type_feature}_y': self.control_type_feature,
                                 f'{self.identify_perturbation_feature}_y': f'{self.identify_perturbation_feature}'})
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )
            self.map2 = (
                _corr_df[['level_0', f'{self.identify_perturbation_feature}_x', f'{self.feature}_x', f'{self.control_type_feature}_x']].copy()
                .rename(columns={'level_0': self.sample_id_feature,
                                 f'{self.feature}_x': self.feature,
                                 f'{self.control_type_feature}_x': self.control_type_feature,
                                 f'{self.identify_perturbation_feature}_x': f'{self.identify_perturbation_feature}'})
                .drop_duplicates()
                .sort_values(by=self.sample_id_feature)
                .reset_index(drop=True)
            )

        _corr_df = _corr_df.pivot('level_1', 'level_0', 0).reset_index().set_index('level_1')
        _corr_df.index.name = None
        _corr_df = _corr_df.rename_axis(None, axis=1)
        return _corr_df

    @staticmethod
    def filter_nan(_y_true, _y_pred):
        arg = np.argwhere(~np.isnan(_y_pred))
        return _y_true[arg].flatten(), _y_pred[arg].flatten()

    def process_self_correlation(self, corr):
        _corr = (
            corr.unstack().reset_index()
            .rename(columns={0:"corr"})
        )
        _corr = _corr.merge(self.map2, left_on='level_0', right_on=self.sample_id_feature,
                                            how='left').drop([self.sample_id_feature], axis=1)
        _corr = _corr.merge(self.map1, left_on='level_1', right_on=self.sample_id_feature,
                                            how='left').drop([self.sample_id_feature], axis=1)
        if self.within and self.mode == "replicability":
            _corr["corr"] = np.where(_corr['level_0'] == _corr['level_1'], np.nan, _corr["corr"])
        elif self.within and self.mode == "matching":
            _corr["corr"] = np.where(_corr[f'{self.identify_perturbation_feature}_x'] == _corr[f'{self.identify_perturbation_feature}_y'], np.nan, _corr["corr"])

        _corr = _corr.pivot('level_1', 'level_0', 'corr').reset_index().set_index('level_1')
        _corr.index.name = None
        _corr = _corr.rename_axis(None, axis=1)

        return _corr

    def cleanup(self):
        keep = list((self.truth_matrix.sum(axis=1)>0))
        self.corr['keep'] = keep
        self.map1['keep'] = keep
        self.truth_matrix['keep'] = keep

        self.corr = self.corr.loc[self.corr.keep].drop(columns=['keep'])
        self.map1 = self.map1.loc[self.map1.keep].drop(columns=['keep'])
        self.truth_matrix = self.truth_matrix.loc[self.truth_matrix.keep].drop(columns=['keep'])


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
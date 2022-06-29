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
    def __init__(self, profile1, profile2, group_by_feature, within=False, rank=False, anti_correlation=False, challenge_negcon=False):
        """
        Parameters:
        -----------
        profile1: pandas.DataFrame
            dataframe of profiles
        profile2: pandas.DataFrame
            dataframe of profiles
        group_by_feature: str
            Name of the column
        within: bool, default: False
            Whether profile1 and profile2 are the same dataframe or not.
        rank: bool, default: False
            Whether to use rank of the correlation values or not.
        anti_correlation: book, default: False
            Whether both anti-correlation and correlation are used in the calculation
        challenge_negcon: bool, default:  False
            Whether to calculate precision scores by challenging negcon.
        """
        self.sample_feature = 'Metadata_sample_id'
        self.control_type_feature = 'Metadata_control_type'
        self.feature = group_by_feature
        self.within = within
        self.rank = rank
        self.anti_correlation = anti_correlation
        self.challenge_negcon = challenge_negcon

        self.profile1 = self.process_profiles(profile1)
        self.profile2 = self.process_profiles(profile2)

        self.map1 = self.profile1[[self.feature, self.sample_feature, self.control_type_feature]].copy()
        self.map2 = self.profile2[[self.feature, self.sample_feature, self.control_type_feature]].copy()

        self.corr = self.compute_correlation()
        self.truth_matrix = self.create_truth_matrix()

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
        _metadata_df = _profile[[self.feature, self.control_type_feature]]
        width = int(np.log10(len(_profile)))+1
        _perturbation_id_df = pd.DataFrame({self.sample_feature: [f'sample_{i:0{width}}' for i in range(len(_metadata_df))]})
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
        _sample_names_1 = list(self.profile1[self.sample_feature])
        _sample_names_2 = list(self.profile2[self.sample_feature])
        _corr = np.corrcoef(_profile1, _profile2)
        _corr = _corr[0:len(_sample_names_1), len(_sample_names_1):]
        if self.anti_correlation:
            _corr = np.abs(_corr)
        if self.within:
            np.fill_diagonal(_corr, 0)
        _corr_df = pd.DataFrame(_corr, columns=_sample_names_2, index=_sample_names_1)
        if self.rank:
            _corr_df = _corr_df.rank(1, method="first")
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
        _truth_matrix = _truth_matrix.merge(self.map2, left_on='level_0', right_on=self.sample_feature, how='left').drop([self.sample_feature,0], axis=1)
        _truth_matrix = _truth_matrix.merge(self.map1, left_on='level_1', right_on=self.sample_feature, how='left').drop([self.sample_feature], axis=1)
        _truth_matrix['value'] = np.where(_truth_matrix[f'{self.feature}_x'] == _truth_matrix[f'{self.feature}_y'], 1, 0)
        if self.within:
            _truth_matrix['value'] = np.where(_truth_matrix['level_0'] == _truth_matrix['level_1'], 0, _truth_matrix['value'])
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
        if self.challenge_negcon:
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
        _corr_df = _corr_df.merge(self.map2, left_on='level_0', right_on=self.sample_feature, how='left').drop([self.sample_feature], axis=1)
        _corr_df = _corr_df.merge(self.map1, left_on='level_1', right_on=self.sample_feature, how='left').drop([self.sample_feature], axis=1)

        if self.challenge_negcon:
            _corr_df['filter'] = np.where(_corr_df[f'{self.feature}_x'] != _corr_df[f'{self.feature}_y'], 0, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_x'] == "negcon", 1, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_y'] == "negcon", 0, _corr_df['filter'])
        else:
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_x'] == "negcon", 0, _corr_df['filter'])
            _corr_df['filter'] = np.where(_corr_df[f'{self.control_type_feature}_y'] == "negcon", 0, _corr_df['filter'])

        _corr_df = _corr_df.query('filter==1').reset_index(drop=True)

        self.map1 = (
            _corr_df[['level_1', f'{self.feature}_y', f'{self.control_type_feature}_y']].copy()
            .rename(columns={'level_1': self.sample_feature, f'{self.feature}_y': self.feature, f'{self.control_type_feature}_y':self.control_type_feature})
            .drop_duplicates()
            .sort_values(by=self.sample_feature)
            .reset_index(drop=True)
        )
        self.map2 = (
            _corr_df[['level_0', f'{self.feature}_x', f'{self.control_type_feature}_x']].copy()
            .rename(columns={'level_0': self.sample_feature, f'{self.feature}_x': self.feature, f'{self.control_type_feature}_y':self.control_type_feature})
            .drop_duplicates()
            .sort_values(by=self.sample_feature)
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


def time_point(modality, time_point):
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
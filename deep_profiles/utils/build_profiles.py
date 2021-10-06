from glob import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

exp_name = 'efn_pretrained'
plates = glob(f'outputs/{exp_name}/features/*')
plates = [p.split('/')[-1] for p in plates]
plates.sort()
NUM_FEATS = 6400

def aggregate_features(plate):
    metadata = []
    npzfiles = list(glob(f'outputs/{exp_name}/features/{plate}/*npz'))
    features = np.empty((len(npzfiles), NUM_FEATS))
    nan_index = []
    for i, pklpath in enumerate(tqdm(npzfiles, leave=False)):
        with np.load(pklpath, allow_pickle=True) as vals:
            if np.any(np.isnan(vals['features'])):
                nan_index.append(i)
            num_cells = vals['locations'].shape[0]
            metadata.append(vals['metadata'].item())
            metadata[-1]['num_cells'] = num_cells
            features[i] = vals['features'].mean(axis=0)
    metadata = pd.DataFrame(metadata)
    features = pd.DataFrame(features)
    dframe = pd.concat([metadata, features], axis=1)
    # Ignore nan
    dframe.drop(nan_index, inplace=True)
    dframe.columns = [str(c) for c in dframe.columns]
    groupby = dframe.groupby(['Metadata_Plate', 'Metadata_Well'])
    features = groupby[[str(i) for i in range(NUM_FEATS)]].mean()
    return features.reset_index()


data = process_map(aggregate_features, plates)
df = pd.concat(data)
df.to_parquet(f'{exp_name}.parquet', engine='pyarrow')

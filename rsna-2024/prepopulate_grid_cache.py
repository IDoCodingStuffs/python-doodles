import math
from tqdm import tqdm
import argparse

from rsna_dataloader import *

DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)

df = TRAINING_DATA.dropna()
# This drops any subjects with nans

filtered_df = pd.DataFrame(columns=df.columns)
for series_desc in CONDITIONS.keys():
    subset = df[df['series_description'] == series_desc]
    if series_desc == "Sagittal T2/STIR":
        subset = subset[subset.groupby(["study_id"]).transform('size') == 5]
    else:
        subset = subset[subset.groupby(["study_id"]).transform('size') == 10]
    filtered_df = pd.concat([filtered_df, subset])

filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform('size') == 25]

series_descs = {e[0]: e[1] for e in df[["series_id", "series_description"]].drop_duplicates().values}


if __name__ == "__main__":
    dirs = glob.glob("./data/rsna-2024-lumbar-spine-degenerative-classification/train_images/*/")
    dirs = sorted(dirs)

    parser = argparse.ArgumentParser(
        prog='GridCachePrepop',
    )
    parser.add_argument('index', type=int)
    parser.add_argument('count', type=int)
    args = parser.parse_args()

    slice_size = math.ceil(len(dirs)/args.count)
    dirslice = dirs[slice_size*args.index:slice_size*(args.index + 1)]

    for dir in tqdm(dirslice):
        read_study_as_voxel_grid(dir, series_type_dict=series_descs)

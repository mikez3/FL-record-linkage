# Import libs
import pandas as pd
import numpy as np
import cupy as cp
from nameparser import HumanName
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist
import cudf
from tqdm import tqdm
import dask.array as da

from numba import njit

@njit(fastmath=True)
def custom_distance(pointA, pointB):
    return np.sum((np.maximum(pointA, pointB) - np.minimum(pointA, pointB)) <= 1)

metric = 'cosine'
# euclidean,cityblock,seuclidean,sqeuclidean,cosine,correlation,hamming,jaccard,jensenshannon,chebyshev,canberra,braycurtis,-NOT-mahalanobis, custom_distance

test_rows = 50000
refrows = 200
chunk_size = 5000

dfA_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None, nrows=test_rows).drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
dfB_test = pd.read_csv('Data/BIASB_200000.csv', delimiter="|", header=None, nrows=test_rows).drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_test = dfA_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_test = dfB_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

# for spanish names
# dfA = pd.read_csv('Data/authors3_A.csv', header=None, encoding='latin-1', nrows=nrows).rename({0:'id', 1:'FirstName', 2:'MiddleName',3:'LastName'}, axis='columns')
# dfB = pd.read_csv('Data/authors3_B.csv', header=None, encoding='latin-1', nrows=nrows).rename({0:'id', 1:'FirstName', 2:'MiddleName',3:'LastName'}, axis='columns')
# dfA = dfA[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
# dfB = dfB[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)


ref = pd.read_csv('reference_set.csv', names=['name'], nrows=refrows)
# Extract the first and last names from the parsed names
ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)
# ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.surnames)
ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
# ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
# ref['name'] = ref['ParsedName']
ref = ref.drop(['ParsedName', 'name'], axis=1)

# A-B
# Α-ref
edit_distances_AtoRef_names = []
edit_distances_AtoRef_lastnames = []
edit_distances_AtoRef_middlenames_first = []
edit_distances_AtoRef_middlenames_last = []
# edit_distances_AtoRef_names_lastnames = []
# edit_distances_AtoRef_lastnames_names = []

for nameA in dfA_test.itertuples(index=False):
# for nameA in dfA.itertuples(index=False):
    # FirstName A - FirstName ref
    edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
    # LastName A - LastName ref
    edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

    # MiddleName A - Firstname ref
    edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

    # MiddleName A - Lastname ref
    edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])
    
    # # FirstName A - LastName ref
    # edit_distances_AtoRef_names_lastnames.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['LastName']])
    # # LastName A - FirstName ref
    # edit_distances_AtoRef_lastnames_names.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['FirstName']])


edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)

# edit_distances_AtoRef_names_lastnames = np.array(edit_distances_AtoRef_names_lastnames)
# edit_distances_AtoRef_lastnames_names = np.array(edit_distances_AtoRef_lastnames_names)

# B-ref
edit_distances_BtoRef_names = []
edit_distances_BtoRef_lastnames = []
edit_distances_BtoRef_middlenames_first = []
edit_distances_BtoRef_middlenames_last = []
# edit_distances_BtoRef_names_lastnames = []
# edit_distances_BtoRef_lastnames_names = []

for nameB in dfB_test.itertuples(index=False):
# for nameB in dfB.itertuples(index=False):
    # FirstName B - FirstName ref
    edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
    # LastName B - LastName ref
    edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
    # MiddleName B - FirstName ref
    edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

    # MiddleName B - Lastname ref
    edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

    # # FirstName B - LastName ref
    # edit_distances_BtoRef_names_lastnames.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['LastName']])
    # # # LastName B - FirstName ref
    # edit_distances_BtoRef_lastnames_names.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['FirstName']])

edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

# edit_distances_BtoRef_names_lastnames = np.array(edit_distances_BtoRef_names_lastnames)
# edit_distances_BtoRef_lastnames_names = np.array(edit_distances_BtoRef_lastnames_names)
# %%
num_chunks = len(edit_distances_AtoRef_names) // chunk_size + 1
for i in tqdm(range (num_chunks)):
        start_i = i * chunk_size
        end_i = (i + 1) * chunk_size
        if end_i > test_rows:
            continue
        for j in tqdm(range (num_chunks)):
            filename = f'/path/to/test_datasets/parquet/parquet_test_{metric}_n{test_rows}_ref{refrows}_chunk_{i}_{j}.parquet'

            start_j = j * chunk_size
            end_j = (j + 1) * chunk_size
            # Convert arrays to Dask arrays
            edit_distances_AtoRef_names_dask = da.from_array(edit_distances_AtoRef_names[start_i:end_i])
            edit_distances_BtoRef_names_dask = da.from_array(edit_distances_BtoRef_names[start_j:end_j])
            edit_distances_AtoRef_lastnames_dask = da.from_array(edit_distances_AtoRef_lastnames[start_i:end_i])
            edit_distances_BtoRef_lastnames_dask = da.from_array(edit_distances_BtoRef_lastnames[start_j:end_j])
            edit_distances_AtoRef_middlenames_first_dask = da.from_array(edit_distances_AtoRef_middlenames_first[start_i:end_i])
            edit_distances_BtoRef_middlenames_first_dask = da.from_array(edit_distances_BtoRef_middlenames_first[start_j:end_j])
            edit_distances_AtoRef_middlenames_last_dask = da.from_array(edit_distances_AtoRef_middlenames_last[start_i:end_i])
            edit_distances_BtoRef_middlenames_last_dask = da.from_array(edit_distances_BtoRef_middlenames_last[start_j:end_j])
            
            distances_names = da.map_blocks(lambda x, y: cdist(x, y, metric), edit_distances_AtoRef_names_dask, edit_distances_BtoRef_names_dask).compute()
            distances_names = cp.asarray(distances_names)

            distances_lastnames = da.map_blocks(lambda x, y: cdist(x, y, metric), edit_distances_AtoRef_lastnames_dask, edit_distances_BtoRef_lastnames_dask).compute()
            distances_lastnames = cp.asarray(distances_lastnames)

            distances_middlenames_first = da.map_blocks(lambda x, y: cdist(x, y, metric), edit_distances_AtoRef_middlenames_first_dask, edit_distances_BtoRef_middlenames_first_dask).compute()
            distances_middlenames_first = cp.asarray(distances_middlenames_first)

            distances_middlenames_last = da.map_blocks(lambda x, y: cdist(x, y, metric), edit_distances_AtoRef_middlenames_last_dask, edit_distances_BtoRef_middlenames_last_dask).compute()
            distances_middlenames_last = cp.asarray(distances_middlenames_last)
            
            # _-------------------------------------

            # For cosine, correlation:
            # GPU accelerated
            # distances_names = (cucdist((edit_distances_AtoRef_names[start_i:end_i]), (edit_distances_BtoRef_names[start_j:end_j]), metric=metric))
            # distances_lastnames = (cucdist((edit_distances_AtoRef_lastnames[start_i:end_i]), (edit_distances_BtoRef_lastnames[start_j:end_j]), metric=metric))
            # distances_middlenames_first = (cucdist((edit_distances_AtoRef_middlenames_first[start_i:end_i]), (edit_distances_BtoRef_middlenames_first[start_j:end_j]), metric=metric))
            # distances_middlenames_last = (cucdist((edit_distances_AtoRef_middlenames_last[start_i:end_i]), (edit_distances_BtoRef_middlenames_last[start_j:end_j]), metric=metric))

            # no GPU
            # distances_names = cdist(edit_distances_AtoRef_names[start_i:end_i], edit_distances_BtoRef_names[start_j:end_j], metric)
            # distances_lastnames = cdist(edit_distances_AtoRef_lastnames[start_i:end_i], edit_distances_BtoRef_lastnames[start_j:end_j], metric)
            # distances_middlenames_first = cdist(edit_distances_AtoRef_middlenames_first[start_i:end_i], edit_distances_BtoRef_middlenames_first[start_j:end_j], metric)
            # distances_middlenames_last = cdist(edit_distances_AtoRef_middlenames_last[start_i:end_i], edit_distances_BtoRef_middlenames_last[start_j:end_j], metric)
            
            # distances_names = 1 - fastdist.matrix_to_matrix_distance(edit_distances_AtoRef_names[start_i:end_i], edit_distances_BtoRef_names[start_j:end_j], fastdist.cosine, metric)
            # distances_lastnames = 1 - fastdist.matrix_to_matrix_distance(edit_distances_AtoRef_lastnames[start_i:end_i], edit_distances_BtoRef_lastnames[start_j:end_j], fastdist.cosine, metric)
            # distances_middlenames_first = 1 - fastdist.matrix_to_matrix_distance(edit_distances_AtoRef_middlenames_first[start_i:end_i], edit_distances_BtoRef_middlenames_first[start_j:end_j], fastdist.cosine, metric)
            # distances_middlenames_last = 1 - fastdist.matrix_to_matrix_distance(edit_distances_AtoRef_middlenames_last[start_i:end_i], edit_distances_BtoRef_middlenames_last[start_j:end_j], fastdist.cosine, metric)

            # distances_first_last = np.array(cdist(edit_distances_AtoRef_names_lastnames, edit_distances_BtoRef_names_lastnames, metric=metric))
            # distances_last_first = np.array(cdist(edit_distances_AtoRef_lastnames_names, edit_distances_BtoRef_lastnames_names, metric=metric))
            # distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel()))

            distances = cp.column_stack((cp.asarray(distances_names).ravel(), cp.asarray(distances_lastnames).ravel(), cp.asarray(distances_middlenames_first).ravel(), cp.asarray(distances_middlenames_last).ravel()))
            comparisons_with_labels = cudf.DataFrame((dfA_test['id'].values[start_i:end_i, None] == dfB_test['id'].values[start_j:end_j]).astype(int).ravel(), columns=['label'])
            distances_with_labels = cp.column_stack([cp.asarray(comparisons_with_labels['label']), cp.asarray(distances)])

            # if np.isnan(cp.asnumpy(distances_with_labels)).any():
            #     print("Result contains NaN values, stopping the run")
            #     nan_count = np.sum(np.isnan(distances))
            #     print(f"Result contains {nan_count} NaN values")
            #     sys.exit()
            # print('NaN check end')
            dataAB = cudf.DataFrame(distances_with_labels)

            # %%
            # Save the data
            dataAB.columns = ['col' + str(i) for i in range(len(dataAB.columns))]
            dataAB.to_parquet(filename)
            # dataAB.to_csv(filename)
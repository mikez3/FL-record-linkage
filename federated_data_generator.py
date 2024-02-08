"""
NOTE:
This script is a working prototype and contains some repetitive code blocks for different data processing steps. 
The repetition was intentional to ensure each step runs smoothly and to make debugging easier during the development phase. 
The code will be refactored and optimized in the future to eliminate repetition and improve efficiency. 
Please keep this in mind while reviewing.

---

FUNCTIONALITY:
The main functionality of this script is to generate data for Federated Learning.
It creates the training data for two clients and the test data that the global model
will use for evaluation. Each client's data is generated independently to simulate 
real-world conditions where each client's data distribution might be different.
"""

# Import libs
import pandas as pd
import numpy as np
from nameparser import HumanName
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist
import os, cudf
import sys

def custom_distance(pointA, pointB):
    return np.count_nonzero((np.abs(pointA - pointB) <= 1))

# metric = custom_distance
metric = 'cosine'
# euclidean,cityblock,seuclidean,sqeuclidean,cosine,correlation,hamming,jaccard,jensenshannon,chebyshev,canberra,braycurtis,-NOT-mahalanobis, custom_distance

# Read csvs
nrows = 500
refrows = 200
dfA = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
dfB = pd.read_csv('Data/BIASB_200000.csv', delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
dfA = dfA[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB = dfB[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

ref = pd.read_csv('reference_set.csv', names=['name'], nrows=refrows)
# ref = ref.sample(frac=1, random_state=41).reset_index(drop=True)

# Extract the first and last names from the reference set
ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)
ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
ref = ref.drop(['ParsedName', 'name'], axis=1)

# Create datasets for local training
dfA2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])
dfB2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])

dfA2['id'] = dfA['id']
dfA2['FirstName'] = dfA['FirstName']
# dfA2['FirstName'] = dfA['FirstName'] + '!'
# dfA2['LastName'] = dfA['LastName']
dfA2['LastName'] = dfA['LastName'] + '!'
# dfA2['MiddleName'] = dfA['MiddleName'] + '!'
dfA2['MiddleName'] = dfA['MiddleName']

dfB2['id'] = dfB['id']
dfB2['FirstName'] = dfB['FirstName']
# dfB2['FirstName'] = dfB['FirstName'] + '!'
# dfB2['LastName'] = dfB['LastName']
dfB2['LastName'] = dfB['LastName'] + '!'
# dfB2['MiddleName'] = dfB['MiddleName'] + '!'
dfB2['MiddleName'] = dfB['MiddleName']

# The reference set lacks a 'MiddleName' field. To compute the edit distance for 'MiddleNames', 
# we separately use the 'FirstName' and 'LastName' from the reference set. This results in 4 total
# dimensions (First-First_ref, Last-Last_ref, Middle-First_ref, Middle-last_ref).

# Client1_train = A-A2
# Client2_train = B-B2
# server_test = A-B

# Calculate edit distances between A-ref, A2-ref and then calculate the distance of their edit distances using a metric (best: cosine or correlation)
# A-A2
# Α-ref
edit_distances_AtoRef_names = []
edit_distances_AtoRef_lastnames = []
edit_distances_AtoRef_middlenames_first = []
edit_distances_AtoRef_middlenames_last = []
# edit_distances_AtoRef_names_lastnames = []
# edit_distances_AtoRef_lastnames_names = []

for nameA in dfA.itertuples(index=False):
    # FirstName A - FirstName ref
    edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
    # LastName A - LastName ref
    edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

    # MiddleName A - Firstname ref
    edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

    # MiddleName A - Lastname ref
    edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])
    
    # FirstName A - LastName ref
    # edit_distances_AtoRef_names_lastnames.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['LastName']])

    # LastName A - FirstName ref
    # edit_distances_AtoRef_lastnames_names.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['FirstName']])


edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)
# edit_distances_AtoRef_names_lastnames = np.array(edit_distances_AtoRef_names_lastnames)
# edit_distances_AtoRef_lastnames_names = np.array(edit_distances_AtoRef_lastnames_names)

# A2-ref
edit_distances_BtoRef_names = []
edit_distances_BtoRef_lastnames = []
edit_distances_BtoRef_middlenames_first = []
edit_distances_BtoRef_middlenames_last = []
# edit_distances_BtoRef_names_lastnames = []
# edit_distances_BtoRef_lastnames_names = []

for nameB in dfA2.itertuples(index=False):
    # FirstName B - FirstName ref
    edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
    # LastName B - LastName ref
    edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
    # MiddleName B - FirstName ref
    edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

    # MiddleName B - Lastname ref
    edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

    # FirstName B - LastName ref
    # edit_distances_BtoRef_names_lastnames.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['LastName']])

    # # LastName B - FirstName ref
    # edit_distances_BtoRef_lastnames_names.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['FirstName']])

edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)
# edit_distances_BtoRef_names_lastnames = np.array(edit_distances_BtoRef_names_lastnames)
# edit_distances_BtoRef_lastnames_names = np.array(edit_distances_BtoRef_lastnames_names)

distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))
# distances_first_last = np.array(cdist(edit_distances_AtoRef_names_lastnames, edit_distances_BtoRef_names_lastnames, metric=metric))
# distances_last_first = np.array(cdist(edit_distances_AtoRef_lastnames_names, edit_distances_BtoRef_lastnames_names, metric=metric))

distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))
# distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel(), (distances_first_last).ravel(), (distances_last_first).ravel()))

if np.isnan(distances).any():
    print("Result contains NaN values, stopping the run")
    nan_count = np.sum(np.isnan(distances))
    print(f"Result contains {nan_count} NaN values")
    sys.exit()

comparisons_with_labels = pd.DataFrame((dfA['id'].values[:, None] == dfA2['id'].values).astype(int).ravel(), columns=['label'])

# save
distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
data = cudf.DataFrame(distances_with_labels)
filename = '/data/train_A.csv'
os.makedirs(os.path.dirname(filename), exist_ok=True)
data.to_csv(filename, index=False, header=None)

# A-B
# Α-ref
edit_distances_AtoRef_names = []
edit_distances_AtoRef_lastnames = []
edit_distances_AtoRef_middlenames_first = []
edit_distances_AtoRef_middlenames_last = []
# edit_distances_AtoRef_names_lastnames = []
# edit_distances_AtoRef_lastnames_names = []

for nameA in dfA.itertuples(index=False):
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

for nameB in dfB.itertuples(index=False):
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


distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))
# distances_first_last = np.array(cdist(edit_distances_AtoRef_names_lastnames, edit_distances_BtoRef_names_lastnames, metric=metric))
# distances_last_first = np.array(cdist(edit_distances_AtoRef_lastnames_names, edit_distances_BtoRef_lastnames_names, metric=metric))


distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))
# distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel(), (distances_first_last).ravel(), (distances_last_first).ravel()))

if np.isnan(distances).any():
    print("Result contains NaN values, stopping the run")
    nan_count = np.sum(np.isnan(distances))
    print(f"Result contains {nan_count} NaN values")
    sys.exit()

comparisons_with_labels = pd.DataFrame((dfA['id'].values[:, None] == dfB['id'].values).astype(int).ravel(), columns=['label'])
# save
distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
data = cudf.DataFrame(distances_with_labels)
filename = '/data/test_AB.csv'
os.makedirs(os.path.dirname(filename), exist_ok=True)
data.to_csv(filename, index=False, header=None)


# B-B2
# B-ref
edit_distances_AtoRef_names = []
edit_distances_AtoRef_lastnames = []
edit_distances_AtoRef_middlenames_first = []
edit_distances_AtoRef_middlenames_last = []
# edit_distances_AtoRef_names_lastnames = []
# edit_distances_AtoRef_lastnames_names = []

for nameA in dfB.itertuples(index=False):
    # FirstName A - FirstName ref
    edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
    # LastName A - LastName ref
    edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

    # MiddleName A - Firstname ref
    edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

    # # MiddleName A - Lastname ref
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


# B2-ref
edit_distances_BtoRef_names = []
edit_distances_BtoRef_lastnames = []
edit_distances_BtoRef_middlenames_first = []
edit_distances_BtoRef_middlenames_last = []
# edit_distances_BtoRef_names_lastnames = []
# edit_distances_BtoRef_lastnames_names = []

for nameB in dfB2.itertuples(index=False):
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


distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))
# distances_first_last = np.array(cdist(edit_distances_AtoRef_names_lastnames, edit_distances_BtoRef_names_lastnames, metric=metric))
# distances_last_first = np.array(cdist(edit_distances_AtoRef_lastnames_names, edit_distances_BtoRef_lastnames_names, metric=metric))

distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))
# distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel(), (distances_first_last).ravel(), (distances_last_first).ravel()))


if np.isnan(distances).any():
    print("Result contains NaN values, stopping the run")
    nan_count = np.sum(np.isnan(distances))
    print(f"Result contains {nan_count} NaN values")
    sys.exit()

comparisons_with_labels = pd.DataFrame((dfB['id'].values[:, None] == dfB2['id'].values).astype(int).ravel(), columns=['label'])
# save
distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
data = cudf.DataFrame(distances_with_labels)
filename = '/data/train_B.csv'
os.makedirs(os.path.dirname(filename), exist_ok=True)
data.to_csv(filename, index=False, header=None)

# Read the CSV files
test = cudf.DataFrame(pd.read_csv('/data/test_AB.csv', header=None))
trainA = cudf.DataFrame(pd.read_csv('/data/train_A.csv', header=None))
trainB = cudf.DataFrame(pd.read_csv('/data/train_B.csv', header=None))

# Combine the DataFrames
df = cudf.concat([test, trainA, trainB])
if isinstance(metric, str):
    csv_name = metric+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
elif callable(metric):
    csv_name = str(metric.__name__)+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"

# csv_name = metric+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
filename = '/data/'+ csv_name

# Save the combined DataFrame to a new CSV file
df.to_csv(filename, index=False, header=None, chunksize=100000)
os.makedirs(os.path.dirname('/tmp/dataset/data.csv'), exist_ok=True)
df.to_csv('/tmp/dataset/data.csv', index=False, header=None, chunksize=100000)
import os
import argparse

import pandas as pd
import cudf
import numpy as np  
from nameparser import HumanName

from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist

from distutils.util import strtobool

def custom_distance(pointA, pointB, threshold = 1):
    return np.count_nonzero(np.abs(pointA - pointB) <= threshold)


def read_and_procese_csv(path, nrows, ref_nrows):
    # Read csvs
    dfA = pd.read_csv(path[0], delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1)\
        .rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
    dfB = pd.read_csv(path[1], delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1)\
        .rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
    dfA = dfA[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
    dfB = dfB[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
    if (len(path)==3):
        print("Used",nrows, "records")
        print("Used",ref_nrows, "names from reference set")
        ref = pd.read_csv(path[2], names=['name'], nrows=ref_nrows)
        # Extract the first and last names from the parsed names of reference set
        ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
        ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)
        ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
        ref = ref.drop(['name', 'ParsedName'], axis=1)
        return dfA, dfB, ref
    return dfA, dfB


def edit_distance(df, ref):
    edit_distances_names = []
    edit_distances_lastnames = []
    edit_distances_middlenames_first = []
    edit_distances_middlenames_last = []

    for name in df.itertuples(index=False):
        # FirstName df - FirstName ref
        edit_distances_names.append([Levenshtein.distance(name[1], nameRef) for nameRef in ref['FirstName']])
        # LastName df - LastName ref
        edit_distances_lastnames.append([Levenshtein.distance(name[2], nameRef) for nameRef in ref['LastName']])
        # MiddleName df - FirstName ref
        edit_distances_middlenames_first.append([Levenshtein.distance(name[3], nameRef) for nameRef in ref['FirstName']])
        # MiddleName df - LastName ref
        edit_distances_middlenames_last.append([Levenshtein.distance(name[3], nameRef) for nameRef in ref['LastName']])

    edit_distances_names = np.array(edit_distances_names)
    edit_distances_lastnames = np.array(edit_distances_lastnames)
    edit_distances_middlenames_first = np.array(edit_distances_middlenames_first)
    edit_distances_middlenames_last = np.array(edit_distances_middlenames_last)

    return edit_distances_names, edit_distances_lastnames, edit_distances_middlenames_first, edit_distances_middlenames_last

def edit_distance_no_ref(dfA, dfB):
    edit_distances_names = []
    edit_distances_lastnames = []
    edit_distances_middlenames = []

    for nameA in dfA.itertuples(index=False):
        # FirstName df - FirstName B
        edit_distances_names.append([Levenshtein.distance(nameA[1], nameB) for nameB in dfB['FirstName']])
        # LastName A - LastName B
        edit_distances_lastnames.append([Levenshtein.distance(nameA[2], nameRef) for nameRef in dfB['LastName']])

        # Middlename A - Middlename B
        edit_distances_middlenames.append([Levenshtein.distance(nameA[3], nameB) for nameB in dfB['MiddleName']])

    edit_distances_names = np.array(edit_distances_names)
    edit_distances_lastnames = np.array(edit_distances_lastnames)
    edit_distances_middlenames = np.array(edit_distances_middlenames)
    
    return edit_distances_names, edit_distances_lastnames, edit_distances_middlenames


def calculate_distance_matrices(metric, threshold, ref=True, *args):
    distances = []
    if (metric != 'custom_distance'):
        # metrics = euclidean, cityblock, seuclidean, sqeuclidean, cosine, correlation, hamming, jaccard, jensenshannon, chebyshev, canberra, braycurtis
        if ref:
            for edit_distances_AtoRef, edit_distances_BtoRef in args:
                distances.append(np.array(cdist(edit_distances_AtoRef, edit_distances_BtoRef, metric=metric)).ravel())            

        else:
            for edit_distances_AtoB in args:
                distances.append(np.array(cdist(edit_distances_AtoB, edit_distances_AtoB, metric=metric)).ravel())        
    else:
    # Custom distance calculator  
        distances = []
        for edit_dist in args:
            distances.append(np.array([custom_distance(edit_dist[0][i], edit_dist[1][j], threshold) 
                                    for i in range(len(edit_dist[0])) 
                                    for j in range(len(edit_dist[1]))]).ravel()) 
        distances = np.column_stack(distances)
    return np.column_stack(distances)

def add_labels(dfA, dfB):
    # labels = pd.DataFrame((dfA['id'].values[:, None] == dfB['id'].values).astype(int).ravel(), columns=['label'])
    labels = cudf.DataFrame((dfA['id'].values[:, None] == dfB['id'].values).astype(int).ravel(), columns=['label'])
    return labels


def save_to_csv(distances, labels, filename):
    
    # Add the label column to the numpy array
    distances_with_labels = np.column_stack([labels['label'].astype(int), distances])

    # Shuffle
    rng = np.random.default_rng(41)
    rng.shuffle(distances_with_labels)

    # data = pd.DataFrame(distances_with_labels)
    data = cudf.DataFrame(distances_with_labels)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # If data_loader.py from nvflare/app_opt/sklearn is not fixed, uncomment:
    # data.to_csv(filename, index=False)
    data.to_csv(filename, index=False, header=None)


def main():
    parser = argparse.ArgumentParser(description='Create data points to train the model')
    parser.add_argument('--path', type=str, nargs=3, default=['Data/BIASA_200000.csv', 'Data/BIASB_200000.csv', 'Data/reference_set.csv'], help='Path of each CSV file (data and reference set)')
    parser.add_argument('--rows', type=int, default=1000, help='Number of rows to read from each CSV file')
    parser.add_argument('--refrows', type=int, default=200, help='Number of rows to read from each CSV file')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for distance calculation')
    parser.add_argument('--threshold', type=str, default=1, help='Threshold for the custom_distance function')
    parser.add_argument('--filename', type=str, default="/tmp/dataset/data.csv", help='The name of the output CSV file')
    parser.add_argument('--ref', type=strtobool, default=True, help='Use a reference set in the calculations. Set to True to use a reference set, or False to not use a reference set.')
    args = parser.parse_args()
    
    dfA, dfB, ref = read_and_procese_csv(args.path, args.rows, args.refrows)
    if (args.ref):
        edit_distances_AtoRef_names, edit_distances_AtoRef_lastnames, edit_distances_AtoRef_middlenames_first, edit_distances_AtoRef_middlenames_last\
            = edit_distance(dfA, ref)
        edit_distances_BtoRef_names, edit_distances_BtoRef_lastnames, edit_distances_BtoRef_middlenames_first, edit_distances_BtoRef_middlenames_last\
            = edit_distance(dfB, ref)
        
        distances = calculate_distance_matrices(args.metric, args.threshold, args.ref, (edit_distances_AtoRef_names, edit_distances_BtoRef_names), (edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames)\
                                                , (edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first), (edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last))
    else:
        edit_distances_AtoB_names, edit_distances_AtoB_lastnames, edit_distances_AtoB_middlenames\
            = edit_distance_no_ref(dfA, dfB)
        distances = calculate_distance_matrices(args.metric, args.threshold, args.ref, (edit_distances_AtoB_names), (edit_distances_AtoB_lastnames), (edit_distances_AtoB_middlenames))
    labels = add_labels(dfA, dfB)
    save_to_csv(distances, labels, args.filename)
    print("Created",len(distances), "data points")

if __name__ == "__main__":
    main()

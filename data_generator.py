import os
import argparse

import pandas as pd
import numpy as np
from nameparser import HumanName

from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist


def custom_distance(pointA, pointB, threshold):
    return np.count_nonzero(np.abs(pointA - pointB) <= threshold)


def read_and_procese_csv(path, nrows):
    pathA, pathB, pathRef = path
    # Read csvs
    dfA = pd.read_csv(pathA, delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1)\
        .rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
    dfB = pd.read_csv(pathB, delimiter="|", header=None, nrows=nrows).drop([3,4,5,6], axis=1)\
        .rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
    
    ref = pd.read_csv(pathRef, names=['name'])
    dfA = dfA[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
    dfB = dfB[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

    # Extract the first and last names from the parsed names of reference set
    ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
    ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)
    ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
    ref = ref.drop(['name', 'ParsedName'], axis=1)
    return dfA, dfB, ref


def edit_distance(df, ref):
    edit_distances_names = []
    edit_distances_lastnames = []
    edit_distances_middlenames_first = []
    edit_distances_middlenames_last = []
    edit_distances_names_lastnames = []
    edit_distances_lastnames_names = []

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


def calculate_distance_matrices(metric, threshold, *args):
    distances = []
    if (metric != 'custom_distance'):
        # metrics = euclidean, cityblock, seuclidean, sqeuclidean, cosine, correlation, hamming, jaccard, jensenshannon, hebyshev, canberra, braycurtis
        for edit_distances_AtoRef, edit_distances_BtoRef in args:
            distances.append(np.array(cdist(edit_distances_AtoRef, edit_distances_BtoRef, metric=metric)).ravel())
    else:
    # Custom distance calculator
       for edit_distances_AtoRef, edit_distances_BtoRef in args:
            distances.append(np.array([custom_distance(i, j, threshold) for i in edit_distances_AtoRef for j in edit_distances_BtoRef]).ravel())
    return np.column_stack(distances)


def add_labels(dfA, dfB):
    labels = pd.DataFrame((dfA['id'].values[:, None] == dfB['id'].values).astype(int).ravel(), columns=['label'])
    return labels


def save_to_csv(distances, labels, filename):
    
    # Determine the directory from the filename
    dir_name = os.path.dirname(filename)
    
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)
   
    np.savetxt(filename, distances, delimiter=',')
    data = pd.read_csv(filename, header=None)
    data['label'] = labels['label']

    # Reorder the columns to put 'label' first
    cols = ['label'] + [col for col in data.columns if col != 'label']
    data = data.reindex(columns=cols)

    # Save the DataFrame back to the CSV file
    data.to_csv(filename, index=False, header=None)


def main():
    parser = argparse.ArgumentParser(description='Create data points to train the model')
    parser.add_argument('--path', type=str, nargs=3, default=['Data/BIASA_200000.csv', 'Data/BIASB_200000.csv', 'Data/reference_set.csv'], help='Path of each CSV file (data and reference set)')
    parser.add_argument('--rows', type=int, default=1000, help='Number of rows to read from each CSV file')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for distance calculation')
    parser.add_argument('--threshold', type=str, default=1, help='Threshold for the custom_distance function')
    parser.add_argument('--filename', type=str, default="/tmp/dataset/data.csv", help='The name of the output CSV file')
    args = parser.parse_args()

    dfA, dfB, ref = read_and_procese_csv(args.path, args.rows)
    edit_distances_AtoRef_names, edit_distances_AtoRef_lastnames, edit_distances_AtoRef_middlenames_first, edit_distances_AtoRef_middlenames_last\
          = edit_distance(dfA, ref)
    edit_distances_BtoRef_names, edit_distances_BtoRef_lastnames, edit_distances_BtoRef_middlenames_first, edit_distances_BtoRef_middlenames_last\
          = edit_distance(dfB, ref)
    
    distances = calculate_distance_matrices(args.metric, args.threshold, (edit_distances_AtoRef_names, edit_distances_BtoRef_names), (edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames)\
                                            , (edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first), (edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last))
    labels = add_labels(dfA, dfB)
    save_to_csv(distances, labels, args.filename)


if __name__ == "__main__":
    main()

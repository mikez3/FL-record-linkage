import joblib
import dask_cudf
import cuml
import numpy as np
import cupy as cp
from dask.diagnostics import ProgressBar
from fastdist import fastdist

metric = 'cosine'
kernel = 'rbf'
lines_trained = '500'
ref = '200'
lines_to_test = 50000

print(kernel, ',lines trained:', lines_trained)
model = joblib.load('/path/to/trained_models/'+kernel+'/'+metric+'_n'+lines_trained+'_ref'+ref+'.joblib')
# %%

ref = '200'

data_path = '/path/to/tests_datasets/test_'+metric+'_n'+str(lines_to_test)+'_ref'+ref+'.csv'
# data_path = f'/media/mike/corsair/correct_data/new_dataset/only_tests/parquet/parquet_test_cosine_n50000_ref200_chunk_*_*.parquet'
chunksize = 500000
csv_lines = lines_to_test**2
num_chunks = csv_lines // chunksize + 1
print('num_chunks',num_chunks)
tp = fp = fn = 0

def process_chunk(chunk):
    if chunk.empty:
        # print("Why?")
        chunk_fp = chunk_fn = chunk_tp = 0
        return chunk_fp, chunk_fn, chunk_tp
    predictions_joblib = model.predict(chunk.drop(chunk.columns[0], axis=1)).astype(np.int64)
    test_true_label = chunk.iloc[:, 0].astype('int64')
    if cp.unique(predictions_joblib).size > 1:
         _, chunk_fp, chunk_fn, chunk_tp = fastdist.confusion_matrix(test_true_label.to_numpy(), predictions_joblib.to_numpy()).ravel()
    else:
        chunk_fp = chunk_fn = chunk_tp = 0
    return chunk_fp, chunk_fn, chunk_tp

chunk = dask_cudf.read_csv(data_path, header=None, blocksize=chunksize)
# chunk = dask_cudf.read_parquet(data_path, blocksize=chunksize)
with ProgressBar():
    results = chunk.map_partitions(process_chunk, meta=(None, 'int64')).compute()

tp = fp = fn = 0
for _, (chunk_fp, chunk_fn, chunk_tp) in results.items():
    tp += chunk_tp
    fp += chunk_fp
    fn += chunk_fn
# %%
if not(tp+fp == 0):  
    precision = cp.round(tp/(tp+fp),4)
else:
    precision = 0
if not(tp+fn == 0):
    recall = cp.round(tp/(tp+fn),4)
else:
    recall = 0
print("P =", precision)
print("R =", recall)

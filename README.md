# Federated SVM with Scikit-learn for Entity Resolution

Please make sure you set up virtual environment and Jupyterlab follows [example from NVFlare](https://github.com/NVIDIA/NVFlare/blob/main/examples/README.md)


## Scikit-learn, tabular data, and federated SVM
### Scikit-learn
This application uses [Scikit-learn](https://scikit-learn.org/). [cuML](https://docs.rapids.ai/api/cuml/stable/) can also be used as backend instead of Scikit-learn.
### Tabular data
The data used in this application is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples.
- the first column represents the label. 
- the other columns cover the features.    


Each client is expected to have one local data file containing both training and validation samples. 
To load the data for each client, the following parameters are expected by local learner:
- data_file_path: (`string`) the full path to the client's data file. 
- train_start: (`int`) start row index for the training set.
- train_end: (`int`) end row index for the training set.
- valid_start: (`int`) start row index for the validation set.
- valid_end: (`int`) end row index for the validation set.

### Federated SVM
The machine learning algorithm used is [SVM for Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
Under this setting, federated learning can be formulated in two steps:
- local training: each client trains a local SVM model with their own data
- global training: server collects the support vectors from all clients and 
  trains a global SVM model based on them

Unlike other iterative federated algorithms, federated SVM only involves 
these two training steps. Hence, in the server config, we have
```json
"num_rounds": 2
```
The first round is the training round, performing local training and global aggregation. 
Next, the global model will be sent back to clients for the second round, 
performing model validation and local model update. 
If this number is set to a number greater than 2, the system will report an error and exit.


## Data generator
You can run the script from the command line with the following command:
```commandline
python data_generator.py --path [pathA, pathB, pathRef] --rows [rows] --metric [metric] --threshold [threshold] --filename [filename]
```
For example, to read 1000 rows from 'Data/BIASA_200000.csv', 'Data/BIASB_200000.csv', and 'Data/reference_set.csv', calculate distances using the cosine metric, and save the resulting dataset to '/tmp/dataset/data.csv', you would use the following command:

```commandline
python data_generator.py --path Data/BIASA_200000.csv Data/BIASB_200000.csv Data/reference_set.csv --rows 1000 --metric cosine --filename /tmp/dataset/data.csv
```
### How it works
1. The script reads data from three CSV files: two data files and a reference set. The data files contain columns for 'id', 'FirstName', 'LastName', and 'MiddleName'. The reference set contains a single column 'name'.

2. The script calculates the Levenshtein distance between the names in the data files and the names in the reference set in order to create numerical data points.

3. Then it calculates distance matrices using the [cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) function from scipy. If the custom distance function is used, then a threshold is required. The default value for threshold is 1.

4. The script adds labels to the data points. A label of 1 indicates that two data points have the same 'id', and a label of 0 indicates that they have different 'ids'.

5. The script saves the resulting dataset to a CSV file, which is located at `/temp/dataset/data.csv`.


## Prepare clients' configs with proper data information 
For real-world FL applications, the config JSON files are expected to be 
specified by each client, according to their own local data path and splits for training and validation.

In this simulated study, to efficiently generate the config files for a 
study under a particular setting, we provide a script to automate the process. 
Note that manual copying and content modification can achieve the same.

For an experiment with `K` clients, we split one dataset into `K+1` parts in a non-overlapping fashion: `K` clients' training data and `1` common validation data. 
To simulate data imbalance among clients, we provided several options for client data splits by specifying how a client's data amount correlates with its ID number (from `1` to `K`):
- Uniform
- Linear
- Square
- Exponential

These options can be used to simulate no data imbalance (`uniform`), 
moderate data imbalance (`linear`), and high data imbalance (`square` for 
larger client number e.g., `K=20`, exponential for smaller client number e.g., 
`K=5` as it will be too aggressive for larger client numbers)

This step is performed by 
```commandline
bash prepare_job_config.sh
```
In this example, we chose the Radial Basis Function (RBF) kernel to experiment with three clients under the uniform data split. 

Below is a sample config for site-1, saved to `./jobs/sklearn_svm_3_uniform/app_site-1/config/config_fed_client.json`:
```json
{
  "format_version": 2,
  "executors": [
    {
      "tasks": [
        "train"
      ],
      "executor": {
        "id": "Executor",
        "path": "nvflare.app_opt.sklearn.sklearn_executor.SKLearnExecutor",
        "args": {
          "learner_id": "svm_learner"
        }
      }
    }
  ],
  "task_result_filters": [],
  "task_data_filters": [],
  "components": [
    {
      "id": "svm_learner",
      "path": "svm_learner.SVMLearner",
      "args": {
        "data_path": "/tmp/dataset/data.csv",
        "train_start": 100000,
        "train_end": 400000,
        "valid_start": 0,
        "valid_end": 100000
      }
    }
  ]
}
```

## Run experiment with FL simulator
[FL simulator](https://nvflare.readthedocs.io/en/2.3/user_guide/fl_simulator.html) is used to simulate FL experiments or debug codes, not for real FL deployment.
We can run the FL simulator with three clients under the uniform data split with
```commandline
nvflare simulator ./jobs/sklearn_svm_3_uniform -w ./workspace -n 3 -t 3
```
or
```commandline
bash run_experiment_simulator.sh
```
You can view the AUC, Precision, and Recall of the resulting global model in the clients' logs and on Google TensorBoard. To access TensorBoard, use the following command:
```commandline
python3 -m tensorboard.main --logdir='workspace'
```

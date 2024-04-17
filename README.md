# Federated Learning approach for Privacy Preserving Record Linkage
> **Notice**: This project is currently under development.

This project provides a privacy-preserving solution for record matching, eliminating the need for dataholders to transfer their data externally in any form. By leveraging a Federated Learning approach with Support Vector Machines and using a reference set, it achieves high-quality matching comparable to non-Federated Learning setups for plain record linkage.

## Quick info:
[`federated_data_generator.py`](#data-generator): Generates data for federated learning simulator

[`local_data_generator.py`](#data-generator): Generates data for local training (no Federated Learning here)

[`prepare_job_config.sh`](#prepare-clients-configs-with-proper-data-information): Generates the configuration files

[`run_experiment_simulator.sh`](#run-experiment-with-fl-simulator): Runs the FL simulator

[`test_only_data_generator.py`](#test-trained-models-on-bigger-datasets): Creates test datasets. Suitable for big tests files

[`tester.py`](#test-trained-models-on-bigger-datasets): Tests saved models on selected datasets


## NVFLare

This project was implemented using NVFlare.

For more detailed information about the framework, you may refer to the [Scikit-learn SVM example](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-svm) in the [NVIDIA NVFlare](https://github.com/NVIDIA/NVFlare/tree/main) repository. This example provides a detailed walkthrough of how to use Scikit-learn's SVM with NVFlare.

<!-- ## cuML - Scikit-learn
For faster execution times with large datasets, it is recommended to use [cuML](https://docs.rapids.ai/api/cuml/stable/). Alternatively, [Scikit-learn](https://scikit-learn.org/) can be used as a backend instead of cuML. -->

## Data generator

<!-- Here the [cudf](https://github.com/rapidsai/cudf) library was used for faster execution. -->

There are 2 options for data generation depanding if the data is going to be used for local training or in a Federated environment.

For the first option you can run the script from the command line with the following command:
```commandline
python local_data_generator.py --path [pathA, pathB, pathRef] --rows [rows] --metric [metric] --threshold [threshold] --filename [filename]
```
For example, to read 1000 rows from 'Data/BIASA_200000.csv', 'Data/BIASB_200000.csv', and 'Data/reference_set.csv', calculate distances using the cosine metric, and save the resulting dataset to '/tmp/dataset/data.csv', you would use the following command:

```commandline
python local_data_generator.py --path Data/BIASA_200000.csv Data/BIASB_200000.csv Data/reference_set.csv --rows 1000 --metric cosine --filename /tmp/dataset/data.csv
```

For the second option (i.e. the dataset for federated learning), the data generation process is integrated directly into the `federated_data_generator.py` file. To change parameters such as the path, number of rows, metric, or filename, you will need to do it manually in the code.

### How it works
<!-- Add Differential Privacy as an option here -->
1. The script reads data from three CSV files: two data files with the real records and a reference set with random names. The data files contain columns for 'id', 'FirstName', 'LastName', and 'MiddleName'. The reference set contains a single column 'name' which is a Full Name.

2. The script calculates the Levenshtein distance (edit distance) between the names in the data files and the names in the reference set in order to create numerical data points.

3. The script then calculates distance matrices using the [cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) function from scipy. 
<!-- A custom distance metric is also implemented as an alternative, which calculates the total number of edit distances between two records that have a difference of &le; 1 ([see below for more details](#custom-distance-metric)).  -->

4. Next, labels are assigned. A label of 1 is given when two data points have the same 'id', indicating that they represent the same entity. Conversely, a label of 0 is given when they have different 'ids', indicating that they are different entities.

5. The script saves the resulting dataset to a CSV file, which is located at `/tmp/dataset/data.csv`.

<!-- #### Custom Distance Metric
The custom metric is a function that calculates the number of records in the reference set for which two real database records have an edit distance within a threshold of ±1.
The value range of the function is [0, len(reference_set)], with larger values of the function implying greater similarity.

For example, with 2 names in the reference set, let's say the points:
Point A: [ [1, 2], [8, 8] ].
Point B: [ [1, 3], [7, 13] ].
Here [1, 2] contrasts the edit distances of the FirstName of the first real record (pointA) with the 2 FirstNames in the reference set; [8, 8] is for the LastNames. Similarly for the second data record (Point B).
For each dimension we calculate how many edit distances are "close" for each record.
In the 1st dimension we have |1-1|=0 &le;1 and |2-3|=1 &le; 1. We are interested in how many results are less than or equal to 1. So here it is 2 and this is the value the function returns.
In the 2nd dimension, half of the records are similar, i.e. with one reference set name, the records have very similar or the same edit distance, while with the other they do not.
Larger threshold values give a greater tolerance for the function to consider larger edit distance differences as "close". -->



## Prepare clients' configs with proper data information 
A script is used to automatically generate the configuration files for a specific setting, following the approach suggested in the NVFlare documentation. This script simplifies the process and eliminates the need for manual copying and modification of the files.

You can run the script with the following command:

```bash
bash prepare_job_config.sh
```

Please note that this script will recreate the `jobs/sklearn_svm_base/` folder for each client and also for the server. For instance, if you modify the number of clients in this bash file to 2, it will create a new folder under the `jobs/` directory named `sklearn_svm_2_uniform`.

The newly created folder will contain the same files and code as in the `jobs/sklearn_svm_base/` directory. If you wish to make more detailed modifications, such as changing the model kernel for the client and server, you will need to modify the `jobs/sklearn_svm_base/app/config/config_fed_server.json` file.

In this example, we chose the Radial Basis Function (RBF) kernel to experiment with three clients under the uniform data split. 

Below is a sample config for site-1, saved to `./jobs/sklearn_svm_2_uniform/app_site-1/config/config_fed_client.json`:
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
        "train_start": 250000,
        "train_end": 500000,
        "valid_start": 0,
        "valid_end": 250000
      }
    }
  ]
}
```

## Differential Privacy
Differential Privacy can also be added using Randomized Response. This is applied to the labels of support vectors after local training and before they are sent to the server. The code for this option is currently commented out in the `/jobs/sklearn_svm_base/app/custom/svm_learner.py` file.


## Run experiment with FL simulator
We can run the [FL simulator](https://nvflare.readthedocs.io/en/2.3/user_guide/fl_simulator.html) with three clients under the uniform data split with
```commandline
nvflare simulator ./jobs/sklearn_svm_2_uniform -w ./workspace -n 2 -t 2
```
or
```commandline
bash run_experiment_simulator.sh
```
You can monitor the Precision and Recall metrics of the resulting global model through the clients' logs and Google TensorBoard. To launch TensorBoard, execute the following command:
```bash
python3 -m tensorboard.main --logdir='workspace'
```

## Test trained models on bigger datasets
`test_only_data_generator.py`: Creates big test datasets
`tester.py`: Tests saved models on selected datasets


# Federated SVM with Scikit-learn for Entity Resolution


## Additional Resources

For more detailed information, you may refer to the [Scikit-learn SVM example](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/sklearn-svm) in the [NVIDIA NVFlare](https://github.com/NVIDIA/NVFlare/tree/main) repository. This example provides a detailed walkthrough of how to use Scikit-learn's SVM with NVFlare.

## Scikit-learn
This application uses [Scikit-learn](https://scikit-learn.org/). [cuML](https://docs.rapids.ai/api/cuml/stable/) can also be used as backend instead of Scikit-learn.

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
In this simulated study, to efficiently generate the config files for a 
study under a particular setting, a script is provided to automate the process. 
Note that manual copying and content modification can achieve the same.

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
We can run the [FL simulator](https://nvflare.readthedocs.io/en/2.3/user_guide/fl_simulator.html) with three clients under the uniform data split with
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

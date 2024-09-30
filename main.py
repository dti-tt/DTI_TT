from sklearn.metrics import mean_squared_error
from DeepPurpose import utils, dataset
import pandas as pd
import warnings
import os
import numpy as np
import torch
import sys
warnings.filterwarnings("ignore")

#Execution: python main_dti.py tt cold
#sys.argv[1] = {"tt", "bs"}
#sys.argv[2] == {"cold", "warm"}


if sys.argv[1] == "tt":
    from DeepPurpose import DTI_TwoTower as models
elif sys.argv[1] == "bs":
    from DeepPurpose import DTI_Baseline as models

DATA_PATH = "./data/"

if sys.argv[2] == "cold":
    TRAIN_FILE_NAME = "coldstart_train.csv"
    TEST_FILE_NAME = "coldstart_test.csv"
elif sys.argv[2] == "warm":
    TRAIN_FILE_NAME = "warmstart_train.csv"
    TEST_FILE_NAME = "warmstart_test.csv"


def save_to_csv(ground_truth, prediction, file_name='output.csv'):
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    # Create a DataFrame with the two arrays
    data = pd.DataFrame({
        'ground_truth': ground_truth,
        'prediction': prediction
    })
    
    # Write the DataFrame to a CSV file
    data.to_csv(file_name, index=False)
    print(f'CSV file saved as {file_name}')


def data_repreprocess(path_to_dataset, file_name, new_file_name):
    """
    Input: path and file_name
    Output: new txt file with the same format of DeepPurpose
    """
    data = pd.read_csv(path_to_dataset + "/" + file_name)
    new_data = data[["Smiles", "Sequence", "pIC50"]]
    if file_name.split("_")[0] == "new":
        new_data.to_csv(path_to_dataset + "/" + new_file_name, sep=' ', index=False)
    else:
        new_data.to_csv(path_to_dataset + "/" + new_file_name, sep='\t', index=False)
    # delete first row as it makes error
    file_in = open(path_to_dataset + "/" + new_file_name, "r")
    file_out = open(path_to_dataset + "/new_" + new_file_name, "w")
    count = 0
    for line in file_in.readlines():
        if count > 0:
            file_out.write(line)
        count = count + 1
    file_in.close()
    file_out.close()


def test_with_our_testset(path_to_testset):
    """
    Input: path to testset
    Output: list of drug, target, score
    """
    file_in = open(path_to_testset, "r")
    drug = []
    target = []
    score = []
    for line in file_in.readlines():
        try:
            drug.append(line.split()[0])
            target.append(line.split()[1])
            score.append(float(line.split()[2]))
        except:
            print(line)
    return drug, target, score


def pic50_to_ic50_nM(pic50):
    """
    Convert pIC50 to IC50 in nanomolar (nM).

    Parameters:
    pic50 (float): The pIC50 value.

    Returns:
    float: The IC50 value in nanomolar (nM).
    """
    ic50_nM = 10.0 ** (9.0 - pic50)
    return ic50_nM


def normalized_rmse(y_true, y_pred, norm_type='range'):
    """
    Compute normalized RMSE.

    Parameters:
    y_true (array): Actual values.
    y_pred (array): Predicted values.
    norm_type (str): Type of normalization ('range' or 'mean'). Defaults to 'range'.

    Returns:
    float: Normalized RMSE.
    """
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Normalize based on the selected method
    if norm_type == 'range':
        norm_value = np.max(y_true) - np.min(y_true)
    elif norm_type == 'mean':
        norm_value = np.mean(y_true)
    else:
        raise ValueError("Invalid norm_type. Choose 'range' or 'mean'.")

    # Calculate normalized RMSE
    nrmse = rmse / norm_value
    return nrmse


def error_metrics(pic50_gt, pic50_pr):
    pic50_nrmse = normalized_rmse(np.asarray(pic50_gt), np.asarray(pic50_pr))
    ic50_gt = [pic50_to_ic50_nM(val) for val in pic50_gt]
    ic50_pr = [pic50_to_ic50_nM(val) for val in pic50_pr]
    ic50_nrmse = normalized_rmse(np.asarray(ic50_gt), np.asarray(ic50_pr))
    return pic50_nrmse, ic50_nrmse



def main():
    # Convert data format from our data to DeepPurpose
    data_repreprocess(DATA_PATH, TRAIN_FILE_NAME, "train.txt")
    data_repreprocess(DATA_PATH, TEST_FILE_NAME, "test.txt")
    
    # Data read
    X_drugs_test, X_targets_test, y_test = dataset.read_file_training_dataset_drug_target_pairs(
        DATA_PATH + 'new_test.txt')
    X_drugs_train, X_targets_train, y_train = dataset.read_file_training_dataset_drug_target_pairs(
        DATA_PATH + 'new_train.txt')

    fout = open("RESULT_" + sys.argv[1] + "_" + sys.argv[2] + ".csv", "w")
    drug_encoding_list = ["MPNN", "Transformer", "KPGT"]
    target_encoding_list = [ "Transformer", "ProteinBERT"]
    for target_encoding in target_encoding_list:
        fout_drug = open("RESULT_" + sys.argv[1] + "_" + sys.argv[2] + "_protein_" + str(target_encoding) + ".csv", "w")
        for drug_encoding in drug_encoding_list:
            print(drug_encoding, target_encoding)
            # data split (lets just use their function for now)
            train = utils.data_process(X_drugs_train, X_targets_train, y_train, drug_encoding, target_encoding,
                                       split_method='no_split')
            val = utils.data_process(X_drugs_test, X_targets_test, y_test, drug_encoding, target_encoding,
                                     split_method='no_split')
            test = utils.data_process(X_drugs_test, X_targets_test, y_test, drug_encoding, target_encoding,
                                      split_method='no_split')
            # Define model
            config = utils.generate_config(drug_encoding=drug_encoding,
                                           target_encoding=target_encoding,
                                           cls_hidden_dims=[1024],
                                           train_epoch=60,
                                           test_every_X_epoch = 5,
                                           LR=0.0002,
                                           mlp_hidden_dims_drug=[1024, 256, 64],
                                           mlp_hidden_dims_target=[1024, 64],
                                           batch_size=128,
                                           hidden_dim_drug=128,
                                           mpnn_hidden_size=128,
                                           mpnn_depth=4,
                                           cnn_target_filters=[32, 64, 96],
                                           cnn_target_kernels=[4, 8, 12]
                                           )
            model = models.model_initialize(**config)
            # Train model
            model.train(train, val, test)
            # Test model with our custom data
            drug, target, y = test_with_our_testset(DATA_PATH + "new_test.txt")
            X_pred = utils.data_process(drug, target, y,
                                        drug_encoding, target_encoding,
                                        split_method='no_split')
            y_pred_custom = model.predict(X_pred)
            pic50_nrmse, ic50_nrmse = error_metrics(y, y_pred_custom)
            save_to_csv(y, y_pred_custom, file_name = "RESULT_" + sys.argv[1] + "_" + sys.argv[2] + str(drug_encoding)+ "_" + str(target_encoding) + ".csv")
 
            print("DRUG ENCODING:", drug_encoding, "TARGET ENCODING:", target_encoding,
                  "PIC_50 Normalized RMSE:", pic50_nrmse, "IC_50_nM Normalized RMSE:", ic50_nrmse)
            fout.write("DRUG ENCODING:," + str(drug_encoding) + ",TARGET ENCODING:," + str(
                target_encoding) + ",PIC_50 Normalized RMSE:," + str(pic50_nrmse) + ",IC_50_nM Normalized RMSE:," + str(
                ic50_nrmse) + "\n")
            fout_drug.write("DRUG ENCODING:," + str(drug_encoding) + ",TARGET ENCODING:," + str(
                target_encoding) + ",PIC_50 Normalized RMSE:," + str(pic50_nrmse) + ",IC_50_nM Normalized RMSE:," + str(
                ic50_nrmse) + "\n")

        fout_drug.close()
    fout.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

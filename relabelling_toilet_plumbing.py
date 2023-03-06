from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from argparse import ArgumentParser
import warnings
import mlflow
import json


warnings.filterwarnings("ignore")

def log_artifact(df,path,condition):
    df.rename(columns={'index':'class','class':'count'},inplace=True)
    df.to_csv(path+'_'+condition+'_distribution.csv',index=False)
    mlflow.log_artifact(path+'_'+condition+'_distribution.csv')
    os.remove(path+'_'+condition+'_distribution.csv')
    return None



def convert_to_dataframe(filenames, path):
    """
    The function reads the truth labels from txt files and forms a dataframe.
    :param filenames: list
    :param path: str
    :return: gt_df1: dataframe
    """
    gt_df1 = pd.DataFrame()
    content_info = []
    for file in tqdm(filenames):
        with open(os.path.join(path, file)) as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip("\n")
                l = l.split()
                l.insert(0, file)
                content_info.append(l)
                g_df = pd.DataFrame(content_info, columns=["filenames", "class", "x1", "y1", "x2", "y2"])
        li = ["x1", "y1", "x2", "y2"]
        for u in li:
            g_df[u] = g_df[u].astype(float)
            g_df[u] = round(g_df[u], 3)
        gt_df1 = pd.concat([gt_df1, g_df], axis=0, ignore_index=True)
        content_info.clear()
    print(gt_df1['class'].value_counts())
    df = gt_df1['class'].value_counts().reset_index()
    if "train" in path:
        log_artifact(df, "train", "original")
    if "test" in path:
        log_artifact(df, "test", "original")
    if "val" in path:
        log_artifact(df, "val", "original")

    return gt_df1


def relabel_lines_valves(gt_df1,path):
    """
    The function replaces the label values to ensure that the labels could be used for stacked model training.
    :param gt_df1: dataframe
    :return: gt_df1: dataframe
    """
    gt_df1["class"] = gt_df1["class"].str.replace("3", "2", regex=True)
    gt_df1["class"] = gt_df1["class"].str.replace("5", "4", regex=True)
    gt_df1["class"] = gt_df1["class"].str.replace("7", "6", regex=True)
    gt_df1["class"] = gt_df1["class"].str.replace("1", "0", regex=True)
    df = gt_df1['class'].value_counts().reset_index()
    if "train" in path:
        log_artifact(df, "train", "relabelled")
    if "test" in path:
        log_artifact(df, "test", "relabelled")
    if "val" in path:
        log_artifact(df, "val", "relabelled")
    return gt_df1


def recorrect_objects(gt_df1):
    """
    The function replaces the label values to ensure that the labels could be used for stacked model training.
    :param gt_df1: dataframe
    :return: gt_df1: dataframe
    """
    gt_df1["class"] = gt_df1["class"].str.replace("2", "1", regex=True)
    gt_df1["class"] = gt_df1["class"].str.replace("4", "2", regex=True)
    gt_df1["class"] = gt_df1["class"].str.replace("6", "3", regex=True)
    print("relabelled class distribution")
    return gt_df1


def write_label_files(gt_df1, path1):
    """
    The function writes the relabelled values as label txt files to a new path.
    :param gt_df1: dataframe
    :param path1: str
    :return: None
    """
    new_filenames = gt_df1["filenames"].unique()
    for file in tqdm(new_filenames):
        df = gt_df1[gt_df1["filenames"] == file]
        df = df.drop(["filenames"], axis=1)
        df.to_csv(os.path.join(path1, file), sep=" ", header=False, index=False)

def change_folder_names(path):
     if "relabels" in path:
        print("duper")
        old_path = path.split('/')
        old_path = old_path[0:-2]
        old_path[1]='/home'
        old_path=os.path.join(*old_path)
        old_name = "relabels"
        new_name = "labels"
        os.rename(os.path.join(old_path,old_name),os.path.join(old_path,new_name))
     else:
        print("super")
        old_path1 = path.split('/')
        old_path1 = old_path1[0:-2]
        old_path1[1]='/home'
        old_path1 = os.path.join(*old_path1)
        old_name1 = "labels"
        new_name1 = "orig_labels"
        os.rename(os.path.join(old_path1, old_name1), os.path.join(old_path1, new_name1))


def main(args):
    """
    This is the main function which takes parameters from argument parser and calls the functions described below.
    :param args: dict
    :return: None
    """

    mlflow_data=''
    with open(args.mlflow_setting, 'r') as f:
        mlflow_data = json.load(f)
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(mlflow_data['experiment_name'])
    run_id=mlflow_data['run_id']
    mlflow.start_run(run_id=run_id)
    Path(args.relabel_train_path).mkdir(parents=True, exist_ok=True)
    Path(args.relabel_val_path).mkdir(parents=True, exist_ok=True)
    Path(args.relabel_test_path).mkdir(parents=True, exist_ok=True)
    train_filenames = os.listdir(args.train_path)
    gt_train_df = convert_to_dataframe(train_filenames, args.train_path)
    gt_train_df = relabel_lines_valves(gt_train_df,args.train_path)
    gt_train_df = recorrect_objects(gt_train_df)
    # change_folder_names(args.train_path)
    # train_file_path=change_folder_names(args.relabel_train_path)
    print("Writing relabelled train labels")
    write_label_files(gt_train_df,args.relabel_train_path)
    val_filenames = os.listdir(args.val_path)
    gt_val_df = convert_to_dataframe(val_filenames, args.val_path)
    gt_val_df = relabel_lines_valves(gt_val_df,args.val_path)
    gt_val_df = recorrect_objects(gt_val_df)
    # change_folder_names(args.val_path)
    # val_file_path = change_folder_names(args.relabel_val_path)
    print("Writing relabelled val labels")
    write_label_files(gt_val_df, args.relabel_val_path)
    test_filenames = os.listdir(args.test_path)
    gt_test_df = convert_to_dataframe(test_filenames, args.test_path)
    gt_test_df = relabel_lines_valves(gt_test_df,args.test_path)
    gt_test_df = recorrect_objects(gt_test_df)
    # change_folder_names(args.test_path)
    # test_file_path = change_folder_names(args.relabel_test_path)
    print("Writing relabelled test labels")
    write_label_files(gt_test_df, args.relabel_test_path)
    change_folder_names(args.train_path)
    change_folder_names(args.relabel_train_path)
    mlflow.end_run()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/labels/train",
        type=str,
        help="Path to the test label folder",
    )
    parser.add_argument(
        "--val_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/labels/val",
        type=str,
        help="Path to the test label folder",
    )
    parser.add_argument(
        "--test_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/labels/test",
        type=str,
        help="Path to the test label folder",
    )
    parser.add_argument(
        "--relabel_train_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/relabels/train",
        type=str,
        help="Path to save the relabelled train files",
    )
    parser.add_argument(
        "--relabel_val_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/relabels/val",
        type=str,
        help="Path to save the relabelled test files",
    )
    parser.add_argument(
        "--relabel_test_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111/relabels/test",
        type=str,
        help="Path to save the relabelled test files",
    )
    parser.add_argument(
        "--mlflow_setting",
        default='mlflow_setting.json',
        type=str,
        help='JSON file name')
    args = parser.parse_args()
    main(args=args)


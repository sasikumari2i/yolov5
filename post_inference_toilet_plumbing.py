from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np
import warnings
import mlflow
import json
from pathlib import Path
import shutil
warnings.filterwarnings("ignore")


def compute_stats(path):
    """
    The function prepares a dataframe from the labels of ground truth after collating all labels in the file path.
    :param path: str
    :return: gt_df1:dataframe
             len(gt_df1): int
             len(gt_filename): int
    """
    gt_filename = os.listdir(path)
    gt_filename = [file for file in gt_filename if "aug" not in file]
    gt_df1 = pd.DataFrame()
    content_info = []
    for file in gt_filename:
        with open(os.path.join(path, file), "r") as f:
            line = f.readlines()
            for l in line:
                l = l.strip("\n")
                l = l.split()
                l.insert(0, file)
                content_info.append(l)
                g_df = pd.DataFrame(
                    content_info, columns=["file", "class", "x1", "y1", "x2", "y2"]
                )
        li = ["x1", "y1", "x2", "y2"]
        for u in li:
            g_df[u] = g_df[u].astype(float)
            g_df[u] = round(g_df[u], 3)
        gt_df1 = pd.concat([gt_df1, g_df], axis=0, ignore_index=True)
        content_info.clear()
    gt_df1.dropna(inplace=True)
    return len(gt_filename), gt_df1, len(gt_df1)


def compute_mapped_stats(gt_filename, path):
    """
    The function filters the ground truth information according to the number of predictions mapped.
    :param gt_filename: list
    :param path: str
    :return: len(gt_filename): int
             gt_df1: dataframe
             len(gt_df1): int
    """
    gt_filename = [file for file in gt_filename if "aug" not in file]
    gt_df1 = pd.DataFrame()
    content_info = []
    for file in gt_filename:
        with open(os.path.join(path, file), "r") as f:
            line = f.readlines()
            for l in line:
                l = l.strip("\n")
                l = l.split()
                l.insert(0, file)
                content_info.append(l)
                g_df = pd.DataFrame(
                    content_info, columns=["file", "class", "x1", "y1", "x2", "y2"]
                )
        li = ["x1", "y1", "x2", "y2"]
        for u in li:
            g_df[u] = g_df[u].astype(float)
            g_df[u] = round(g_df[u], 3)
        gt_df1 = pd.concat([gt_df1, g_df], axis=0, ignore_index=True)
        content_info.clear()
    gt_df1.dropna(inplace=True)
    return len(gt_filename), gt_df1, len(gt_df1)


def compute_stats1(path):
    """
    The function collates all predicted labels and forms a prediction dataframe.
    :param path: str
    :return: len(pr_filename): int
             pr_df: dataframe
             len(pr_df): int
    """
    pr_df = pd.DataFrame()
    pr_filename = os.listdir(path)
    content_info1 = []
    for file in pr_filename:
        with open(os.path.join(path, file), "r") as f:
            line = f.readlines()
            for l in line:
                l = l.strip("\n")
                l = l.split()
                l.insert(0, file)
                content_info1.append(l)
                p_df = pd.DataFrame(
                    content_info1,
                    columns=["file", "class", "x1", "y1", "x2", "y2", "conf"],
                )
        li = ["x1", "y1", "x2", "y2"]
        for u in li:
            p_df[u] = p_df[u].astype(float)
            p_df[u] = round(p_df[u], 3)
        pr_df = pd.concat([pr_df, p_df], axis=0, ignore_index=True)
        content_info1.clear()
    return len(pr_filename), pr_df, len(pr_df)


def compute_inference_stats(gt_df1, path, path1):
    """
    The function calls routine which will provide prediction dataframe.
    Ensures both prediction dataframe and ground truth dataframe are alinged for comparison.
    :param gt_df1: dataframe
    :param path: str
    :param path1: str
    :return: num_mapped_actual_images: int
             gt_df: dataframe
             num_mapped_actual_instances: int
             num_pr_images: int
             pr_df: dataframe
             num_pred_instances: int
    """
    pr_filename = os.listdir(path)
    gt_filename = gt_df1["file"].unique()
    not_detected = list(set(gt_filename) - set(pr_filename))
    gt_filename = [ele for ele in gt_filename if ele not in not_detected]
    sorted(pr_filename, key=gt_filename.index)
    for i in range(len(gt_filename)):
        assert gt_filename[i] == pr_filename[i]
    num_mapped_actual_images, gt_df, num_mapped_actual_instances = compute_mapped_stats(
        gt_filename, path1
    )
    num_pr_images, pr_df, num_pred_instances = compute_stats1(path)
    return (
        num_mapped_actual_images,
        gt_df,
        num_mapped_actual_instances,
        num_pr_images,
        pr_df,
        num_pred_instances,
    )


def compute_stacked_inference_stats(gt_df1, path, path1):
    """
    The function calls routine which will provide prediction dataframe.
    Ensures both prediction dataframe and ground truth dataframe are alinged for comparison.
    :param gt_df1: dataframe
    :param path: str
    :param path1: str
    :return: num_mapped_actual_images: int
             gt_df: dataframe
             num_mapped_actual_instances: int
             num_pr_images: int
             pr_df: dataframe
             num_pred_instances: int
    """
    pr_df = pd.read_csv(path)
    pr_df["file"] = pr_df["file"].str.replace(".jpg", ".txt")
    pr_filename = pr_df["file"].unique().tolist()
    gt_filename = gt_df1["file"].unique()
    not_detected = list(set(gt_filename) - set(pr_filename))
    gt_filename = [ele for ele in gt_filename if ele not in not_detected]
    pr_name = sorted(pr_filename, key=lambda e: gt_filename.index(e))
    for i in range(len(gt_filename)):
        assert gt_filename[i] == pr_name[i]
    num_mapped_actual_images, gt_df, num_mapped_actual_instances = compute_mapped_stats(
        gt_filename, path1
    )
    num_pr_images = len(pr_name)
    num_pred_instances = len(pr_df)
    return (
        num_mapped_actual_images,
        gt_df,
        num_mapped_actual_instances,
        num_pr_images,
        pr_df,
        num_pred_instances,
    )


def organize_dataframe(hdf):
    """
    The function orgnaizes the dataframe facilitating instance to instance mapping.
    :param hdf: dataframe
    :return: hdf_c: dataframe
    """
    clist = []
    pr_clas = []
    for index, row in hdf.iterrows():
        if len(row["gt_label"]) > len(row["pr_class"]):
            while len(row["pr_class"]) < len(row["gt_label"]):
                row["pr_class"] += ["50"]
        row["gt_label"].sort()
        row["pr_class"] = [str(u) for u in row["pr_class"]]
        row["pr_class"].sort()
        if len(row["gt_label"]) < len(row["pr_class"]):
            while len(row["pr_class"]) > len(row["gt_label"]):
                row["gt_label"].sort()
                row["pr_class"].sort()
                row["pr_class"].pop()
        clist.append((row["gt_file"], row["gt_label"], row["pr_class"]))
    hdf_c = pd.DataFrame(clist)
    hdf_c.columns = hdf.columns
    return hdf_c


def reorganize_dataframes(gt_df, pr_df):
    """
    The function concats both ground truth and predicted dataframe and ensures easy instance mapping.
    :param gt_df: dataframe
    :param pr_df: dataframe
    :return: compre_df:dataframe
             mdf: dataframe
    """
    gt = gt_df[["file", "class"]]
    gt_grp = gt.groupby("file")["class"].apply(list).reset_index()
    pr = pr_df[["file", "class"]]
    pr_grp = pr.groupby("file")["class"].apply(list).reset_index()
    gt_grp.sort_values("file", inplace=True)
    pr_grp.sort_values("file", inplace=True)
    mdf = pd.concat([gt_grp, pr_grp], axis=1, ignore_index=True)
    mdf.rename(
        columns={0: "gt_file", 1: "gt_label", 2: "pr_file", 3: "pr_class"}, inplace=True
    )
    hdf = mdf[["gt_file", "gt_label", "pr_class"]]
    compre_df = organize_dataframe(hdf)
    return compre_df, mdf

def add_images(df,path):
    df['images'] = df['file'].str.replace('.txt','.jpg')
    df['images'] = path+df["images"]
    return df


def compute_business_metrics(mdf,path):
    """
    The function computes the business metric.
    :param mdf: dataframe
    :return: matched: dataframe
             not_matched:dataframe
    """
    li = []
    li1 = []
    matched = pd.DataFrame()
    not_matched = pd.DataFrame()
    for index, row in mdf.iterrows():
        x, y = row["gt_label"], row["pr_class"]
        x = [int(i) for i in x]
        y = [int(i) for i in y]
        x.sort()
        y.sort()
        if x == y:
            li1.append((row["gt_file"], x, y))
            gh_y = pd.DataFrame(li1, columns=["file", "actual", "predicted"])
            matched = pd.concat([matched, gh_y], axis=0, ignore_index=True)
            li1.clear()
        else:
            li.append((row["gt_file"], x, y))
            gh_y1 = pd.DataFrame(li, columns=["file", "actual", "predicted"])
            not_matched = pd.concat([not_matched, gh_y1], axis=0, ignore_index=True)
            li.clear()
    matched=add_images(matched,path)
    not_matched=add_images(not_matched,path)
    return matched, not_matched


def compute_instances_images_matched(
    num_actual_images,
    num_actual_instances,
    num_mapped_actual_images,
    num_mapped_actual_instances,
    hdf,
    mdf,
    images_path
):
    """
    The function prints all the metrics related to inference.
    :param num_actual_images: int
    :param num_actual_instances: int
    :param num_mapped_actual_images: int
    :param num_mapped_actual_instances: int
    :param hdf: dataframe
    :param mdf: dataframe
    :return:detected :dataframe
    """
    ydf = hdf.set_index(["gt_file"]).apply(pd.Series.explode).reset_index()
    ydf["match"] = np.where(ydf["gt_label"] == ydf["pr_class"], "yes", "no")
    print("Number of Ground Truth Images:---------", num_actual_images)
    print("Number of Predicted Images:----------", num_mapped_actual_images)
    print(
        "Number of Non Detected Images:--------",
        num_actual_images - num_mapped_actual_images,
    )
    print("Number of Actual Instances:------", num_actual_instances)
    print("Number of Predicted Instances:-------", num_mapped_actual_instances)
    print(
        "Percentage Instance Detected:---------",
        100 * ((num_mapped_actual_instances) / (num_actual_instances)),
    )
    # print("Number of Instance Matched:--------", ydf.match.value_counts())
    print("*" * 120)
    matched, not_matched = compute_business_metrics(mdf,images_path)
    print("Number of Images Matched:---------", len(matched))
    print("The Business KPI:-----------", 100 * (len(matched) / num_actual_images))
    mlflow.log_metric("Stacked_Model_KPI_V4",100 * (len(matched) / num_actual_images))
    return matched, not_matched

def extract_not_matched_images(df):
    files=df['images'].values.tolist()
    Path('missed_detections').mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy(file,'missed_detections')
    print("completed the copying of missed detections")


def main(args):
    """
    The main function takes in parameters from the argument parser and can run in 2 ways. One for a
    generic inference and another way for stacked inference depending on parameter stacked-model. It saves the
    respective inference files as comma seperated files in the same directory.
    :param args: dict
    :return: None
    """
    mlflow_data = ''
    with open(args.mlflow_setting, 'r') as f:
        mlflow_data = json.load(f)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = mlflow_data['experiment_name']
    mlflow.set_experiment(experiment_name)
    run_id = mlflow_data['run_id']
    mlflow.start_run(run_id=run_id)
    if args.stacked_model == "no":
        num_actual_images, gt_df1, num_actual_instances = compute_stats(
            args.ground_truth_label_file_path
        )
        (
            num_mapped_actual_images,
            gt_mapped_df1,
            num_mapped_actual_instances,
            num_predicted_images,
            pr_df,
            num_predicted_instances,
        ) = compute_inference_stats(
            gt_df1, args.predicted_label_file_path, args.ground_truth_label_file_path
        )
        compre_df, mdf = reorganize_dataframes(gt_mapped_df1, pr_df)
        matched, not_matched = compute_instances_images_matched(
            num_actual_images,
            num_actual_instances,
            num_mapped_actual_images,
            num_predicted_instances,
            compre_df,
            mdf,
        )
        print("saving images which had exact match")
        matched.to_csv("matched.csv")
        print("saving images which did not have exact match")
        not_matched.to_csv("not_matched.csv")
    else:
        num_actual_images_s, gt_df1, num_actual_instances_s = compute_stats(
            args.ground_truth_label_file_path
        )
        (
            num_mapped_actual_images_s,
            gt_mapped_df1_s,
            num_mapped_actual_instances_s,
            num_predicted_images_s,
            pr_df_s,
            num_predicted_instances_s,
        ) = compute_stacked_inference_stats(
            gt_df1, args.stacked_model_file, args.ground_truth_label_file_path
        )
        print(
            num_mapped_actual_images_s,
            num_mapped_actual_instances_s,
            num_predicted_images_s,
            num_predicted_instances_s,
        )
        compre_df_s, mdf_s = reorganize_dataframes(gt_mapped_df1_s, pr_df_s)
        matched, not_matched = compute_instances_images_matched(
            num_actual_images_s,
            num_actual_instances_s,
            num_mapped_actual_images_s,
            num_predicted_instances_s,
            compre_df_s,
            mdf_s,
            args.images_file_path,
        )
        print("saving images which had exact match")
        matched.to_csv("stack_matched.csv")
        mlflow.log_artifact("stack_matched.csv",artifact_path='predicted_outcomes')
        print("saving images which did not have exact match")
        not_matched.to_csv("stack_not_matched.csv")
        mlflow.log_artifact("stack_not_matched.csv",artifact_path="predicted_outcomes")
        extract_not_matched_images(not_matched)
        mlflow.end_run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--images_file_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/trial2/images/test/",
        type=str,
        help="Path to the images test files",
    )
    parser.add_argument(
        "--ground_truth_label_file_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/trial2/orig_labels/test",
        type=str,
        help="Path to the ground truth label file",
    )
    parser.add_argument(
        "--predicted_label_file_path",
        default="/content/yolov5/runs/detect/exp2/labels",
        type=str,
        help="Path to the predicted label folder",
    )
    parser.add_argument(
        "--stacked_model",
        default="yes",
        type=str,
        help="Whether stacked model is going to be used to generate outputs",
    )
    parser.add_argument(
        "--stacked_model_file",
        default="toilet_plumbing_recent.csv",
        type=str,
        help="Name of the Stacked model prediction file",
    )
    parser.add_argument(
        "--mlflow_setting",
        default='mlflow_setting.json',
        type=str,
        help='JSON file name')
    args = parser.parse_args()
    main(args=args)

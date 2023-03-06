import shutil
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
from tqdm import tqdm
from argparse import ArgumentParser
from aug_images import aug_image, save_augmentations
import cv2
import warnings
import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch import nn, optim
from jcopdl.callback import Callback, set_config
from torch.utils.data import DataLoader
import pandas as pd
import mlflow
import json
import torchmetrics
from torchsummary import summary

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_ground_truth_file(path):
    """
    The function collates all the label files present in a folder and converts it into single dataframe.
    :param path: str
    :return: gt_df1: dataframe
    """
    gt_df1 = pd.DataFrame()
    content_info = []
    labelnames = os.listdir(path)
    for label in labelnames:
        with open(os.path.join(path, label)) as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip("\n")
                l = l.split()
                l.insert(0, label)
                content_info.append(l)
                g_df = pd.DataFrame(content_info, columns=["label", "class", "x1", "y1", "x2", "y2"])
        li = ["x1", "y1", "x2", "y2"]
        for u in li:
            g_df[u] = g_df[u].astype(float)
            g_df[u] = round(g_df[u], 3)
        gt_df1 = pd.concat([gt_df1, g_df], axis=0, ignore_index=True)
        content_info.clear()
    return gt_df1


def map_dataframe(df, supply_line_path, shutoff_valve_path, tankbolt_path, coupler_path):
    """
    Maps the corresponding folder paths for supply lines and shutoff valves with filesnames present in
    the dataframe.
    :param df: dataframe
    :param supply_line_path: str
    :param shutoff_valve_path: str
    :return: df: dataframe
    """
    df["label"] = df["label"].str.replace(".txt", ".jpg", regex=True)
    df.loc[((df["class"] == "0") | (df["class"] == "1")), "path"] = shutoff_valve_path
    df.loc[((df["class"] == "2") | (df["class"] == "3")), "path"] = supply_line_path
    df.loc[((df["class"] == "4") | (df["class"] == "5")), "path"] = tankbolt_path
    df.loc[((df["class"] == "6") | (df["class"] == "7")), "path"] = coupler_path
    return df


def identify_minority_images(df):
    """
    The function filters minority class from the dataframe, which subsequently would be augmented.
    :param df: dataframe
    :return: minority_df: dataframe
    """
    minority_df = df[(df["class"] == "1") | (df["class"] == "3") | (df["class"] == "5") | (df["class"] == "7")]
    return minority_df


def perform_augmentation(augmented_df, source_image_path, destination_folder, num_augmentation):
    """
    The function calls the augmentation function and performs augmentation of images.
    At present augmentation of minority class is only done.
    :param augmented_df: dataframe
    :param source_image_path: str
    :param destination_folder: str
    :return: aug_df: dataframe
    """
    augmented_df.rename(
        columns={"img_path": "filename", "scaled_xmin": "xmin", "scaled_ymin": "ymin", "scaled_xmax": "xmax", "scaled_ymax": "ymax", "img_width": "width", "img_height": "height"}, inplace=True
    )
    print(augmented_df.columns)
    augmented_df.drop(["label", "x1", "y1", "x2", "y2", "path"], axis=1, inplace=True)
    augmented_df.reset_index(drop=True, inplace=True)
    resize = False
    new_shape = (512, 512)
    output_folder = os.path.join(destination_folder, "aug")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    img_list = augmented_df.filename.unique()
    cols = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    aug_df = pd.DataFrame(columns=cols)
    print(augmented_df["class"].value_counts())
    print("Generating augmented images")
    for filename in tqdm(img_list):
        # augment image
        category = augmented_df.loc[augmented_df["filename"] == filename, "class"]
        category = list(set(category))
        if len(category) > 1:
            print(f"Skipping file - {filename} , since it has multiple categories - {category}")
            continue
        assert len(category) == 1
        category = category[0]
        num_of_augmentations = int(num_augmentation)
        aug_images, aug_bbs = aug_image(filename, augmented_df, source_image_path, num_of_augmentations)
        aug_df = save_augmentations(aug_images, aug_bbs, aug_df, filename, output_folder, resize, new_shape)
    return aug_df


def add_image_width_height(df):
    """
    The function adds image width and height to each file/image present in the dataframe.
    Augmentation needs the dataframe/csv to be in a specific structure. Width and Height of image is a must.
    :param df: dataframe
    :return: df: dataframe
    """
    df["img_path"] = df["path"] + "/" + df["label"]
    height_list = []
    width_list = []
    for row in df["img_path"]:
        img = cv2.imread(row)
        if img is None:
            height = np.nan
            width = np.nan
        else:
            height, width = img.shape[:2]
        height_list.append(height)
        width_list.append(width)
    df["img_height"] = height_list
    df["img_width"] = width_list
    df.dropna(inplace=True, axis=0)
    return df


def scale_bbox_values(df):
    """
    The function scales the bounding box coordinates.
    :param df: dataframe
    :return: df: dataframe
    """
    df["scaled_xmin"] = df["x1"] * df["img_width"]
    df["scaled_ymin"] = df["y1"] * df["img_height"]
    df["scaled_xmax"] = df["x2"] * df["img_width"]
    df["scaled_ymax"] = df["y2"] * df["img_height"]
    return df


def identify_majority_images(mapped_dataframe):
    """
    The function filters the majority class from the dataframe, which will not be augmented.
    :param mapped_dataframe: dataframe
    :return: df: dataframe
    """
    majority_df = mapped_dataframe[(mapped_dataframe["class"] == "0") | (mapped_dataframe["class"] == "2") | (mapped_dataframe["class"] == "4") | (mapped_dataframe["class"] == "6")]
    majority_df["filename"] = majority_df["path"] + "/" + majority_df["label"]
    df = majority_df[["filename", "class"]]
    return df


def split_dataset(df_new):
    """
    The function splits the given dataframe into 3 namely train dataframe,test dataframe and valid dataframe.
    The ratio of split is 80-10-10.
    :param df: dataframe
    :return: train_df: dataframe
             test_df: dataframe
             val_df: dataframe
    """
    print(df_new["class"].value_counts())
    sorted_class_df = df_new["class"].value_counts().reset_index()
    sorted_class_df.rename(columns={"class": "count", "index": "class"}, inplace=True)
    least_samples_list = sorted_class_df[sorted_class_df["count"] == 1]["class"].values
    print(least_samples_list)
    least_samples_df = df_new[df_new["class"].isin(least_samples_list)]
    df_new = df_new[~df_new["class"].isin(least_samples_list)]
    print("after----->", df_new["class"].value_counts())
    train_df, test_df = train_test_split(df_new, test_size=0.05, random_state=0, stratify=df_new[["class"]])
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=0, stratify=train_df[["class"]])
    print(train_df.shape, test_df.shape, val_df.shape)
    return train_df, test_df, val_df


def copy_file(df, path):
    """
    The function helps to copy the files from parent folder to another folder with a specific data structure.
    This is essentially done to reorganize the image contents.
    :param df: dataframe
    :param path: str
    :return: None
    """
    print(df["class"].value_counts())
    classes = df["class"].unique()
    print("classes---------------", classes)
    for cl in classes:
        path1 = os.path.join(path, cl)
        print(path1)
        df1 = df[df["class"] == cl]
        for row in df1["renamed_filename"]:
            print(row)
            img = row.split("/")[-1]
            my_file = Path(row)
            if my_file.is_file():
                shutil.copy(row, os.path.join(path1, img))
            else:
                continue
    print("Copying of files completed")


def convert_image_2_array(path):
    """
    The function converts the images into numpy array.
    :param path: str
    :return: image_array : numpy array
    """
    image_array = []
    for folder in os.listdir(path):
        sub_path = path + "/" + folder
        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            if img_arr is None:
                continue
            img_arr = cv2.resize(img_arr, (224, 224))
            image_array.append(img_arr)
    return image_array


def normalize_array(x_train, x_val, x_test):
    """
    Normalizes the given image input.
    :param x_train: list
    :param x_val: list
    :param x_test: list
    :return: train_x: numpy array
             test_x: numpy array
             val_x:  numpy array
    """
    train_x = np.array(x_train)
    test_x = np.array(x_test)
    val_x = np.array(x_val)
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    val_x = val_x / 255.0
    return train_x, val_x, test_x


def map_test_files(test_set):
    """
    The function does mapping of the datafiles and targets it represents.
    :param test_set: DirectoryIterator
    :return: test_dataframe : dataframe
    """
    testfiles = test_set.filenames
    files = []
    classes = []
    test_dataframe = pd.DataFrame()
    for file in testfiles:
        fileparts = file.split("/")
        file_name = fileparts[1]
        clas = fileparts[0]
        files.append(file_name)
        classes.append(clas)
    test_dataframe["filename"] = files
    test_dataframe["class"] = classes
    print(test_dataframe["class"].value_counts())
    return test_dataframe


def rename_filenames(df):
    """
    The function renames files that were augmented.
    :param df: dataframe
    :return: df: dataframe
    """
    df.dropna(inplace=True)
    rename_list = df["filename"].tolist()
    renamed_list = []
    for file in rename_list:
        term = "jpg"
        c = Counter([file[i : i + len(term)] for i in range(len(file))])
        if c[term] > 1:
            file1 = file.replace(".jpg", "_jpg", 1)
        else:
            file1 = file
        renamed_list.append(file1)
    df["renamed_filename"] = renamed_list
    return df


class CustomMobileNetv2(nn.Module):
    """
    Class definition of the MobileNetV2 Model.
    """

    def __init__(self, output_size=2):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.freeze()
        self.mnet.classifier = nn.Sequential(nn.Linear(1280, output_size), nn.LogSoftmax(1))

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True


def data_loaders(train_path, valid_path, test_path, bs, crop_size):
    """
    The function prepares the data that can be easily consumed by the model.
    :param train_path: str
    :param valid_path: str
    :param test_path: str
    :return:
            train_x : numpy array
            val_x: numpy array
            test_x: numpy array
            test_dataframe: dataframe
            train_y: numpy array
            val_y: numpy array
            test_y: numpy array
    """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((crop_size, crop_size)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((crop_size, crop_size)),
        ]
    )
    train_set = datasets.ImageFolder(train_path, transform=train_transform)
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_set = datasets.ImageFolder(valid_path, transform=test_transform)
    valloader = DataLoader(val_set, batch_size=bs, shuffle=True)
    test_set = datasets.ImageFolder(test_path, transform=test_transform)
    testloader = DataLoader(test_set, shuffle=True)
    return train_set, val_set, test_set, trainloader, valloader, testloader


def prepare_data_for_model(input_df, path, bs, crop_size):
    """
    The function takes dataframe containing image file names as input and splits the data into train,test and val.
    Maps the classes and copies the file into the folders that keras model mandates.
    :param input_df: dataframe
    :return: output of dataloader function
    """
    input_df = rename_filenames(input_df)
    train, test, val = split_dataset(input_df)
    print("aaaaa--------------->", len(train), len(test), len(val))
    train["class"] = train["class"].astype(str).astype(int)
    test["class"] = test["class"].astype(str).astype(int)
    val["class"] = val["class"].astype(str).astype(int)
    di = {0: "GOOD", 1: "BAD", 2: "GOOD", 3: "BAD", 4: "GOOD", 5: "BAD", 6: "GOOD", 7: "BAD"}
    train["class"] = train["class"].map(di)
    test["class"] = test["class"].map(di)
    val["class"] = val["class"].map(di)
    print(train.dtypes)
    print(train["class"].value_counts())
    print("*" * 75)
    print(test["class"].value_counts())
    print("*" * 75)
    print(val["class"].value_counts())
    copy_file(train, os.path.join(path, "train"))
    copy_file(test, os.path.join(path, "test"))
    copy_file(val, os.path.join(path, "val"))
    train_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    return data_loaders(train_path, valid_path, test_path, bs, crop_size)


def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device, epoch):
    """
    The function performs the training and validation of the model.

    :param mode: str ('train'/'val')
    :param dataset: torchvision.datasets.folder.ImageFolder
    :param dataloader: torch.utils.data.dataloader.DataLoader
    :param model: __main__.CustomMobileNetv2
    :param criterion: torch.nn.modules.loss.CrossEntropyLoss
    :param optimizer: torch.optim.adamw.AdamW
    :param device: torch.device
    :return: cost : flaot
             acc: float
    """
    if mode == "train":
        model.train()
    elif mode == "val":
        model.eval()

    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        # target = target.unsqueeze(1)
        # target = target.float()
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)

        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    print(f"{mode.title()} - Epoch: {epoch}, Cost: {cost:.4f}, Accuracy: {acc:.4f}")
    return cost, acc


def train_val_model(train_set, val_set, trainloader, valloader, model, criterion, optimizer, device, callback):
    print("welcome to training")
    num_epochs = 1

    for epoch in range(num_epochs):
        train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device, epoch)
        test_cost, test_score = loop_fn("val", val_set, valloader, model, criterion, optimizer, device, epoch)

        # Logging
        callback.log(train_cost, test_cost, train_score, test_score)
        mlflow.log_metric("classifier_train_cost",float(train_cost),step=epoch)
        mlflow.log_metric("classifier_train_score",float(train_cost),step=epoch)
        mlflow.log_metric("classifier_val_cost",float(train_cost),step=epoch)
        mlflow.log_metric("classifier_val_score",float(train_score),step=epoch)

        # Checkpoint
        callback.save_checkpoint()

        # Runtime Plotting
        callback.cost_runtime_plotting()
        callback.score_runtime_plotting()

        # Early Stopping
        if callback.early_stopping(model, monitor="test_score"):
            mlflow.log_param("classifier_training_epochs",epoch)
            callback.plot_cost()
            callback.plot_score()
            break
    pass


def model_activity(model,train_set, val_set, test_set, trainloader, valloader, testloader, config):
    """
    The function calls different functions related to model building,model summary, training, plotting
    model loss and accuracy and reports related to classification and computes business metrics.
    :param train_x: numpy array
    :param val_x: numpy array
    :param test_x: numpy array
    :param test_dataframe: dataframe
    :param train_y: numpy array
    :param val_y: numpy array
    :param test_y: numpy array
    :return: None
    """



    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    criterion_name = str(type(criterion)).split(".")[-1][:-2]
    # Withoout class weights
    # criterion = FocalLoss(gamma=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = str(type(optimizer)).split(".")[-1][:-2]
    callback = Callback(model, config, early_stop_patience=5, outdir="model")
    patience = callback.early_stop_patience
    mlflow.log_param("classifier_lr",float(lr))
    mlflow.log_param("classifier_optimizer",optimizer_name)
    mlflow.log_param("classifier_criterion_name",criterion_name)
    mlflow.log_param("classifier_early_stop_patience",patience)
    mlflow.log_param("classifier_device",device)
    train_val_model(train_set, val_set, trainloader, valloader, model, criterion, optimizer, device, callback)
    pass

def rename_augmented_files(path):
    """
    The function renames the augmented files (since it has 2 '.' in the filename)
    :param path: str
    :return:None
    """
    filenames1 = os.listdir(path)
    for file in filenames1:
        from collections import Counter

        term = "jpg"
        c = Counter([file[i : i + len(term)] for i in range(len(file))])
        if c[term] > 1:
            file1 = file.replace(".jpg", "_jpg", 1)
        else:
            file1 = file
        my_file = Path(os.path.join(path, file))
        if my_file.is_file():
            os.rename(os.path.join(path, file), os.path.join(path, file1))
        else:
            continue
    print("Completed Renaming Augmented Files in Cropped Image Folder")
    pass

def plot_roc(roc_auc):
    '''
    The function plots the roc curve for the model and logs the artifact into mlflow server.
    Args:
        roc_auc:

    Returns:

    '''
    fpr, tpr, thresholds = roc_auc.compute()
    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    fig = plt.gcf()
    fig.savefig("Classifier_ROC.jpg")
    mlflow.log_artifact("Classifier_ROC.jpg", artifact_path="plots")
    plt.close(fig)

def plot_cm(confusion_matrix,label2cat):
    '''
    The function computes confusion matrix and plots the same and logs it into mlflow server.
    Args:
        confusion_matrix:
        label2cat:

    Returns:

    '''
    confusion_matrix_tensor = confusion_matrix.compute()
    confusion_matrix_numpy = confusion_matrix_tensor.numpy().astype(int)
    labels = label2cat
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(confusion_matrix_numpy, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # Loop over the data and add annotations to the heatmap
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix_numpy[i, j],
                           ha='center', va='center', color='black')
    plt.title('Confusion Matrix')
    plt.show()
    fig.savefig("Classifier_Confusion_Matrix.jpg")
    mlflow.log_artifact("Classifier_Confusion_Matrix.jpg", artifact_path="plots")
    plt.close()



def predict(model, testloader, device, label2cat):
    """
    The function loads the model and provides prediction for the test data with labels.
    :param model:__main__.CustomMobileNetv2
    :param testloader: torch.utils.data.dataloader.DataLoader
    :param device:torch.device
    :param label2cat: list
    :return: None
    """
    accuracy=[]
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2,task='binary')
    roc_auc = torchmetrics.ROC(num_classes=2,task='binary')
    model.eval()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(testloader):
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            y_pred_probs = torch.softmax(outputs, dim=1)
            y_pred_pos = y_pred_probs[:, 1]
            y_pred = torch.argmax(y_pred_probs, dim=1)
            confusion_matrix.update(y_pred,classes)
            roc_auc.update(y_pred_pos,classes)
            if classes == preds:
                acc = 1
            else:
                acc = 0
            accuracy.append(acc)
    accuracy = np.array(accuracy)
    print("Accuracy--------------->:",accuracy.mean())
    mlflow.log_metric("Classifier_Accuracy",accuracy.mean())
    plot_roc(roc_auc)
    plot_cm(confusion_matrix,label2cat)




def main(args):
    """
    This function is the main function which takes input parameters from the argument parser and calls functions described below.
    :param args: dict
    :return: None
    """
    mlflow_data = ''
    with open(args.mlflow_setting, 'r') as f:
        mlflow_data = json.load(f)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = mlflow_data['experiment_name']
    mlflow.set_experiment(experiment_name)
    run_id=mlflow_data['run_id']
    mlflow.start_run(run_id=run_id)
    args = vars(args)
    Path(args["destination_file_path"], "train/GOOD").mkdir(parents=True, exist_ok=True)
    Path(args["destination_file_path"], "train/BAD").mkdir(parents=True, exist_ok=True)
    Path(args["destination_file_path"], "test/GOOD").mkdir(parents=True, exist_ok=True)
    Path(args["destination_file_path"], "test/BAD").mkdir(parents=True, exist_ok=True)
    Path(args["destination_file_path"], "val/GOOD").mkdir(parents=True, exist_ok=True)
    Path(args["destination_file_path"], "val/BAD").mkdir(parents=True, exist_ok=True)
    print("Reading Ground Truth Labels")
    ground_truth_df = read_ground_truth_file(args["src_ground_truth_labels_file_path"])
    print("Mapping Ground Truth Label DataFrame With The Cropped Image Paths")
    mapped_dataframe = map_dataframe(
        ground_truth_df,
        args["crop_supply_line_images_file_path"],
        args["crop_shutoff_valve_images_file_path"],
        args["crop_toilet_tankbolt_images_file_path"],
        args["crop_toilet_coupler_images_file_path"],
    )
    print("Filtering Minority Label Images")
    minority_df = identify_minority_images(mapped_dataframe)
    print("Filtering Good Objects")
    majority_df = identify_majority_images(mapped_dataframe)
    print("Reading cropped image dimensions and adding the information")
    minority_df = add_image_width_height(minority_df)
    print("Scaling BBOX Co_ordinates of Minority Dataframe")
    minority_df = scale_bbox_values(minority_df)
    print(minority_df.columns)
    print("Performing Augmentation Of Minority Images")
    minority_df1 = minority_df[minority_df["class"] == "1"]
    minority_df3 = minority_df[minority_df["class"] == "3"]
    minority_df5 = minority_df[minority_df["class"] == "5"]
    minority_df7 = minority_df[minority_df["class"] == "7"]
    augment_success_df1 = perform_augmentation(minority_df1, args["crop_shutoff_valve_images_file_path"], args["destination_file_path"], args["num_augmentation_bad_shutoff_valve"])
    augment_success_df3 = perform_augmentation(minority_df3, args["crop_supply_line_images_file_path"], args["destination_file_path"], args["num_augmentation_bad_supply_line"])
    augment_success_df5 = perform_augmentation(minority_df7, args["crop_toilet_tankbolt_images_file_path"], args["destination_file_path"], args["num_augmentation_bad_tankbolt"])
    augment_success_df7 = perform_augmentation(minority_df5, args["crop_toilet_coupler_images_file_path"], args["destination_file_path"], args["num_augmentation_bad_coupler"])
    augment_success_df = pd.concat([augment_success_df1, augment_success_df3, augment_success_df5, augment_success_df7], axis=0)
    augment_success_df1 = augment_success_df[["filename", "class"]]
    compre_df = pd.concat([majority_df, augment_success_df1], axis=0)
    rename_augmented_files(args["crop_supply_line_images_file_path"])
    rename_augmented_files(args["crop_shutoff_valve_images_file_path"])
    rename_augmented_files(args["crop_toilet_tankbolt_images_file_path"])
    rename_augmented_files(args["crop_toilet_coupler_images_file_path"])
    bs = 32
    crop_size = 224
    train_set, val_set, test_set, trainloader, valloader, testloader = prepare_data_for_model(compre_df, args["destination_file_path"], bs, crop_size)
    config = set_config({"batch_size": bs, "crop_size": crop_size, "output_size": len(train_set.classes)})
    config_dict = config.__dict__
    for k,v in config_dict.items():
        k="classifier_"+str(k)
        mlflow.log_param(k,v)
        # build the model with pretrained weights
    model = CustomMobileNetv2(config.output_size).to(device)
    input_size = (3, crop_size, crop_size)  # replace with your model's input shape
    file_name="classifier_model_summary.txt"
    with open(file_name,'w') as fp:
        sys.stdout = fp
        summary(model,input_size=input_size,device=device.type)
        sys.stdout = sys.__stdout__
    mlflow.log_artifact(file_name,artifact_path='model_info')
    model_activity(model,train_set, val_set, test_set, trainloader, valloader, testloader, config)
    model = CustomMobileNetv2()
    model.load_state_dict(torch.load("model/weights_best.pth", map_location=device))
    label2cat = train_set.classes
    predict(model, testloader, device, label2cat)
    # mlflow.end_run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src_ground_truth_labels_file_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/trial2/orig_labels/test",
        type=str,
        help="Path to the ground truth label file",
    )
    parser.add_argument(
        "--crop_shutoff_valve_images_file_path",
        default="/home/karthik/yolov5/runs/detect/exp/crops/SHUTOFF_VALVE",
        type=str,
        help="Path to the cropped shutoff valve image file",
    )
    parser.add_argument(
        "--crop_supply_line_images_file_path",
        default="/home/karthik/yolov5/runs/detect/exp/crops/SUPPLY_LINE",
        type=str,
        help="Path to the cropped supply lines image file",
    )
    parser.add_argument(
        "--crop_toilet_coupler_images_file_path",
        default="/home/karthik/yolov5/runs/detect/exp/crops/TOILET_COUPLER",
        type=str,
        help="Path to the cropped coupler image file",
    )
    parser.add_argument(
        "--crop_toilet_tankbolt_images_file_path",
        default="/home/karthik/yolov5/runs/detect/exp/crops/TOILET_TANKBOLT",
        type=str,
        help="Path to the cropped tank bolt image file",
    )

    parser.add_argument(
        "--destination_file_path",
        default="/home/karthik/yolov5/runs/detect/exp",
        type=str,
        help="Path to the folder where data is stored in accordance to the pytorch input file structure",
    )
    parser.add_argument(
        "--num_augmentation_bad_shutoff_valve",
        default=1,
        type=int,
        help="Number of augmentations to be generated for each bad shutoff valve image",
    )
    parser.add_argument(
        "--num_augmentation_bad_supply_line",
        default=1,
        type=int,
        help="Number of augmentations to be generated for each bad supply line image",
    )
    parser.add_argument(
        "--num_augmentation_bad_tankbolt",
        default=1,
        type=int,
        help="Number of augmentations to be generated for each bad tank bolt image",
    )
    parser.add_argument(
        "--num_augmentation_bad_coupler",
        default=1,
        type=int,
        help="Number of augmentations to be generated for each bad coupler image",
    )
    parser.add_argument(
        "--mlflow_setting",
        default='mlflow_setting.json',
        type=str,
        help='JSON file name')
    args = parser.parse_args()
    main(args=args)

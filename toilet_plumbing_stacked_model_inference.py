from argparse import ArgumentParser
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch import nn
from torchvision.models import mobilenet_v2
import mlflow
import json


class CustomMobileNetv2(nn.Module):
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


def load_models(path1, path2):
    """
    The function loads the object detection and classifier model.
    :param path1:str
    :param path2:str
    :return:
           object_detection_model:models.common.AutoShape
           classifier:__main__.CustomMobileNetv2
    """
    object_detection_model = torch.hub.load("ultralytics/yolov5", "custom", path=path1)
    classifier = CustomMobileNetv2()
    classifier.load_state_dict(torch.load(path2, map_location="cpu"))
    return object_detection_model, classifier


def model_inference(detector, classifier, path, iou, conf):
    """
    The function provides inference by stacking the object detection and classifier.
    Returns the inference as a dataframe along wit number of images and number of instances.
    :param detector:models.common.AutoShape
    :param classifier:tensorflow.python.keras.engine.functional.Functional
    :param path: str
    :param iou: str
    :param conf: str
    :return:
           df: dataframe
           c: int
           ins: int
    """
    images = os.listdir(path)
    response = []
    c = 0
    ins = 0
    for img in tqdm(images):
        filename = img
        c += 1
        imgss = [cv2.imread(path + img)[..., ::-1]]
        detector.conf = float(conf)
        detector.iou = float(iou)
        result = detector(imgss, augment=True)
        mapping = {
            0: "BAD",
            1: "GOOD",
        }
        object_result = result.pandas().xyxy[0]
        ins += len(result.xyxy[0])
        for i in range(0, len(result.xyxy[0])):
            bbox_raw = result.xyxy[0][i]
            object_name = object_result["name"].iloc[i]
            bbox = []
            for bound in bbox_raw:
                bbox.append(int(bound.item()))
                bbox = bbox[:4]
            image = imgss[0]
            new_image = image.copy()
            cropped_image = new_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            im = Image.fromarray((cropped_image * 1).astype(np.uint8)).convert("RGB")
            # im=Image.fromarray(cropped_image)
            input_tensor = preprocess(im)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to("cuda")
                classifier.to("cuda")
            classifier.eval()

            with torch.no_grad():
                output = classifier(input_batch)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = mapping[torch.argmax(probabilities).item()]
            pred_class = str(pred_class) + "_" + str(object_name)
            print(pred_class)
            result_dict = {"file": filename, "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3], "confidence": round(bbox_raw[4].item(), 2), "class": pred_class}
            response.append(result_dict)
    pred_df = pd.DataFrame(response)
    return pred_df, c, ins


def convert_class_int(df):
    """
    The function converts the class names into integer representation of classes.
    :param df: dataframe
    :return:df: dataframe
    """
    mapping = {
        "GOOD_SHUTOFF_VALVE": 0,
        "BAD_SHUTOFF_VALVE": 1,
        "GOOD_SUPPLY_LINE": 2,
        "BAD_SUPPLY_LINE": 3,
        "GOOD_TOILET_TANKBOLT": 4,
        "BAD_TOILET_TANKBOLT": 5,
        "GOOD_TOILET_COUPLER": 6,
        "BAD_TOILET_COUPLER": 7,
    }
    df = df.applymap(lambda s: mapping.get(s) if s in mapping else s)
    print(df["class"].value_counts())
    return df


def main(args):
    """
    This is the main function that takes the parameters from argument parser and calls the functions described below.
    Saves the inference as a comma seperated file in the same directory.
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
    detector, classifier = load_models(args.object_detection_model_path, args.classifier_model)
    df, num_img_files, num_instances = model_inference(detector, classifier, args.images_path, args.iou, args.conf)
    print("No of images files:----------", num_img_files)
    print("No of Instances Detected:-------------", num_instances)
    mlflow.log_metric("Stacked_Model_Number_Images_Detected",num_img_files)
    mlflow.log_metric("Stacked_Model_Instance_Detected",num_instances)
    df = convert_class_int(df)
    df.to_csv(args.save_inference_file_path_name)
    mlflow.end_run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--object_detection_model_path",
        default="/home/karthik/Downloads/tp_stacked/best.pt",
        type=str,
        help="Path to the object detection model",
    )

    parser.add_argument(
        "--classifier_model",
        default="/home/karthik/Downloads/tp_stacked/weights_best.pth",
        type=str,
        help="Path to the classification model file",
    )

    parser.add_argument(
        "--images_path",
        default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/trial2/images/test/",
        type=str,
        help="Path to the image files",
    )

    parser.add_argument(
        "--iou",
        default=0.5,
        type=str,
        help="Path to the folder where data is stored in accordance to the pytorch input file structure",
    )

    parser.add_argument(
        "--conf",
        default=0.7,
        type=str,
        help="Model confidence level for predicted objects",
    )
    parser.add_argument(
        "--save_inference_file_path_name",
        default="toilet_plumbing_recent.csv",
        type=str,
        help="Path to which the inference file will be saved",
    )
    parser.add_argument(
        "--mlflow_setting",
        default='mlflow_setting.json',
        type=str,
        help='JSON file name')

    args = parser.parse_args()
    main(args=args)

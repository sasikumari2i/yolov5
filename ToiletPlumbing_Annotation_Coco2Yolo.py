import cv2
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from pathlib import Path
from aug_images import aug_image, save_augmentations
from rebox import BBox
from rebox.formats import yolo, pascal, coco
import warnings
import json
import shutil
import mlflow
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")




def read_annotation_file(src_annotation_file_path: str):
    """
    The function reads the annotation file and extract details related to annotated
    images,values(bbox,area etc) and also the category.
    :param src_annotation_file_path: str
    :return: annotated images:list
             annotated_values:list
             annotated_categories:list
    """
    if not os.path.exists(src_annotation_file_path):
        raise FileNotFoundError(f"{src_annotation_file_path} not found")
    with open(src_annotation_file_path) as fp:
        annotation_dict = json.load(fp)
    annotated_images = annotation_dict["images"]
    annotated_values = annotation_dict["annotations"]
    annotated_categories = annotation_dict["categories"]
    return annotated_images, annotated_values, annotated_categories


def create_image_mapping(annotated_images: list, src_image_folder: str):
    """
    The function creates a map between the image files in the source and the annotated image id.
    :param annotated_images: list
    :param src_image_folder: str
    :return: annotated_images: list
    """
    for ann_details in tqdm(annotated_images):
        del ann_details["coco_url"]
        del ann_details["date_captured"]
        del ann_details["absolute_url"]
        # Rename the image with only filename
        ann_details["file_name"] = ann_details["file_name"].split("/")[-1]
        # Create filename to save each image with unique number
        ann_details["target_image_name"] = f"img{ann_details['id']}.jpg"
        # Rectify images where width or height value is missing
        if ann_details["width"] == 0 or ann_details["height"] == 0:
            image_full_path = os.path.join(src_image_folder, ann_details["file_name"])
            if not os.path.exists(image_full_path):
                raise FileNotFoundError(f"{image_full_path} not found")
            image_cv2 = cv2.imread(image_full_path)
            ann_details["width"], ann_details["height"], _ = image_cv2.shape
    return annotated_images


def get_image_annotation(annotated_images: list, annotated_values: list):
    """
    The function re-adjusts the category_id,re-organizes the list.
    :param annotated_images: list
    :param annotated_values: list
    :return: annotated_images: list
    """
    not_found = []
    for image_details in tqdm(annotated_images):
        image_id = image_details["id"]
        ann_list = []
        found = False
        categories = []
        areas = []
        annotation_ids = []
        for ann_details in annotated_values:
            if ann_details["image_id"] == image_id:
                areas.append(ann_details["area"])
                annotation_ids.append(ann_details["id"])
                categories.append(ann_details["category_id"] - 1)
                ann_list.append(ann_details)
                found = True
        image_details["ann_list"] = ann_list
        image_details["categories"] = categories
        image_details["areas"] = areas
        image_details["annotation_ids"] = annotation_ids
        if not found:
            not_found.append(image_id)
    print(f"Total images: {len(annotated_images)}")
    print(f"Not Found: {len(not_found)}")
    assert len(not_found) == 0
    return annotated_images


def identify_tobe_augmented(df, categories, percentage):
    """
    The function reads the dataframe and relabels the bounding box for a category as belonging to a new category
    using the percentile of ratio of area bbox and image. Threshold for relabelling as been kept as 5%.
    :param df: dataframe
    :param categories: list
    :return: df: dataframe
    """
    for category_id in categories:
        sub_df = df[df.category_id == category_id]
        sub_df["bbox_image_ratio_perc"] = sub_df["bbox_image_ratio"].rank(pct=True)
        sub_df.sort_values(["bbox_image_ratio_perc"], inplace=True)
        category_4_details = sub_df.loc[sub_df.bbox_image_ratio_perc < float(percentage), ["image_id", "id"]]
        df.loc[category_4_details.index, "category_id"] = 8
    return df


def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    """
    The function converts the coco format bbox co-ordinates as yolo-format
    co-ordinates using rebox utility.
    :param x1: float
    :param y1: float
    :param w: float
    :param h: float
    :param image_w: float
    :param image_h: float
    :return: coco_bbox.as_format(yolo, image_w, image_h).value ---(x_center,y_center,w,h):(float,loat,float,float)
    """
    coco_bbox = BBox([x1, y1, w, h], coco)
    return coco_bbox.as_format(yolo, image_w, image_h).value


def pascal_to_yolo(x1, y1, w, h, image_w, image_h):
    """
    The function converts pascal format cooridnates to yolo-format using the rebox utility.
    :param x1: float
    :param y1: float
    :param w: float
    :param h: float
    :param image_w: float
    :param image_h: float
    :return: pascal_bbox.as_format(yolo, image_w, image_h).value ----(x_center,y_center,w,h):(float,float,float,float)
    """
    pascal_bbox = BBox([x1, y1, w, h], pascal)
    return pascal_bbox.as_format(yolo, image_w, image_h).value


def convert_to_yolo_format_for_augmented_images(image_df):
    """
    The function calls the yolo format cnverter to convert the bbox details present in the augmented images.
    The bbox details of augmented images would be in pascal format.
    :param image_df: dataframe
    :return: image_df: dataframe
    """
    yolo_list = []
    x_center_list = []
    y_center_list = []
    w_list = []
    h_list = []
    for index, image_details in tqdm(image_df.iterrows()):
        img_w = image_details["width"]
        img_h = image_details["height"]
        yolo_str = ""
        x_min = image_details["xmin"]
        y_min = image_details["ymin"]
        x_max = image_details["xmax"]
        y_max = image_details["ymax"]
        current_category = image_details["category_id"]
        print(x_min,y_min,x_max,y_max,img_w,img_h)
        x_center, y_center, w, h = pascal_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
        x_center = format(x_center, ".6f")
        y_center = format(y_center, ".6f")
        w = format(w, ".6f")
        h = format(h, ".6f")
        yolo_str += f"{current_category} {x_center} {y_center} {w} {h}\n"
        yolo_list.append(yolo_str)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        w_list.append(w)
        h_list.append(h)
    image_df["yolo_str"] = yolo_list
    image_df["x_center"] = x_center_list
    image_df["y_center"] = y_center_list
    image_df["w"] = w_list
    image_df["h"] = h_list
    image_df["x_center"] = image_df["x_center"].astype(float)
    image_df["y_center"] = image_df["y_center"].astype(float)
    image_df["w"] = image_df["w"].astype(float)
    image_df["h"] = image_df["h"].astype(float)
    return image_df


def convert_to_yolo_format(df):
    """
    The function would call the yolo converter for converting bbox details of non-augmented images to
    yolo-format. The non augmented images bbox co-ordinates would be in coco-format.
    :param df: dataframe
    :return: df: dataframe
    """
    df["scaled_xmin"] = df["xmin"] * df["width"]
    df["scaled_ymin"] = df["ymin"] * df["height"]
    df["scaled_xmax"] = df["xmax"] * df["width"]
    df["scaled_ymax"] = df["ymax"] * df["height"]
    yolo_list = []
    x_center_list = []
    y_center_list = []
    w_list = []
    h_list = []
    for index, image_details in tqdm(df.iterrows()):
        yolo_str = ""
        x_min = image_details["scaled_xmin"]
        y_min = image_details["scaled_ymin"]
        x_max = image_details["scaled_xmax"]
        y_max = image_details["scaled_ymax"]
        img_w = image_details["width"]
        img_h = image_details["height"]
        current_category = image_details["category_id"]
        x_center, y_center, w, h = coco_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
        x_center = format(x_center, ".6f")
        y_center = format(y_center, ".6f")
        w = format(w, ".6f")
        h = format(h, ".6f")
        yolo_str += f"{current_category} {x_center} {y_center} {w} {h}\n"
        yolo_list.append(yolo_str)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        w_list.append(w)
        h_list.append(h)
    df["yolo_str"] = yolo_list
    df["x_center"] = x_center_list
    df["y_center"] = y_center_list
    df["w"] = w_list
    df["h"] = h_list
    df["x_center"] = df["x_center"].astype(float)
    df["y_center"] = df["y_center"].astype(float)
    df["w"] = df["w"].astype(float)
    df["h"] = df["h"].astype(float)
    return df


def copy_images_and_annotations_to_dest(source_image_path, image_df, dest_image_path, dest_annotation_path):
    """
    The function copies all the augmented images and their corresponding labels to the given destination folder.
    :param source_image_path: str
    :param image_df: dataframe
    :param dest_image_path: str
    :param dest_annotation_path: str
    :return: None
    """
    Path(dest_image_path).mkdir(parents=True, exist_ok=True)
    Path(dest_annotation_path).mkdir(parents=True, exist_ok=True)
    for index, row in tqdm(image_df.iterrows()):
        source_image_name = row["file_name"]
        source_full_path = os.path.join(source_image_path, source_image_name)
        dest_image_name = row["target_image_name"]
        dest_full_path = os.path.join(dest_image_path, dest_image_name)
        if not os.path.exists(source_full_path):
            raise FileNotFoundError(f"{source_full_path} not found")
        else:
            shutil.copy(source_full_path, dest_full_path)
        yolo_str = row["yolo_str"]
        target_annotation_name = row["target_annotation_name"]
        annotation_full_path = os.path.join(dest_annotation_path, target_annotation_name)
        with open(annotation_full_path, "w") as file_object:
            file_object.write(yolo_str)


def split_dataset(df,split_size=0.10, column_name="class_name", by_category=True):
    """
    The function splits the dataset into train,test and val.
    :param df: dataframe
    :param column_name: str
    :param by_category: boolean
    :return: train_df: dataframe
             test_df:  dataframe
             val_df:   dataframe
    """
    if by_category:
        df["sorted_categories"] = df.category_id.apply(lambda x: sorted(x))
        df["sorted_categories"] = df["sorted_categories"].astype(str)
        sorted_categories_df = df["sorted_categories"].value_counts().reset_index()
        least_samples_list = sorted_categories_df[sorted_categories_df.sorted_categories == 1]["index"]
        least_samples_df = df[df.sorted_categories.isin(least_samples_list)]
        df = df[~df.index.isin(least_samples_df.index)]
    train_df, test_df = train_test_split(df, test_size=split_size, random_state=0, stratify=df[[column_name]])
    train_df, val_df = train_test_split(train_df, test_size=split_size, random_state=0, stratify=train_df[[column_name]])
    if by_category:
        train_df = pd.concat([train_df, least_samples_df], axis=0)
    print(train_df.shape, val_df.shape)

    test_df["minority_present_1"] = test_df.category_id.apply(lambda x: check_if_one_minority_classes_present(x))
    test_df["minority_present_3"] = test_df.category_id.apply(lambda x: check_if_three_minority_classes_present(x))
    train_df = pd.concat([train_df, test_df[test_df["minority_present_1"] == True], test_df[test_df["minority_present_3"] == True]], axis=0)
    test_df = test_df.loc[~(test_df["minority_present_1"] == True) & ~(test_df["minority_present_3"] == True)]
    print(test_df.shape)
    print("*" * 120)
    return train_df, val_df, test_df


def convert_image_details_to_df(annotated_images):
    """
    The function converts annotated image details in a list to dataframe.
    :param annotated_images: list
    :return: image_df: dataframe
    """
    column_names = annotated_images[0].keys()
    image_df = pd.DataFrame(annotated_images, columns=column_names)
    image_df.rename(columns={"id": "image_id"}, inplace=True)
    return image_df


def convert_annotation_details_to_df(annotated_values):
    """
    The function converts the annotated values present as list to a dataframe.
    :param annotated_values: list
    :return: annotation_df: dataframe
    """
    column_names = annotated_values[0].keys()
    annotation_df = pd.DataFrame(annotated_values, columns=column_names)
    annotation_columns = ["id", "category_id", "image_id", "bbox"]
    annotation_df = annotation_df[annotation_columns]
    annotation_df[["xmin", "ymin", "xmax", "ymax"]] = pd.DataFrame(annotation_df.bbox.tolist(), index=annotation_df.index)
    annotation_df["category_id"] -= 1
    annotation_df.drop(columns=["bbox"], inplace=True, axis=1)
    return annotation_df


def identify_image_class(image_df, annotation_df):
    """
    The function creates a map between the image_details (image_df) and annotated_values(annotation_df)
    by merging them and returns the resultant dataframe.
    :param image_df: dataframe
    :param annotation_df: dataframe
    :return: image_df: dataframe
    """
    image_category_df = annotation_df.groupby(["image_id"])["category_id"].unique().reset_index()
    image_category_df["sorted_categories"] = image_category_df.category_id.apply(lambda x: sorted(x))
    total_image_rows = image_df.shape[0]
    image_df = image_df.merge(image_category_df, how="inner", on="image_id")
    assert image_df.shape[0] == total_image_rows
    return image_df


def expand_df_into_multiple_rows(image_df):
    """
    The function explodes the categories column of the image_df to ensure that
    each catgeory present in an image file corresponds to a row in the dataframe.
    :param image_df: dataframe
    :return: image_df: dataframe
    """
    print(image_df.columns)
    image_df = image_df.explode(["categories", "ann_list"]).reset_index(drop=True)
    print("After exploding Shape: ", image_df.shape)
    print(image_df.class_name.value_counts())
    return image_df


def split_bounding_box_coordinates(df):
    """
    The function reorganizes the given datarame essentially to ensure that nested list of bbox co-ordinates
    are split into seperate columns.
    :param df: dataframe
    :return: df: dataframe
    """
    df["bbox"] = df["ann_list"].apply(lambda x: x["bbox"])
    bbox_df = pd.DataFrame(df["bbox"].to_list(), columns=["xmin", "ymin", "xmax", "ymax"])
    df = pd.concat([df, bbox_df], axis=1)
    return df


def calculate_bbox_details(image_df, annotation_df):
    """
    The function computes the bbox area,image area, and ratio of the two after scaling the bbox co-ordinates.
    :param image_df: dataframe
    :param annotation_df: dataframe
    :return: dataframe
    """
    annotation_df = annotation_df.merge(image_df[["image_id", "file_name", "width", "height"]], how="inner", on="image_id")
    annotation_df["image_area"] = annotation_df["width"] * annotation_df["height"]
    annotation_df["bbox_area"] = (annotation_df["xmax"] * annotation_df["width"]) * (annotation_df["ymax"] * annotation_df["height"])
    annotation_df["bbox_image_ratio"] = annotation_df["bbox_area"] / annotation_df["image_area"]
    return annotation_df


def check_if_both_minority_classes_present(categories):
    """
    The function identifies if both the minority classes are present in the same image.
    :param categories: list
    :return: True/False: boolean
    """
    if 1 in categories and 3 in categories:
        return True
    else:
        return False


def check_if_one_minority_classes_present(categories):
    """
    The function identifies if both the minority classes are present in the same image.
    :param categories: list
    :return: True/False: boolean
    """
    if 1 in categories and 3 not in categories:
        return True
    else:
        return False


def check_if_three_minority_classes_present(categories):
    """
    The function identifies if both the minority classes are present in the same image.
    :param categories: list
    :return: True/False: boolean
    """
    if 3 in categories and 1 not in categories:
        return True
    else:
        return False


def find_rows_to_augment(image_df, annotation_df):
    """
    The function identifies the rows(images) that need to be augmented.
    It also splits the original dataframe as df to be augmented and df not to be augmented.
    :param image_df: dataframe
    :param annotation_df: dataframe
    :return: to_augment_df: dataframe
             do_not_augment_df: dataframe
    """
    image_df["is_1_and_3_present"] = image_df.category_id.apply(lambda x: check_if_both_minority_classes_present(x))
    print(image_df.is_1_and_3_present.value_counts())
    minority_image_ids = image_df.loc[image_df.is_1_and_3_present == True, "image_id"]
    # image_df["is_1_present"] = image_df.category_id.apply(lambda x: check_if_one_minority_classes_present(x))
    print(image_df.is_1_and_3_present.value_counts())
    minority_image_ids = image_df.loc[image_df.is_1_and_3_present == True, "image_id"]
    total_rows = annotation_df.shape[0]
    minority_df = annotation_df.loc[annotation_df.image_id.isin(minority_image_ids)]
    annotation_df = annotation_df.loc[~annotation_df.image_id.isin(minority_image_ids)]
    to_augment_df = annotation_df.loc[~annotation_df.category_id.isin([0, 2, 4])]
    print("Rows to augment: ", to_augment_df.shape)
    do_not_augment_df = annotation_df.loc[annotation_df.category_id.isin([0, 2, 4])]
    do_not_augment_df = pd.concat([do_not_augment_df, minority_df], axis=0)
    print("Rows not to augment: ", do_not_augment_df.shape)
    assert total_rows == (to_augment_df.shape[0] + do_not_augment_df.shape[0])
    return to_augment_df, do_not_augment_df


def convert_variables(df):
    """
    convert the bbox co-ordinates to pascal format. This is done for images that needs to be augmented.
    The augmentation function accepts in pascal format.
    :param df: dataframe
    :return: df: dataframe
    """
    df["xmax"] = df["xmax"] + df["xmin"]
    df["ymax"] = df["ymax"] + df["ymin"]
    return df


def generate_augmentation_details(to_augment_df):
    """
    The function reorganizes the dataframe and scales the bbox co-ordinates before augmentation.
    :param to_augment_df: dataframe
    :return: to_augment_df: dataframe
    """
    augment_columns = ["file_name", "width", "height", "category_id", "xmin", "ymin", "xmax", "ymax"]
    to_augment_df = to_augment_df[augment_columns]
    to_augment_df.rename(columns={"category_id": "class", "file_name": "filename"}, inplace=True)
    to_augment_df["xmin"] *= to_augment_df["width"]
    to_augment_df["ymin"] *= to_augment_df["height"]
    to_augment_df["xmax"] *= to_augment_df["width"]
    to_augment_df["ymax"] *= to_augment_df["height"]
    return to_augment_df


def perform_augmentation(mappings, augmented_df, source_image_path, destination_folder, num_augmentation):
    """
    The function calls the augmentation function and performs augmentation of images.
    At present augmentation of minority class is only done.
    :param augmented_df: dataframe
    :param source_image_path: str
    :param destination_folder: str
    :return: aug_df: dataframe
    """
    resize = False
    new_shape = (512, 512)
    output_folder = os.path.join(destination_folder, "aug")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    img_list = augmented_df.filename.unique()
    aug_df = pd.DataFrame(columns=augmented_df.columns.tolist())
    print("Generating augmented images")
    if num_augmentation>1:
        for filename in tqdm(img_list):
            # augment image
            category = augmented_df.loc[augmented_df["filename"] == filename, "class"]
            # print(category)
            category = list(set(category))
            if len(category) > 1:
                print(f"Skipping file - {filename} , since it has multiple categories - {category}")
                continue
            assert len(category) == 1
            category = category[0]
            if num_augmentation>1:
                aug_images, aug_bbs = aug_image(filename, augmented_df, source_image_path, num_augmentation)
                aug_df = save_augmentations(aug_images, aug_bbs, aug_df, filename, output_folder, resize, new_shape)
                return aug_df
    return augmented_df

def compute_splitfraction(size):
    train_init_split = 1.00 - size
    test_split = size
    val_split = train_init_split * size
    train_split = train_init_split - val_split
    di={"train":train_split,"val":val_split,"test":test_split}
    with open('split_fraction.json', 'w') as fp:
        json.dump(di, fp)
    mlflow.log_artifact("split_fraction.json")
    path=os.getcwd()
    os.remove(os.path.join(path,"split_fraction.json"))
    return None

def collate_augmented_dataframes(mappings,to_augment_df,args):
    print(args)
    compre_augment_success_df_1 = perform_augmentation(
        mappings,
        to_augment_df,
        args["src_image_folder"],
        args["destination_folder"],
        args["num_augmentation_shut_off_valve"],
    )
    compre_augment_success_df_3 = perform_augmentation(
        mappings,
        to_augment_df,
        args["src_image_folder"],
        args["destination_folder"],
        args["num_augmentation_supply_line"],
    )
    compre_augment_success_df_5 = perform_augmentation(
        mappings,
        to_augment_df,
        args["src_image_folder"],
        args["destination_folder"],
        args["num_augmentation_bolt"],
    )
    compre_augment_success_df_7 = perform_augmentation(
        mappings,
        to_augment_df,
        args["src_image_folder"],
        args["destination_folder"],
        args["num_augmentation_coupler"],
    )
    compre_augment_sucess_df=pd.concat([compre_augment_success_df_1,compre_augment_success_df_3,compre_augment_success_df_5,compre_augment_success_df_7],axis=1)
    if len(compre_augment_sucess_df)>0:
        return compre_augment_sucess_df
    else:
        return to_augment_df





def main(args):
    """
    The Main function which recieves the annotation-file-path,source-image-path and destination-path.
    The function ends by copying all the images and labels into the destination path.
    :param args: str
    :return: None
    """
    args = vars(args)
    experiment_name = args['mlflow_experiment_name']
    mlflow.set_experiment(experiment_name)
    run_name=args['mlflow_run_name']
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print("Current run ID:", run_id)
        mlflow_setting = {"experiment_name": experiment_name,"run_name":run_name,"run_id":run_id}
        with open('mlflow_setting.json', 'w') as fp:
            json.dump(mlflow_setting, fp)

        mappings={0:'GOOD SHUT OFF VALVE',1:'BAD SHUT OFF VALVE',2: 'GOOD SUPPLY LINE',3:'BAD SUPPLY LINE',4:'GOOD TANK BOLT',5:'BAD TANK BOLT',6: 'GOOD TOILET COUPLER',7:'BAD TOILET COUPLER'}
        with open('class_description.json', 'w') as fp:
            json.dump(mappings, fp)
        mlflow.log_artifact("class_description.json")
        path=os.getcwd()
        os.remove(os.path.join(path,"class_description.json"))
        print("Reading annotation file")
        annotated_images, annotated_values, annotated_categories = read_annotation_file(src_annotation_file_path=args["src_annotation_file_path"])
        print("Capturing image details")
        annotated_images = create_image_mapping(annotated_images=annotated_images, src_image_folder=args["src_image_folder"])
        print("converting image details to dataframe")
        image_df = convert_image_details_to_df(annotated_images)
        print("Image DataFrame: ", image_df.shape)
        print("converting annotation details to dataframe")
        annotation_df = convert_annotation_details_to_df(annotated_values)
        print("Annotation df:", annotation_df.shape)
        print("Identify image class")
        image_df = identify_image_class(image_df=image_df, annotation_df=annotation_df)
        print("Calculate bounding box area")
        annotation_df = calculate_bbox_details(image_df=image_df, annotation_df=annotation_df)
        # print("Creating new categories for good supply lines")
        print(annotation_df.shape)
        # annotation_df = identify_tobe_augmented(df=annotation_df, categories=[4], percentage=args["percentage_supply_lines"])
        do_not_augment_df = annotation_df
        do_not_augment_df = convert_to_yolo_format(do_not_augment_df)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(do_not_augment_df.describe())
        yolo_str_df = do_not_augment_df.groupby(["image_id"])["yolo_str"].apply("".join).reset_index()
        category_df = do_not_augment_df.groupby(["image_id"])["category_id"].apply(list).reset_index()
        do_not_augment_df = yolo_str_df.merge(category_df, on="image_id", how="inner")
        do_not_augment_df["target_image_name"] = str("img") + do_not_augment_df["image_id"].astype(str) + ".jpg"
        do_not_augment_df["target_annotation_name"] = str("img") + do_not_augment_df["image_id"].astype(str) + ".txt"
        do_not_augment_df = do_not_augment_df.merge(image_df[["image_id", "file_name"]], on="image_id", how="left")
        # split the augmented images into train, test and validation
        compute_splitfraction(args['split_size'])
        train_df, val_df, test_df = split_dataset(do_not_augment_df,split_size=args['split_size'],column_name="sorted_categories")
        print("Not augmented Split: ", train_df.shape, val_df.shape, test_df.shape)
        di2 = {"train_size": train_df.shape[0], "val_size": val_df.shape[0], "test_size": test_df.shape[0]}
        with open('data_split.json', 'w') as fp:
            json.dump(di2, fp)
        mlflow.log_artifact("data_split.json")
        print("Copying images and annotations to destination")
        copy_images_and_annotations_to_dest(
            source_image_path=args["src_image_folder"],
            image_df=train_df,
            dest_image_path=os.path.join(args["destination_folder"], "images", "train"),
            dest_annotation_path=os.path.join(args["destination_folder"], "labels", "train"),
        )
        copy_images_and_annotations_to_dest(
            source_image_path=args["src_image_folder"],
            image_df=test_df,
            dest_image_path=os.path.join(args["destination_folder"], "images", "test"),
            dest_annotation_path=os.path.join(args["destination_folder"], "labels", "test"),
        )
        copy_images_and_annotations_to_dest(
            source_image_path=args["src_image_folder"],
            image_df=val_df,
            dest_image_path=os.path.join(args["destination_folder"], "images", "val"),
            dest_annotation_path=os.path.join(args["destination_folder"], "labels", "val"),
        )


if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument(
            "--src_annotation_file_path",
            default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/TOILET_PLUMBING_V1.json",
            type=str,
            help="Path to the annotation file",
        )
        parser.add_argument(
            "--src_image_folder",
            default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/TP_data/images",
            type=str,
            help="Path to the image folder",
        )
        parser.add_argument(
            "--destination_folder",
            default="/home/karthik/Desktop/chrp_deliverables/chrp_datasets/toilet_plumbing/tp_model_input_data_sasi111",
            type=str,
            help="Path to save the image and annotation in yolo format",
        )
        parser.add_argument(
            "--percentage_supply_lines",
            default=0,
            type=float,
            help="Percentage of supply lines to be relabelled.",
        )
        parser.add_argument(
            "--split_size",
            default=0.10,
            type=float,
            help="Percentage of supply lines to be relabelled.",
        )
        parser.add_argument(
            "--mlflow_experiment_name",
            default="Toilet_Plumbing_Now",
            type=str,
            help="Name of the mlflow experiment run",
        )
        parser.add_argument(
            "--mlflow_run_name",
            default="trial1",
            type=str,
            help="Name of the mlflow experiment run",
        )
        args = parser.parse_args()
        print(args)
        main(args=args)


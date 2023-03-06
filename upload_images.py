import os
from azure.storage.blob import BlockBlobService

root_path = '/home/karthik/input_data/images'
dir_name = 'train'
path = f"{root_path}/{dir_name}"
file_names = os.listdir(path)

account_name = '<your account name>'
account_key = '<your account key>'
container_name = '<your container name, such as `test` for me>'

block_blob_service = BlockBlobService(
    account_name=account_name,
    account_key=account_key
)

for file_name in file_names:
    blob_name = f"{dir_name}/{file_name}"
    file_path = f"{path}/{file_name}"
    block_blob_service.create_blob_from_path(container_name, blob_name, file_path)
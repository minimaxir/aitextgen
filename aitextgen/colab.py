import sys
import shutil
import os
import tarfile

try:
    from google.colab import drive
except ImportError:
    pass


def mount_gdrive():
    """Mounts the user's Google Drive in Colaboratory."""
    assert (
        "google.colab" in sys.modules
    ), "You must be in Colaboratory to mount your Google Drive"

    drive.mount("/content/drive")


def is_mounted():
    """Checks if the Google Drive is mounted."""
    assert os.path.isdir("/content/drive"), "You must mount first using mount_gdrive()"


def get_tarfile_name(model_folder):
    """Converts a folder path into a filename for a .tar archive"""
    tarfile_name = model_folder.replace(os.path.sep, "_") + ".tar"

    return tarfile_name


def copy_model_to_gdrive(model_folder="aitextgen", copy_folder=False):
    """Copies the model folder to a mounted Google Drive."""
    is_mounted()

    if copy_folder:
        shutil.copytree(model_folder, "/content/drive/My Drive/" + model_folder)
    else:
        file_path = get_tarfile_name(model_folder)

        # Reference: https://stackoverflow.com/a/17081026
        with tarfile.open(file_path, "w") as tar:
            tar.add(model_folder)

        shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_model_from_gdrive(model_folder="aitextgen", copy_folder=False):
    """Copies the model folder from a mounted Google Drive."""
    is_mounted()

    if copy_folder:
        shutil.copytree("/content/drive/My Drive/" + model_folder, model_folder)
    else:
        file_path = get_tarfile_name(model_folder)

        shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)

        with tarfile.open(file_path, "r") as tar:
            tar.extractall()


def copy_file_to_gdrive(file_path):
    """Copies a file to a mounted Google Drive."""
    is_mounted()

    shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_file_from_gdrive(file_path):
    """Copies a file from a mounted Google Drive."""
    is_mounted()

    shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)

import sys
import shutil
import os

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


def copy_file_to_gdrive(file_path, to_folder=None):
    """Copies a file to a mounted Google Drive."""
    is_mounted()

    if to_folder:
        dest_path = os.path.join("/content/drive/My Drive/", to_folder, file_path)
    else:
        dest_path = os.path.join("/content/drive/My Drive/", file_path)

    shutil.copyfile(file_path, dest_path)


def copy_file_from_gdrive(file_path, from_folder=None):
    """Copies a file from a mounted Google Drive."""
    is_mounted()

    if from_folder:
        source_path = os.path.join("/content/drive/My Drive/", from_folder, file_path)
    else:
        source_path = os.path.join("/content/drive/My Drive/", file_path)

    shutil.copyfile(source_path, file_path)


def create_gdrive_folder(folder_name):
    """Creates a folder in a mounted Google Drive."""
    is_mounted()

    folder_path = os.path.join("/content/drive/My Drive/", folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

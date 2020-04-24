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


def get_tarfile_name(checkpoint_folder):
    """Converts a folder path into a filename for a .tar archive"""
    tarfile_name = checkpoint_folder.replace(os.path.sep, "_") + ".tar"

    return tarfile_name


def copy_checkpoint_to_gdrive(run_name="run1", copy_folder=False):
    """Copies the checkpoint folder to a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join("checkpoint", run_name)

    if copy_folder:
        shutil.copytree(
            checkpoint_folder, "/content/drive/My Drive/" + checkpoint_folder
        )
    else:
        file_path = get_tarfile_name(checkpoint_folder)

        # Reference: https://stackoverflow.com/a/17081026
        with tarfile.open(file_path, "w") as tar:
            tar.add(checkpoint_folder)

        shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_checkpoint_from_gdrive(run_name="run1", copy_folder=False):
    """Copies the checkpoint folder from a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join("checkpoint", run_name)

    if copy_folder:
        shutil.copytree(
            "/content/drive/My Drive/" + checkpoint_folder, checkpoint_folder
        )
    else:
        file_path = get_tarfile_name(checkpoint_folder)

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

import os
import errno
import argparse
import shutil
import subprocess
import stat
from pathlib import Path

REPO_URL = "https://github.com/MorrisCheng0414/RSNA2023-Challenge-Pretrain-Weights.git"

def git_lfs_clone(repo_url, dest_path):
    """
    Clones the pretrain weights from Github using git lfs clone.
    
    Args:
        repo_url (str): The URL of the Git repository.
        dest_path (Path): The path where the repository should be cloned. 
    """
    command = ["git", "lfs", "clone", repo_url, str(dest_path)]

    try:
        # Execute the git lfs clone command
        subprocess.run(command,
                       check=True,
                       capture_output=True,
                       text=True,
                       shell=False) # shell=False is safer and recommended
        print("Clone successful.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during cloning:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: 'git' or 'git lfs' command not found. Please ensure they are installed and in your PATH.")
        return False

def remove_readonly_and_retry(func, path, exc_info):
    """
    Error handler for shutil.rmtree that changes a file's permission
    from read-only to writable and retries the deletion.
    """
    if not os.path.exists(path):
        return

    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Failed to delete {path} after changing permissions: {e}")
        raise

def main():
    ### Part 1: Setup and Cloning ###
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description = "Downloads, moves, and cleans up pretrain weight files.")
    parser.add_argument("--repo_dest", default = str(script_dir),
                        help = "The destination folder of the cloned repository")
    parser.add_argument("--repo_name", default = "RSNA2023-Challenge-Pretrain-Weights",
                        help = "The name of the new directory where the repo will be cloned")
    args = parser.parse_args()

    dest_path = Path(args.repo_dest) / args.repo_name

    if dest_path.exists():
        print(f"Error: The destination folder '{dest_path}' already exists.")
        print("Please delete the existing folder or choose a different name.")
        return False

    # Perform the clone with git lfs
    if not git_lfs_clone(REPO_URL, dest_path):
        return False
    
    ### Part 2: Moving the Weight Folders ###
    try:
        print("\nMoving pretrain weights folders...")
        
        repo_root = dest_path # Correct path is the destination itself
        
        weight_folder_2d_src = repo_root / "2d_pretrain_weights"
        weight_folder_3d_src = repo_root / "3d_pretrain_weights"
        
        weight_folder_2d_dest = script_dir / "2d_pretrain" / "pretrain_weights"
        weight_folder_3d_dest = script_dir / "3d_pretrain" / "pretrain_weights"

        if not weight_folder_2d_src.exists():
            raise FileNotFoundError(f"Source folder not found: {weight_folder_2d_src}")
        
        if not weight_folder_3d_src.exists():
            raise FileNotFoundError(f"Source folder not found: {weight_folder_3d_src}")

        shutil.move(weight_folder_2d_src, weight_folder_2d_dest)
        shutil.move(weight_folder_3d_src, weight_folder_3d_dest)
        
        print("Weights moved successfully.")

    except (FileNotFoundError, shutil.Error) as e:
        print(f"Error during move operation: {e}")
        return False

    ### Part 3: Deleting the Repository Folder ###
    print("\nStarting cleanup...")
    if dest_path.exists():
        try:
            shutil.rmtree(dest_path, onerror=remove_readonly_and_retry)
            print(f"Successfully deleted the cloned folder '{dest_path}'.")
        except OSError as e:
            print(f"Error deleting folder '{dest_path}': {e}")
    else:
        print(f"Cleanup skipped: '{dest_path}' not found.")

if __name__ == "__main__":
    main()
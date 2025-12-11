import os
import gdown

def download_weights():
    """Download model weights from Google Drive"""
    
    # Create weights directory if it doesn't exist
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Google Drive file IDs and corresponding filenames
    downloads = [
        {
            "file_id": "1LzZJal2BMjFcOpwI9uOsOC0IQcZVzmEP",
            "filename": "multilevel_tempsal.pt",
            "url": "https://drive.google.com/uc?id=1LzZJal2BMjFcOpwI9uOsOC0IQcZVzmEP",
            "directory": "../src/model/TempSal/weights/"
        },
        {
            "file_id": "14czAAQQcRLGeiddPOM6AaTJTieu6QiHy",
            "filename": "TranSalNet_Res.pth",
            "url": "https://drive.google.com/uc?id=14czAAQQcRLGeiddPOM6AaTJTieu6QiHy",
            "directory": "../src/model/TranSalNet/weights/"
        },
        {
            "file_id": "1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T",
            "filename": "TranSalNet_Dense.pth",
            "url": "https://drive.google.com/uc?id=1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T",
            "directory": "../src/model/TranSalNet/weights/"
        }
    ]
    
    for download in downloads:
        # Create the target directory if it doesn't exist
        target_dir = download["directory"]
        os.makedirs(target_dir, exist_ok=True)
        
        output_path = os.path.join(target_dir, download["filename"])
        
        if os.path.exists(output_path):
            print(f"File {download['filename']} already exists, skipping...")
            continue
            
        print(f"Downloading {download['filename']}...")
        try:
            gdown.download(download["url"], output_path, quiet=False)
            print(f"Successfully downloaded {download['filename']}")
        except Exception as e:
            print(f"Error downloading {download['filename']}: {str(e)}")

if __name__ == "__main__":
    download_weights()
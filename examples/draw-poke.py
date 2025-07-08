# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.55.0",
#     "mopaint==0.2.1",
#     "numpy==2.3.1",
#     "pillow==11.3.0",
#     "requests==2.32.4",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.14.9"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from mopaint import Paint
    return (Paint,)


@app.cell
def _(Paint, mo):
    paint_widget = mo.ui.anywidget(Paint(height=400))
    paint_widget
    return (paint_widget,)


@app.cell
def _(encoder, paint_widget):
    query = encoder.transform([paint_widget.get_pil()])[0]
    return (query,)


@app.cell
def _(X_image, image_dict, mo, np, query):
    # Calculate cosine similarity between query and each row in X_image
    query_norm = query / np.linalg.norm(query)
    X_image_norm = X_image / np.linalg.norm(X_image, axis=1, keepdims=True)
    cosine_similarities = np.dot(X_image_norm, query_norm)

    # Get indices of rows sorted by similarity (highest first)
    most_similar_indices = np.argsort(-cosine_similarities)

    # Get the top N most similar rows
    top_n = 5  # Adjust as needed
    closest_indices = most_similar_indices[:top_n]
    closest_distances = 1 - cosine_similarities[closest_indices]

    keys = [list(image_dict.keys())[i] for i in closest_indices]
    mo.hstack([image_dict[k] for k in keys])
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Image to vector""")
    return


@app.cell
def _(folder_out):
    from PIL import Image
    import numpy as np

    # Create a dictionary to store images as numpy arrays
    image_dict = {}

    # Loop through all files in the directory
    for file_path in folder_out.glob("*"):
        # Check if the file is an image (common image extensions)
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            try:
                # Open the image using PIL/Pillow
                img = Image.open(file_path)
                image_dict[file_path.name] = img
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

    # Return the dictionary of images
    image_dict["54.webp"]
    return image_dict, np


@app.cell
def _(image_dict, np):
    class ColorHistogramEncoder:
        def __init__(self, n_buckets=56):
            self.n_buckets = n_buckets

        def transform(self, X, y=None):
            output = np.zeros((len(X), self.n_buckets * 3))
            for i, x in enumerate(X):
                arr = np.array(x)
                # Create mask to ignore white pixels (R=G=B=255)
                white_mask = ~((arr[:,:,0] == 255) & (arr[:,:,1] == 255) & (arr[:,:,2] == 255))
            
                # Apply mask to each channel before creating histograms
                r_channel = arr[:,:,0][white_mask]
                g_channel = arr[:,:,1][white_mask]
                b_channel = arr[:,:,2][white_mask]
            
                output[i, :] = np.concatenate(
                    [
                        np.histogram(
                            r_channel,
                            bins=np.linspace(0, 255, self.n_buckets + 1),
                        )[0],
                        np.histogram(
                            g_channel,
                            bins=np.linspace(0, 255, self.n_buckets + 1),
                        )[0],
                        np.histogram(
                            b_channel,
                            bins=np.linspace(0, 255, self.n_buckets + 1),
                        )[0],
                    ]
                )
            return output

    encoder = ColorHistogramEncoder()
    X_image = encoder.transform(image_dict.values())
    return X_image, encoder


@app.cell
def _(X_image):
    X_image
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Downloading images""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(download_file, unzip_file):
    download_file("https://github.com/koaning/fun-data/raw/refs/heads/main/poke.zip", save_path="poke.zip")

    folder_out = unzip_file("poke.zip", extract_to="poke")
    return (folder_out,)


@app.cell
def _():
    import requests
    from pathlib import Path
    import os
    from tqdm import tqdm

    def download_file(url, save_path=None, chunk_size=1024):
        """
        Download a file from a URL and save it locally.
    
        Parameters:
        -----------
        url : str
            The URL of the file to download
        save_path : str or Path, optional
            The path where the file should be saved. If None, uses the filename from the URL
            and saves in the current directory.
        chunk_size : int, optional
            Size of chunks to use when streaming the download
        
        Returns:
        --------
        Path
            The path where the file was saved
        """
        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
        
            # Get the filename from the URL if save_path is not provided
            if save_path is None:
                filename = os.path.basename(url.split('?')[0])  # Remove query parameters
                save_path = Path(filename)
            else:
                save_path = Path(save_path)
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
            # Get the total file size if available
            total_size = int(response.headers.get('content-length', 0))
        
            # Download the file with progress bar
            with open(save_path, 'wb') as file, tqdm(
                desc=f"Downloading {save_path.name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    bar.update(size)
                
            print(f"✅ File downloaded successfully to {save_path.absolute()}")
            return save_path
    
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading file: {e}")
            return None
    return Path, download_file, tqdm


@app.cell
def _(Path, tqdm):
    import zipfile

    def unzip_file(zip_path, extract_to=None, remove_after=False):
        """
        Extract contents of a zip file to a directory.
    
        Parameters:
        -----------
        zip_path : str or Path
            Path to the zip file to extract
        extract_to : str or Path, optional
            Directory where contents should be extracted. If None, extracts to a directory
            with the same name as the zip file (without the .zip extension).
        remove_after : bool, optional
            Whether to remove the zip file after extraction (default: False)
        
        Returns:
        --------
        Path
            The path where the contents were extracted
        """
        try:
            zip_path = Path(zip_path)
        
            # Determine extraction directory
            if extract_to is None:
                extract_to = zip_path.with_suffix('')  # Remove .zip extension
            else:
                extract_to = Path(extract_to)
            
            # Create extraction directory if it doesn't exist
            extract_to.mkdir(parents=True, exist_ok=True)
        
            # Get file size for progress reporting
            zip_size = zip_path.stat().st_size
        
            # Open the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files for progress reporting
                file_list = zip_ref.namelist()
            
                # Extract all files with progress bar
                for file in tqdm(file_list, desc=f"Extracting {zip_path.name}"):
                    zip_ref.extract(file, extract_to)
        
            print(f"✅ Files extracted successfully to {extract_to.absolute()}")
        
            # Remove zip file if specified
            if remove_after:
                zip_path.unlink()
                print(f"🗑️ Removed zip file: {zip_path}")
            
            return extract_to
    
        except zipfile.BadZipFile:
            print(f"❌ Error: {zip_path} is not a valid zip file")
            return None
        except Exception as e:
            print(f"❌ Error extracting zip file: {e}")
            return None
    return (unzip_file,)


if __name__ == "__main__":
    app.run()

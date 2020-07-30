# Image Compression
<img src="ImageCompression_GIF.gif" width="700">

Karhunen–Loève transform (KLT) is best choice for performing image compression because it results in least possible mean square error. However, KLT depends on the input image, which makes compression impractical. DCT is the closest approximation of KLT, therefore we use DCT for JPEG compression. In this project, it is shown that KLT is perfectly able to reconstruct compressed image, while DCT and DFT do not reconstruct perfectly. We have used DCT for JPEG compression with threshold and zonal coding. 

## How to Run Code:
   1. Clone Repo
   ```
   git clone "https://github.com/dkjangid910/Image_Compression_Algos.git"
   ```
   2. Create Virtual environment
   ```
    virtualenv -p /usr/bin/python3.6 venv(name of virtual environment)
   ```
   3. Activate Virtual environment
   ```
   source venv/bin/activate
   ```
   4. Download and Install dependencies 
   ```
   pip install -r requirements.txt 
   ```
   5. Run Code
   
      5.1 for KLT
      ```
      python klt.py -i ./Data/set_1 
      ``` 
      Reconstructed images will be saved in results/klt folder.
      
      5.2 for DFT and DCT
      ```
      python dft_dct.py -i ./Data/set_1
      ```
      Reconstructed images will be saved in results/dft and results/dct folder
      
      5.3 For JPEG Compression 
      ```
      python jpeg_comp.py -i Data/set_1/Nikon_D70s_1_23105.png  -n 64 -m zonal_coding
      ```
      where n is number of coefficients and m is methods either zonal coding or threshold coding. Results will be saved in results/jpeg folder
   

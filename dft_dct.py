from PIL import Image
import numpy as np
from scipy import fftpack
import cv2
import glob
import os
import argparse 


def largest_coeff_value_dft(img_fft, num_of_coefficient):
    rows_1 = img_fft.shape[0]
    cols_1 = img_fft.shape[1]
    
    img_fft_m = np.zeros((rows_1, cols_1), dtype = complex)
     
    img_fft_abs = np.abs(img_fft)

    temp = np.reshape(img_fft_abs, (rows_1 * cols_1))
    
    j =0
    if temp[num_of_coefficient -1] == temp[num_of_coefficient]:
       tot_coeff = num_of_coefficient  + 1
    else:
       tot_coeff = num_of_coefficient 

    while (j < tot_coeff): 
           
        index = np.where(img_fft_abs == np.amax(img_fft_abs))
        img_fft_m [index[0][0], index[1][0]]= img_fft[index[0][0], index[1][0]] 
        img_fft_abs[index[0][0], index[1][0]]= -10  
        j = j + 1
    return img_fft_m 


def largest_coeff_value_dct(img_dct, num_of_coefficient):
    rows_1 = img_dct.shape[0]
    cols_1 = img_dct.shape[1]
    
    img_dct_m = np.zeros((rows_1, cols_1))
     
    img_dct_abs = np.abs(img_dct)
    
    j =0
   
    while (j < num_of_coefficient ):
        index = np.where(img_dct_abs == np.amax(img_dct_abs))
        img_dct_m [index[0][0], index[1][0]]= img_dct[index[0][0], index[1][0]] 
        img_dct_abs[index[0][0], index[1][0]]= -10 
        j = j + 1
    return img_dct_m 



def main(args):
  
    num_of_coefficient = args["num_of_coeff"]
    print(f'number of coefficients are {num_of_coefficient}')  
  
    PATH = args["input_image_path"]
    
    sum_1 = 0
    sum_2 = 0
       
    filename_list = [] 
  
    results_dft = f'results/dft'
    results_dct = f'results/dct'

    if not os.path.isdir(f'{results_dft}'):
        os.makedirs(results_dft) 
         
    if not os.path.isdir(f'{results_dct}'):
        os.makedirs(results_dct) 
       
     
    for filename in sorted(glob.glob(f'{PATH}/*')):
       
        img  = Image.open(filename)
        img = np.asarray(img)
        rows, cols = img.shape[0], img.shape[1] 
        no_of_pixels = rows * cols
          
        # fft 
        img_fft = fftpack.fft2(img)
        img_fft_reconstructed = largest_coeff_value_dft(img_fft, num_of_coefficient)
        img_reconstructed_f   = fftpack.ifft2(img_fft_reconstructed)
        sum_1 = sum_1 + np.sqrt( np.sum((np.abs(img_reconstructed_f - img)) ** 2 )/no_of_pixels) 
             
        image_recon_f  = Image.fromarray(np.uint8(img_reconstructed_f))
        img_name = os.path.basename(filename)
 
        image_recon_f.save(f'{results_dft}/{num_of_coefficient}_{img_name}')

        # DCT   
        img1 = np.float32(img) / 255.0
        img_dct_1 = cv2.dct(img1)
        img_dct = img_dct_1 * 255
        img_dct_reconstructed = largest_coeff_value_dct(img_dct, num_of_coefficient)  
        img_reconstructed_d =  cv2.idct(img_dct_reconstructed)   
        sum_2 = sum_2 + np.sqrt(np.sum((np.abs(img_reconstructed_d - img)) ** 2 )/ no_of_pixels)
        
        image_recon_d  = Image.fromarray(np.uint8(img_reconstructed_d))
        image_recon_d.save(f'{results_dct}/{num_of_coefficient}_{img_name}')
         
        filename_list.append(filename)    
             
    num_of_images = len(filename_list)
    MSE_fft = sum_1 / num_of_images
    MSE_dct = sum_2 / num_of_images
    print("Mean Square Error FFT: ", MSE_fft)
    print("Mean Square Error DCT: ", MSE_dct)


if __name__ == "__main__":
     
     arg_parse = argparse.ArgumentParser()
     arg_parse.add_argument("-i", "--input_image_path", required=True,
                            help = "Path of Input Image")
     arg_parse.add_argument("-n", "--num_of_coeff", default= 100,
			    help = "number of coefficients")
     args = vars(arg_parse.parse_args())

     main(args)
     

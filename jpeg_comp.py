import argparse
import numpy as np
from PIL import Image
from scipy import fftpack
import cv2
import os
import collections
from dahuffman import HuffmanCodec
import ast

def largest_N_value_DCT(img_dct, num_of_coefficients):
   
    
   rows_1, cols_1, no_of_blocks = img_dct.shape[0], img_dct.shape[1], img_dct.shape[2]
   
   img_dct_abs = np.abs(img_dct)
 
   img_dct_m = np.zeros((rows_1, cols_1, no_of_blocks))
     
   i = 0
   
   while (i < no_of_blocks):
         j =0
   
         while (j < num_of_coefficients):
              index = np.where(img_dct_abs[:,:,i] == np.amax(img_dct_abs[:,:,i]))
              k = 0 
              while k < len(index[0]) and np.amax(img_dct_abs[:,:,i]) !=0:
                  img_dct_m [index[0][k], index[1][k], i]= img_dct[index[0][k], index[1][k],i] 
              
                  img_dct_abs[index[0][k], index[1][k],i]= -100
                  k = k + 1
              
              j = j + 1
         i = i + 1  
      
   return img_dct_m 


def check_and_zero_pad_img(img):
   
   rows, cols = img.shape[0], img.shape[1] 
 
   if rows == cols:
       if rows % 8 != 0:
          # zero padding both rows and cols
          temp = rows % 8
          zero_pad_img = np.pad(img, (temp), 'constant', constant_values= (0) )
       else:
          # No zero padding
          zero_pad_img = img
   else:
       if rows % 8 != 0 and cols % 8 ==0: 
         # zero pad row
          temp = rows % 8
          zero_pad_img = np.pad(img, (temp,0), 'constant', constant_values= (0))
          zero_pad_img = zero_pad_img[temp:,:]
       elif rows % 8 == 0 and cols % 8 != 0:
         # zero pad columns
          temp = cols % 8
          zero_pad_img =  np.pad(img, (0, temp), 'constant', constant_values=(0))
          zero_pad_img = zero_pad_img[0:rows, :]
       elif rows % 8 !=0 and cols % 8 != 0:
          # zero pad both rows and cols with different padding size
          temp_1 = rows % 8
          temp_2 = cols % 8
          zero_pad_img = np.pad(img, (temp_1, temp_2), 'constant', constant_values = (0)) 
       else:
          # No zero Padding
          zero_pad_img = img


   return zero_pad_img    


def partitions_in_8X8(zero_pad_image, block_size = 8):

   #import pdb; pdb.set_trace()  
   rows, cols = zero_pad_image.shape[0] , zero_pad_image.shape[1] 
                       
   no_of_blocks = int(rows*cols/(block_size*block_size))
   img_subblock = np.zeros((block_size,block_size,no_of_blocks))
   
   count = 0
   i  = 0

   while(i < int(rows/8)):
        j = 0
        while(j < int(cols/8)):
               
           img_subblock[:,:,count] = zero_pad_image[8*i:8*(i+1), 8*j:8*(j+1)]
           j  =  j + 1
           count = count + 1
        i = i + 1
        
   return img_subblock


def convert_back_to_original_image(img_subblock, size):
    
    rows, cols, no_of_blocks = img_subblock.shape[0], img_subblock.shape[1], img_subblock.shape[2]
    desired_rows = size[0]
    desired_cols = size[1]
   
    image_reconstructed = np.zeros((size[0], size[1])) 
    count = 0
    i = 0
     
    while(i < int(desired_rows/8)):
         j = 0
         while(j < int(desired_cols/8)):
          
            image_reconstructed[8*i:8*(i+1), 8*j:8*(j+1)] = img_subblock[:,:,count] 
            j = j + 1
            count = count + 1
         i = i + 1
    return image_reconstructed 

   
def DCT_of_each_subblock(img_subblock):
    
   normalized_img = np.float32(img_subblock) / 255.0
   rows, cols, total_blocks = img_subblock.shape[0], img_subblock.shape[1], img_subblock.shape[2]
   img_dct = np.zeros((rows, cols, total_blocks), dtype = 'float32')
   
   for i in range(total_blocks):
        
      img_dct[:,:,i] = cv2.dct(normalized_img[:,:,i])
    
   img_dct = img_dct * 255
   img_dct = img_dct.astype(dtype='int32')
   
   return img_dct 
   

def IDCT_of_each_subblock(img_subblock_dct_N):
   
    
   rows, cols, total_blocks =  img_subblock_dct_N.shape[0], img_subblock_dct_N.shape[1], img_subblock_dct_N.shape[2]
   img_subblock_reconstructed = np.zeros((rows, cols, total_blocks), dtype = 'int32')
   
   for i in range(total_blocks):
         img_subblock_reconstructed[:,:,i] = cv2.idct(img_subblock_dct_N[:,:,i])
   
   
   return img_subblock_reconstructed 


def Quantization(img_dct, q_matrix):
     
   rows, cols, no_of_blocks = img_dct.shape[0], img_dct.shape[1], img_dct.shape[2]
   img_dct_q = np.zeros((rows,cols, no_of_blocks))

   for i in range(no_of_blocks):
          img_dct_q[:,:,i] = np.divide(img_dct[:,:, i], q_matrix)
   img_dct_q = img_dct_q.astype(dtype='int32')
   
   return img_dct_q


def denorm_Quantization(revert_zig_zag, q_matrix):
    
    
      
   rows, cols, no_of_blocks = revert_zig_zag.shape[0], revert_zig_zag.shape[1], revert_zig_zag.shape[2]
   
   denorm_q  = np.zeros((rows, cols, no_of_blocks))   
   for i in range(no_of_blocks):
          denorm_q[:,:,i] = np.multiply(revert_zig_zag[:,:,i],  q_matrix) 
   
   return denorm_q
  

 

def zig_zag_index(k ,n):
    # credit to Tomas Bouda (Coells)
    if k >= n * (n+1) // 2:
       i , j = zig_zag_index(n * n - 1 -k, n)
       return n - 1 - i, n - 1 - j
     
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    
    return (j, i -j) if i & 1 else (i -j, j)


def zig_zag_scanning(img_dct_q):
  
    rows, cols, no_of_blocks = img_dct_q.shape[0], img_dct_q.shape[1], img_dct_q.shape[2]
    img_zig_zag = np.zeros((rows*cols, no_of_blocks))   
 
    for count in range(no_of_blocks):
         for i in range(rows * cols):
            index = zig_zag_index(i, rows)
            img_zig_zag[i, count]  = img_dct_q[index[0], index[1], count]
            
    return img_zig_zag


def revert_zig_zag_scanning(RLE_total_decode):
    
     no_of_elements, total_no_blocks = RLE_total_decode.shape[0], RLE_total_decode.shape[1]
     rows, cols = int(np.sqrt(no_of_elements)), int(np.sqrt(no_of_elements))

     revert_zig_zag = np.zeros((rows,cols,total_no_blocks))
     for count in range(total_no_blocks):
        
            for i in range(no_of_elements):
                 
                  index = zig_zag_index(i, rows)
                  revert_zig_zag[index[0], index[1], count]  = RLE_total_decode[i, count]
 

     return revert_zig_zag


def run_length_encoding(img_zig_zag):
     
    length, total_no_blocks = img_zig_zag.shape[0], img_zig_zag.shape[1] 

   # import pdb; pdb.set_trace()  
    RLE_total = []

    for count in range(total_no_blocks):
        
        RLE = []
        zero_count = 0

        for i in range(length):
                       
              if img_zig_zag[i, count] == 0 and i < length -1:
                  
                    zero_count = zero_count + 1       
                   
              else:
                    
                    RLE.append((zero_count,img_zig_zag[i, count])) 
                    zero_count = 0  
    #    import pdb; pdb.set_trace()
        RLE_total.append(RLE) 
   #  import pdb; db.set_trace()
    return RLE_total   
 

     
def run_length_decoding(RLE_total):
    
    #import pdb; pdb.set_trace() 
    total_no_blocks = len(RLE_total)
    no_elements_in_each_block = 64
     
    RLE_decode = np.zeros((no_elements_in_each_block, total_no_blocks))
    for count in range(total_no_blocks):
        i = 0
        j_1 = 0
  
        while i < len(RLE_total[count]):
                  
             if RLE_total[count][i][0] == 0:

                   RLE_decode[j_1,count] = RLE_total[count][i][1]
                   j_1 = j_1+1
                   i = i + 1
             else :
                  j_2 = RLE_total[count][i][0]
                  j_1 = j_1 + j_2
                  RLE_decode[j_1, count] = RLE_total[count][i][1]
                  j_1 = j_1 + 1
                  i = i + 1
       
    return RLE_decode               
     

def huffman_coding_decoding(RLE_total):
    

     RLE_total_new = str(RLE_total)
     
     codec = HuffmanCodec.from_data(RLE_total_new)
     print("Huffman Code Table: \n")

     codec.print_code_table()
     coded_string  = codec.encode(RLE_total_new)
     decoded_string = codec.decode(coded_string)
     return codec, coded_string, decoded_string


def main(args):

   PATH = args["input"]
   num_of_coefficients = args["no_of_coefficient"]

   q_matrix  = np.array( [[16,  11,  10,  16,  24,  40,  51,  61],
                          [12,  12,  14,  19,  26,  58,  60,  55],
                          [14,  13,  16,  24,  40,  57,  69,  56],
                          [14,  17,  22,  29,  51,  87,  80,  62],
                          [18,  22,  37,  56,  68, 109, 103,  77],
                          [24,  35,  55,  64,  81, 104, 113,  92],
                          [49,  64,  78,  87, 103, 121, 120, 101],
                          [72,  92,  95,  98, 112, 100, 103,  99]])


   
   # Reading the Image
   img  = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
   rows, cols = img.shape[0], img.shape[1] 
   

   # Condition to check the image size is multiple of 8 X 8 

   zero_pad_image = check_and_zero_pad_img(img)

   zero_pad_image = zero_pad_image.astype('int32')
   
   zero_pad_image = zero_pad_image - 128

  
   # partitions into 8 X 8 blocks
    
   img_subblock = partitions_in_8X8(zero_pad_image)

     
    # DCT
   img_dct = DCT_of_each_subblock(img_subblock)  
    

   if (args["method"] == 'threshold_coding'):
   # Keep only N coefficients for each subblock
   
         img_subblock_dct_N = largest_N_value_DCT(img_dct, num_of_coefficients)
         denorm_q = img_subblock_dct_N
   else: 
        # Quantization 
        img_dct_q =  Quantization(img_dct, q_matrix)

        # Zig-Zag Scanning
    
        img_zig_zag = zig_zag_scanning(img_dct_q)

        # Run Length Encoding
    
        RLE_total = run_length_encoding(img_zig_zag)

        # Huffman Coding
  
        codec, coded_string, decoded_string  = huffman_coding_decoding(RLE_total)
  
        decoded_list = ast.literal_eval(decoded_string)
     
        # Run Length Decoding

        RLE_total_decode = run_length_decoding(decoded_list)
   
        # Invert Zig-Zag Operation
        revert_zig_zag  = revert_zig_zag_scanning(RLE_total_decode)
  
        # Denormalization by Quantization
        denorm_q =  denorm_Quantization(revert_zig_zag, q_matrix)
       
   # IDCT
   
   img_subblock_reconstructed = IDCT_of_each_subblock(denorm_q) 
 
   # convet back to 3 Dimension to 2 Dimension 
   img_reconstructed = convert_back_to_original_image(img_subblock_reconstructed, (rows, cols)) 
  
     
   # Level Shifting   
   img_reconstructed = img_reconstructed + 128

   img_reconstructed = img_reconstructed.astype(dtype ='uint8')
    
         
   # Error 
      
   rmse = np.sqrt(np.sum((img - img_reconstructed)** 2) / (rows * cols))
     
   print("RMSE : ", rmse)

   # save reconstructed Image
   img_name = os.path.basename(PATH)
   img_name = img_name.replace('.png', '.jpg')
   #save_img = '{}_'.format(N) + img_name
  
   results_jpeg = f'results/jpeg'

   if not os.path.isdir(f'{results_jpeg}'):
        os.makedirs(results_jpeg) 
     
   save_img = f'{results_jpeg}/{num_of_coefficients}_{args["method"]}_{img_name}'

   cv2.imwrite( save_img, img_reconstructed)



if __name__ == "__main__":
   
   ap = argparse.ArgumentParser()
   ap.add_argument("-i", "--input", required=True, help="path to input image for jpeg compression", type =str)
   ap.add_argument("-n", "--no_of_coefficient", required=True, help = "number of coefficients between 1 to 64", type =int)
   ap.add_argument("-m", "--method", required=True, help = "enter zonal_coding or threshold_coding", type = str) 
   args = vars(ap.parse_args())

   main(args)   
   


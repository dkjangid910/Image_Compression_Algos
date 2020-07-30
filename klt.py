from PIL import Image
import numpy as np
import glob
import os
import argparse 


def mean_and_norm(imgs):
    # Mean
  
    imgs_norm = np.squeeze(imgs)
    imgs_norm = (1/255) * imgs_norm
    mean = np.mean(imgs_norm, axis = 1)
    mean = np.reshape(mean, (mean.shape[0], 1))
    imgs_norm = imgs_norm - mean

    return mean, imgs_norm 


def KLT(imgs_norm, num_of_coefficient):
	    
  
    # Covariance Matrix 
    cov_mat = np.dot(imgs_norm.T, imgs_norm)
    val, vec = np.linalg.eig(cov_mat)
    val = np.sqrt(val)
    val = np.reciprocal(val)
    val = np.diag(val)
       
    eigen_vec = np.dot(np.dot(imgs_norm,vec),val)

    # Kl_Coefficient
    KL_Coef = np.dot(imgs_norm.T, eigen_vec)
    KL_Coef = KL_Coef[:, 0:num_of_coefficient]
    eigen_vec = eigen_vec[:, 0:num_of_coefficient]
    
    return KL_Coef, eigen_vec


def reconstruct_image(KL_Coef, eigen_vec, mean, 
		      imgs_norm, total_imgs, num_of_pixels):
      
    image_recon = np.dot(eigen_vec, KL_Coef.T)
    sum_square	= np.sqrt(np.sum(np.abs(image_recon - imgs_norm) ** 2)/num_of_pixels)
    MSE	= sum_square /	total_imgs
    image_recon = (image_recon + mean) * 255 
    return MSE, image_recon


def save_recon_imgs(image_recon, total_imgs, filename_list, num_of_coefficient):

    for i in range(total_imgs):
        img_recon = np.reshape(image_recon[:, i], (120,160), order = 'F')
        img_recon = Image.fromarray(np.uint8(img_recon))
        img_name  = os.path.basename(filename_list[i])
           
        result_dir = f'results/klt'
        if not os.path.isdir(f'{result_dir}'):
            os.makedirs(result_dir)
        img_recon.save(f'{result_dir}/{num_of_coefficient}_coeff_recons_{img_name}')
	 


def reading_images(PATH):
    image_list = []
    filename_list =[]
     
    for filename in sorted(glob.glob(f'{PATH}/*')):	  
	    img  = Image.open(filename)
	    img = np.asarray(img)
	    img = np.reshape(img, (img.shape[0] * img.shape[1], 1), order = 'F')
	    image_list.append(img)
	    filename_list.append(filename)
 
    rows = img.shape[0]
    cols = img.shape[1]
    total_imgs = len(image_list)
     
    imgs = np.asarray(image_list, dtype = np.float32)
    imgs = imgs.T

    return imgs, filename_list, rows, cols, total_imgs 

 
def main(args):
  
    num_of_coefficient = args["num_of_coeff"]
    print(f'number of coefficients are {num_of_coefficient}')  
  
    PATH = args["input_img_path"]
 
    # Reading the Images  
    imgs, filename_list, rows, cols, total_imgs  = reading_images(PATH)

    num_of_pixels = rows * cols

    mean, imgs_norm = mean_and_norm(imgs)

    KL_Coef, eigen_vec = KLT(imgs_norm, num_of_coefficient)

    MSE, image_recon = reconstruct_image(KL_Coef, eigen_vec, mean, 
					 imgs_norm, total_imgs, num_of_pixels)

    # MSE 
    print("MSE is :", MSE) 

    # saving reconstructed image
    print("Saving reconstructed images in results folder")
    save_recon_imgs(image_recon, total_imgs, filename_list, num_of_coefficient)	   



if __name__ == "__main__":

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--input_img_path", required=True,
			   help = "Path of Input Image")
    arg_parse.add_argument("-n", "--num_of_coeff", default= 20,
			   help = "number of coefficients")

    args = vars(arg_parse.parse_args())
    main(args)
		
      

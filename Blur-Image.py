#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#David Lipkin
#11/22/2019
#This program was written as part of a project for PIC 16:
#Python with Applications under professor Hangji Ji at UCLA

def blur(im, method):
    """
    blurring(im, method) takes a gray-scale picture and offers two
    options for noise blurring: uniform or Gaussian.
    
    The uniform blur uses a uniform kernel of size k x k to determine
    the value of each pixel in the output.
    
    The Gaussian blur uses a weighted kernel of size k x k, and
    weight-intensity sigma, to determine the value of each pixel in 
    the output.
    """

    import numpy as np
    
    n,m,d = im.shape
    output = np.empty([n,m], dtype = 'float')
    
    if method == 'uniform':
        
        k = 7 #Determines the size of the filter
        
        for i in range(0 + k/2, n - k/2):
            for j in range(0 + k/2, m - k/2):
                #Sets each output pixel to the uniform sum of the k x k surrounding pixels
                output[i,j] = np.sum(im[(i-k/2): (i+k/2)+1, (j-k/2): (j+k/2)+1])
        #Normalizes the output
        output /= k**2
        #Converts the output to an n x m x 3 greyscale image
        grey_output = np.dstack((output/3, output/3, output/3))
        
    if method == 'Gaussian':
        
        k = 25 #Determines the size of the filter
        sigma = 2 #Determines the intensity of the weights for the Gaussian method
        
        #Creates a Gaussian filter based on k and sigma values above
        filter = np.empty([k,k], dtype='float')
        for x in range(k):
            for y in range(k):
                filter[x,y] = np.exp(-((x-(k-1)*0.5)**2+(y-(k-1)*0.5)**2)/(2.0*sigma**2))
                filter_sum = np.sum(filter)
                filter = filter/filter_sum
    
        for i in range(0 + k/2, n - k/2):
            for j in range(0 + k/2, m - k/2):
                #Sets weights of the k x k surrounding pixels
                filtered_tiles = filter*(im[(i-k/2): (i+k/2)+1, (j-k/2): (j+k/2)+1, 0])
                #Sets each output pixel to the weighted sum of the k x k surrounding pixels
                output[i,j] = np.sum(filtered_tiles)
        
        #Converts the output to an n x m x 3 greyscale image
        grey_output = np.dstack((output, output, output))
        grey_output = grey_output[0 + k/2: n - k/2, 0 + k/2: m - k/2, :]
        
    return grey_output.astype(int)


# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


#Test-Code 1:
#This cell prints a noisy image

img = mpimg.imread("kitty-cat-bw.jpg")
plt.imshow(img)


# In[ ]:


#Test-Code 2:
#This cell prints the image above smoothed using the uniform blur method

img_u = blur(img, 'uniform')
print "Uniform Blurred Image:"
plt.imshow(img_u)


# In[ ]:


#Test-Code 3:
#This cell prints the image above smoothed using the Gaussian blur method

img_g = blur(img, 'Gaussian')
print "Gaussian Blurred Image:"
plt.imshow(img_g)


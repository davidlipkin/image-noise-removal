#!/usr/bin/env python
# coding: utf-8

# In[13]:


#David Lipkin
#11/20/2019
#Homework 6 for PIC 16
#Professor Ji


# In[14]:


#Question 1
def heart(im):
    """
    heart(im) takes an image im as input and outputs a heart-shaped
    cut-out of it on a pink background. The shape of the heart depends
    on the dimensions of the image.
    """
    
    get_ipython().magic(u'matplotlib inline')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    
    im = im.copy()
    n,m,d = im.shape
    y,x = np.ogrid[0:n,0:m]

    mask1 = y*(3*m) > x*(4*n) + n*m #Cuts out a triangle from the bottom left
    mask2 = y*(3*m) > -x*(4*n) + 5*n*m #Cuts out a triangle from the bottom right
    mask3 = ((x - m/4)**2 + (y - n/4)**2 > (m/4)**2) & (x <= m/2) & (y < n/3) #Cuts out a hump from the top left
    mask4 = ((x - 3*m/4)**2 + (y - n/4)**2 > (m/4)**2) & (x >= m/2) & (y < n/3) #Cuts out a hump from the top right

    #Sets cut-out regions to pink
    im[mask1] = (255,200,240)
    im[mask2] = (255,200,240)
    im[mask3] = (255,200,240)
    im[mask4] = (255,200,240)
    
    return im


# In[15]:


#Question 2
def blurring(im, method):
    """
    blurring(im, method) takes a gray-scale picture and offers two
    options for noise blurring: uniform or Gaussian.
    
    The uniform blur uses a uniform kernel of size k x k to determine
    the value of each pixel in the output.
    
    The Gaussian blur uses a weighted kernel of size k x k, and
    weight-intensity sigma, to determine the value of each pixel in 
    the output.
    """
    
    get_ipython().magic(u'matplotlib inline')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    
    k = 9 #Determines the size of the filter
    sigma = 1 #Determines the intensity of the weights for the Gaussian method
    
    n,m,d = im.shape
    output = np.empty([n,m], dtype = 'float')
    
    if method == 'uniform':
        for i in range(0 + k/2, n - k/2):
            for j in range(0 + k/2, m - k/2):
                output[i,j] = np.sum(im[(i-k/2): (i+k/2)+1, (j-k/2): (j+k/2)+1]) #Sets each output pixel to the uniform sum of the k x k surrounding pixels
        output /= k**2 #Normalizes the output
        grey_output = np.dstack((output/3, output/3, output/3)) #Converts the output to an n x m x 3 greyscale image
        
    if method == 'Gaussian':
        
        #Creates a Gaussian filter based on k and sigma values above
        filter = np.empty([k,k], dtype='float')
        for x in range(k):
            for y in range(k):
                filter[x,y] = np.exp(-((x-(k-1)*0.5)**2+(y-(k-1)*0.5)**2)/(2.0*sigma**2))
                filter_sum = np.sum(filter)
                filter = filter/filter_sum
    
        for i in range(0 + k/2, n - k/2):
            for j in range(0 + k/2, m - k/2):
                filtered_tiles = filter*(im[(i-k/2): (i+k/2)+1, (j-k/2): (j+k/2)+1, 0]) #Sets weights of the k x k surrounding pixels
                output[i,j] = np.sum(filtered_tiles) #Sets each output pixel to the weighted sum of the k x k surrounding pixels
                
        grey_output = np.dstack((output, output, output)) #Converts the output to an n x m x 3 greyscale image
        
    return grey_output.astype(int)


# In[16]:


#Question 3
def detect_edge(im, method):
    """
    detect_edge(im, method) takes a gray-scale image and detects edges,
    with the option of horizontal, vertical or both.
    """

    get_ipython().magic(u'matplotlib inline')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    
    n,m = im.shape
    output = np.empty([n,m], dtype = 'float64')
    
    if method == 'vertical':
        filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Sets standard vertical edge filter
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                output[i,j] = np.sum(filter * (im[i-1: i+2, j-1: j+2]))
                if output[i,j] > 0:
                    output[i,j] = 1 #Sets positive edges to white
                if output[i,j] == 0:
                    output[i,j] = 0.5 #Sets neutral pixels to grey
                if output[i,j] < 0:
                    output[i,j] = 0 #Sets negative edges to black
                
    if method == 'horizontal':
        filter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #Sets standard horizontal edge filter
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                output[i,j] = np.sum(filter * (im[i-1: i+2, j-1: j+2]))
                if output[i,j] > 0:
                    output[i,j] = 1 #Sets positive edges to white
                if output[i,j] == 0:
                    output[i,j] = 0.5 #Sets neutral pixels to grey
                if output[i,j] < 0:
                    output[i,j] = 0 #Sets negative edges to black

    if method == 'both':
        filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) + np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #Combines vertical and horizontal edge filters
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                output[i,j] = np.sum(filter * (im[i-1: i+2, j-1: j+2]))
                if output[i,j] < 0:
                    output[i,j] *= -1 #Removes distinction between positive and negative edges
                
    grey_output = np.dstack((output*255, output*255, output*255)) #Converts output to n x m x 3 greyscale image
    return grey_output.astype(int)


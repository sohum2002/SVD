
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets, svm, metrics


# #Homework 4. (100 points)
# In this homework, you will be making system of hand written digit recognition. The data is given in usps_resampled directory.
# 
# Your goal is to answer the question: given an image of hand written digit. What is the number?
# 
# There are many ways to do this. But you will end up using either neural network or SVM to do this. Do not implement it yourself. Use packages. You may use PyBrain or Scikit-learn for the learning algrithm. Let me know if you want to use ither package. Pick one read the documentation. If you want to learn about something new, pick an algorithm I haven't taught. Ex: KNN is easy to understand. (Again don't implement it yourself. It will be super slow.)
# 
# There are thre routes for this. Again, pick one:
# 
# First(Full credit): you could use the raw input of each pixel as feature. This is a dumb thing to do but it should work with algorithm like K-mean clustering and KNN. This is quite slow though but the performance should be OK. See first few pages of this link https://www.math.ucdavis.edu/~saito/courses/167.s12/Lecture21.pdf
# 
# Second (Recommended: Full credit+20 points.): you could extract numbers from each image. For example you could extract
#   - The luminosity(kind of how bright an image is) of the whole image.
#   - The luminosity of each 4x4 block.
#   - The symmetry of vertical axis. (If you flip the image vertically, do you get the same value for the pixel). This can be easily done by computing the sum of difference square of the correponding opposite pixel.
#   - The symmetry on horizontal axes.
#   - The symmetry on the two diagonal axes.
#   - These should give you a really good performance already.
# Then all you need to do is to train SVM or Neural Network on these data. You may need to do ECOC if your package doesn't do multiclass automatically for you.
#   
# Third (Fancy, complicated math but really cool) (Full credit + 50 points): You could try to do single value decomposition of the image for the features instead of trying to guess the feature yourself as in the second route. This is quite cool. Try google for single value decomposition image processing.
# 
# If you do all three, I'll give you 150 points plus ability to skip 2 exercises (full credit))
# 
# #Your Task
# Train some sort of classifier to guess the number for the given picture. Once you get it print out confusion matrix for the test dataset(not train) in any format that I could understand. See example of confusion matrix here: https://www.math.ucdavis.edu/~saito/courses/167.s12/Lecture21.pdf .

# ##Here is how to read the data.
# Let start with how to read the data. This is a terrible format though.
# 
# 
# I strongly suggest preprocessing the data and save it in the format that you like first. You will find 
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
# very useful. See the last example.
# 

# In[2]:

D = loadmat('usps_resampled/usps_resampled.mat')
#it is a dictionary of four elements
# train_labels for the number of each training image
# train_patterns for the train images
# test_labels
# test_patterns


# In[3]:

#this is where the images are
images = D['train_patterns'].T #T is for transpose they keep it in funny format
images2 = D['train_patterns']
images3 = D['test_patterns']

U,s,V =  np.linalg.svd(images2, full_matrices=True)
U2,s2,V2 =  np.linalg.svd(images3, full_matrices=True)

#this is the shape of the first image
#it represents 16x16 pixel image but the author flatten it out.
print images[0].shape

#So we should reshape the image
im = images[0].reshape(16,16) #you can try print it

plt.imshow(im, interpolation='nearest', cmap=cm.Greys)

#here is how you access the pixel
#if you wonder which index correspond to which pixel
imm = im.copy()
imm[2,12] = 1 # 1 is black and -1 is white try print it.
plt.imshow(imm, interpolation='nearest', cmap=cm.Greys)


# In[4]:

def showPic(img):
        imgg = img.copy()
        im = imgg.reshape(16,16) #you can try print it
        plt.imshow(im, interpolation='nearest', cmap=cm.Greys)

class USPS:
    def __init__(self, labels, labels2, U, U2):
        self.labels = labels
        self.labels2 = labels2
        self.U = U
        self.U2 = U2
    
    def findLabels(self):
        for i in range(len(self.labels)):
            for j in range(len(self.labels[0])):
                if(self.labels[i,j]==1):
                    self.labels[i] = j

        self.numIdentifier = labels[:,1]
        
    def findLabels2(self):
        for i in range(len(self.labels2)):
            for j in range(len(self.labels2[0])):
                if(self.labels2[i,j]==1) :
                    self.labels2[i] = j

        self.numIdentifier2 = self.labels2[:,1]


    def train(self, images2):
        twentyU = self.U[:,:20]
        C1 = []

        for i in range(len(self.labels)):
            twentyImages = images2[:,i]
            C1.append(np.dot(twentyU.T, twentyImages))
        
        return C1

    def test(self, images3):
        twentyU2 = U2[:,:20]

        C2 = []

        for i in range(len(labels)):
            twentyImages2 = images3[:,i]
            C2.append(np.dot(twentyU2.T, twentyImages2))
        
        return C2

    def findSuccess(self, C1, C2):
        C1 = np.asarray(C1)
        C2 = np.asarray(C2)

        svc = svm.SVC()
        svc.fit(C1,self.numIdentifier)

        output = []
        prob = 0

        for i in range(len(C2)):
            output.append(svc.predict(C2[i]))

        for i in range(len(output)):
            if(output[i] == self.numIdentifier2[i]):
                prob += 1

        return prob/float(len(output))


labels = D['train_labels'].T
labels2 = D['test_labels'].T

usps = USPS(labels, labels2, U, U2)
usps.findLabels()
usps.findLabels2()
C1 = usps.train(images2)
C2 = usps.test(images3)
probSuccess = usps.findSuccess(C1, C2)

print "The success rate for this model is: ", probSuccess * 100, "% from", len(labels), "inputs"


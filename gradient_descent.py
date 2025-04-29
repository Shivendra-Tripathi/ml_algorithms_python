# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:01:34 2025

@author: Shivendra Tripathi
"""
import numpy as np
import matplotlib.pyplot as plt

class Data:
  features=0  #Stores the actual data for gradient descent
  toggleVar=0 #Stores the output or toggle values
  nSamples=0  #Stores the number of data points
  nFeatures=0
  
  #Constructor
  def  __init__(self,nFeatures,nSamples,dtype='float'):
    self.features=np.ones((nSamples,nFeatures+1),dtype)
    self.toggleVar=np.zeros(nSamples,dtype)
    self.nSamples=nSamples
    self.nFeatures=nFeatures
    
  #Initialising the features
  def set_feature(self,featureIndex,feature):
    try:
      self.features[:,featureIndex+1]=feature
    except:
      print("Exception in feature Initalisation(featureIndex={0})___".format(featureIndex))
  
  #Initialising the toggleVar
  def set_toggle_var(self,toggleVar):
    try:
      self.toggleVar=toggleVar
    except:
      print("Exception in toggle Initialisation____")
  
  #Function to get the feature 
  def get_feature(self,index):
    return np.array(self.features[:,index+1]).reshape(self.nSamples)
  
  #Function to get the toggle Vars
  def get_toggle_var(self):
    return np.array(self.toggleVar)
      
  #Print the data
  def print_(self,sampleIndex=-1):
    if sampleIndex>=0:
      try:
       print("Features<{0}> = Toggle<{1}>".format(self.features[sampleIndex,1:],self.toggleVar[sampleIndex]))
      except:
        print("Error in printing(sampleIndex={0})".format(sampleIndex))
    else:
      for i in range(self.nSamples):
        print("Features<{0}> = Toggle<{1}>".format(self.features[i,1:],self.toggleVar[i]))

   

#Class to handle matplotlib plottings
class Plot:
   def __init__(self,title=''):
       plt.title(title)
     
   def plot_points(self,x,y,_marker='x',markersize=1,_linestyle=' ',color='b'):
     plt.plot(x,y,_marker,_linestyle,color)
     
   def plot_line(self,c,m):
     x=np.array(plt.xlim())
     y=m*x+c
     plt.plot(x,y,marker='',linestyle='-')
     
   def make_visible(self):
     plt.show()
     
        
    
    
def gradient_descent(data,stepsize=0.01,steps=100):
  
  alpha=stepsize
  theta=np.zeros((data.nFeatures+1,1))

  
  for i in range(steps):
    variation=np.zeros((data.nFeatures+1,1))
                       
    for j in range(data.nFeatures+1):
      predicted=(data.features @ theta).flatten()  #Value based on current theta
          
      diff=predicted-data.toggleVar    #Difference between predicted val and actual val
      xj=(data.features[:,j]).flatten()
      expectance = (alpha / data.nSamples) * np.dot(diff, xj)
      variation[j,0]=expectance
    
    theta=theta-variation
    
    
    
  return theta
      
  
    
  #Function to mean normalise the data
def mean_normalize(arr):
  range=arr.max()-arr.min()
  mean=arr.mean()
  return (arr-mean)/range
    
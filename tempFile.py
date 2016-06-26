# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:16:08 2016

@author: sasa
"""

from PyQt4 import QtCore, QtGui
from PyQt4 import *
import sys
from PyQt4.QtGui import QMainWindow
from Tkinter import *
#import easygui
import math


try:
    from PIL import Image
except ImportError:
    import Image

import cv2
import collections
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance

# k-means
from sklearn.cluster import KMeans


# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab

def load_image(path):
    return cv2.cvtColor(cv2.imread(path),2)  #ovde je bilo RGB
        
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def remove_noise(binary_image):
    ret_val = erode(dilate(binary_image))
    #ret_val = invert(binary_image)
    return ret_val

def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin


def invert(image):
    return 255-image


def display_image(image,title, color= False):
    #resized_image = cv2.resize(image, (400, 400))
    #plt.figure(figsize=(400,400))
    if color:
        plt.title(title)
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        plt.title(title)


def dilate(image):
    kernel = np.ones((8,8)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=10)


def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=3)

#Funkcionalnost implementirana u V2
def resize_region(region):
    resized = cv2.resize(region,(80,80), interpolation = cv2.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
    
# TODO - ROI3
def select_roi3(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_dic = {}
    
    xt=0
    yt=0

    
    list_contours=[]
    regions_color=[]
    
    
    maxArea=0
        
    for contour in contours:
        if(cv2.contourArea(contour)>maxArea):
            maxArea=cv2.contourArea(contour)
            
#    contours = []
#    contour_angles = []
#    contour_centers = []
#    contour_sizes = []
#    for contour in contours:
#        center, size, angle = cv2.minAreaRect(contour)
#        xt,yt,h,w = cv2.boundingRect(contour)
#
#        region_points = []
#        for i in range (xt,xt+h):
#            for j in range(yt,yt+w):
#                dist = cv2.pointPolygonTest(contour,(i,j),False)
#                if dist>=0 and image_bin[j,i]==255: # da li se tacka nalazi unutar konture?
#                    region_points.append([i,j])
#        contour_centers.append(center)
#        contour_angles.append(angle)
#        contour_sizes.append(size)
#        contours.append(region_points)
#    
#    # postavljanje kontura u vertikalan polozaj
#    contours = rotate_regions(contours, contour_angles, contour_centers, contour_sizes)
    #display_image(contours[0])
    for contour in contours:
#        display_image(contour)
        center, size, angle = cv2.minAreaRect(contour)
        xt,yt,h,w = cv2.boundingRect(contour)
    
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if((cv2.contourArea(contour) >300) and (isTriangleP(contour) or isOsmougao(contour) or isSquareP(contour)  or isCircleP(contour)) and (cv2.contourArea(contour)*4 > maxArea)):

            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            #region = image_bin[y:y+h+1,x:x+w+1];
            #regions_dic[x] = resize_region(region)
            
            #x,y,h,w = cv2.boundingRect(contour)
            region = image_bin[y:y+h+1,x:x+w+1]
            region_color = image_orig[y:y+h+1,x:x+w+1];
            regions_color.append(region_color)
            #regions_dic[x] = resize_region(region)
            regions_dic[x] = [resize_region(region), (x,y,w,h)]

            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,255,0),5)
            list_contours.append(contour)
            #cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),5)
            
#    maxArea=0
#        
#    for contour in list_contours:
#        if(cv2.contourArea(contour)>maxArea):
#            maxArea=cv2.contourArea(contour)
#    
#    for contour in list_contours:
#        #x,y,w,h=cv2.boundingRect(contour)
#        #print "povrsina ove konture " + str(cv2.contourArea(contour))
#        if cv2.contourArea(contour)*4 >=maxArea:
#            xt,yt,h,w = cv2.boundingRect(contour)    
#            #print x,y,w,h
#            region = image_bin[y:y+h+1,x:x+w+1]
#            regions_dic[x] = resize_region(region) 
#            print xt,yt,w,h
            
            #cv2.rectangle(image_orig,(xt,yt),(xt+w,yt+h),(255,255,0),5)    
    
    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()
    sorted_regions = np.array(sorted_regions_dic.values())
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions[:,0], regions_color


def rotate_regions(contours,angles,centers,sizes):
    '''Funkcija koja vrši rotiranje regiona oko njihovih centralnih tačaka
    Args:
        contours: skup svih kontura [kontura1, kontura2, ..., konturaN]
        angles:   skup svih uglova nagiba kontura [nagib1, nagib2, ..., nagibN]
        centers:  skup svih centara minimalnih pravougaonika koji su opisani 
                  oko kontura [centar1, centar2, ..., centarN]
        sizes:    skup parova (height,width) koji predstavljaju duzine stranica minimalnog
                  pravougaonika koji je opisan oko konture [(h1,w1), (h2,w2), ...,(hN,wN)]
    Return:
        ret_val: rotirane konture'''
    ret_val = []
    
    #print angles
    #print "angle"
    #print len(contours)
    #print len(angles)
    for idx,contour in enumerate(contours):
        #print "print enumeration"             
        angle = angles[idx]
        cx,cy = centers[idx]
        height, width = sizes[idx]
        if width<height:
            print "return nnn"
            angle+=90
            
        # Rotiranje svake tačke regiona oko centra rotacije
        alpha = np.pi/2 - abs(np.radians(angle))
        
        print str(alpha) + "alfaaaaaaaaaaaaaaaaaaaaaa"
        region_points_rotated = np.ndarray((len(contour), 2), dtype=np.int16)
        #print "con len"
        
        for i, point in enumerate(contour):
            
            
            #print  "point blabla"
            #print point
            #print cx,cy
            x = point[0]
            y = point[1]
            #print x,y
            #TODO 1 - izračunati koordinate tačke nakon rotacije
            rx = np.sin(alpha)*(x-cx) - np.cos(alpha)*(y-cy) + cx
            ry = np.cos(alpha)*(x-cx) + np.sin(alpha)*(y-cy) + cy
            
            #print "rxry"
            #print rx , ry
            
            
            region_points_rotated[i]=([rx,ry])
        ret_val.append(region_points_rotated)
        

    return ret_val
    
def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_dict = {}
    i=0
    image_width=image_orig.shape[0]
    image_height=image_orig.shape[1]
    area=image_width*image_height
    itera = 0
    #x,y,w,h=0

    listContours=[]
    contours2 = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    center=0 
    size=0
    angle=0
    region=0
    #print len(contours)
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        region = (image_bin.copy())[y:y+h+1,x:x+w+1];
        #center, size, angle = cv2.minAreaRect(contour)
        #print region
        #print cv2.contourArea(contour)
        if cv2.contourArea(contour)>300:
                print str(cv2.contourArea(contour))+" povrsina konture"
                
                
                regions_dict[i] = [resize_region(region), (x,y,w,h)]   #x bilo
                i+=1
                #if (cv2.isContourConvex(contour)):#if isTriangle(contour) or isCircle(contour) or isSquare(contour):
                if isTriangleP(contour) or isOsmougao(contour) or isSquareP(contour)  or isCircleP(contour) :                
                    #cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
                    #print "Jeste krug"
                    listContours.append(contour)

                    
                    
    maxArea=0
        
    for contour in listContours:
        if(cv2.contourArea(contour)>maxArea):
            maxArea=cv2.contourArea(contour)
            
            
    for contour in listContours:
        #x,y,w,h=cv2.boundingRect(contour)
        #print "povrsina ove konture " + str(cv2.contourArea(contour))
        if cv2.contourArea(contour)*4 >=maxArea:
            #print "nacrtao sam i sad mi pusi kurac"
            #print x,y,w,h
        
            xt,yt,h,w = cv2.boundingRect(contour)
            cv2.rectangle(image_orig,(xt,yt),(xt+w,yt+h),(255,255,0),5)
            
    print "max area                          fadsfasdfasfasdfasf"  + str(maxArea)
            
    for contour in listContours:
        #x,y,w,h=cv2.boundingRect(contour)
        #print "povrsina ove konture " + str(cv2.contourArea(contour))
        if cv2.contourArea(contour)*4 >=maxArea:
            #print "nacrtao sam i sad mi pusi kurac"
            #print x,y,w,h
        
            xt,yt,h,w = cv2.boundingRect(contour)
            cv2.rectangle(image_orig,(xt,yt),(xt+w,yt+h),(255,255,0),5)
            
            region_points = []
            for i in range (xt,xt+h):
                for j in range(yt,yt+w):
                    dist = cv2.pointPolygonTest(contour,(i,j),False)
                    if dist>=0 and image_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                        region_points.append([i,j])
            contour_angles.append(angle)
            contour_centers.append(center)
            contour_sizes.append(size)
            contours2.append(region_points)
            

    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    print "contours2" + str(len(contours2))    
    
    #contours2 = rotate_regions(contours2, contour_angles, contour_centers, contour_sizes)
    
#    sorted_rectangles = sorted_regions[:,1]
   # region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
#    for x,y,w,h in sorted_regions[1:-1, 1]:
#        region_distances[-1] += x
#        region_distances.append(-x-w)
#    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, contours2, regions_dict
    
#Shape detecting
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist     
    
    
def isTriangle(contour):
    
    x,y,w,h = cv2.boundingRect(contour)
    
    area=cv2.contourArea(contour)
    mathArea=(w*w*math.sqrt(3))/4    
    ratio=(area/mathArea)*100
    
    print "triangle"
    h2= w*math.sqrt(3)/2
    print h2
    print h
    aSide=math.sqrt((w/2)*(w/2)+math.pow((w*math.sqrt(3)/2),2))
    #print aSide
    #print w
    area=w*w * math.sqrt(3)/4
    print area
    
    print cv2.contourArea(contour)
    areaRelative=area*0.05
    
    print areaRelative
    relative=h2*0.05
    print relative
    if h>=h2-relative and h<= h2+relative and cv2.contourArea(contour) >= area-areaRelative and cv2.contourArea(contour) <= area+areaRelative :
        return True
    else:
        return False
        

def isSquare(contour):
    
    print "quadric"
    area=cv2.contourArea(contour)
    print area
    x,y,w,h=cv2.boundingRect(contour)
    
    cArea=w*h
    print cArea
    ratio=cArea*0.04
    
    if(area >= cArea-ratio and area <= cArea+ratio):
        return True
    else:
        return False
    
    
    
def isCircle(contour):
    #broj=broj+1
    print("\n\n\n")
    sumDistances=0
    
    x,y,w,h=cv2.boundingRect(contour)
    
    centerX=x+(w/2)
    centerY=y+(h/2)
    radius=w/2
    
    print str(radius)+"radius"
    
    relative=radius*0.1
    
    for point in contour:
        
        #print point
        x1=point[0][0]
        y1=point[0][1]
        print(str(calculateDistance(x1,y1,centerX,centerY)) + "str")
        distance=calculateDistance(x1,y1,centerX,centerY)
        if(distance<(radius-relative) or distance>(radius+relative) ):
            return False        
        
    return True
    
    avgRadius=sumDistances/len(contour)
    
    value=(avgRadius/(w/2)*100)
    
    radius=w/2
    
    relative=radius*0.01
    #print (str(broj)+ "kontura")
    #print (str(value)+" value"+ str((w/2)))
    
    if value>=  value<=100.50  :
        return False
    else:
        return False
            
        
        
###############################################################################
        
    #param p1 is point1 (x,y)
    #param p2 is point2 (x,y)
    #param p3 is point3 (x,y)
    #return angle betwen two vectors from 3 different points in radians
def getAngle(p1,p2,p3):
    
    #vector between p1 and p2
#>>> x = np.array([1,2,3])
#>>> y = np.array([-7,8,9])
#>>> np.dot(x,y)
#36
#>>> dot = np.dot(x,y)
#>>> x_modulus = np.sqrt((x*x).sum())
#>>> y_modulus = np.sqrt((y*y).sum())
#>>> cos_angle = dot / x_modulus / y_modulus # cosine of angle between x and y
#>>> angle = np.arccos(cos_angle)
#>>> angle
#0.80823378901082499
#>>> angle * 360 / 2 / np.pi # angle in degrees
    
    
    v12=np.array([p2[0]-p1[0],p2[1]-p1[1]])
    v13=np.array([p3[0]-p1[0],p3[1]-p1[1]])
    
    #print str(v12)+"v1" 
    
    #print str(v13)+"v2"
    
    
    
    scalarProduct=np.dot(v12,v13)
    v12_modul=np.sqrt((v12*v12).sum())
    v13_modul=np.sqrt((v13*v13).sum())
    
    #print str(scalarProduct)+" sp"
    #print str(v12_modul)+" modul "+ str(v13_modul)
   # print str((scalarProduct)/(v12_modul*v13_modul)) +"resultttt"
    
    return np.arccos((scalarProduct)/(v12_modul*v13_modul))*(180/math.pi)
   
def isCircleP(contour):
    
    print "aproksimacija circle"
    
    epsilon = 0.008*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    #print approx
    print len(approx)
    if len(approx) >=12 and not(cv2.isContourConvex(contour)) and len(approx)<20 :
        print "ovo je krug"
        return True
    else:
        return False 
   
   
def isSquareP(contour):
    
    
    epsilon = 0.015*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    #print approx 
    if len(approx) == 4 :
        alpha1=getAngle(approx[0][0],approx[1][0],approx[3][0])
        alpha2=getAngle(approx[1][0],approx[0][0],approx[2][0])
        alpha3=getAngle(approx[2][0],approx[1][0],approx[3][0])
        alpha4=getAngle(approx[3][0],approx[0][0],approx[2][0])
        
        

        
        if alpha1>85.0 and alpha1 <95.0 and alpha2>85.0 and alpha2 <95.0 and alpha3>85.0 and alpha3 <95.0 and alpha4>85.0 and alpha4 <95.0:
            return True
        else:
            return False
    else:
        return False    
    
    
    
def isOsmougao(contour):
    #print "aproksimacija oct"
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    #print approx
    if len(approx) == 8 :
        alpha=getAngle(approx[1][0],approx[2][0],approx[0][0])
       
        if alpha>100.0 and alpha <140.0 :
           
            return True
        else:
            return False
    else:
        return False            

def isTriangleP(contour):
    print "aproximacija trougao"
    epsilon = 0.035*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    
    print len(approx)
    
    print approx

    if(len(approx) == 3):
        alpha1=getAngle(approx[0][0],approx[1][0],approx[2][0])
        print str(alpha1) +" a1"
        alpha2=getAngle(approx[1][0],approx[0][0],approx[2][0])
        print str(alpha2) +" a2"
        alpha3=getAngle(approx[2][0],approx[0][0],approx[1][0])
        print str(alpha3) +" a3"
        
        if( alpha1>53 and alpha1<68 and alpha2>53 and alpha2<68 and alpha3 >53 and alpha3 <68):
            return True
        else:
            return False
    else: 
        return False
    
    #pogledati dokumentaciju opencv za matchshapes
def matchShapesFunction(shape) :
    img1 = load_image('circleCheck.png')
    img2 = load_image('triangleCheck.png')
    img3 = load_image('squareCheck.png')
    img4 = load_image('octagCheck.png')
    print(" test")
    #print(img1.size)
    img11=remove_noise(image_bin(image_gray(img1)))
    img21=remove_noise(image_bin(image_gray(img2)))
    img31=remove_noise(image_bin(image_gray(img3)))
    img41=remove_noise(image_bin(image_gray(img4)))    
    img,contours,hierarchy = cv2.findContours(img11,2,1)
    
    #vraca true ako je oblik neki od ovih, ako nije onda vraca false
    state=False
    
    
    #display_image()    
    #display_image(a,"imgcircle")
    
    #print (len(contours)+55)
    
    
    cnt1 = contours[0]
    img,contours,hierarchy = cv2.findContours(img11,2,1)    
    
    cnt1=contours[0]
   
    img,contours,hierarchy = cv2.findContours(img21,2,1)
    cnt2 = contours[0]
    
    img,contours,hierarchy = cv2.findContours(img31,2,1)
    cnt3 = contours[0]
    
    img,contours,hierarchy = cv2.findContours(img41,2,1)
    cnt4 = contours[0]
    
    print ("start testing")
    
        
    
    print "end testing"
    
    ret = cv2.matchShapes(shape,cnt1,1,0.0)
    print str(ret)+"prvi"
    if ret < 0.01:
        state = True
        return state
        
    ret = cv2.matchShapes(shape,cnt2,1,0.0)
    print str(ret) +"drugi"
    if ret < 0.01:
        state = True
        return state
        
    
    bla3 = cv2.matchShapes(shape,cnt3,1,0.0)
    print str(bla3) +"treci"
    if ret < 0.01:
        state = True
        return state
    
    
    bla = cv2.matchShapes(shape,cnt4,1,0.0)
    print str(bla)+"cetvrti"
    if ret < 0.01 :
        state = True
        return state
    
    
    return state
    

# TODO 3
def select_roi2(image_orig, image_bin):
    
    img, contours_borders, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    for contour in contours_borders:
        
        if((cv2.contourArea(contour) >200) and (isTriangleP(contour) or isOsmougao(contour) or isSquareP(contour)  or isCircleP(contour))):
            center, size, angle = cv2.minAreaRect(contour)
            xt,yt,h,w = cv2.boundingRect(contour)
            cv2.rectangle(image_orig,(xt,yt),(xt+w,yt+h),(255,0,0),10)
    
            region_points = []
            for i in range (xt,xt+h):
                for j in range(yt,yt+w):
                    dist = cv2.pointPolygonTest(contour,(i,j),False)
                    if dist>=0 and image_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                        region_points.append([i,j])
            contour_centers.append(center)
            contour_angles.append(angle)
            contour_sizes.append(size)
            contours.append(region_points)
    
    #Postavljanje kontura u vertikalan polozaj
    contours = rotate_regions(contours, contour_angles, contour_centers, contour_sizes)
    
    #spajanje kukica i kvacica
    #contours = merge_regions(contours)
    
    regions_dict = {}
    for contour in contours:
    
        min_x = min(contour[:,0])
        max_x = max(contour[:,0])
        min_y = min(contour[:,1])
        max_y = max(contour[:,1])

        region = np.zeros((max_y-min_y+1,max_x-min_x+1), dtype=np.int16)
        for point in contour:
            x = point[0]
            y = point[1]
            
             # TODO 3 - koordinate tacaka regiona prebaciti u relativne koordinate
            '''Pretpostavimo da gornja leva tačka regiona ima apsolutne koordinate (100,100).
            Ako uzmemo tačku sa koordinatama unutar regiona, recimo (105,105), nakon
            prebacivanja u relativne koordinate tačka bi trebala imati koorinate (5,5) unutar
            samog regiona.
            '''
            region[y-min_y,x-min_x] = 255

        
        regions_dict[min_x] = [resize_region(region), (min_x,min_y,max_x-min_x,max_y-min_y)]
        
    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    return image_orig, sorted_regions[:, 0]
    
# TODO - NEURAL NETWORK
# TODO - create_ann
def create_ann():
    
    ann = Sequential()
    # Postavljanje slojeva neurona mreže 'ann'
    ann.add(Dense(input_dim=6400, output_dim=128,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    ann.add(Dense(input_dim=128, output_dim=6,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    return ann
   
# TODO - train_ann
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.001, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze  nb_epoch=500
    ann.fit(X_train, y_train, nb_epoch=2000, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
    return ann

# TODO - display_result
def display_result_ann(outputs, alphabet):
    print '\n>>>display_result_ann'
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]

        result += alphabet[winner(output)]
        result += ' i '
    return result

    
#######################################################################################   
    
            
image_test_original = load_image('C:\Users\sbstb\Desktop\stopSign6.jpg')

width,height= image_test_original.shape[0:2]


image_test_temp=image_test_original.copy()
#print str(sizeX) +"    fadfakdsgfjadskjfaksjfiajj"

#for x in range(0, width - 1):
#
#    for y in range(0, height - 1):
#        
#        pixel=image_test_original[x,y]
#        #if( x >130 and y > 130 and y <200):
#            #print str(pixel)+" pixel pre transformacije"
#        
#        #if  pixel[1]*0.75> pixel[2] or pixel[1]*0.75>pixel[0]:
#        if not(pixel[0]*0.5 > pixel[1] and pixel[0]*0.5 > pixel[2]):
#            image_test_temp[x,y]=[255, 255, 255]
            
        #print str(image_test_original[x,y]) +" pixel posle transformacije"
            

#print image_test_original[100,150]

image_test = remove_noise(image_bin(image_gray(image_test_temp)))
plt.figure(1)
display_image(image_test_original,"Original")
plt.figure(2)
display_image(image_test_temp,"Procesuirana")
plt.figure(3)
display_image(image_test,"image test")


image_test_hsv=image_test_original.copy()

hsv = cv2.cvtColor(image_test_hsv, cv2.COLOR_BGR2HSV)



    # define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
mask2 = cv2.inRange(hsv_black,lower_black,upper_black)

mask= mask1


    # Bitwise-AND mask and original image
res = cv2.bitwise_and(image_test_hsv,image_test_hsv, mask= mask)

plt.figure(4)

display_image(mask,"res")

#print "greeeeeeeeeeeeeen"

red = np.uint8([[[255,0,0 ]]])

hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

#print hsv_red


selected_regions, letters, colorRegions=select_roi3(image_test_original.copy(),mask)



print 

plt.figure(5)
display_image(selected_regions,"regioni")

j=6
k=0
for letter in letters:
    plt.figure(j)
    #print letter
    
    k+=1
    display_image(letter,"region"+str(k))
    j+=1

shape=0

# /-/-/-/-
#selected_test_obucavanje, signs_obucavanje, region_distances_obucavanje, regions_color_obucavanje = my.select_roiV3(image_original_obucavanje.copy(), image_obucavanje)

inputs_obucavanje = prepare_for_ann(letters)

print 'ovdefdasfffffffffffffffffffffffffffffffffffffdasfsdffas'
signs_alphabet = ['pet', 'deset','5najst','2deset','zSmer', 'stop']
outputs_obucavanje = convert_output(signs_alphabet)

print '\nlen(inputs_obucavanje)=', len(inputs_obucavanje), ' len(outputs_obucavanje)=', len(outputs_obucavanje)
ann = create_ann()
ann = train_ann(ann, inputs_obucavanje, outputs_obucavanje)

print 'after train'

# predikcija na osnovu obucene neuronske mreze
image_test_original = load_image('C:\Users\sbstb\Desktop\sm.jpg')

width,height= image_test_original.shape[0:2]


image_test_temp=image_test_original.copy()
image_test = remove_noise(image_bin(image_gray(image_test_temp)))
plt.figure(30)
display_image(image_test_original,"Original")
plt.figure(31)
display_image(image_test_temp,"Procesuirana")
plt.figure(32)
display_image(image_test,"image test")


image_test_hsv=image_test_original.copy()

hsv = cv2.cvtColor(image_test_hsv, cv2.COLOR_BGR2HSV)

black = np.uint8([[[0,0,0 ]]])
hsv_black = cv2.cvtColor(black,cv2.COLOR_BGR2HSV)
print hsv_black, 'ovo je black kolor value'

lower_black1=np.array([0,0,0])
upper_black1=np.array([40,255,255])

    # define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
#mask2 = cv2.inRange(hsv_black,lower_black1,upper_black1)

mask=mask1

    # Bitwise-AND mask and original image
res = cv2.bitwise_and(image_test_hsv,image_test_hsv, mask= mask)

plt.figure(33)

display_image(mask,"res")

#print "greeeeeeeeeeeeeen"

red = np.uint8([[[255,0,0 ]]])

hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

#print hsv_red


selected_regions, signs_test,colorRegions =select_roi3(image_test_original.copy(),mask)

plt.figure(44)
display_image(selected_regions,"regioni")

j=45
k=0
for letter in colorRegions:
    plt.figure(j)
    #print letter
    
    k+=1
    display_image(letter,"region"+str(k))
    j+=1

shape=0


#selected_regions_test, signs_test, region_distances_test, regions_color_test = my.select_roiV3(image_test_original.copy(), image_test)

inputs_test = prepare_for_ann(signs_test)
results_test = ann.predict(np.array(inputs_test, np.float32))
print display_result_ann(results_test, signs_alphabet)
print '\nresults_test=', results_test


# /-/-/-/-

#print "distancesssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"
#print len(distances)
#print len(letters[0])






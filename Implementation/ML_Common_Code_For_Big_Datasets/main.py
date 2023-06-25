import cv2
import numpy as np
import pandas as pd
import glob
from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import time

df = pd.DataFrame()
df_Features = pd.DataFrame()

def readImagesFromFolder():
    dataset = []
    directoryPath = "C:/Users/user/PycharmProjects/FYPJune2023/datasets/"
    global datasetTypeOptionNumber
    datasetTypeOptionNumber = int(input("Enter 1 for Deforestation | 2 for Flood | 3 for Volcano: "))
    global datasetBatchNumber
    datasetBatchNumber = int(input("Enter 1 for 1st batch | 2 for 2nd batch | 3 for 3rd batch | 4 for 4th batch | 5 for 5th batch | 6 for test: "))

    if datasetTypeOptionNumber == 1:
        datasetType = "deforestationLargeDataset"
    elif datasetTypeOptionNumber == 2:
        datasetType = "floodLargeDataset"
    elif datasetTypeOptionNumber == 3:
        datasetType = "volcanoLargeDataset"
    else:
        print("Invalid Input")
        exit()

    print("The dataset batch number", str(datasetBatchNumber))
    print("The dataset batch number type", type(str(datasetBatchNumber)))
    filenames = glob.glob(directoryPath + datasetType + "/Unlabeled/" + str(datasetBatchNumber) + "/*.jpg")
    filenames.sort()
    print("filenames", filenames)
    for img in filenames:
        n = cv2.imread(img, 0)
        if n is not None:
            dataset.append(n)
    print("The length of the image dataset: ", len(dataset))
    return dataset

def normalize(image):
  normalizedImg = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
  return normalizedImg

def calcHistogram(image):
    hist, edges, patches = plt.hist(image.ravel(), 256, [0, 256])
    return hist

def gaussianBlur(image):
    gaussian = cv2.GaussianBlur(image,(5,5),0)
    return gaussian

def findAverage(list):
    avg = sum(list)/len(list)
    return avg

#**********************************************************************************************************************
#Extracting features
G = 256 #NUmber of gray levels in an image

#Statistical features
def calc_Probability_Density(hist_i_value, size):
    return (hist_i_value/size)

def stat_Mean(i, prob_density):
    mean = (i * (prob_density))
    return mean

def stat_Features_Set1(i, mean, prob_density):
    avg_contrast = pow((i - mean), 2) * prob_density
    skewness_component = pow((i - mean), 3) * prob_density
    kurtosis_component = (pow((i - mean), 4) * prob_density) - 3
    energy = pow(prob_density, 2)
    return avg_contrast, skewness_component, kurtosis_component, energy

def calc_Stat_Features(img):
    stat_mean = 0
    stat_avg_contrast =0
    skewness_component=0
    kurtosis_component=0
    hist = calcHistogram(img)
    size = img.size
    for i in range(G):
        p_d = calc_Probability_Density(hist[i], size)
        stat_mean += stat_Mean(i, p_d)
    for i in range(G):
        p_d = calc_Probability_Density(hist[i], size)
        p1, p2, p3, p4 = stat_Features_Set1(i, stat_mean, p_d)
        stat_avg_contrast += p1  # variance^2 (avg_contrast)
        skewness_component += p2 #skewness_component
        kurtosis_component += p3 #kurtosis_component
    skewness = pow(stat_avg_contrast,(-3/2)) * skewness_component
    kurtosis = pow(stat_avg_contrast,(-4/2)) * kurtosis_component
    return stat_mean, stat_avg_contrast, skewness, kurtosis

def gaborFilter(img):
    num = 1
    fimg_list = []
    for theta in range(3):  # 0, 45, 90
        theta = theta / 4 * np.pi  # theta 0, 1/4, 1/2
        for theta_i in (theta - 10, theta - 5, theta):
            for sigma in (3, 5):
                for lamda in np.arange(np.pi / 2, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.5):
                        gabor_label = 'Gabor' + str(num)
                        kernel = cv2.getGaborKernel((5, 5), sigma, theta_i, lamda, gamma, 0,
                                                    ktype=cv2.CV_32F)
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        df[gabor_label] = filtered_img
                        fimg_list.append(fimg)
                        num += 1
    return fimg_list

def cannyEdge(img):
    edges = cv2.Canny(img, 100, 200)
    edges2 = edges.reshape(-1)
    df['Canny edges'] = edges2
    return edges2

def edge_roberts(img):
    edge_robert = roberts(img)
    edge_robert2 = edge_robert.reshape(-1)
    df['Roberts'] = edge_robert2
    return edge_robert2

def edge_sobel(img):
    edge_sobel1 = sobel(img)
    edge_sobel2 = edge_sobel1.reshape(-1)
    df['Sobel'] = edge_sobel2
    return edge_sobel2

def edge_scharr(img):
    edge_scharr1 = scharr(img)
    edge_scharr2 = edge_scharr1.reshape(-1)
    df['Scharr'] = edge_scharr2
    return edge_scharr2

def edge_prewitt(img):
    edge_prewitt1 = prewitt(img)
    edge_prewitt2 = edge_prewitt1.reshape(-1)
    df['Prewitt'] = edge_prewitt2
    return edge_prewitt2

def glcmFeatures(img):
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    diss_sim = (graycoprops(glcm, 'dissimilarity')[0, 0])
    corr = (graycoprops(glcm, 'correlation')[0, 0])
    homogen = (graycoprops(glcm, 'homogeneity')[0, 0])
    energy = (graycoprops(glcm, 'energy')[0, 0])
    contrast = (graycoprops(glcm, 'contrast')[0, 0])
    return diss_sim, corr, homogen, energy, contrast

if __name__ == '__main__':

    images = readImagesFromFolder()
    contrast_stretched_image_list = []
    num_CS = 0
    Canny_Edges_list = []
    Edge_Roberts_List = []
    Edge_Sobel_List = []
    Edge_Scharr_List = []
    Edge_Prewitt_List = []
    Gaussian_img1_List = []
    Gaussian_img3_List = []
    Variance_list = []
    Stat_Mean_List = []
    Stat_Avg_Contrast_List = []
    Stat_Skewness_List = []
    Stat_Kurtosis_List = []
    Stat_Entropy_List = []
    Gabor_Individual_List = []
    GLCM_diss_sim_List = []
    GLCM_corr_List = []
    GLCM_homogen_List = []
    GLCM_energy_List = []
    GLCM_contrast_List = []

    if datasetTypeOptionNumber == 1:
        crop_y = images[0].shape[0] - 40
        crop_x = images[0].shape[1]
        check_crop = 0
    elif datasetTypeOptionNumber == 2:
        crop_y = images[0].shape[0] - 30
        crop_x = images[0].shape[1]
    elif datasetTypeOptionNumber == 3:
        crop_y = 25
        crop_x = images[0].shape[1]

    startTime_1 = time.time()
    for img in images:

        print("length of images list, ", len(images))
        
        if datasetTypeOptionNumber == 1:
            cropped_image = img[0:crop_y, 0:crop_x]
        elif datasetTypeOptionNumber == 2:
            cropped_image = img[0:crop_y, 0:crop_x]
        elif datasetTypeOptionNumber == 3:
            cropped_image = img[crop_y:images[0].shape[0], 0:crop_x]
        CS_img = normalize(cropped_image)
        contrast_stretched_image_list.append(CS_img)

        gaussian_img = gaussianBlur(CS_img)

        canny_edges2 = cannyEdge(gaussian_img)
        avg_canny_edges2 = findAverage(canny_edges2)
        Canny_Edges_list.append(avg_canny_edges2)

        edge_robert2 = edge_roberts(gaussian_img)
        avg_edge_roberts2 = findAverage(edge_robert2)
        Edge_Roberts_List.append(avg_edge_roberts2)

        edge_sobel2 = edge_sobel(gaussian_img)
        avg_edge_sobel2 = findAverage(edge_sobel2)
        Edge_Sobel_List.append(avg_edge_sobel2)

        edge_scharr2 = edge_scharr(gaussian_img)
        avg_edge_scharr2 = findAverage(edge_scharr2)
        Edge_Scharr_List.append(avg_edge_scharr2)

        edge_prewitt2 = edge_prewitt(gaussian_img)
        avg_edge_prewitt2 = findAverage(edge_prewitt2)
        Edge_Prewitt_List.append(avg_edge_prewitt2)

        gabor_images = gaborFilter(gaussian_img)
        total_gabor_image_values = []
        for img in gabor_images:
            g = img.reshape(-1)
            avg_g = findAverage(g)
            total_gabor_image_values.append(avg_g)
        Gabor_Individual_List.append(total_gabor_image_values)

        stat_mean, stat_avg_contrast, skewness, kurtosis = calc_Stat_Features(gaussian_img)
        Stat_Mean_List.append(stat_mean)
        Stat_Avg_Contrast_List.append(stat_avg_contrast)
        Stat_Skewness_List.append(skewness)
        Stat_Kurtosis_List.append(kurtosis)

        diss_sim, corr, homogen, energy, contrast = glcmFeatures(gaussian_img)
        GLCM_diss_sim_List.append(diss_sim)
        GLCM_corr_List.append(corr)
        GLCM_homogen_List.append(homogen)
        GLCM_energy_List.append(energy)
        GLCM_contrast_List.append(contrast)

        print(num_CS)
        num_CS = num_CS + 1

    executionTime_1 = (time.time() - startTime_1)
    print("Time taken to complete pre-processing and feature extraction of ", len(images), " images = ", executionTime_1, " seconds")

    Gabor_Total_Dict = {}
    for i in range(len(Gabor_Individual_List[0])):
        Gabor_Total_Dict['GB'+str(i)]=[]


    for i in Gabor_Individual_List:
        image_values = i
        num = 0
        for j in image_values:
            label = 'GB'+str(num)
            val = Gabor_Total_Dict[label]
            val.append(j)
            Gabor_Total_Dict[label] = val
            num += 1

    print("Length of the Gabor dictionary", len(Gabor_Total_Dict))
    Gray_Images_List = []
    for i in range(len(images)):
        Gray_Images_List.append(i+1)

    df_Features['Gray Level Images'] = Gray_Images_List
    df_Features['CannyEdge'] = Canny_Edges_list
    df_Features['EdgeRoberts'] = Edge_Roberts_List
    df_Features['EdgeSobel'] = Edge_Sobel_List
    df_Features['EdgeScharr'] = Edge_Scharr_List
    df_Features['EdgePrewitt'] = Edge_Prewitt_List
    df_Features['StatMean'] = Stat_Mean_List
    df_Features['StatAvgContrast'] = Stat_Avg_Contrast_List
    df_Features['StatSkewness'] = Stat_Skewness_List
    df_Features['StatKurtosis'] = Stat_Kurtosis_List

    Gabor_num = 0
    for i in range(len(Gabor_Total_Dict)):
        label = 'GB' + str(Gabor_num)
        val = Gabor_Total_Dict[label]
        df_Features[label] = val
        Gabor_num += 1

    df_Features['GLCM_Diss_similarity'] = GLCM_diss_sim_List
    df_Features['GLCM_Correlation'] = GLCM_corr_List
    df_Features['GLCM_Homogeneity'] = GLCM_homogen_List
    df_Features['GLCM_Energy'] = GLCM_energy_List
    df_Features['GLCM_Contrast'] = GLCM_contrast_List

    if datasetTypeOptionNumber == 1:
        datasetType = "deforestationLargeDataset"
    elif datasetTypeOptionNumber == 2:
        datasetType = "floodLargeDataset"
    elif datasetTypeOptionNumber == 3:
        datasetType = "volcanoLargeDataset"

    df_Features.to_csv('C:/Users/user/PycharmProjects/FYPJune2023/files/' + datasetType + str(datasetBatchNumber) + '.csv')

    executionTime_2 = (time.time() - startTime_1)
    print("Time taken to plot graphs and create csv files = ", executionTime_2, " seconds")



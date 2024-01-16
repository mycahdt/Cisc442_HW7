import cv2 as cv
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 1) Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice. 
def multiResolution(img, n):

    # Array that holds the n levels of images
    multiResImgs = [img]

    # Loops through for n-1 levels
    for i in range(n-1):

        # Uses resize() to reduce the side of the image by half its width and height
        resizedImg = cv.resize(img, None, fx = 0.5, fy = 0.5, interpolation=cv.INTER_AREA)

        # Uses GaussianBlur() as the Gaussian kernel
        img = cv.GaussianBlur(resizedImg,(5,5),0)

        # Adds the new image to the array of multi-resolution images
        multiResImgs.append(img)

    return multiResImgs



# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 2) Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
def multiScale(img, n):
    # Array that holds the n levels of images
    multiScaleImgs = [img]

    # Loops through for n-1 levels
    for i in range(n-1):

        # Uses GaussianBlur() as the Gaussian kernel
        img = cv.GaussianBlur(img,(5,5),0)

        # Adds the new image to the array of multi-resolution images
        multiScaleImgs.append(img)

    return multiScaleImgs



# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3) Generate Laplacian planes using a Laplacian kernel of your choice 
# (can use: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#cv2.Laplacian).
def laplPlane(img):

    # Uses GaussianBlur() on the image
    blurImg = cv.GaussianBlur(img,(3,3),0)

    # Uses Laplacian operator to generate laplacian planes
    laplPlaneImg = cv.Laplacian(blurImg,cv.CV_64F)

    return laplPlaneImg




# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 4) Generate an approximation to Laplacian using the difference of Gaussian planes from (1). 
# Note: you need to do 'Expand' on images before taking the difference.
def diffOfGausPlanesA(imgs):

    diffOfGausImgs = [None]*len(imgs)
    diffOfGausImgs[len(imgs)-1] = imgs[len(imgs)-1]

    for i in range(len(imgs)-1, 0, -1):
        expanded = cv.resize(imgs[i], None, fx = 2, fy = 2, interpolation=cv.INTER_AREA)
        newImg = imgs[i-1] - expanded
        diffOfGausImgs[i-1] = newImg

    return diffOfGausImgs




# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 5) Generate an approximation to Laplacian using the difference of Gaussian planes from (2)
def diffOfGausPlanesB(imgs):

    finalImg = imgs[2] - imgs[3]
    finalImg += imgs[1] - imgs[2]
    finalImg += imgs[0] - imgs[1]
    return finalImg




def main():
    image = cv.imread('Einstein.jpg')

    # 1) Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice. 
    multiResImages = multiResolution(image, 4)


    # 2) Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
    multiScaleImages = multiScale(image, 4)

    # 3) Generate Laplacian planes using a Laplacian kernel of your choice 
    # (can use: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#cv2.Laplacian).
    laplPlaneImage = laplPlane(image)

    # 4) Generate an approximation to Laplacian using the difference of Gaussian planes from (1). 
    # Note: you need to do 'Expand' on images before taking the difference.
    dogImgsA = diffOfGausPlanesA(multiResImages)
    
    # 5) Generate an approximation to Laplacian using the difference of Gaussian planes from (2)
    dogImgB = diffOfGausPlanesB(multiScaleImages)


    # Displays Images
    cv.imshow('Original Image', image)

    # 1.
    # Shows the 4 levels of multi-resolution
    cv.imshow('Multi-resolution Image 1', multiResImages[0])
    cv.imshow('Multi-resolution Image 2', multiResImages[1])
    cv.imshow('Multi-resolution Image 3', multiResImages[2])
    cv.imshow('Multi-resolution Image 4', multiResImages[3])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 2.
    # Shows the 4 levels of multi-scale
    cv.imshow('Multi-scale Image 1', multiScaleImages[0])
    cv.imshow('Multi-scale Image 2', multiScaleImages[1])
    cv.imshow('Multi-scale Image 3', multiScaleImages[2])
    cv.imshow('Multi-scale Image 4', multiScaleImages[3])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 3.
    # Shows the Lalacian Planes using a Laplacian kernel
    cv.imshow('Laplacian Planes Image', laplPlaneImage)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 4.
    # Shows the Difference of Gaussian Planes from the multi-resolution
    cv.imshow('Difference of Gaussian Planes Image A1', dogImgsA[0])
    cv.imshow('Difference of Gaussian Planes Image A2', dogImgsA[1])
    cv.imshow('Difference of Gaussian Planes Image A3', dogImgsA[2])
    cv.imshow('Difference of Gaussian Planes Image A4', dogImgsA[3])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 5.
    # Shows the Difference of Gaussian Planes from the multi-scale
    cv.imshow('Difference of Gaussian Planes Image B1', dogImgB)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
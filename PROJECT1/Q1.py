import cv2
import numpy as np
import matplotlib.pyplot as plt


def noise_remove_median(img):
    new_img = img.copy()
    height, width = img.shape[:]

    # truncate the image to fit edges
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            new_img[i][j] = np.median(img[i-1:i+2, j-1:j+2])
    # cv2.imwrite('Q1_image/output_noise.png', new_img)
    return new_img


def histogram_equalization(img):
    img = img.astype(float)
    img = img / 255
    height, width = img.shape[:]

    unique_ele, cnt = np.unique(img, return_counts=True)
    cnt = cnt/(height*width)
    cnt = np.cumsum(cnt)
    # print(cnt)

    # replace the original pixel with new one
    for i in range(len(unique_ele)):
        img[img == unique_ele[i]] = cnt[i]
    img = img*255
    # cv2.imwrite('Q1_image/output_hist.png', img.astype(np.uint8))

    return img.astype(np.uint8)


def spatial_Laplacian_filter_variant(img):
    height, width = img.shape[:]

    new_img = img.copy().astype(int)
    for i in range(1, height-1):
        for j in range(1, width-1):
            new_img[i][j] = 9*img[i][j]-img[i-1][j]-img[i+1][j]-img[i][j-1]\
                            - img[i][j+1]-img[i-1][j-1]-img[i-1][j+1]-img[i+1][j-1]-img[i+1][j+1]
            new_img[i][j] = np.clip(new_img[i][j], 0, 255)

    return new_img.astype(np.uint8)


def spatial_sobel_operator(img):
    height, width = img.shape[:]

    new_img = img.copy().astype(int)
    new_img2 = img.copy().astype(int)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            new_img[i][j] = - 2*img[i - 1][j] + 2*img[i + 1][j] - img[i - 1][j - 1] - img[i - 1][j + 1] + img[i + 1][j - 1] + img[i + 1][j + 1]
            new_img[i][j] = np.clip(new_img[i][j], 0, 255)
            new_img2[i][j] = - 2 * img[i][j - 1] + 2 * img[i][j + 1] - img[i - 1][j - 1] + img[i - 1][j + 1] - img[i + 1][j - 1] + img[i + 1][j + 1]
            new_img2[i][j] = np.clip(new_img2[i][j], 0, 255)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            img[i][j] = np.clip(img[i][j]-new_img[i][j]-new_img2[i][j], 0, 255)

    # cv2.imwrite('Q1_image/sobel.png', new_img+new_img2)
    return img.astype(np.uint8)


def log_brightness_enhancement(img):

    # brightness enhancement
    img = img.astype(float)
    img = img/255
    # img = np.power(img, 0.2) # histogram
    img = np.power(img, 0.8) # Laplacian
    # img = np.power(img, 0.6) # sobel
    img = img*255

    return img.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread('Q1_image/test.png', cv2.IMREAD_GRAYSCALE)
    # noise removing
    img = noise_remove_median(img)

    # different contrast enhancement
    # img = histogram_equalization(img)
    img = spatial_Laplacian_filter_variant(img)
    # img = spatial_sobel_operator(img)

    # brightness enhancement
    img = log_brightness_enhancement(img)
    cv2.imwrite('Q1_image/output_1.png', img)


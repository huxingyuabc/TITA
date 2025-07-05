import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def y2rgb(image_A, image_B, fused):
    # Read the image_A, image_B, and fused images in cv2 format (H, W, C) with range [0, 255]
    # assert fused.shape[2] == 1

    # Convert to YCrCb color space
    ycbcr_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2YCrCb).astype(float)
    ycbcr_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2YCrCb).astype(float)

    # Extract Cb and Cr channels
    cb_A = ycbcr_A[:, :, 2]
    cr_A = ycbcr_A[:, :, 1]
    cb_B = ycbcr_B[:, :, 2]
    cr_B = ycbcr_B[:, :, 1]

    # Initialize the Cb and Cr fusion
    height, width = image_A.shape[:2]
    cb_fused = np.zeros((height, width), dtype=float)
    cr_fused = np.zeros((height, width), dtype=float)

    # Fuse Cb/Cr channels
    mask_A = np.abs(cb_A - 128)
    mask_B = np.abs(cb_B - 128)
    mask_combined = mask_A + mask_B
    cb_fused[mask_combined != 0] = (cb_A[mask_combined != 0] * mask_A[mask_combined != 0]
                                    + cb_B[mask_combined != 0] * mask_B[mask_combined != 0]) / mask_combined[mask_combined != 0]
    cb_fused[mask_combined == 0] = 128

    mask_A = np.abs(cr_A - 128)
    mask_B = np.abs(cr_B - 128)
    mask_combined = mask_A + mask_B
    cr_fused[mask_combined != 0] = (cr_A[mask_combined != 0] * mask_A[mask_combined != 0]
                                    + cr_B[mask_combined != 0] * mask_B[mask_combined != 0]) / mask_combined[mask_combined != 0]
    cr_fused[mask_combined == 0] = 128

    # Construct the final fused YCbCr image and convert it to BGR
    img_fused_ycbcr = np.empty((height, width, 3), dtype=np.uint8)
    img_fused_ycbcr[:, :, 0] = fused
    img_fused_ycbcr[:, :, 2] = cb_fused
    img_fused_ycbcr[:, :, 1] = cr_fused
    img_fused = cv2.cvtColor(img_fused_ycbcr, cv2.COLOR_YCrCb2BGR)

    return img_fused

if __name__ == '__main__':
    # Define the directories
    folder1 = 'far_near/test_imgs/far'
    folder2 = 'far_near/test_imgs/near'
    folderf = 'far_near/results/'

    # Get the list of image file paths
    filepath1 = glob.glob(os.path.join(folder1, "*.jpg"))
    filepath2 = glob.glob(os.path.join(folder2, "*.jpg"))
    filepathf = glob.glob(os.path.join(folderf, "*.jpg"))

    # Process the images
    for pic in range(1, 21):  # Assuming you have 20 images numbered from 1 to 20
        img1 = cv2.imread(os.path.join(folder1, f'{pic}.jpg'))
        img2 = cv2.imread(os.path.join(folder2, f'{pic}.jpg'))
        imgf_y = cv2.imread(os.path.join(folderf, f'{pic}.jpg'), cv2.IMREAD_GRAYSCALE)

        height, width, channel = img1.shape
        if channel == 3:
            ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb).astype(float)
            cb1 = ycbcr1[:, :, 2]
            cr1 = ycbcr1[:, :, 1]

        height, width, channel = img2.shape
        if channel == 3:
            ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb).astype(float)
            cb2 = ycbcr2[:, :, 2]
            cr2 = ycbcr2[:, :, 1]

        # Fuse Cb/Cr
        if 'cb1' in locals() and 'cb2' in locals():
            cbf = np.where((cb1 == 128) & (cb2 == 128), 128,
                           (cb1 * np.abs(cb1 - 128) + cb2 * np.abs(cb2 - 128)) /
                           (np.abs(cb1 - 128) + np.abs(cb2 - 128)))
        elif 'cb1' in locals():
            cbf = cb1
        elif 'cb2' in locals():
            cbf = cb2

        if 'cr1' in locals() and 'cr2' in locals():
            crf = np.where((cr1 == 128) & (cr2 == 128), 128,
                           (cr1 * np.abs(cr1 - 128) + cr2 * np.abs(cr2 - 128)) /
                           (np.abs(cr1 - 128) + np.abs(cr2 - 128)))
        elif 'cr1' in locals():
            crf = cr1
        elif 'cr2' in locals():
            crf = cr2

        if 'cbf' in locals() and 'crf' in locals():
            imgf_ycbcr = np.empty((height, width, 3), dtype=float)
            imgf_ycbcr[:, :, 0] = imgf_y
            imgf_ycbcr[:, :, 2] = cbf
            imgf_ycbcr[:, :, 1] = crf
            imgf = cv2.cvtColor(imgf_ycbcr.astype('uint8'), cv2.COLOR_YCrCb2BGR)

        # # Plot the images
        # plt.figure(1)
        # plt.subplot(131), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # plt.subplot(132), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # plt.subplot(133), plt.imshow(cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB))
        # plt.show()

        # Save the fused image
        savepath = os.path.join(folderf, 'rgb', f'{pic}.jpg')
        cv2.imwrite(savepath, imgf)

        # Clear variables for the next iteration
        if 'cbf' in locals():
            del cbf
        if 'crf' in locals():
            del crf

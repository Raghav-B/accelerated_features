from modules.xfeat import XFeat
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# Returns the percentage of keypoint matches found relative to all keypoints detected in img1
# img1 should be completely visible in img2
def repeatability_test(out1, out2, img1, img2):
    min_cossim = -1
    idxs0, idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
    num_detected = out1['keypoints'].shape[0]
    num_matched = out2['keypoints'][idxs1].shape[0]

    keypoints_np = out1['keypoints'][idxs0].cpu().numpy()
    keypoints_cv = [cv2.KeyPoint(float(x), float(y), 3) for x, y in keypoints_np]
    outputImage = cv2.drawKeypoints(img1, keypoints_cv, 0, (0, 255, 0), None)
    plt.figure()
    plt.imshow(outputImage)
    print("Repeatability Test:", num_matched/num_detected)

def saliency_heatmap(out, img1, overlay = False):
    heatmap_original = out['heatmap'].squeeze().cpu().numpy() # 4d array of B, C H, W becomes 2D of H, W
    salient = out['filtered_salient']
    heatmap = np.zeros_like(heatmap_original)
    print(img1.shape)

    for(x, y) in salient[0]:
        x = int(x.item())
        y = int(y.item())
        heatmap[y, x] = heatmap_original[y, x]

    display_heatmap(img1, heatmap=heatmap, overlay=overlay)

    
def reliability_heatmap(out, img1, overlay = False):
    keypoints = out['keypoints'].squeeze().cpu().numpy() #2D Array of keypoint coordinates
    scores = out['scores'].squeeze().cpu().numpy() # 1D array of reliability scores
    heatmap = np.zeros((img1.shape[0], img1.shape[1]))

    print(keypoints.shape)
    for i, (x, y) in enumerate(keypoints):
        x = int(x.item())
        y = int(y.item())
        heatmap[y, x] = scores[i]

    display_heatmap(img1, heatmap=heatmap, overlay=overlay)

    
def display_heatmap(img1, heatmap, overlay):
    # Apply blurring to heatmap
    heatmap = gaussian_filter(heatmap, sigma=3) # Increase sigma if more blur is required

    # Normalise the values [0, 1] and then scale it between [0, 255]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8) 
    
    # Apply colourmap    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO) # plasma is good too
     
    # Resize the heatmap, if it doesnt match the image dimensions due to downsampling
    heatmap = cv2.resize(heatmap, (img1.shape[1], img1.shape[0]))  

    if(overlay):
        heatmap = cv2.addWeighted(img1, 0.5, heatmap, 0.5, 0)
    
    plt.figure()
    plt.imshow(heatmap, cmap="inferno")
    plt.colorbar()  # Add a colorbar for reference
    plt.title('Smoothed Heatmap')


def display_image(out, img1):
    keypoints = out["keypoints"]
    keypoints_np = keypoints.cpu().detach().numpy()
    keypoints_cv = [cv2.KeyPoint(float(x), float(y), 3) for x, y in keypoints_np]
    outputImage = cv2.drawKeypoints(img1, keypoints_cv, 0, (0, 255, 0), None)
    plt.figure()
    plt.imshow(outputImage)


xfeat = XFeat()

# Load the images
im1 = cv2.imread('./assets/ref.png')
im2 = cv2.imread('./assets/tgt.png')
out1 = xfeat.detectAndCompute(im1, top_k=4096)[0]
out2 = xfeat.detectAndCompute(im2, top_k=4096)[0]

repeatability_test(out1, out2, im1, im2)
saliency_heatmap(out1, im1, overlay=True)    
reliability_heatmap(out1, im1, overlay=True)
display_image(out1, im1)
plt.show()


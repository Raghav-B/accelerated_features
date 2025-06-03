from modules.xfeat import XFeat
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np

xfeat = XFeat()


# Load the images
im1 = cv2.imread('./assets/ref.png')
im2 = cv2.imread('./assets/tgt.png')

#Simple inference with batch sz = 1
# output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]

output = xfeat.detectAndCompute(im1, top_k = 4096, detection_threshold = 0.7)[0]

keypoints = output["keypoints"]
keypoints_np = keypoints.cpu().detach().numpy()
keypoints_cv = [cv2.KeyPoint(float(x), float(y), 3) for x, y in keypoints_np]

heatmap = output["heatmap"] # keypoint heatmap, probability that each pixel is a potential keypoint
heatmap = heatmap.squeeze() # converting a 4d array into a 2d array tensor
heatmap = heatmap.cpu().detach().numpy() # converting into numpy array

logits = output["logits"]




# rmap = output["scores"]
# rmap = rmap.cpu().detach().numpy()
# rmap = rmap.squeeze()
# rmap = rmap.reshape(64,64)

# sns.heatmap(rmap, cmap='viridis', annot=False)
# plt.show()

# Scores is mapped to keypoints, so can use keypoints coordinates to populate the heatmap for reliability map


# #heatmap = cv2.GaussianBlur(heatmap, (0,0), sigmaX=7, sigmaY=7, borderType=cv2.BORDER_REPLICATE)
# heatmap = heatmap / heatmap.max()
# # Convert heatmap float [0,1] to uint8 [0,255]
# heatmap_uint8 = np.uint8(255 * heatmap)
# # Apply a colormap (e.g., COLORMAP_JET or COLORMAP_VIRIDIS)
# colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
# plt.figure(figsize=(8, 6))
# plt.imshow(colored_heatmap)

# # Shows the circled keypoints
outputImage = cv2.drawKeypoints(im1, keypoints_cv, 0, (0, 255, 0), None)
print(outputImage.shape)
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
# for some reason the heatmap doesnt have the same number of pixels as the original image
heatmap_resized = cv2.resize(heatmap_color, (outputImage.shape[1], outputImage.shape[0]))  # to (600, W)
overlay = cv2.addWeighted(outputImage, 0.4, heatmap_resized, 0.6, 0)
plt.figure(figsize=(12,12))
plt.imshow(outputImage[..., ::-1])#, plt.show()
plt.imshow(overlay)


# # Create heatmap 
# plt.figure(figsize=(12, 12))
# sns.heatmap(heatmap, cmap='viridis', annot=False)
# plt.title('Heatmap of PyTorch Tensor')
# plt.show()


# # Reliability Map



plt.figure(figsize=(12, 12))
H, W = heatmap.shape
#heatmap = cv2.resize(heatmap, (int(W/8), int(H/8)), interpolation=cv2.INTER_LINEAR)
sns.heatmap(heatmap, cmap='viridis', annot=False)
plt.title('Heatmap of PyTorch Tensor')
plt.show()
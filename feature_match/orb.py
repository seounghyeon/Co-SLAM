import cv2
import torch
import numpy as np


"""
SIFT Feature Matching Class

- Inputs are index of images
- image1 is query image, image2 is train image
- both images where keypoints should be found and matched
- outputs:  - uv_1      uv coordinates of keypoints in image 1 (in order of best to worst match) 
            - uv_2      uv coordinates of keypoints in image 2 (in order of best to worst match) 
            - index     indices of matched uv coordinate in 1D tensor
            - matches

"""
class ORBMatcher:
    def __init__(self):
        self.orb = cv2.ORB.create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, image1, image2, idx, Hedge, Wedge, gt_color):
        # print("inside match")

        debug_uv = False
        if image1 is None:
            print("\nTHIS IS NONE IN IMAGE1IN no previous image saved up\n\n")
            return None, None, None, None
        
        # Upload images to GPU memory
        image1_gpu = cv2.cuda_GpuMat()
        image2_gpu = cv2.cuda_GpuMat()
        image1_gpu.upload(image1)
        image2_gpu.upload(image2)

  
        debug = False

        # Detect ORB keypoints and descriptors
        keypoints_1, descriptors_1 = self.orb.detectWithDescriptorsAsync(image1_gpu, None)
        keypoints_2, descriptors_2 = self.orb.detectWithDescriptorsAsync(image2_gpu, None)

        # Download keypoints and descriptors from GPU to CPU
        keypoints_1 = keypoints_1.download()
        keypoints_2 = keypoints_2.download()
        descriptors_1 = descriptors_1.download()
        descriptors_2 = descriptors_2.download()

        if descriptors_1 is None or descriptors_2 is None:
            return None, None, None, None, None, None, None, None

        # Create matches using brute force algorithm
        matches = self.bf.match(descriptors_1, descriptors_2)
        # Sort matches by their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # # matches saved in tensor
        # matches_tensor = torch.tensor([(match.queryIdx, match.trainIdx, match.distance) for match in matches], dtype=torch.float32)
        # if(debug):
        #     print("size of matches",matches_tensor.size())

        """
        UV NEEDS TO ADD 20 because the uv here starts at 0 but in reality it starts at 20
        it was reduced down
        """





        # hier

        u_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)
        uv_1 = torch.stack((u_reshaped_1, v_reshaped_1), dim=1)

        u_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)

        uv_2 = torch.stack((u_reshaped_2, v_reshaped_2), dim=1)
        
        
        
        # print("\nuv_2 in sift: \n", uv_2[:10])
        if(debug):
            # print("u_reshaped first 10: ",u_reshaped_1[:10])
            # print("v_reshaped first 10: ",v_reshaped_1[:10])
            print("combined tensor keypoints1:", uv_1[:10])
            print("u kp2 first 10: ",u_reshaped_1[:10])
            print("v kp2 first 10: ",v_reshaped_1[:10])
            print("kp2 first 10:", uv_1[:10])

            # shows image 2 with the first 10 keypoints of the matches 
            for uv in uv_2[:10]:
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(image2, (u, v), radius=4, color=(0, 255, 0), thickness=-1)  # Draw a green circle
            # cv2.imshow('image2', image2)


            for uv in uv_1[:10]:
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(image1, (u, v), radius=4, color=(0, 255, 0), thickness=-1)  # Draw a green circle


        # starts at Wedge is 20
        # uv_2 += 20
        # print("\nuv_2 in sift: \n", uv_2[:10])





        # print("IMAGE SHAPE ",image1.shape[0], image1.shape[1])          # (420, 580, 3) with -20 on each side and top and bottom
    

        # (row(u) * width) + col(v) computes the index of uv coord (in the 1D tensor)
        W1 = image1.shape[1]
        index_1 = (v_reshaped_1 * W1) + u_reshaped_1
        index_2 = (v_reshaped_2 * W1) + u_reshaped_2
        if(debug):
            print("index size: ", index_1.size())
            print("index first: ", index_1)
            print("index size: ", index_2.size())
            print("index second: ", index_2)


        index_1 = index_1.cpu()
        index_2 = index_2.cpu()


        if (debug_uv):     
            matched_image = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches, None, flags=2)
            cv2.imwrite(f'/home/shham/Pictures/cs_match/match_{idx}.jpg', matched_image)
            cv2.imwrite(f'/home/shham/Pictures/cs_match/match_{idx}.jpg', matched_image)
            cv2.imwrite(f'/home/shham/Pictures/cs_match2/match_{idx}_prev.jpg', image1)

            cv2.imwrite(f'/home/shham/Pictures/cs_match2/match_{idx}.jpg', image2)
        uv_1 = uv_1.to(torch.float32)
        uv_2 = uv_2.to(torch.float32)

        # Extract colors from the images at the UV coordinates
        colors_1 = torch.tensor([image1[int(v), int(u)] for u, v in uv_1], dtype=torch.float32, device=self.device)
        colors_2 = torch.tensor([image2[int(v), int(u)] for u, v in uv_2], dtype=torch.float32, device=self.device)

        # for uncropped frame uv coordinate calculation current frame
        u_full = u_reshaped_2 + Wedge 
        v_full = v_reshaped_2 + Hedge 
        uv__full = torch.stack((u_full, v_full), dim=1)

        Width = W1 + 2 * Wedge
        index_full_cur = (v_full * Width) + u_full


        # for uncropped frame uv coordinate calculation current frame
        u_full_p = u_reshaped_1 + Wedge 
        v_full_p = v_reshaped_1 + Hedge 
        uv__full_prev = torch.stack((u_full_p, v_full_p), dim=1)

        index_full_prev = (v_full_p * Width) + u_full_p


        if (debug_uv):     
            for uv in uv__full[:10]:
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(np_gt, (u, v), radius=4, color=(0, 255, 0), thickness=-1)  # Draw a green circle
            cv2.imwrite(f'/home/shham/Pictures/gt_image/match_{idx}cur.jpg', np_gt)

        u_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)


        return uv_1, uv_2, index_1, index_2, colors_2, colors_1, index_full_cur, index_full_prev


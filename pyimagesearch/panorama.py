# import the necessary packages
import numpy as np
import cv2
import imutils

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # Unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # If the match is None, then there aren’t enough keypoints to create a panorama
        if M is None:
            return None

        # Otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M

        # Get the dimensions of the two images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        # Compute the canvas size by transforming the corners of imageA
        corners = np.array([
            [0, 0],
            [0, hA],
            [wA, hA],
            [wA, 0]
        ], dtype="float32")

        # Apply homography to the corners of imageA
        warped_corners = cv2.perspectiveTransform(np.array([corners]), H)[0]

        # Get the bounding box of the new panorama
        [min_x, min_y] = np.int32(warped_corners.min(axis=0))
        [max_x, max_y] = np.int32(warped_corners.max(axis=0))

        # Calculate the size of the resulting canvas
        width = max(max_x, wB) - min(min_x, 0)
        height = max(max_y, hB) - min(min_y, 0)

        # Adjust the homography to shift the panorama within the visible canvas
        translation_matrix = np.array([
            [1, 0, -min(min_x, 0)],
            [0, 1, -min(min_y, 0)],
            [0, 0, 1]
        ])

        # Apply the translation to the homography matrix
        H = translation_matrix @ H

        # Warp imageA to the new canvas size
        result = cv2.warpPerspective(imageA, H, (width, height))

        # Paste imageB into the panorama
        result[-min(min_y, 0):hB - min(min_y, 0), -min(min_x, 0):wB - min(min_x, 0)] = imageB

        # Check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        # Return the stitched image
        return result

    def detectAndDescribe(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # If we're using OpenCV 3.X, use SIFT
        if self.isv3:
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        else:
            # Use SIFT for OpenCV 2.4.X
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # Convert the keypoints to a NumPy array
        kps = np.float32([kp.pt for kp in kps])

        # Return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # Loop over the raw matches
        for m in rawMatches:
            # Ensure the distance is within a certain ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        # Otherwise, no homography could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # Loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

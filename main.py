import argparse
import logging
import cv2
import numpy as np
import os
import sys
import warnings


class Stitcher():
    def __init__(self, args):
        """
        image_dir: 'directory' containing all images
        key_image_path: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        """
        self.input_dir = args.input_dir

        if args.sequence is None:
            logging.info("No sequence given. Auto computing will be used")
            try:
                self.image_files = os.listdir(self.input_dir)
            except:
                logging.error(f"Unable to open directory: {self.input_dir}")
                sys.exit(-1)
            self.auto_sequence = True
        else:
            logging.info(f"We got a sequence {args.sequence}")
            self.image_files = args.sequence.split(',')
            self.image_files = [image_file+'.jpg' for image_file in self.image_files]
            self.auto_sequence = False
        self.image_paths = list(map(lambda x: os.path.join(self.input_dir, x), self.image_files))

        if not args.sequence is None:
            logging.info(f"Key image is the first element of sequence {self.image_files[0]}")
            self.key_image_path = self.image_paths[0]
        elif args.key_image:
            logging.info(f"Key image given by the user {args.key_image+'.jpg'}")
            self.key_image_path = os.path.join(self.input_dir,args.key_image+'.jpg')
        else:
            logging.info(f"No key image was given. Auto computing")
            self.key_image_file = str(self.find_anchor_img(self.image_paths)) + '.jpg'
            self.key_image_path = os.path.join(self.input_dir, self.key_image_file)

        self.image_paths = list(filter(lambda x: x != self.key_image_path, self.image_paths))
        base_img_rgb = cv2.imread(self.key_image_path)

        if base_img_rgb is None:
            raise IOError(f"{self.key_image_path} doesn't exist")
        logging.info("Starting stitching...")
        self.final_img = self.stitch_images(base_img_rgb)

    def find_anchor_img(self, dir_list):
        imgs = []
        for i in dir_list:
            imgs.append(cv2.imread(i))
        count = len(imgs)
        score = list(range(count))
        dscs_list = []

        for i in imgs:
            detector = cv2.xfeatures2d.SIFT_create()
            kps, dscs = detector.detectAndCompute(i, None)
            dscs_list.append(dscs)

        flann_params = dict(algorithm=1,
                            trees=5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        logging.info('Looking for anchor images... ')
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                matches = matcher.knnMatch(dscs_list[j], trainDescriptors=dscs_list[i], k=2)
                matches = self.filter_matches(matches)
                score[i] += self.image_ditance(matches)
        logging.info("Found Anchor image: " + str(score.index(min(score))))
        return score.index(min(score))

    def filter_matches(self, matches, k=2, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == k and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        return filtered_matches

    def image_ditance(self, matches):
        distances = [match.distance for match in matches]
        return sum(distances)

    def compute_image_features(self, detector, matcher, base_descs, base_features, next_img):
        logging.debug("Finding points...")

        # Find points in the next frame
        next_features, next_descs = detector.detectAndCompute(next_img, None)

        matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

        logging.debug("Match Count: ", len(matches))

        matches_subset = self.filter_matches(matches)

        logging.debug("Filtered Match Count: ", len(matches_subset))

        distance = self.image_ditance(matches_subset)

        logging.debug("Distance from Key Image: ", distance)

        averagePointDistance = distance / float(len(matches_subset))

        logging.debug("Average Distance: ", averagePointDistance)

        kp1 = []
        kp2 = []

        for match in matches_subset:
            kp1.append(base_features[match.trainIdx])
            kp2.append(next_features[match.queryIdx])

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        logging.debug(f'{np.sum(status)} / {len(status)}  inliers/matched')

        inlierRatio = float(np.sum(status)) / float(len(status))

        return H, inlierRatio, averagePointDistance, next_features, next_descs, matches_subset

    def find_next_image(self, detector, matcher, base_descs, base_features):
        closest_image = None
        # Find the best next image from the remaining images
        for next_img_path in list(self.image_paths):

            logging.debug("Reading %s..." % next_img_path)

            # Read in the next image...
            next_img_rgb = cv2.imread(next_img_path)
            next_img = cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY)

            H, inlierRatio, averagePointDistance, next_features, next_descs, matches_subset = self.compute_image_features(
                detector, matcher, base_descs, base_features, next_img)

            # if ( closest_image == None or averagePointDistance < closest_image['dist'] ):
            if closest_image is None or inlierRatio > closest_image['inliers']:
                closest_image = {}
                closest_image['h'] = H
                closest_image['inliers'] = inlierRatio
                closest_image['dist'] = averagePointDistance
                closest_image['path'] = next_img_path
                closest_image['rgb'] = next_img_rgb
                closest_image['img'] = next_img
                closest_image['feat'] = next_features
                closest_image['desc'] = next_descs
                closest_image['match'] = matches_subset

        return closest_image

    def warpTwoImages(self, img1, img2, H):
        '''warp img2 to img1 with homograph H'''
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        Ht = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])  # translate

        result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))
        result[-ymin:-ymin + h1, -xmin:-xmin + w1] = img1
        return result

    def crop(self, img):
        x = np.nonzero(img)
        yy = x[0]
        xx = x[1]
        minX = np.min(xx)
        maxX = np.max(xx)
        minY = np.min(yy)
        maxY = np.max(yy)
        return img[minY:maxY, minX:maxX]

    def stitch_images(self, base_img_rgb):

        if len(list(self.image_paths)) < 1:
            return base_img_rgb
        base_img = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2GRAY)

        # Use the SIFT feature detector
        detector = cv2.xfeatures2d.SIFT_create()

        # Find key points in base image for motion estimation
        base_features, base_descs = detector.detectAndCompute(base_img, None)

        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})

        logging.debug("Iterating through next images...")

        if self.auto_sequence:
            closest_image = self.find_next_image(detector=detector, matcher=matcher, base_descs=base_descs,
                                                 base_features=base_features)
        else:
            next_img_path = self.image_paths.pop()
            next_img_rgb = cv2.imread(next_img_path)
            next_img = cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY)

            H, inlierRatio, averagePointDistance, next_features, next_descs, matches_subset = \
                self.compute_image_features(detector=detector, matcher=matcher, base_descs=base_descs,base_features=base_features, next_img=next_img)
            closest_image = {}
            closest_image['h'] = H
            closest_image['inliers'] = inlierRatio
            closest_image['dist'] = averagePointDistance
            closest_image['path'] = next_img_path
            closest_image['rgb'] = next_img_rgb
            closest_image['img'] = next_img
            closest_image['feat'] = next_features
            closest_image['desc'] = next_descs
            closest_image['match'] = matches_subset

        logging.debug("Closest Image: ", closest_image['path'])
        logging.debug("Closest Image Ratio: ", closest_image['inliers'])
        self.image_paths = list(filter(lambda x: x != closest_image['path'], self.image_paths))
        H = closest_image['h']
        # Warp the new image given the homography from the old image
        final_img = self.warpTwoImages(closest_image['rgb'], base_img_rgb, H)
        final_img = self.crop(final_img)
        return self.stitch_images(final_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to save the ouput')
    parser.add_argument('--sequence', type=str, required=False,
                        help='Sequence of the images to process. If not given, the code tries to compute it '
                             'automatically')
    parser.add_argument('--key_image', type=str, required=False,
                        help='File name of the key image to start stitching with. If not given, the code tries to '
                             'compute it automatically.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stitcher = Stitcher(args)
    final_img = stitcher.final_img
    final_img = stitcher.crop(final_img)
    args.input_dir = args.input_dir[:-1] if args.input_dir[-1]=='/' else args.input_dir
    output_path = os.path.join(args.output_dir,os.path.split(args.input_dir)[-1]+'.jpg')
    cv2.imwrite(output_path,final_img)

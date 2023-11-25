# Steps

0. Develop object tracking algorithm based on features similarities comparison with rgb only

   1. Made by Nacim Bouia, but unavailable at the moment

1. Develop object tracking algorithm based on features similarities comparison with depth only

   1. Use detection or segmentation to extract object from frame (done)
      1. Detection (done)
   2. Extract depth (features) from center or centroid of the object
   3. Replace depth's object for center depth
   4. Save image
   5. Apply feature extraction over frame
   6. Compare using depth-only
   7. Apply masks over original frame
   8. Generate video with mask

2. Develop object tracking algorithm based on features similarities comparison with depth + rgb features
   ...

## Tasks

= Develop single object tracking based in feature matching

- generate similarity matrix for cropped images:(done)
  - SURF - SIFT optimized
    -SIFT (preferred) (done)
    -siamese network (need of training the nn)
- apply hungarian matching to cropped images (done)
- extracting cropped detection masks(done)
- Hungarian Matching(done)
- Optical flow solves occlusion problem, not applying it is bad
- use Lowe ratio test to ensure that the best match is distinguished enough of the second match (done)
- create a video with the matches (done):
  - track bboxes with a dictionary with key = id and value = label - the dict changes each iteration and it's used in runtime
    (done)
  - adapt method which create videos(done)
- solve overlap bbox problem (non maximum suppression)
- add depth
- IDEA: apply evaluation metric (iou - a lot of work, need to annotate the frames ground truth) and train the weights over the cost_matrixes over VisDroneDataSet
- compress track frames at end of tracking

## Problems

- When an object goes out of the image, it has less features for doing the pairing. A solve would be apply the position metric to the cost_matrix
- Occlusion. A solve would be apply optical flow
- Overlapping bounding boxes (it's worse in human crowd videos) (solved with NMS)
- bboxes should be an attribute of the frame, otherwise it will leave residues of old bboxes in the frames (solved)
- NMS should consider the estimated direction of the velocity and features of mask prediction (after it will have a virtual one created with optical flow ) so in crossing or ultrapassing the detection wouldn't be pruned

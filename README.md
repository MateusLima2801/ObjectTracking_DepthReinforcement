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

## Matching methods advantages

### Feature

1. Allows to distinguish different looking objects

### Position

1. Allows to distinguish distant objects in the frame

### Depth

1. Allows to distinguish objects in different depths, it'd work well in the case we have similar objects near to each other but in different depths (crossing)
2. If the camera turns...
3. You can use depth to adjust threshold (?)

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
- solve overlap bbox problem (non maximum suppression) (done)
- add center depth ( )
- IDEA: apply evaluation metric (iou) and train the weights over the cost_matrixes over VisDroneDataSet
- compress track frames at end of tracking
- apply Parallel NMS(0.6ms) instead of NMS (137ms) (done)
- send results (done)
- apply validation with ground truth annotations provided by VisDrone

## Problems

- When an object goes out of the image, it has less features for doing the pairing. A solve would be apply the position metric to the cost_matrix
- Occlusion. A solve would be apply optical flow
- Overlapping bounding boxes (it's worse in human crowd videos) (solved with NMS)
- bboxes should be an attribute of the frame, otherwise it will leave residues of old bboxes in the frames (solved)
- NMS should consider the estimated direction of the velocity and features of mask prediction (after it will have a virtual one created with optical flow ) so in crossing or ultrapassing the detection wouldn't be pruned
- detection outdoors has lower confidence scores

## References

1. H. Lee, J. -S. Lee and H. -C. Choi, "Parallelization of Non-Maximum Suppression," in IEEE Access, vol. 9, pp. 166579-166587, 2021, doi: 10.1109/ACCESS.2021.3134639.

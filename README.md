# Steps

0. Develop object tracking algorithm based on features similarities comparison with rgb only

   1. Made by Nacim Bouia, but unavailable at the moment

1. Develop object tracking algorithm based on features similarities comparison with depth only

   1. Use detection or segmentation to extract object from frame
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

- adapt cifar siamese network to this project,
  - train in crops (?): need to label the data
  - train in car-people-truck labeled data: find it
- generate similarity matrix for cropped images:
  -SIFT
  -siamese network (preferred)
- apply hungarian matching to cropped images
- extracting cropped detection masks(done)
- Hungarian Matching(done)
- Meeting with Lepetit: try to inspect few images at first
- Optical flow solves occlusion problem, not applying it is bad
  s

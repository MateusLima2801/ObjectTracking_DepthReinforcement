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

- generate similarity matrix for cropped images:(done)
  - SURF - SIFT optimized
    -SIFT (preferred) (done)
    -siamese network (need of training the nn)
- apply hungarian matching to cropped images (done)
- extracting cropped detection masks(done)
- Hungarian Matching(done)
- Meeting with Lepetit: try to inspect few images at first
- Optical flow solves occlusion problem, not applying it is bad
- use Lowe ratio test to ensure that the best match is distinguished enough of the second match (done)
- create a video with the matches:
  - track bboxes with a dictionary with key = id and value = label - the dict changes each iteration and it's used in runtime
    (done)
  - adapt method which create videos

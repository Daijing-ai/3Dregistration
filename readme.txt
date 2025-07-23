1. extract_keyframes(extract_keyframes.py)
--input: video file
--output: keyframes

2. run_depthanything(ndt_ply_v3.py)
--input: keyframes
--output: depth maps, cloudpoints, merged cloudpoint

3. ICP registration(ndt_ply_v3.py)
--input: cloudpoints(MRI, video)
--output: error, transformation matrix, registered cloudpoints
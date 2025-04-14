Use main.py to process 1.txt which you need to cluster the clould points

The parameters you need to change include:

# 1.file_path 
If you want to change the file name 

# 2.column_range 
 Specify the column range. If you dont want to use all of the points 
column_range = (0, 2)

# 3.row_range 
 Specify the row range. If you dont want to use all of the points 

# 4.eps=0.1
 If a point's distance from kernal point is more than this, the point is not as same as the class of th kernal

# 5.min_samples
 Min points in each kernal

# 6.visualize
 If you want to visualize the points just choose True

# 7.files
 If you want to save the clustered points as identical txt, you need to choose True

 **ps.visualizer can be used to show scatters figure**

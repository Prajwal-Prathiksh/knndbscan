## kNN-DBSCAN
    a parallel implementation of kNN-DBSCAN given k-NNG using MPI and OpenMP.

### Compile the source code
    1. cd test/
    2. make

### Usage
    To run knn-DBSCAN (with input parameters $\epsilon$=1300.0, $k$=100$) an an existing knn graph ("mnist70k.knn.txt") of a dataset (with 7,000 points) with 4 MPI tasks and 4 threads per MPI task:
        cd test/
        ibrun -np 4 ./knndbscan -n 70000 -e 1300.0 -m 100 -i mnist70k.knn.txt -k 100 -o labels.txt -t 4
    
    The descriptions each arguements:
       -n num_points  : number of points
       -e epsilon     : input parameter of kNN-DBSCAN, raidus of consideration for core points
       -m minpts      : input parameter of kNN-DBSCAN, min points
       -i knngraph_name    : file containing kNN graph
       -k num_neighbors    : number of neighbors for each point in input file,k should be no less than m (default of k is m is k is not set)
       -o output      : output file with label for each point, the order of points is as the same with knn graph
       -t threads  : number of threads per MPI task

    Also, run the following to get detail description on the program arguments
        ibrun -np 1 ./knndbscan ?


### Input kNN-G file format
    For ASCII file of kNN-G, each line has all k neighbors of a point. The beginning is the point id. Then all k neighbors has the format of distance followed by the neighbor id.
    For example, k = 3, point 7 with 3-nearest neighbors are (0.0, 7) (1.0, 10) (2.0,11). Line for point 7 is: 7 0.00 7 1.0 10 2.0 11


### Example kNN-G files
    Some example kNN-G files are available from <https://drive.google.com/open?id=1osnWytsjqfwBWyFs70neaYgR9V3BKXUI>.
    There are kNN-Gs for four datasets: MNIST70K, MNIST2M, CIFAR-10 and Uniform2S. For examople, "MNIST70K.knn100.txt.zip" contatins kNN-G graph of dataset MNIST70K with k=100.    


### Output label file 
    -each line has an intger. Sequence of the labels follow the point id from the knn file.
        "-1" represents noise 

    -Normalized Mutual Information (NMI) values can be computed by using python script NMI.py in /test. This script requires numpy (1.16.4) and (0.21.2).



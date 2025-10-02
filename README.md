## kNN-DBSCAN
A parallel implementation of kNN-DBSCAN given k-NNG using MPI and OpenMP.

### Python Package

**Install:**
```sh
make install
```

**Test:**
```sh
make test           # Unit tests
make test-examples  # Benchmarking & visualizations
```

**Development:**
```sh
make lint     # Code linting
make format   # Code formatting  
make build    # Build distributions
make clean    # Clean artifacts
```

### Standalone C++ Build
```sh
cd test/ && make
```

### C++ Usage
Run kNN-DBSCAN with ε=1300.0, k=100 on kNN graph with 4 MPI tasks and 4 threads:
```sh
cd test/
ibrun -np 4 ./knndbscan -n 70000 -e 1300.0 -m 100 -i mnist70k.knn.txt -k 100 -o labels.txt -t 4
```

**Arguments:**
- `-n`: Number of points
- `-e`: Epsilon (radius for core points)  
- `-m`: MinPts (minimum points for dense region). Note: MinPts should be k + 1.
- `-i`: Input kNN graph file
- `-k`: Number of neighbors per point
- `-o`: Output labels file
- `-t`: Threads per MPI task


### File Formats

**kNN Graph:** Each line contains a point's k neighbors:
```
point_id distance1 neighbor1_id distance2 neighbor2_id ...
```

For example, k = 3, point 7 has 3 nearest neighbors, being (7, 0.00), (10, 1.0), and (11, 2.0).
This point is represented as:
```
7 0.00 7 1.0 10 2.0 11
```
> [!NOTE]
> For a given point, its own ID **must** be included as a neighbor with distance 0.0.
> This requirement stems from the core point detection algorithm in [`src/clusters.cpp:43`](src/clusters.cpp#L43), 
> which uses direct array indexing `A[p*maxk + minPts-2]` to find the distance to the (minPts-1)th 
> nearest neighbor, assuming the self-neighbor is always at index 0.

**Output:** One integer per line (cluster ID or -1 for noise)

## Citation

If you use this software in your research, please cite the following paper:

**BibTeX:**
```bibtex
@misc{chen2024knndbscandbscanhighdimensions,
      title={KNN-DBSCAN: a DBSCAN in high dimensions}, 
      author={Youguang Chen and William Ruys and George Biros},
      year={2024},
      eprint={2009.04552},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2009.04552}, 
}
```

**Plain text:**
> Chen, Y., Ruys, W., & Biros, G. (2024). KNN-DBSCAN: a DBSCAN in high dimensions. arXiv preprint arXiv:2009.04552.

**Paper:** [KNN-DBSCAN: a DBSCAN in high dimensions](https://arxiv.org/abs/2009.04552)



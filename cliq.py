from pyclustering.cluster.clique import clique, clique_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
# read two-dimensional input data 'Target'
print(type(FCPS_SAMPLES.SAMPLE_TARGET))
data = read_sample("xy_col.data")
print(FCPS_SAMPLES.SAMPLE_TARGET)

# create CLIQUE algorithm for processing
#print(data)

intervals = 100 # defines amount of cells in grid in each dimension
threshold = 0  # lets consider each point as non-outlier
clique_instance = clique(data, intervals, threshold)
# start clustering process and obtain results
clique_instance.process()
clusters = clique_instance.get_clusters()  # allocated clusters
noise = clique_instance.get_noise()     # points that are considered as outliers (in this example should be empty)
cells = clique_instance.get_cells()     # CLIQUE blocks that forms grid
print("Amount of clusters:", len(clusters))
# visualize clustering results
print(clusters)

print("\n________________________coordinates____________________________\n")

x = list()
y = list()
centroids = list()
cnt = 1
for cluster in clusters:
    print("cluster :" + str(cnt))
    for i in range(len(cluster)):
        a = data[cluster[i]]
        print(a)
        x.append(a[0])
        y.append(a[1])
    xcenter = int(sum(x)/len(x))
    ycenter = int(sum(y)/len(y))
    centroids.append((xcenter, ycenter))
    print(" ")
    cnt += 1


with open('centroid.data', 'w') as var:
    for centroid in centroids:
        var.write(str(centroid).replace("'", '').replace(",", ' ').replace("(", '').replace(")", '')+'\n')

cells = clique_instance.get_cells()
#print(cells)
clique_visualizer.show_grid(cells, data)    # show grid that has been formed by the algorithm
clique_visualizer.show_clusters(data, clusters, noise)  # show clustering results
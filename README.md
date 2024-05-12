# UAV Navigation In Agriculture for Weed Removal

# Abstract 
To optimize agricultural activities, increase crop production and manage agricultural problems, unmanned aerial
vehicles (agricultural drones) are used. Weed detection systems are important solutions to one of the profound and serious
agricultural problems that is weed growth. These systems can be further strengthened by the use of point clouds and image
processing to enhance crop maintenance. In this paper, we present a simplified way to detect weed and enhance the navigation of a drone to oversee
weed growth. The pretrained DeepLabv3+ model (pretrained on PASCAL VOC 2012 dataset) is used for the weed semantic segmentation task. Point clouds represent objects or space in the form of points representing the X and Y geometric coordinates
of a single point on an underlying sampled surface. The areas
containing weeds are formed into clusters based on the CLIQUE
grid clustering algorithm. The clusters are identified as “nodes”
and an optimum shortest path is devised to travel through all
the nodes. All the weed sections are taken care of by navigating
through all the clusters using the shortest path found. When the
current cluster is taken care of, the UAV navigates to the next
cluster.

# Dataset
We have used the CoFly-WeedDB dataset (~436MB) consists of 201 aerial images, capturing different weed types that interfere among the row crop (cotton), as well as their corresponding annotated images.

# Architecture
![image](https://github.com/saroja77/UAV-Navigation-in-Agriculture-for-Weed-Removal/assets/110019339/49937f4c-c07b-4914-a2ef-7fc3351bdf71)





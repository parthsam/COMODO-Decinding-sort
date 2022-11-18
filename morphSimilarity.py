#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from skimage import measure
from skimage.future import graph
import networkx as nx
import sys
import scipy.misc
import imageio
from scipy.spatial import distance as dist
from PIL import Image, ImageOps


def compute_distance(image_set_directory, 
                     signature_function='surface_volume_ratio_sig', 
                     visualize_graphs=False,
                     weighted=False,
                     cosine=False, 
                     visualize_padded_vector=False,
                     max_degree_node_color=1):
    image_vectors = []
    components = []
    rags = []
    rags_dict = {}
    


    # generate images from plt files
    # source_directory = Path(image_set_directory)
    # for idx, image in enumerate(source_directory.iterdir()):
    #     # read the image file and process it
    #     img_data = np.genfromtxt(image, skip_header=3, skip_footer=(x * y))
    #     img = img_data[:, 2].reshape(x + 1, y + 1)
    #     plt.imsave(f'{idx}.jpg', img, cmap='gray')

    # `image_set_directory` can either be a string containing path to the files or
    # a list of strings containing path to multiple files
    if isinstance(image_set_directory, str):
        sorted_img_files = sorted([i for i in Path(image_set_directory).iterdir()])
    elif isinstance(image_set_directory, list):
        sorted_img_files = []
        for directory in image_set_directory:
            sorted_directory_files = sorted([i for i in Path(directory).iterdir()], key=lambda x: int(str(x).split("/")[-1].split(".")[-2]))
            sorted_img_files += sorted_directory_files
            


    for p in sorted_img_files:
        
        if p.is_file():
            print(p)

            # read image data and convert the image to grayscale
            img = io.imread(p)
            # if the image has an alpha (tansparency) chanel remove it
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)
          # #  print(img)
           # imagetest = (img > 0.5).astype(int)
           # im_invert=255-img
           # plt.imshow(im_invert, cmap='gray')
            #im_invert=np.invert(img)
            #plt.imsave(f"./images/{p.stem}_inverted.png", im_invert,cmap='gray')

            # read raw bitwise data 
            # img_data = np.genfromtxt(p)
            # img = img_data.reshape((100, 200))
            # img = np.where(img > 0.5, 1, 0)
           # plt.imsave(f"./images/{p.stem}_Comodo.png", imagetest, cmap='gray')


            # generate region adjacency graph
            rag = generate_region_adjacency_graph(img, signature_function)
            # rags.append(rag)
            # rags_dict[p.stem] = rag
            # rag, component = generate_region_adjacency_graph(img, signature_function)
            # components.append(component)

            # visualize graphs if required
            if visualize_graphs:
                generate_graph_visualization_images([rag], p.stem, max_degree_node_color)
            # generate bfs vector
            vector = generate_bfs_vector(rag, max_degree_node_color)
            image_vectors.append(vector)
            # target_dir = Path(image_set_directory)/"txtfiles"
            # print(target_dir)
            # #fname=Path(image_set_directory/(f"{p.stem}.txt"))
            # fname=(p.stem + '.txt')
            # print(fname)
            # np.savetxt(fname,generate_padded_vectors_reversable(image_vectors))
           

    # visualize padded vectors
    if visualize_padded_vector:
        if len(image_vectors) != 2:
            print("Error: No. of morphologies != 2. Padded vector visualization aborted")
        else:
            generate_padded_vector_visualization(generate_padded_vectors_reversable(image_vectors))

    # generate bfs vector
    distances = []
    for vector_1 in image_vectors:
        distance_row = []
        for vector_2 in image_vectors:
            # make the 2 vectors of equal length
            vector_pair = [vector_1, vector_2]
           # print(vector_pair)
            padded_vectors = generate_padded_vectors_reversable(vector_pair)
            # print(padded_vectors)
            if cosine:
                v1, v2 = padded_vectors
                abs_padded_vectors = [[np.abs(i) for i in v1], [np.abs(j) for j in v2]]
                distance = dist.cosine(*padded_vectors)
            else:
                # compute the euclidean distance of the 2 vectors
                if weighted:
                    distance = weightedL2(padded_vectors[0], padded_vectors[1])
                else:
                    distance = np.linalg.norm(padded_vectors[0] - padded_vectors[1])
                    # compute d squared for coparison
                    # distance = np.power(distance, 2)
                    
                    # fname=(p.stem + '.txt')
                    # print(fname)
                    # np.savetxt(fname,padded_vectors)
                    
            distance_row.append(distance)
        distances.append(distance_row)
        

    return distances #image_vectors # 
    #return generate_padded_vectors_reversable(image_vectors)
    # return components

def weightedL2(vector_1, vector_2):
    diff = vector_1 - vector_2
    weight_array = [damping_function(i) for i in np.arange(len(diff))]
    return np.sqrt((weight_array*diff*diff).sum())

def damping_function(x):
    return 10*np.exp(-0.05*x)

def generate_region_adjacency_graph(image, signature_function):
    """
    Create a region adjacency graph from a given image
    Args:
        image (ndarray):
            grayscale image
        signature_function (int):
            function to be used to calculate the signal of each
            component/blob
    Return:
        rag (RAG):
            region adjacency graph
    """
    # identify neighbouring pixels with the same pixel value,
    # assign them labels and split them
    components, binary_image, label_image = extract_components(image, allow_blank_images=True, return_images=True)

    # make sure number of components = number of unique labels
    # make sure no component is being pruned
    assert len(components) == len(np.unique(label_image)), "Total components != Total labels"

    # generate the Region Adjacency Graph
    rag = graph.rag_mean_color(binary_image, label_image)

    # calculate the signature of each component
    sigs = []
    for component in components:
        sig = apply_signatures(component, signature_function, allow_empty_sig=True)
        sigs.append(sig[0])

    # make sure number of signatures = number of components
    # Ensure no sig value is getting pruned
    assert len(components) == len(sigs), "Total signatures != Total components"

    # delete components whose signature is None
    # those components need to be pruned
    for idx, sig in enumerate(sigs):
        if not sig:
            rag.remove_node(idx + 1)

    # create a pruned component list that only 
    # has components with valid signatures
    pruned_components = []
    for idx, sig in enumerate(sigs):
        if sig:
            pruned_components.append(components[idx])

    # add signatures as node weights
    for idx, sig in enumerate(sigs):
        # only add signature when signature is not None
        # None signature means the component needs to 
        # be ignored
        if sig:
            rag.nodes[idx + 1]['weight'] = sig

            # remove unwanted data in the nodes
            del rag.nodes[idx + 1]['total color']

    # remove edge weights since they are not required
    for edge in rag.edges:
        node_1, node_2 = edge
        del rag[node_1][node_2]['weight']

    # inform user about the components that were neglected
    total_components_pruned = len(components) - len(pruned_components)
    print(f"Pruned {total_components_pruned}/{len(components)} component(s)")

    # return rag, pruned_components
    # return rag, components
    return rag

def generate_graph_visualization_images(graphs, filename, max_degree_node_color, combined=True):
  """
  Save graph visualizations as images.

  Args:
      graphs: list of graphs to be visualized
      combined: if False, directory name will
                be determined based on index number 
                of the graph. Default=True.
  """
  # trajectory_name, trajectory_index = trajectory_name.split("_")
  # target_dir = Path("/home/namit/codes/Entropy-Isomap/outputs/constDt-5replicas-noPer-4x4_graphs")/trajectory_name/(f"{trajectory_index}.png")
  # if target_dir.exists():
    # return None

  root_nodes = []
  for graph_num,gg in enumerate(graphs):
      
      fig,axes = plt.subplots()

      # import ipdb; ipdb.set_trace()
      
      # setting node size
      node_size = [i[1]['pixel count'] for i in gg.nodes(data=True)]
      sum_node_size = sum(node_size)
      node_size_normalized = [(i/sum_node_size)*5000 for i in node_size]
      
      # setting node color
      node_color = []
      for i in gg.nodes(data=True):
          current_color = i[1]['mean color'][0]
          if current_color == 1:
              # this is white
              # set to light grey
              node_color.append(np.array([0.7,0.7,0.7]))
          elif current_color == 0:
              # this is black
              # set to dark grey
              node_color.append(np.array([0.3,0.3,0.3]))
          else:
              # this should never happen
              print("Unknown color of node.")
      
      # setting node label
      node_labels = {}
      for index, size in enumerate(node_size):
          node_labels[index+1] = f"{size}"
          # node_labels[index+1] = f"{size} ({index+1})"
          
      # setting node edge colors
      edgecolors = ['k']*len(node_color)
      root_node = get_max_degree_node(gg, max_degree_node_color)
      # print(f"{graph_num} - {root_node}")
      # try:
      #     edgecolors[root_node-1] = 'r'
      # except:
      #     nx.draw_kamada_kawai(graph)
      #     return None
      root_nodes.append(root_node)
      

      # create the graph and save it
      nx.draw_kamada_kawai(gg, 
                            node_color  = node_color,
                            edgecolors  = edgecolors,
                            with_labels = True,
                            ax          = axes)
      
      # target_dir = Path("/home/namit/codes/Entropy-Isomap/outputs/constDt-5replicas-noPer-4x4_graphs")/trajectory_name
      target_dir = Path("./")/"graphs"
  
      # create if path does not exist
      target_dir.mkdir(parents=True, exist_ok=True)
      
      # title = trajectory_name+f" #{graph_num%80} ({graph_num})"
      # plt.title(title, y=-0.1)
      
      plt.savefig(target_dir/(f"{filename}.png"))
      # plt.savefig(f'/home/namit/codes/meads/morphology-similarity/playground/Results/organic_morphology_graph_{m_idx}.pdf')

      print("generated graph file: ", target_dir/(f"{filename}.png"))
      
      plt.cla()
      plt.close()
  return root_nodes

def generate_padded_vector_visualization(padded_vectors):
  fig, ax = plt.subplots(figsize=(12,2))
  ticks = np.arange(0, len(padded_vectors[0]), 20)
  minor_ticks = np.arange(0, len(padded_vectors[0]), 5)
  ax.set_xticks(ticks)
  ax.set_xticks(ticks, minor=True)
  ax.set_yticklabels(['',f'10x80 morph #7', f'10x80 morph #3'])
  plt.grid(True, axis='x', linestyle='-',zorder=0)

  plt.setp(ax.get_xminorticklabels(), visible=False)

  plt.xlabel("Vector Dimension")
  plt.title("Comparing vectors post padding")

  legend = [Patch(facecolor=[8/255,46/255,20/255],     edgecolor='k',label='Black Node'),
            Patch(facecolor=[232/255,236/255,236/255], edgecolor='k',label='White Node'),
            Patch(facecolor=[255/255,90/255,95/255],   edgecolor='k',label='Padding')]
  plt.legend(handles=legend, bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)

  plt.tight_layout()

  for i in range(len(padded_vectors[0])):
      if padded_vectors[0][i]==0:
          c1 = [255/255,90/255,95/255]
      elif padded_vectors[0][i]>0:
          c1 = [232/255,236/255,236/255]
      else:
          c1 = [8/255,46/255,20/255]
          
      if padded_vectors[1][i]==0:
          c2 = [255/255,90/255,95/255]
      elif padded_vectors[1][i]>0:
          c2 = [232/255,236/255,236/255]
      else:
          c2 = [8/255,46/255,20/255]
          
      color=[c1,c2]
      plt.barh([0,1], [1,1], left=i, color=color, height=0.75, orientation='horizontal',zorder=3)
      a = plt.xticks(np.arange(0, len(padded_vectors[0])+1, 5.0))
  plt.savefig("vector_visualization.jpg")

def generate_bfs_vector(graph, max_degree_node_color, return_traversal_order=False):
    """
    Vecotorize a given graph using the priority BFS algorithm

    Args:
        graph (networkx.Graph):
            The input graph to use
        return_traversal_order (bool):
            Whether to return traversal order of the nodes.
            (default=False)

    Returns:
        vector (list):
            vector representation of the graph
        traversal_order (list):
            A list containing the indices of the nodes in the order they
            were traversed. Only returned when return_traversal_order is True.
    """
    # determine the root node
    root = get_max_degree_node(graph, max_degree_node_color)

    # generate BFS vector
    return priority_bfs(graph, root, return_traversal_order=return_traversal_order)

def generate_padded_vectors(vectors):
    """
    Implements layerwise padding.
    Note this function only pads any two vectors at a time by aligning
    all the white nodes one under the other and filing empty spaces
    with empty nodes in between.
    Args:
        vectors (list): A list of length 2 containing the 2 vectors to be padded
    Returns:
        padded_vectors (list): A list of length 2 containing the 2 padded vectors
    """

    split_vectors = []

    for vector in vectors:
        # split the nodes based on sign
        split_indices = []

        for index,d in enumerate(vector):
            if d >= 0:
                split_indices.extend([index, index+1])

        split_vector = np.split(vector, split_indices)
        split_vectors.append(split_vector)


    # get the maximum length of split at each
    # position for all the vectors
    max_split_length = {}

    for split_vector in split_vectors:
        for index,split in enumerate(split_vector):
            max_split_length[index] = max([len(split), max_split_length.get(index, 0)])


    # pad all the splits to their
    # respective max lengths
    padded_split_vectors = []
    for split_vector in split_vectors:

        padded_split_vector = []
        for index,split in enumerate(split_vector):
            padded_split = front_pad(split, max_split_length[index])
            padded_split_vector.append(padded_split)

        padded_split_vectors.append(padded_split_vector)


    # merge all splits into single vector
    merged_vectors = []
    for padded_split_vector in padded_split_vectors:
        merged_vector = np.concatenate(padded_split_vector)
        merged_vectors.append(merged_vector)


    # over all frontpad to compensate for
    # different number of layers in each graph
    max_dimension = max(map(len, merged_vectors))

    padded_vectors = []
    for merged_vector in merged_vectors:

        padded_vector = front_pad(merged_vector, max_dimension)
        padded_vectors.append(padded_vector)


    # make all vector dimension magnitudes
    # positive irrespective of color
    # positive_vectors = []
    # for padded_vector in padded_vectors:
    #   positive_vector = np.abs(padded_vector)
    #   positive_vectors.append(positive_vector)

    # write some tests maybe
    return padded_vectors

def extract_components(image, 
                       binarize=True, 
                       background=-1,
                       allow_blank_images=False,
                       return_images=False):
    """
    Extract morpholgoical components from an image.

    Arguments:
        image (ndarray): 
            A grayscale image 
        binarize (boolean): 
            Flag to binarize the image before extraction.Defaults to
            True.
        background (int): 
            Pixels of this value will be considered background and will
            not be labeled. By default every pixel is considered for
            labeling. Set to -1 for no background. (default=-1)
        allow_blank_images (boolean):
            Whether a blank image should be considered as a single
            component. (default=False) 
        return_images (boolean): 
            Wheter to return the labeled image & binary
            image.(default=False)

    Returns:
        components (list): 
            A list of component images with the same shape as the input
            image.
        image (ndarray): 
            Original image (and binarized if binarize=True). Only
            returned when return_images is set to True.
        labeled_sample (ndarray): 
            A labelled image where all connected ,regions are assigned
            the same integer value. Only returned when return_images is
            set to True.
    """
    components = []
    if binarize:
        image = (image > 0.5).astype(int)

    labeled_sample = measure.label(image, background=background)

    for label in np.unique(labeled_sample):
        # extract companent into a separate image
        component = (labeled_sample == label).astype(np.float64)
        
        if not allow_blank_images:
            if (component == 0).all():
                continue 
                
        components.append(component)

    # remove the first background component if background pixels needs
    # to be neglected. 
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#label
    if background >= 0:
        components = components[1:]
    
    if return_images:
        return components, image, labeled_sample
    else:
        return components

def apply_signatures(image, sig_funcs, allow_empty_sig=False):
    """
    Applies the provided signature functions to the given image.

    Arguments:
        image: An image, represented as a 2D Numpy array.
        sig_funcs: List of signature extraction functions.
        allow_empty_sig: If signature values of zero should be allowed
                         (default=False)

    Returns:
        A list of signatures for the given image.

    Raises:
        AssertionError: All signatures returned by the extractors need
        to be non-empty. Only when allow_empty_sig is False.
    """
    if isinstance(sig_funcs, str):
        # For convenience, we can pass in a single signature function.
        # This converts it into a list, with it being the only element.
        sig_funcs = [sig_funcs]
    sigs = []
    for sig_func in sig_funcs:
        sig = eval(sig_func+"(image)")
        if not allow_empty_sig:
            assert len(sig) > 0, 'Signatures need to be non-empty.'
        sigs.append(sig)
    sigs = np.array(sigs).T
    return sigs


def surface_volume_ratio_sig(component):
    """The surface to volume (perimeter to area) ratio"""
    perimeter = measure.perimeter(component)
    area = np.sum(component == 1)
    # Note: We are guranteed to have at least 1 pixel of value 1
    # the perimeter of a single pixel is also 1
    if area < 100:
        return None
    return perimeter/area

def shape_ratio_minbymax_sig(component):
    """The ratio of the width of the component to it's height"""
    # The component is the only part of the full image that is 1
    # we need to generate a bounding box around the component 
    # measure the ratio of its height and width

    # convert component array into integer array
    component = component.astype(np.int64)

    # generate region properties of the component
    regions = measure.regionprops(component)

    # the component image will have only one component
    min_row, \
    min_col, \
    max_row, \
    max_col = regions[0]['BoundingBox']
    area = np.sum(component == 1)
    
    lengthXdirection= (max_row-min_row)
    lengthYdirection= (max_col-min_col)
    minlength=min(lengthXdirection,lengthYdirection)
    maxlength=max(lengthXdirection,lengthYdirection)
    if area < 100:
        return None
    
    return(minlength/maxlength)
    
    
def shape_ratio_sig(component):
    """The ratio of the width of the component to it's height"""
    # The component is the only part of the full image that is 1
    # we need to generate a bounding box around the component 
    # measure the ratio of its height and width

    # convert component array into integer array
    component = component.astype(np.int64)

    # generate region properties of the component
    regions = measure.regionprops(component)

    # the component image will have only one component
    min_row, \
    min_col, \
    max_row, \
    max_col = regions[0]['BoundingBox']
    area = np.sum(component == 1)
    if area < 100:
        return None
    lengthXdirection= (max_row-min_row)
    lengthYdirection= (max_col-min_col)
    x=lengthXdirection/lengthYdirection
    #minlength=min(lengthXdirection,lengthYdirection)
    #maxlength=max(lengthXdirection,lengthYdirection)
    if (x<1):
        return((x*(0.5+(0.5*np.tanh(x-1)))))
    else:
        return((0.5+(0.5*np.tanh(x-1))))
    

def fractal_dimension_sig(component):
    """The fractal dimension of the component"""
    if np.isnan(fractal_dimension(component)):
        return None
    area = np.sum(component == 1)
    # if area < 100:
    #     return None
    return fractal_dimension(component)

def get_max_degree_node(graph, max_degree_node_color):
    """
    Determine the node with the maximum degree irrespective of its color

    Args:
        graph (networkx.Graph): 
            The input graph to use.
    Returns:
        max_degree_node (int): 
            index of the node with the maximum degree 
    """

    nodes = list(graph.nodes(data=True))
    max_degree_node = nodes[0][0]

    # Iterate over the nodes and find the most connected node
    for current_node in nodes:

        current_node_degree = graph.degree[current_node[0]]
        max_degree_node_degree = graph.degree[max_degree_node]

        if current_node_degree > max_degree_node_degree:
            max_degree_node = current_node[0]

        elif current_node_degree == max_degree_node_degree:
            # settle a tie by choosing the node of a pre-determined color
            # default is white

            if max_degree_node_color == 0: #black
                prefered_color = np.array([0,0,0])
            else: #white
                prefered_color = np.array([1,1,1])

            current_max_degree_node_color = graph.nodes(data=True)[max_degree_node]['mean color']
            current_node_color = current_node[1]['mean color']

            if np.array_equal(current_node_color,current_max_degree_node_color):
                # both the nodes are tied on connectivity and color
                # we use signature value to break the tie
                if current_node[1]['weight'] > graph.nodes(data=True)[max_degree_node]['weight']:
                    max_degree_node = current_node[0]
            elif np.array_equal(current_node_color, prefered_color):
                max_degree_node = current_node[0]

            # the above logic would only work when we have binary morphologies, because if the
            # two nodes are not of the same color then atleast one has to match the prefered_color
    return max_degree_node

def priority_bfs(graph, root, return_traversal_order=False):
    """
    Implementation of the priority BFS algorithm

    Args:
        graph (networkx.Graph): 
            The input graph to use.
        root (int):
            index of the node to be considered as the root
        return_traversal_order (bool): 
            Whether to return traversal order of the nodes.
            (default=False)

    Returns:
        vector (list): 
            vector representation of the graph
        traversal_order (list): 
            A list containing the indices of the nodes in the order they
            were traversed. Only returned when return_traversal_order is
            True.

        
    """
    vector  = []
    visited = []
    queue   = []

    # Queue element storage format 
    # [ (<node>, <node_signature>) , (<node>, <node_signature>), ... ]
    queue.append((root,graph.nodes[root]['weight']))

    while queue:

        # Step A: Dequeue it to the vector
        current_node_index, current_node_signature = queue.pop(0)
        current_node_color = graph.nodes[current_node_index]['mean color'][0]
        visited.append(current_node_index)


        # Step B: Append it to the vector
        vector.append(get_node_color_sign(current_node_color) *
                      current_node_signature)


        # Step C: Get all of elements children
        current_node_neighbors = []
        for neighbor in graph.neighbors(current_node_index):
            current_node_neighbors.append(
                (neighbor, graph.nodes[neighbor]['weight']))


        # Step D: Sort them by their signature and enqueue them
        current_node_neighbors.sort(key = lambda x: x[1], reverse=True)
        # enqueueing - make sure that node has not been visited first
        # althugh that should not happen since the graph is always
        # acyclic
        for neighbor in current_node_neighbors:
            if neighbor[0] not in visited:
                queue.append(neighbor)

    vector = np.array(vector)

    if return_traversal_order:
        return vector, visited
    else:
        return vector

def get_node_color_sign(node_color):
    """
    Returns -1 for black color (node_color = 0) and 
    1 for white (node_color = 1 or 255)

    Args:
        node_color (int): 
            node color value should be 0, 1 or 255
    
    Returns:
        sign (int): -1 or 1 based in pixel value

    Raises:
        AssertionError: 
            node_color needs to be either black or white.
    """
    assert node_color in [0,1,255], "node_color can only be 0, 1 or 255"

    if node_color < 2:
        # when node color value is 0 or 1 that means it is a black node
        # black node is -1
        return ((-1) ** (node_color+1))
    else:
        # 255 is always white
        # white node is 1
        return 1

def generate_padded_vectors_reversable(vectors):
    """
    Implements layerwise padding. 
    Note this function only pads any two vectors at a time by aligning 
    all the white nodes one under the other and filing empty spaces
    with empty nodes in between. 
    NOTE:
    This is a modified version of the generate_padded_vectors function. 
    This function instead of aligning all white peaks one under the other
    aligns all the continous chunks of elements one under the other by 
    making them of the same length. 
    Args:
        vectors (list): A list of length 2 containing the 2 vectors to be padded
    Returns:
        padded_vectors (list): A list of length 2 containing the 2 padded vectors
    """
     
    split_vectors = []

    for vector in vectors:
        # split the nodes into continous chunks of values having the same sign
        # same sign means same color
        split_vector = []

        previous = vector[0]
        sub_split_vector = []

        for element in vector:
            # component's signature value should always be non-zero
            assert element != 0, 'Element with 0 signature found'

            if element*previous > 0:
                # current and the previous element have the same sign
                sub_split_vector.append(element)
            else:
                # current and the previous do not have the same sign
                split_vector.append(sub_split_vector)
                sub_split_vector = [element]

            previous = element

        # appending the last sub split 
        # since there was no sign change to trigger the sign channge
        split_vector.append(sub_split_vector)

        split_vectors.append(split_vector)
      
    # verify that the first node of both the vectors is of the same color
    # if not, add a padding of 0's equal in lenght with the number of 
    # components of the same color in the same vector
    # before
    # [-1,-1,1, 1, 1,-1]
    # [ 1, 1,1,-1,-1,-1]
    # after
    # [-1,-1,1,1,1,-1]
    # [ 0, 0,1,1,1,-1,-1,-1]
    if (split_vectors[0][0][0]*split_vectors[1][0][0] < 0): # implies root node is not of the same color
      # always add padding to the vector that begins with black (negative)
      # to keep the process deterministic otherwise the distance matrix becomes a symmetrical
      if split_vectors[0][0][0] > 0:
        initial_pad = [0]*len(split_vectors[0][0])
        split_vectors[1].insert(0, initial_pad)
      else:
        initial_pad = [0]*len(split_vectors[1][0])
        split_vectors[0].insert(0, initial_pad)

    # for vector in vectors:
    #     # split the nodes based on sign
    #     split_indices = []

    #     for index,d in enumerate(vector):
    #         if d >= 0:
    #             split_indices.extend([index, index+1])

    #     split_vector = np.split(vector, split_indices)
    #     split_vectors.append(split_vector)


    # print([print(i) for i in split_vectors])



    # get the maximum length of split at each 
    # position for all the vectors
    max_split_length = {}

    for split_vector in split_vectors:
        for index, split in enumerate(split_vector):
            max_split_length[index] = max([len(split), max_split_length.get(index, 0)])



    # pad all the splits to their 
    # respective max lengths
    padded_split_vectors = []
    for split_vector in split_vectors:

        padded_split_vector = []
        for index,split in enumerate(split_vector):

            padded_split = front_pad(split, max_split_length[index], 0)
            padded_split_vector.append(padded_split)

        padded_split_vectors.append(padded_split_vector)



    # Merge all splits into single vector
    merged_vectors = []
    for padded_split_vector in padded_split_vectors:
        merged_vector = np.concatenate(padded_split_vector)
        merged_vectors.append(merged_vector)


    # over all frontpad to compensate for
    # different number of layers in each graph
    max_dimension = max(map(len, merged_vectors))

    padded_vectors = []
    for merged_vector in merged_vectors:

        padded_vector = front_pad(merged_vector, max_dimension, 0)
        padded_vectors.append(padded_vector)


    # make all vector dimension magnitudes 
    # positive irrespective of color
    # positive_vectors = []
    # for padded_vector in padded_vectors:

    #   positive_vector = np.abs(padded_vector)
    #   positive_vectors.append(positive_vector)

    # write some tests maybe
    verify_padded_vectors(padded_vectors)
    return padded_vectors

# test to ensure padding function works as intended
def verify_padded_vectors(padded_vectors):
  # assert length of both the vectors is the same
  assert len(padded_vectors[0]) == len(padded_vectors[1]), \
    "PADDING FAILED: padded vectors differ in length"

  # assert component at each index of the both padded vecors
  # is of the same phase
  for ind in range(len(padded_vectors[0])):
    assert padded_vectors[0][ind]*padded_vectors[1][ind] >= 0, \
    f"PADDING FAILED: two component of different color found side by side.\n \
      vector 1 component: {padded_vectors[0][ind]},\n \
      vector 2 component: {padded_vectors[1][ind]},\n \
      component index   : {ind}"


def front_pad(vector, max_dimension, pad_value=0):
    """
    Add zeroes in front of the given vector
    """
    # make np array if not
    if not isinstance(vector, np.ndarray): vector=np.array(vector)

    return np.pad(vector, 
                  (0,max_dimension-len(vector)),
                  mode='constant', 
                  constant_values=(0,pad_value))

def fractal_dimension(Z, threshold=0.5):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# if len(sys.argv) != 5:
#     print("usage: compute_distance.py <x> <y> <path_to_morphology_set_directory> <path_to_query_image>")
#     sys.exit(-1)

# print(compute_distance(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))

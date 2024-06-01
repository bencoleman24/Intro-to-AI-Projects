from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x_centered = x - np.mean(x, axis=0)
    return x_centered

def get_covariance(dataset):
    S = np.dot(dataset.T, dataset) / (len(dataset) - 1)
    return S

#  Gets m largest eigenvalues and corresponding eigenvectors from the covariance matrix S
def get_eig(S, m):
    eig_values, eig_vectors  = eigh(S, subset_by_index=(len(S)-m, len(S)-1))

    # Get in descending order
    asc_indices = eig_values.argsort()
    desc_indices = asc_indices[::-1]
    eig_values = eig_values[desc_indices]
    eig_vectors = eig_vectors[:, desc_indices]

    # Create a diagonal matrix from m largest eigenvalues (var explained by each  principal component).
    Lambda = np.diag(eig_values)
    U = eig_vectors

    return Lambda, U

# Gets eigenvalues and their eigenvectors from covariance matrix S that explain more than a  proportion 'prob' of the total variance
def get_eig_prop(S, prop):

    # Compute eigenvalues and eigenvectors
    eig_values, eig_vectors = eigh(S)
    
    # Get in descending order
    asc_indices = eig_values.argsort()
    desc_indices = asc_indices[::-1]
    sorted_eig_values = eig_values[desc_indices]
    sorted_eig_vectors = eig_vectors[:, desc_indices]

    # Calculate total variance 
    total_variance = np.sum(sorted_eig_values)
    
    # Calculate  proportion of variance for each eigenvalue
    variance_proportions = sorted_eig_values / total_variance
    
    # Get indices of eigenvalues that explain > prob of the variance
    selected_indices = np.where(variance_proportions > prop)[0]
    
    # Create a diagonal matrix
    Lambda = np.diag(sorted_eig_values[selected_indices])
    
    # Get eigenvectors
    U = sorted_eig_vectors[:, selected_indices]

    return Lambda, U


def project_image(image, U): 
    a_ij = np.dot(U.T, image)
    x_pca = np.dot(U, a_ij)
    return x_pca

def display_image(orig, proj):
    # fig, ax1, ax2 = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    #1
    orig = orig.reshape(64,64).T
    proj = proj.reshape(64,64).T
    #2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    #3
    ax1.set_title('Original')
    ax2.set_title('Projection')
    #4
    orig_image = ax1.imshow(orig, aspect='equal')
    proj_image = ax2.imshow(proj, aspect='equal')
    #5
    plt.colorbar(orig_image, ax=ax1)
    plt.colorbar(proj_image, ax=ax2)
    #6
    return fig, ax1, ax2


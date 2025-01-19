import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import warnings
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def plot_venn_diagram():
    fig, ax = plt.subplots(subplot_kw=dict(frameon=False, xticks=[], yticks=[]))
    ax.add_patch(plt.Circle((0.3, 0.3), 0.3, fc='red', alpha=0.5))
    ax.add_patch(plt.Circle((0.6, 0.3), 0.3, fc='blue', alpha=0.5))
    ax.add_patch(plt.Rectangle((-0.1, -0.1), 1.1, 0.8, fc='none', ec='black'))
    ax.text(0.2, 0.3, '$x$', size=30, ha='center', va='center')
    ax.text(0.7, 0.3, '$y$', size=30, ha='center', va='center')
    ax.text(0.0, 0.6, '$I$', size=30)
    ax.axis('equal')
    plt.show()


def plot_example_decision_tree():
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 0.8, 1], frameon=False, xticks=[], yticks=[])
    ax.set_title('Example Decision Tree: Animal Classification', size=24)

    def text(ax, x, y, t, size=20, **kwargs):
        ax.text(
            x, y, t,
            ha='center', va='center', size=size,
            bbox=dict(boxstyle='round', ec='k', fc='w'), **kwargs
        )

    text(ax, 0.5, 0.9, "How big is\nthe animal?", 20)
    text(ax, 0.3, 0.6, "Does the animal\nhave horns?", 18)
    text(ax, 0.7, 0.6, "Does the animal\nhave two legs?", 18)
    text(ax, 0.12, 0.3, "Are the horns\nlonger than 10cm?", 14)
    text(ax, 0.38, 0.3, "Is the animal\nwearing a collar?", 14)
    text(ax, 0.62, 0.3, "Does the animal\nhave wings?", 14)
    text(ax, 0.88, 0.3, "Does the animal\nhave a tail?", 14)

    text(ax, 0.4, 0.75, "> 1m", 12, alpha=0.4)
    text(ax, 0.6, 0.75, "< 1m", 12, alpha=0.4)

    text(ax, 0.21, 0.45, "yes", 12, alpha=0.4)
    text(ax, 0.34, 0.45, "no", 12, alpha=0.4)
    plt.show()


def visualize_tree(estimator, X, y, boundaries=True, xlim=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    
    if boundaries:
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()


def plot_tree_interactive(X, y):
    def interactive_tree(depth=1):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X, y)
        visualize_tree(clf, X, y)
    
    interact(interactive_tree, depth=(1, 10))


def plot_kmeans_interactive(min_clusters=1, max_clusters=6):
    from sklearn.cluster import KMeans

    def _kmeans_step(n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.title(f'K-Means Clustering with {n_clusters} Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    interact(_kmeans_step, n_clusters=(min_clusters, max_clusters))


def plot_image_components(x, coefficients=None, mean=0, components=None,
                          imshape=(8, 8), n_components=6, fontsize=12):
    plt.figure(figsize=(15, 6))
    for i in range(n_components):
        plt.subplot(2, 3, i + 1)
        plt.imshow(components[i].reshape(imshape), cmap='viridis')
        plt.title(f'Component {i + 1}', fontsize=fontsize)
        plt.axis('off')
    plt.suptitle('Image Components', fontsize=fontsize)
    plt.show()


def plot_pca_interactive(data, n_components=6):
    pca = PCA(n_components=n_components)
    Xproj = pca.fit_transform(data)

    def show_decomp(i=0):
        plot_image_components(
            data[i],
            coefficients=Xproj[i],
            mean=pca.mean_,
            components=pca.components_,
            imshape=(8, 8),
            n_components=n_components
        )

    interact(show_decomp, i=(0, data.shape[0] - 1))


# Ensure that plots are displayed correctly in non-interactive environments
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
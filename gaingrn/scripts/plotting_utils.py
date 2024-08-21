## scripts/plotting_utils.py
# contains different utilities for plotting

from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FixedLocator)
import numpy as np
import scipy
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import logomaker

def plot_segment_hist(datarow, color, name, length):
    #plots a histogram for the occupancy of each segment residue with total occupancy in the dataset.
    max = np.max(datarow)
    try: 
        dens = stats.gaussian_kde(datarow)
    except:
        print(np.unique(datarow))
        return
    fig = plt.figure(figsize=[4,2])
    fig.set_facecolor('w')
    n, x, _ = plt.hist(datarow, bins=np.linspace(0,max,max+1), histtype=u'step', density=True, color='white',alpha=0)
    plt.plot(x, dens(x),linewidth=2,color=color,alpha=1)
    plt.fill_between(x,dens(x), color=color,alpha=0.1)
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    val_string = f'{round(np.average(datarow),2)}±{round(np.std(datarow),2)}'
    plt.text(max, ymax*0.95, name, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.8, val_string, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.65, f"{round(len(datarow)/length*100, 1)}%", horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.xlabel('Element Length [Residues]')
    plt.ylabel('Relative density [AU]')
    plt.savefig(f'../../TESTING/{name}_hist.svg')
    plt.show()
    plt.close(fig)

def plot_matrix(distances, title='', savename=None):
    # plot the RMSD matrix
    fig = plt.figure(figsize=[6,4])
    fig.set_facecolor('w')
    plt.imshow(distances, cmap='Greys')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('RMSD [$\AA$]')
    if savename is not None:
        plt.savefig(f'{savename}.png',dpi=300)
    else:
        plt.show()
    del fig

def plot_heirarchy(distances, groupname='', savename=None):
    # plots the heirarchical clustering to assess groups of structures.
    reduced_distances = squareform(distances, checks=False)
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
    fig = plt.figure(figsize=[18,4], facecolor='w')
    plt.title(f'RMSD Average linkage hierarchical clustering: {groupname}')
    _ = scipy.cluster.hierarchy.dendrogram(linkage, count_sort='descendent', show_leaf_counts=True, leaf_font_size=3)
    if savename is not None:
        plt.savefig(f"{savename}.png", dpi=200)
    else:
        plt.show()
    del fig

def plot_pca(distance_matrix, cluster_intervals, n_components, name, plot3D=False, save=True):
    # Creates and plots a Principal component analysis for assessing variance in between respective clusters
    colorlist = ['blue','red','green','yellow','orange','purple','forestgreen','limegreen','firebrick']
    #X = center_distance_matrix
    X = distance_matrix
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    X_r = pca.fit(X).transform(X)
    print(X_r.shape)

    fig = plt.figure(figsize=[5,5])
    fig.set_facecolor('w')
    if plot3D:
        ax = ax = fig.add_subplot(projection='3d')
        for i, interval in enumerate(cluster_intervals):
            ax.scatter(X_r[interval[0]:interval[1],0], X_r[interval[0]:interval[1],1], X_r[interval[0]:interval[1],2], marker='o', s=8, c=colorlist[i])
    else:
        ax = fig.add_subplot()
        for i, interval in enumerate(cluster_intervals):
            ax.scatter(X_r[interval[0]:interval[1],0], X_r[interval[0]:interval[1],1], marker='o', s=8, c=colorlist[i])
    ax.set_title(f'PCA of MiniBatchKMeans - {name}')
    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')
    if plot3D:
        ax.set_zlabel('PC 2')
    if save:
        plt.savefig(f'{name}_pca.png', dpi=300)

def plot_hist(datarow, color, name, length):
    max = np.max(datarow)
    try: 
        dens = stats.gaussian_kde(datarow)
    except:
        print(np.unique(datarow))
        return
    fig = plt.figure(figsize=[4,2])
    fig.set_facecolor('w')
    n, x, _ = plt.hist(datarow, bins=np.linspace(0,max,max+1), histtype=u'step', density=True, color='white',alpha=0)
    plt.plot(x, dens(x),linewidth=2,color=color,alpha=1)
    plt.fill_between(x,dens(x), color=color,alpha=0.1)
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    val_string = f'{round(np.average(datarow),2)}±{round(np.std(datarow),2)}'
    plt.text(max, ymax*0.95, name, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.8, val_string, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.65, f"{round(len(datarow)/length*100, 1)}%", horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.xlabel('Element Length [Residues]')
    plt.ylabel('Relative density [AU]')
    plt.savefig(f'{name}_hist.svg')
    plt.show()
    plt.close(fig)


def plot_segment_statistics(sse, xvals=None, y_plddt=None, y_occupancy=None, savename=None, show=False):
    fig, ax = plt.subplots(figsize=[5,2])
    fig.set_facecolor('w')
    ax.xaxis.set_minor_locator(MultipleLocator(1)) #AutoMinorLocator())
    ax.xaxis.set_major_locator(FixedLocator([a for a in range(2,100,3)]))#MultipleLocator(3)))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=6)

    if y_plddt is not None:
        plt.bar(xvals, y_plddt, color='silver', alpha=0.7)

    if y_occupancy is not None:
        plt.plot(xvals, y_occupancy, color='dodgerblue')
    
    if y_occupancy is not None and y_plddt is not None:
        normalized_plddt = np.array(y_plddt)*np.array(y_occupancy)
        plt.bar(xvals, normalized_plddt, color='xkcd:lightish red', alpha=0.1)

    plt.title(f'Element Composition ({sse})')
    plt.yticks(ticks = [0, 0.2, 0.4, 0.6, 0.8, 1], labels = ['0%', '20%', '40%', '60%', '80%', '100%'])
    #plt.ylabel('')
    ax.set_xticklabels([f'{sse}.{str(int(v))}' for v in ax.get_xticks()], rotation=90)
    if savename is not None:
        plt.savefig(f'', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_logo_segment(dataframe, sse, threshold=0.05, savename=None):
    # Note down the first and last row where the occupation threshold is met.
    firstval = None
    for i, r in dataframe.iterrows():
        if np.sum(r) > threshold: 
            if firstval is None:
                firstval = i
            lastval = i
    print(firstval, lastval)
    subframe = dataframe.truncate(before=firstval, after=lastval)

    # With the specified interval, create the Logoplot
    fig, ax = plt.subplots(figsize=[5,2])
    cons_logo = logomaker.Logo(subframe,
                                ax=ax,
                                color_scheme='chemistry',
                                show_spines=False,
                                font_name='FreeSans')

    fig.set_facecolor('w')

    ax.xaxis.set_minor_locator(MultipleLocator(1)) #AutoMinorLocator())
    ax.xaxis.set_major_locator(FixedLocator([a for a in range(2,100,3)]))#MultipleLocator(3))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=6)
    ax.set_xticklabels([f'{sse}.{str(int(v))}' for v in ax.get_xticks()], rotation=90)

    cons_logo.draw()

    fig.tight_layout()
    fig.set_facecolor('w')

    plt.savefig(savename, bbox_inches='tight')
    plt.close(fig)

def plot_segment_matrix(segments, sse_matrix, savename, show=False):
    # Plot the element occupancy for each of the SSE elements as a 2D matrix
    plt.imshow(sse_matrix, cmap='gist_yarg')
    plt.xticks(ticks= range(len(segments)), labels=segments, rotation=90)
    plt.yticks(ticks= range(len(segments)), labels=segments)
    plt.xlim(-0.5,20.5)
    plt.ylim(20.5,-0.5)
    cbar = plt.colorbar(shrink=0.5)
    if savename is not None: 
        plt.savefig(savename)
    if show:
        plt.show()
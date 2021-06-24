import numpy as np
import  matplotlib.pyplot as plt


def ax_plot_simplex(ax, optimal_vertex, sim):
    simplex = np.array((optimal_vertex, *(sim + optimal_vertex), optimal_vertex))
    ax.plot(*optimal_vertex, color='limegreen', marker='o')
    ax.plot(simplex[1:-1,0], simplex[1:-1,1], '.', color='blue', marker='o')
    ax.plot(simplex[...,0], simplex[...,1], '-.', color='gray', lw=2)
    

def ax_plot_track(opt, ax, target, n_points, plot_simplex, plot_trure, aspect):
    track = opt.track[-n_points:]
    ax.plot(track[:, 0], track[:, 1], linestyle='-', color='red', marker='o')

    link_best = np.array((track[-1], target))
    ax.plot(link_best[:, 0], link_best[:, 1], linestyle=':', color='red')
    ax.plot(target[0], target[1], color='limegreen', marker='*')
        
    if plot_simplex:
        #simplex = np.array((opt.optimal_vertex, *(opt.sim + opt.optimal_vertex), opt.optimal_vertex))
        ax_plot_simplex(ax, opt.optimal_vertex, opt.sim)
            
    if plot_trure:
        trust_region = plt.Circle(opt.optimal_vertex, opt.rho, color='khaki', fill=True, alpha=0.5)
        ax.add_patch(trust_region)
        
    if aspect:
        ax.set_aspect('equal')
        

def plot_track(opt, target, n_points=5, plot_simplex=True, plot_trure=True, aspect=False):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax_plot_track(opt, ax1, target,
                  n_points=n_points, plot_simplex=plot_simplex,
                  plot_trure=plot_trure, aspect=aspect)
    ax_plot_track(opt, ax2, target,
                  n_points=0, plot_simplex=plot_simplex,
                  plot_trure=plot_trure, aspect=aspect)
    
    plt.show()
    return fig, (ax1, ax2)

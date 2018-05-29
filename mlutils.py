from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_2D_boundary(predict, mins, maxs, line_width=3, line_color="black", line_alpha=1, label=None):
    n = 200
    mins -= np.abs(mins)*.2
    maxs += np.abs(maxs)*.2
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    preds = predict(D)
    levels = np.sort(np.unique(preds))
    levels = [np.min(levels)-1] + [np.mean(levels[i:i+2]) for i in range(len(levels)-1)] + [np.max(levels)+1]
    p = (preds*1.).reshape((n,n))
    plt.contour(gd0,gd1,p, levels=levels, alpha=line_alpha, colors=line_color, linewidths=line_width)
    if label is not None:
        plt.plot([0,0],[0,0], lw=line_width, color=line_color, label=label)
    return np.sum(p==0)*1./n**2, np.sum(p==1)*1./n**2


def plot_2Ddata_with_boundary(predict, X, y, line_width=3, line_alpha=1, line_color="black", dots_alpha=.5, label=None, noticks=False):
    mins,maxs = np.min(X,axis=0), np.max(X,axis=0)    
    plot_2Ddata(X,y,dots_alpha)
    p0, p1 = plot_2D_boundary(predict, mins, maxs, line_width, line_color, line_alpha, label )
    if noticks:
        plt.xticks([])
        plt.yticks([])
        
    return p0, p1



def plot_2Ddata(X, y, dots_alpha=.5, noticks=False):
    colors = cm.hsv(np.linspace(0, .7, len(np.unique(y))))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X[y==label][:,0], X[y==label][:,1], color=colors[i], alpha=dots_alpha)
    if noticks:
        plt.xticks([])
        plt.yticks([])


class Example_Bayes2DClassifier():
    
    def __init__ (self, mean0, cov0, mean1, cov1, w0=1, w1=1):
        self.rv0 = multivariate_normal(mean0, cov0)
        self.rv1 = multivariate_normal(mean1, cov1)
        self.w0  = w0
        self.w1  = w1

    def sample (self, n_samples=100):
        n = int(n_samples)
        n0 = int(n*1.*self.w0/(self.w0+self.w1))
        n1 = int(n) - n0
        X = np.vstack((self.rv0.rvs(n0), self.rv1.rvs(n1)))
        y = np.zeros(n)
        y[n0:] = 1
        
        return X,y
        
    def fit(self, X,y):
        pass
    
    def predict(self, X):
        p0 = self.rv0.pdf(X)
        p1 = self.rv1.pdf(X)
        return 1*(p1>p0)
    
    def score(self, X, y):
        return np.sum(self.predict(X)==y)*1./len(y)
    
    def analytic_score(self):
        """
        returns the analytic score on the knowledge of the probability distributions.
        the computation is a numeric approximation.
        """

        # first get limits for numeric computation. 
        # points all along the bounding box should have very low probability
        def get_boundingbox_probs(pdf, box_size):
            lp = np.linspace(-box_size,box_size,50)
            cp = np.ones(len(lp))*lp[0]
            bp = np.sum([pdf([x,y]) for x,y in zip(lp, cp)]  + \
                        [pdf([x,y]) for x,y in zip(lp, -cp)] + \
                        [pdf([y,x]) for x,y in zip(lp, cp)]  + \
                        [pdf([y,x]) for x,y in zip(lp, -cp)])
            return bp

        rngs = np.exp(np.arange(15))
        for rng in rngs:
            bp0 = get_boundingbox_probs(self.rv0.pdf, rng)
            bp1 = get_boundingbox_probs(self.rv1.pdf, rng)
            if bp0<1e-5 and bp1<1e-5:
                break

        if rng==rngs[-1]:
            print "warning: bounding box prob size",rng,"has prob",np.max([bp0, bp1])        
        
        
        # then, compute numerical approximation by building a grid
        mins, maxs = [-rng, -rng], [+rng, +rng]
        n = 100
        d0 = np.linspace(mins[0], maxs[0],n)
        d1 = np.linspace(mins[1], maxs[1],n)
        gd0,gd1 = np.meshgrid(d0,d1)
        D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))

        p0, p1 = self.rv0.pdf(D), self.rv1.pdf(D)

        # grid points where distrib 1 has greater probability than distrib 0
        gx = (p1-p0>0)*1.

        # true positive and true negative rates
        tnr = np.sum(p0*(1-gx))/np.sum(p0)
        tpr = np.sum(p1*gx)/np.sum(p1)
        return (self.w0*tnr+self.w1*tpr)/(self.w0+self.w1)  

def plot_cluster_predictions(clustering, X, n_clusters = None, cmap = plt.cm.plasma,
                             plot_data=True, plot_centers=True, show_metric=False,
                             title_str=""):

    assert not hasattr(clustering, "n_clusters") or \
           (hasattr(clustering, "n_clusters") and n_clusters is not None), "must specify `n_clusters` for "+str(clustering)

    if n_clusters is not None:
        clustering.n_clusters = n_clusters

    y = clustering.fit_predict(X)
    # remove elements tagged as noise (cluster nb<0)
    X = X[y>=0]
    y = y[y>=0]

    if n_clusters is None:
        n_clusters = len(np.unique(y))

    if plot_data:        
        plt.scatter(X[:,0], X[:,1], color=cmap((y*255./(n_clusters-1)).astype(int)), alpha=.5)
    if plot_centers and hasattr(clustering, "cluster_centers_"):
        plt.scatter(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1], s=150,  lw=3,
                    facecolor=cmap((np.arange(n_clusters)*255./(n_clusters-1)).astype(int)),
                    edgecolor="black")   

    if show_metric:
        sc = silhouette_score(X, y) if len(np.unique(y))>1 else 0
        plt.title("n_clusters %d, sc=%.3f"%(n_clusters, sc)+title_str)
    else:
        plt.title("n_clusters %d"%n_clusters+title_str)

    plt.axis("off")
    return

def experiment_number_of_clusters(X, clustering, show_metric=True,
                                  plot_data=True, plot_centers=True, plot_boundaries=False):
    plt.figure(figsize=(15,6))
    for n_clusters in range(2,10):
        clustering.n_clusters = n_clusters
        y = clustering.fit_predict(X)

        cm = plt.cm.plasma
        plt.subplot(2,4,n_clusters-1)

        plot_cluster_predictions(clustering, X, n_clusters, cm, 
                                 plot_data, plot_centers, show_metric)


def experiment_KMeans_number_of_iterations(X, n_clusters=3,
                                    plot_data=True, plot_centers=True, plot_boundaries=False):
    plt.figure(figsize=(15,6))
    for i in range(10):
        init_centroids = np.vstack((np.linspace(np.min(X[:,0]), np.max(X[:,0])/20, n_clusters), 
                                    [np.min(X[:,1])]*n_clusters)).T

        x0min, x0max = np.min(X[:,0]), np.max(X[:,0])
        x1min, x1max = np.min(X[:,1]), np.max(X[:,1])
        c = np.random.random(size=(n_clusters, 2))/3
        c[:,0] = x0min + c[:,0]*(x0max-x0min)
        c[:,1] = x1min + c[:,1]*(x1max-x1min)
        init_centroids = c

        plt.subplot(2,5,i+1)
        cm = plt.cm.plasma
        
        if i==0:
            
            y = np.argmin(np.vstack([np.sqrt(np.sum((X-i)**2, axis=1)) for i in init_centroids]).T, axis=1)
            
            plt.scatter(X[:,0], X[:,1], color=cm((y*255./(n_clusters-1)).astype(int)), alpha=.5)
            plt.scatter(init_centroids[:,0], init_centroids[:,1], s=150,  lw=3,
                       facecolor=cm((np.arange(n_clusters)*255./(n_clusters-1)).astype(int)),
                       edgecolor="black")   
            plt.axis("off")
            plt.title("initial state")
            

        else:
            n_iterations = i if i<4 else (i-1)*2

            km = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, max_iter=2*n_iterations)
            km.fit(X)

            plot_cluster_predictions(km, X, n_clusters, cm, plot_data, plot_centers, plot_boundaries)

            plt.title("n_iters %d"%(n_iterations))





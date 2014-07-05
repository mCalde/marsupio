
### Confronto tra le 4 acquisizioni del 20/6/2014


    import feature_man as fm
    import plotting as plt
    import preprocessing as prproc
    import mars as ms
    
    import matplotlib.pyplot as pl
    import matplotlib.cm as cm
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import numpy as np


    # ampiezza finestra (50 == 1s)
    window_length = 50
    # overlap tra le finestre
    overlap = window_length/2
    data_115818,sgmdata_115818 = ms.load_dataset(window_length,overlap)
    data_120250,sgmdata_120250 = ms.load_dataset(window_length,overlap,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120250.txt')
    data_120611,sgmdata_120611 = ms.load_dataset(window_length,overlap,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120611.txt')
    data_120922,sgmdata_120922 = ms.load_dataset(window_length,overlap,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120922.txt')
    
    all_data = [(data_115818,"115818"),(data_120250,"120250"),(data_120611,"120611"),(data_120922,"120922")]
    sgm_data = [sgmdata_115818,sgmdata_120250,sgmdata_120611,sgmdata_120922]


##### Plot di tutti i segnali delle 4 acquisizioni senza nessun preprocessing:


    for (data,title) in all_data:
        plt.plot_all(data,"Acquisizione "+title)

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_4_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_4_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_4_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_4_7.png)


##### Plot di tutti i segnali delle 4 acquisizioni filtrate con un filtro mediano:


    
    data_115818,sgmdata_115818 = ms.load_dataset(window_length,overlap,median_filter=True)
    data_120250,sgmdata_120250 = ms.load_dataset(window_length,overlap,median_filter=True,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120250.txt')
    data_120611,sgmdata_120611 = ms.load_dataset(window_length,overlap,median_filter=True,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120611.txt')
    data_120922,sgmdata_120922 = ms.load_dataset(window_length,overlap,median_filter=True,alldatafile='../../acquisizione20062014/acquisizione_20062014/Data_120922.txt')
    
    all_data = [(data_115818,"115818"),(data_120250,"120250"),(data_120611,"120611"),(data_120922,"120922")]
    sgm_data = [sgmdata_115818,sgmdata_120250,sgmdata_120611,sgmdata_120922]
    
    for (data,title) in all_data:
        plt.plot_all(data,"Acquisizione "+title)

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_6_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_6_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_6_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_6_7.png)


##### Plot di alcune variabili:


    cols = ['b','r','g','m']
    for (data,title),c in zip(all_data,cols):
        print "Acquisizione", title
        plt.plot_in_subplots(data,0,1,c,subsize=(25,8))

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_8_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_8_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_8_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_8_7.png)



    cols = ['b','r','g','m']
    for (data,title),c in zip(all_data,cols):
        print "Acquisizione", title
        plt.plot_in_subplots(data,1,2,c,subsize=(25,8))

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_9_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_9_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_9_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_9_7.png)



    cols = ['b','r','g','m']
    for (data,title),c in zip(all_data,cols):
        print "Acquisizione", title
        plt.plot_in_subplots(data,2,3,c,subsize=(25,8))

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_10_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_10_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_10_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_10_7.png)



    cols = ['b','r','g','m']
    for (data,title),c in zip(all_data,cols):
        print "Acquisizione", title
        plt.plot_in_subplots(data,4,5,c,subsize=(25,8))

    Acquisizione 115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_11_1.png)


    Acquisizione 120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_11_3.png)


    Acquisizione 120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_11_5.png)


    Acquisizione 120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_11_7.png)


#### Clustering dei 4 insiemi di dati.


    
    featdim = 2
    ncluster = 10
    for (data,title) in all_data:
        print "Acquisizione ",title
        pca = PCA(n_components=featdim)
        X_r = pca.fit(data).transform(data)
        kmeans = KMeans(n_clusters=ncluster)
        X_r = prproc.scale(X_r)
        kmeans.fit(X_r)
        plt.plot_clustering_result(X_r,kmeans,0,1,(12,8))

    Acquisizione  115818



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_13_1.png)


    Acquisizione  120250



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_13_3.png)


    Acquisizione  120611



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_13_5.png)


    Acquisizione  120922



![png](confronto%20tra%20acquisizioni_files/confronto%20tra%20acquisizioni_13_7.png)



    


    

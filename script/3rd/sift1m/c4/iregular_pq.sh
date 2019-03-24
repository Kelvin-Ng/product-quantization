# hpc1.cse.cuhk.edu.hk:/research/jcheng2/xinyan/github/product-quantization> python run_imi_iregualr_rq.py
# Parameters: dataset = sift1m, topK = 1000, Ks = 256, metric = euclid
# load the base data data/sift1m/sift1m_base.fvecs,
# load the queries data/sift1m/sift1m_query.fvecs,
# load the ground truth data/sift1m/1000_sift1m_euclid_groundtruth.ivecs
#    Training the subspace: 0 / 2, 0 -> 64
#    Training the subspace: 1 / 2, 64 -> 128
# compress items with imi
# train hierarchy rqs on residuals
#    Training the subspace: 0 / 8, 0 -> 16
#    Training the subspace: 1 / 8, 16 -> 32
#    Training the subspace: 2 / 8, 32 -> 48
#    Training the subspace: 3 / 8, 48 -> 64
#    Training the subspace: 4 / 8, 64 -> 80
#    Training the subspace: 5 / 8, 80 -> 96
#    Training the subspace: 6 / 8, 96 -> 112
#    Training the subspace: 7 / 8, 112 -> 128
#    Training the subspace: 0 / 3, 0 -> 43
#    Training the subspace: 1 / 3, 43 -> 86
#    Training the subspace: 2 / 3, 86 -> 128
# compress items with rq

# probe 16 items
# hierarchy_index [ 0  3 16]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
   16,    1,  0.2800,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,  0.3000,
   16,   10,  0.0410,  0.0535,  0.0615,  0.0635,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,  0.0650,
   16,   20,  0.0222,  0.0305,  0.0370,  0.0403,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,  0.0422,
   16,   50,  0.0094,  0.0144,  0.0196,  0.0227,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,  0.0246,
   16,  100,  0.0054,  0.0088,  0.0123,  0.0155,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,  0.0170,
   16, 1000,  0.0008,  0.0015,  0.0026,  0.0042,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,  0.0054,

# probe 256 items
# hierarchy_index [  0  51 256]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
  256,    1,  0.8450,  0.9200,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,  0.9300,
  256,   10,  0.0930,  0.1445,  0.1955,  0.2480,  0.2960,  0.3215,  0.3345,  0.3410,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,  0.3420,
  256,   20,  0.0470,  0.0767,  0.1113,  0.1578,  0.2038,  0.2370,  0.2602,  0.2687,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,  0.2692,
  256,   50,  0.0192,  0.0332,  0.0523,  0.0823,  0.1170,  0.1507,  0.1822,  0.1967,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,  0.1987,
  256,  100,  0.0097,  0.0175,  0.0290,  0.0493,  0.0764,  0.1071,  0.1407,  0.1609,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,  0.1647,
  256, 1000,  0.0010,  0.0019,  0.0037,  0.0071,  0.0133,  0.0237,  0.0403,  0.0605,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,  0.0727,

# probe 1024 items
# hierarchy_index [   0  204 1024]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
 1024,    1,  0.9150,  0.9800,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,  0.9850,
 1024,   10,  0.0990,  0.1650,  0.2390,  0.3175,  0.4015,  0.4805,  0.5440,  0.5845,  0.6020,  0.6050,  0.6055,  0.6055,  0.6055,  0.6055,  0.6055,  0.6055,  0.6055,  0.6055,
 1024,   20,  0.0498,  0.0860,  0.1340,  0.1968,  0.2785,  0.3605,  0.4360,  0.4900,  0.5220,  0.5292,  0.5295,  0.5295,  0.5295,  0.5295,  0.5295,  0.5295,  0.5295,  0.5295,
 1024,   50,  0.0199,  0.0362,  0.0622,  0.0995,  0.1530,  0.2252,  0.3028,  0.3761,  0.4239,  0.4421,  0.4436,  0.4436,  0.4436,  0.4436,  0.4436,  0.4436,  0.4436,  0.4436,
 1024,  100,  0.0100,  0.0186,  0.0336,  0.0570,  0.0945,  0.1510,  0.2231,  0.3019,  0.3635,  0.3921,  0.3959,  0.3959,  0.3959,  0.3959,  0.3959,  0.3959,  0.3959,  0.3959,
 1024, 1000,  0.0010,  0.0020,  0.0039,  0.0075,  0.0142,  0.0266,  0.0489,  0.0854,  0.1388,  0.1955,  0.2203,  0.2203,  0.2203,  0.2203,  0.2203,  0.2203,  0.2203,  0.2203,

# probe 4096 items
# hierarchy_index [   0  819 4096]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
 4096,    1,  0.9100,  0.9900,  0.9900,  0.9950,  0.9950,  0.9950,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
 4096,   10,  0.0990,  0.1755,  0.2765,  0.3800,  0.4835,  0.5855,  0.6770,  0.7515,  0.8050,  0.8405,  0.8495,  0.8525,  0.8525,  0.8525,  0.8525,  0.8525,  0.8525,  0.8525,
 4096,   20,  0.0495,  0.0900,  0.1527,  0.2333,  0.3342,  0.4460,  0.5625,  0.6638,  0.7395,  0.7888,  0.8070,  0.8148,  0.8150,  0.8150,  0.8150,  0.8150,  0.8150,  0.8150,
 4096,   50,  0.0200,  0.0373,  0.0675,  0.1155,  0.1840,  0.2764,  0.3957,  0.5195,  0.6332,  0.7150,  0.7544,  0.7662,  0.7666,  0.7666,  0.7666,  0.7666,  0.7666,  0.7666,
 4096,  100,  0.0100,  0.0189,  0.0355,  0.0634,  0.1090,  0.1798,  0.2825,  0.4054,  0.5372,  0.6414,  0.7012,  0.7233,  0.7253,  0.7253,  0.7253,  0.7253,  0.7253,  0.7253,
 4096, 1000,  0.0010,  0.0020,  0.0039,  0.0076,  0.0148,  0.0284,  0.0534,  0.0977,  0.1712,  0.2796,  0.4069,  0.5026,  0.5256,  0.5256,  0.5256,  0.5256,  0.5256,  0.5256,

# probe 16384 items
# hierarchy_index [    0  3276 16384]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
16384,    1,  0.9400,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
16384,   10,  0.1000,  0.1860,  0.3025,  0.4265,  0.5560,  0.6840,  0.7830,  0.8610,  0.9170,  0.9455,  0.9640,  0.9710,  0.9715,  0.9720,  0.9720,  0.9720,  0.9720,  0.9720,
16384,   20,  0.0500,  0.0945,  0.1655,  0.2675,  0.3917,  0.5298,  0.6645,  0.7768,  0.8610,  0.9083,  0.9418,  0.9592,  0.9645,  0.9652,  0.9652,  0.9652,  0.9652,  0.9652,
16384,   50,  0.0200,  0.0387,  0.0718,  0.1284,  0.2109,  0.3291,  0.4763,  0.6294,  0.7592,  0.8461,  0.9070,  0.9373,  0.9488,  0.9510,  0.9510,  0.9510,  0.9510,  0.9510,
16384,  100,  0.0100,  0.0196,  0.0375,  0.0705,  0.1230,  0.2124,  0.3382,  0.4913,  0.6478,  0.7695,  0.8609,  0.9095,  0.9311,  0.9361,  0.9364,  0.9364,  0.9364,  0.9364,
16384, 1000,  0.0010,  0.0020,  0.0040,  0.0078,  0.0153,  0.0299,  0.0572,  0.1067,  0.1914,  0.3230,  0.4958,  0.6670,  0.7841,  0.8335,  0.8399,  0.8399,  0.8399,  0.8399,

# probe 65536 items
# hierarchy_index [    0 13107 65536]
Probe, top-K, 00001.,  00002.,  00004.,  00008.,  00016.,  00032.,  00064.,  00128.,  00256.,  00512.,  01024.,  02048.,  04096.,  08192.,  16384.,  32768.,  65536.,  131072.,
65536,    1,  0.9450,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
65536,   10,  0.1000,  0.1870,  0.3055,  0.4430,  0.5830,  0.7240,  0.8355,  0.9130,  0.9525,  0.9805,  0.9935,  0.9965,  0.9980,  0.9995,  0.9995,  0.9995,  0.9995,  0.9995,
65536,   20,  0.0500,  0.0953,  0.1685,  0.2755,  0.4160,  0.5695,  0.7225,  0.8408,  0.9175,  0.9595,  0.9832,  0.9925,  0.9963,  0.9988,  0.9990,  0.9992,  0.9992,  0.9992,
65536,   50,  0.0200,  0.0391,  0.0733,  0.1326,  0.2249,  0.3552,  0.5204,  0.6952,  0.8305,  0.9178,  0.9609,  0.9816,  0.9909,  0.9955,  0.9974,  0.9977,  0.9977,  0.9977,
65536,  100,  0.0100,  0.0198,  0.0382,  0.0725,  0.1316,  0.2276,  0.3703,  0.5472,  0.7217,  0.8507,  0.9263,  0.9653,  0.9837,  0.9918,  0.9952,  0.9959,  0.9960,  0.9960,
65536, 1000,  0.0010,  0.0020,  0.0040,  0.0079,  0.0157,  0.0308,  0.0601,  0.1145,  0.2102,  0.3622,  0.5614,  0.7507,  0.8783,  0.9465,  0.9760,  0.9837,  0.9842,  0.9842,
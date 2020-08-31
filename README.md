# SincNet-MER
End-to-End Music Emotion Recognition: Towards Language Sensitive Models

We present here an adaptation of the original SincNet deep learning architecture, that can be found at https://github.com/mravanelli/SincNet/ to the task of Music Emotion Recognition. The original model was created for speaker recognition and verification, therefore our results are not very good, but the end-to-end architecture seems promising for the task with further improvements.

We also provide here our experiments to establish a baseline feature-based model, i.e. the Multi-Layer Perceptron from Scikit-Learn that was used as reference for SincNet.

We used 3 different datasets with music in English (4Q-EMOTION [1]), Mandarin (CH-818 [2]) and Turkish (TR-MUSIC [3]), in an attempt to observe cultural and cross-cultural patterns and propose that MER might be improved with context-based considerations. 

Details about all experiments and results can be found in the associated report. 


This repository was developed as part of the Master's Thesis in Sound and Music Computing, at Universitat Pompeu Fabra, in the 2019-2020 academic year. 

[1] Panda R., Malheiro R., Paiva R. P. (2018). “Musical Texture and Expressivity Features for Music Emotion Recognition”. 19th International Society for Music Information Retrieval Conference – ISMIR 2018, Paris, France

[2] X. Hu and Y. Yang, "Cross-Dataset and Cross-Cultural Music Mood Prediction: A Case on Western and Chinese Pop Songs" in IEEE Transactions on Affective Computing, vol. 8, no. 02, pp. 228-240, 2017. doi: 10.1109/TAFFC.2016.2523503

[3] Er, Mehmet & Aydilek, İbrahim. (2019). Music Emotion Recognition by Using Chroma Spectrogram and Deep Visual Features. International Journal of Computational Intelligence Systems. 12. 10.2991/ijcis.d.191216.001

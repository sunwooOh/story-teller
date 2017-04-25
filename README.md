# 뿅뿅

## Language
- **Learning to Generate Reviews and Discovering Sentiment**, 2017  
Proposes a character(byte)-level language model, which has simplicity and generality.
Uses multiplicative LSTM(mLSTM) instead of LSTM.
- **Skip-thought Vectors**, Kiros et al., NIPS 2015  
Learns representations of phrases/sentences/docs: trains a sentence encoder by predicting the preceding and following sentence.

## Object / Image retrieval
- **Natural Language Object Retrieval**, CVPR'16 [[github](https://github.com/ronghanghu/natural-language-object-retrieval)] [[paper](https://arxiv.org/pdf/1511.04164.pdf)] [[project](http://ronghanghu.com/text_obj_retrieval)]
- **Image Retrieval using Scene Graphs**, CVPR'15 [[paper](http://cs.stanford.edu/people/jcjohns/papers/cvpr2015/JohnsonCVPR2015.pdf)] [[dataset](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip)] [[supplementary](http://cs.stanford.edu/people/jcjohns/cvpr15_supp/)]
- **Particular Object Retrieval with Integral Max-pooling of CNN Activations**, ICLR'16 [[project-MATLAB code](http://cmp.felk.cvut.cz/~toliageo/soft.html)] [[paper](http://arxiv.org/pdf/1511.05879v2.pdf)] [[slides](http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D3L6-ranking.pdf)]
- **Ranking and Retrieval of Image Sequences from Multiple Paragraph Queries**, CVPR'15  
Uses SVM?

## Image / Video captioning
- **End-to-end Concept Word Detection for Video Captioning, Retrieval, and Question Answering**, CVPR'17
- **Sequence to Sequence -- Video to Text**, ICCV'15 [[paper](https://arxiv.org/abs/1505.00487)] [[tensorflow implementation](https://github.com/jazzsaxmafia/video_to_sequence)]
- **From Captions to Visual Concepts and Back**, CVPR'15 [[github](https://github.com/s-gupta/visual-concepts)] [[paper](https://arxiv.org/abs/1411.4952)]
### Storytelling
- **Expressing an Image Stream with a Sequence of Natural Sentences**, NIPS'15  
Takes a stream of photos as an input, outputs a sequence of sentences.
Trains *Coherence recurrent convolutional network* (CRCN), which combines CNN, BiRNN, and entity-based local coherence model.
CNN embeds the image, BiRNN generates natural sentences, and local coherence model smoothens the flow of sentences,
in an unsupervised manner.
Blog posts(20K posts, 140K images on NYC and Disneyland tour) as the dataset, since a travel logs usually have a series of photographs and corresponding descriptions.

## Sentence Representation
- **Distributed Representations of Sentences and Documents**, ICML'14  
Neural net based unsupervised algorithm that learns fixed-length feature representation from variable-length passage.
- **Doc2Vec** [[DL4J tutorial](https://deeplearning4j.org/doc2vec)] [[another tutorial](https://rare-technologies.com/doc2vec-tutorial/)] [[gensim doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)]

## Generative Models
### Painting
- **A Neural Representation of Sketch Drawings**, 2017  
`sketch-rnn`, an RNN that constructs stroke-based drawings. Trained with crude human-drawn images,
which are represented by vectorized strokes. Plans to release the dataset.
- **Portrait Drawing by Paul the Robot**, 2013
- **Artist Agent: A Reinforcement Learning Approach to Automatic Stroke Generation in Oriental Ink Painting**, ICML'12

### Handwritings/Characters
- **Generating Sequences with Recurrent Neural Networks**, 2013
- **Recurrent Net Dreams Up Fake Chinese Characters in Vector Format with TensorFlow**, 2015
- **Drawing and Recognizing Chinese Characters with Recurrent Neural Networks**, 2016

### Image Generation
- **Parallel Multiscale Autoregressive Density Estimation**, 2017 [[paper](https://arxiv.org/pdf/1703.03664.pdf)]  
Parallelizes the Conditional PixelCNN by breaking weak dependencies among pixels
- **Conditional Image Generation with PixelCNN Decoders**, NIPS 2016
- **Pixel Recurrent Neural Networks**, ICLR 2016  
Models the discrete probability of the raw pixel values using autoregressive image modeling.

## Autoencoders
- **Auto-Encoding Variational Bayes**, 2013  
Variational Autoencoder


## Datasets
### Movie related
- **MPII** [[project](http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/mpii-movie-description-dataset/)]
- **MovieQA** [[project](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjctrixp-fSAhWJx7wKHTX3AUIQFggYMAA&url=http%3A%2F%2Fmovieqa.cs.toronto.edu%2F&usg=AFQjCNEcFbWJuLYlhZBzF6HMlCTOTSqR6A&sig2=KDRiu0sCkDx8Fxfq9rnJ-A&bvm=bv.150120842,d.dGc)] [[paper](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjctrixp-fSAhWJx7wKHTX3AUIQFggeMAE&url=https%3A%2F%2Farxiv.org%2Fabs%2F1512.02902&usg=AFQjCNE5v9TUrWx-_Xwk6SR7pVCeiWYMdg&sig2=BXJwrgNEMs_FNjNGmIjBBQ&bvm=bv.150120842,d.dGc)] [[github](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjctrixp-fSAhWJx7wKHTX3AUIQFgglMAI&url=https%3A%2F%2Fgithub.com%2Fmakarandtapaswi%2FMovieQA_CVPR2016&usg=AFQjCNHODqLIvqaHXqhJK9vGrlW3IShI0w&sig2=8O4FCG8DQ6XflMSEEqJ1Wg&bvm=bv.150120842,d.dGc)]

### Image retrieval
- **INRIA Holidays**
- **Oxford Buildings, Oxford Building 105K**
- **Univ. of Kentucky benchmark (UKB)**

## Storytelling
- **Visual Storytelling (SIND)**, NAACL'16 [[paper](https://arxiv.org/abs/1604.03968)] [[project page](http://visionandlanguage.net/VIST/)]
- **Expressing an Image Stream with a Sequence of Natural Sentences (NYC, Disney)**, NIPS'15 [[github](https://github.com/cesc-park/CRCN)]

## Others
- **Neural Turing Machines**, 2014 [[paper](https://arxiv.org/abs/1410.5401)] [[unofficial slides](http://klab.smpp.northwestern.edu/wiki/images/4/43/NTM2.pdf)]
- **Hybrid Computing Using a Neural Network with Dynamic External Memory**, Nautre 2016 [[paper](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)] [[code](https://github.com/deepmind/dnc)]  
Differentiable neural computer.
- [**Deep Learning Textbook**](http://www.deeplearningbook.org/), Ian Goodfellow


## Evaluation metrics
- **Recall@1, Recall@5, Recall@10**: The percentage of groundtruths in the top *k* retrieved objects.
- **Median Rank**

## What to read next
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, **Deep Learning**, 2016.  
Review on unsupervised representation learning
- **Skip-thought Vectors**, Kiros et al. NIPS 2015  
Trains a sentence encoder by predicting the preceding and following sentence.
- **Understanding Neural Networks Through Representation Erasure**, Li et al., 2016  
word2vec has small subsets of dimensions strongly associated with specific tasks
- **Multiplicative LSTM for Sequence Modelling**, Krause et al, 2016 [[paper](https://arxiv.org/abs/1609.07959)]
- **Recurrent Highway Networks** [[paper](https://arxiv.org/abs/1607.03474)]
- **StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks**, 2016

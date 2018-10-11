# CNN-based-gesture-reconition
This project recognizes ASL letters using deep learning method.The project is developed using Sypder and Python 3.5.
The project aims to perform Supervised Learning and corretly classify ASL images.
Firsly I have created a dataset comprising of 25 ASL Letters.Then i have split the dataset into training and testing data.Testing data will contain 25% of training data.For example my Training dataset contails 550 letters,so my testing dataset will contain 110 images(letters).
Next i have used the VGG16 arhitecture to build Deep learning into CNN.
So I gave path for training data,passed it to CNN for training.
Based on the training,CNN will perform classification and will correctly predict the letter for the given sign image.
The output is shown in Text as a "Letter" as well in "Audio".

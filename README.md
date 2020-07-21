# Cats-vs-Dogs - A Machine Learning Algorithm

@Authors:
  - Efrem Yohannes-Mason
  - Beatriz Cangas-Perez

<br>

This project contains the following items:
  - a PDF file containing the Final Project Report
  - a MATLAB file containing the final solutions titled 'finalsolution.m'
  - a MATLAB file containing the pre processing function titled ‘readAndPreprocessImage.m’
  - a MATLAB file containing the pre processing function titled ‘readAndPreprocessImageVgg.m’
  - a README file containing any relevant information regarding this project (current document)

<br>

For this project you will need the following:
  - MATLAB
  - "Deep Learning Toolbox Model for AlexNet Network" package within MATLAB
  - "Deep Learning Toolbox Model for VGG16 Network" package within MATLAB

<br>

The project will account for the following processes:
  01. Download MATLAB.
  02. Download the "Deep Learning Toolbox Model for AlexNet Network" package within MATLAB.
  03. Download the "Deep Learning Toolbox Model for VGG16 Network" package within MATLAB.
  04. Download this project.
  05. Run starter code using the dataset from Kaggle: https://www.kaggle.com/c/dogs-vs-cats/data.
  06. Run "Part 1" of the starter code and ensure that it works as intended. Inspect the contents of the variable model.
  07. (OPTIONAL) Used the “Deep Network Designer” app (within MATLAB) to explore the model interactively.
  08. Run "Part 2" of the starter code and ensure that it works as intended.
  09. Run "Part 3" of the starter code and ensure that it works as intended.
  10. Run "Part 4" of the starter code and ensure that it works as intended.
  11. The starter code used a very small subset of the Kaggle dataset (20 images of dogs and 20 images of cats) out of the 25,000 images (2 x 12,500) available. This has to be changed.  
  12. Download the train.zip data file found here: https://www.kaggle.com/c/dogs-vs-cats/data (this is already included in this project in the root). You can disregard test1.zip (for now) and the  sampleSubmission CSV file (forever).
  13. Write code for “Part 5” of the starter code.
  14. Start a new MATLAB script containing the “final solution”. 
  15. Then, and only then, write code to test the three approaches using the Kaggle test dataset (test1.zip file from Kaggle).

<br> 

During this assignment, as research was performed, the following questions were addressed:
  1. What type of preprocessing is performed by the auxiliary function readAndPreprocessImage ?
  2. What can you say about the montage with network weights for the first convolutional layer ?
  3. How many images are there in each set (training / validation)?
  4. Is the validation accuracy (in my example 66.67%) acceptable for a two-class classifier? Why (not)? If not, what could be the problem?
  5. Did your classifier recognize 'Doge' as a dog? If not, can you tell why?
  6. Is the validation accuracy (in my example 75%) better than before? What could be the reason(s) behind such (modest) improvement? How could you improve it even further?
  7. Did the validation accuracy improve as a result of using a much larger training dataset? How could you improve it even further?

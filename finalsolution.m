%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Summer 2020 
%% Final Project - Final Solution
% Group Members:
% Efrem Yohannes-Mason
% Beatriz Cangas-Perez

%% BASELINE
%% Loading a pre-trained "AlexNet"

model = alexnet;

%% Set up image data
% Images reduced due to training time of neural network
% 1,000 total images
% 500 dog images
% 500 cat images

dataFolder = './CAP_4630_FINALPROJECT_SUMMER2020/train';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Pre-process Images For CNN

% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Divide data into training and validation sets
% 80-percent trainingSet and 20-percent validationSet
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomized');

%% Transfer Learning 

% To retrain a pretrained network to classify new images, we must replace these 
% last layers with new layers adapted to the new data set.

%% Freeze all but last three layers

layersTransfer = model.Layers(1:end-3);
numClasses = 2; % cat and dog

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Configure training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Retrain network

modelTransfer = trainNetwork(trainingSet,layers,options);

%% Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(modelTransfer,validationSet);

%% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy * 100);

%% Test it on unseen images
newImage1 = './CAP_4630_FINALPROJECT_SUMMER2020/dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImage(newImage1);
YPred1 = predict(modelTransfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './CAP_4630_FINALPROJECT_SUMMER2020/cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImage(newImage2);
YPred2 = predict(modelTransfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");

% What about the iconic "Doge"?
newImage3 = './CAP_4630_FINALPROJECT_SUMMER2020/doge.jpg';
img3 = readAndPreprocessImage(newImage3);
YPred3 = predict(modelTransfer,img3);
[confidence3,idx3] = max(YPred3);
label3 = categories{idx3};
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");

%% Data augmentation

% Data augmentation helps prevent the network from overfitting and
% memorizing the exact details of the training images.

%% Defining the imageAugmenter object 
% In our case, we shall use an augmented image datastore to randomly flip
% the training images along the vertical axis and randomly translate them
% up to 30 pixels and scale them up to 10% horizontally and vertically.

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

%% Building the augmented training and validation sets

inputSize = model.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

disp(augimdsTrain.NumObservations) % You should see 28

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);

disp(augimdsValidation.NumObservations) % You should see 12

%% Train the network with augmented datasets

miniBatchSize = 10;
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',3e-4, ...
    'ValidationData',augimdsValidation, ...
    'Verbose',false, ...
    'Plots','training-progress');

modelAug = trainNetwork(augimdsTrain,layers,options);

%% Classify the validation images using the fine-tuned network.

[YPredAug,probsAug] = classify(modelAug,augimdsValidation);

%% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidationAug = validationSet.Labels;
accuracyAug = mean(YPredAug == YValidationAug);
fprintf("The validation accuracy is: %.2f %%\n", accuracyAug * 100);

%% Test it on unseen images
newImage1 = './CAP_4630_FINALPROJECT_SUMMER2020/dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImage(newImage1);
YPred1 = predict(modelTransfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './CAP_4630_FINALPROJECT_SUMMER2020/cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImage(newImage2);
YPred2 = predict(modelTransfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");

% What about the iconic "Doge"?
newImage3 = './CAP_4630_FINALPROJECT_SUMMER2020/doge.jpg';
img3 = readAndPreprocessImage(newImage3);
YPred3 = predict(modelTransfer,img3);
[confidence3,idx3] = max(YPred3);
label3 = categories{idx3};
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");

%% IMPROVED CLASSIFIER 1
%% Loading a pre-trained "VGG-16"

model2 = vgg16;

%% Set up image data
% Images reduced due to training time of neural network
% 1,000 total images
% 500 dog images
% 500 cat images

dataFolder = './CAP_4630_FINALPROJECT_SUMMER2020/train';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Pre-process Images For CNN

% Set the ImageDatastore ReadFcn
% New readAndPreprocessImageVgg function to pre-process
% according to VGG-16 requirements 224-by-224
imds.ReadFcn = @(filename)readAndPreprocessImageVgg(filename);

%% Divide data into training and validation sets
% 80-percent trainingSet and 20-percent validationSet
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomized');

%% Transfer Learning 

% To retrain a pretrained network to classify new images, we must replace these 
% last layers with new layers adapted to the new data set.

%% Freeze all but last three layers

layersTransfer = model2.Layers(1:end-3);
numClasses = 2; % cat and dog

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Configure training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Retrain network

model2Transfer = trainNetwork(trainingSet,layers,options);

%% Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(model2Transfer,validationSet);

%% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy * 100);

%% Test it on unseen images
newImage1 = './CAP_4630_FINALPROJECT_SUMMER2020/dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImageVgg(newImage1);
YPred1 = predict(model2Transfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './CAP_4630_FINALPROJECT_SUMMER2020/cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImageVgg(newImage2);
YPred2 = predict(model2Transfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");

% What about the iconic "Doge"?
newImage3 = './CAP_4630_FINALPROJECT_SUMMER2020/doge.jpg';
img3 = readAndPreprocessImageVgg(newImage3);
YPred3 = predict(model2Transfer,img3);
[confidence3,idx3] = max(YPred3);
label3 = categories{idx3};
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");


%% IMPROVED CLASSIFIER 2
%  MiniBatch size 100
%  Optimizer adam

%% Loading a pre-trained "AlexNet"

model3 = alexnet;

%% Set up image data
% Images reduced due to training time of neural network
% 1,000 total images
% 500 dog images
% 500 cat images

dataFolder = './CAP_4630_FINALPROJECT_SUMMER2020/train';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

%% Pre-process Images For CNN

% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% Divide data into training and validation sets
% 80-percent trainingSet and 20-percent validationSet
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomized');

%% Transfer Learning 

% To retrain a pretrained network to classify new images, we must replace these 
% last layers with new layers adapted to the new data set.

%% Freeze all but last three layers

layersTransfer = model3.Layers(1:end-3);
numClasses = 2; % cat and dog

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Configure training options

options = trainingOptions('adam', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Retrain network

model3Transfer = trainNetwork(trainingSet,layers,options);

%% Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(model3Transfer,validationSet);

%% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy * 100);

%% Test it on unseen images
newImage1 = './CAP_4630_FINALPROJECT_SUMMER2020/dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImage(newImage1);
YPred1 = predict(model3Transfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './CAP_4630_FINALPROJECT_SUMMER2020/cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImage(newImage2);
YPred2 = predict(model3Transfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");

% What about the iconic "Doge"?
newImage3 = './CAP_4630_FINALPROJECT_SUMMER2020/doge.jpg';
img3 = readAndPreprocessImage(newImage3);
YPred3 = predict(model3Transfer,img3);
[confidence3,idx3] = max(YPred3);
label3 = categories{idx3};
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");


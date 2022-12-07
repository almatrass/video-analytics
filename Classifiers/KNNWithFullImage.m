% Load all positive and negative examples
imageStore=imageDatastore('Resources/images','IncludeSubfolders',true,'LabelSource','foldernames');

% Split the dataset randomly into training and testing sets, with 60% used
% for training and 40% for testing
[trainingSet, testingSet] = splitEachLabel(imageStore, 0.6, 'randomize');

% Prepare training images and labels
trainingImagesCells=readall(trainingSet);
numTrainingImages=size(trainingImagesCells,1);
trainingImages=zeros(numTrainingImages,1);
trainingLabels=zeros(numTrainingImages,1);

for index=1:numTrainingImages

    % Get label for current image
    currentLabel=char(trainingSet.Labels(index));
    if currentLabel=="non-face"
        trainingLabels(index)=0;
    else
        trainingLabels(index)=1;
    end

    currentImage=trainingImagesCells{index};

    % Perform preprocessing (if any)
    %currentImage=enhanceContrastALS(currentImage);
    currentImage=getImagePixelArray(currentImage);

    % Add image to matrix
    numPixels=size(currentImage,2);
    trainingImages(index,1:numPixels)=currentImage;
end

% Training
tic;
NNModel=NNtrainingFullImage(trainingImages, trainingLabels);
trainingTime = toc;

% Prepare testing images and labels
testingImagesCells=readall(testingSet);
numTestingImages=size(testingImagesCells,1);
testingImages=zeros(numTestingImages,1);
testingLabels=zeros(numTestingImages,1);
correctClassifications=0;

tic;
for index=1:numTestingImages

    % Get label for current image
    expectedLabel=char(testingSet.Labels(index));
    if expectedLabel=="non-face"
        expectedLabel=0;
    else
        expectedLabel=1;
    end

    currentImage=testingImagesCells{index};

    % Perform preprocessing (if any)
    %currentImage=enhanceContrastALS(currentImage);
    currentImage=getImagePixelArray(currentImage);

    % Test image and get predicted label
    actualLabel=KNNTestingFullImage(currentImage,NNModel,3);
    
    if expectedLabel==actualLabel
        correctClassifications=correctClassifications+1;
    end
end
testingTime=toc;

accuracy=correctClassifications/numTestingImages*100;


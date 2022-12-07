% Load all positive and negative examples
images=imageDatastore('Resources\images\','IncludeSubfolders',true,'LabelSource','foldernames');

% Define the feature extraction method to be used
extractorFunction = @getImagePixelArray;

% Split the dataset randomly into training and testing sets, with 75% used
% for training and 25% for testing
[trainingSet, testingSet] = splitEachLabel(images, 0.75, 'randomize');

% Define options to be used by SVM classifier (if any)
SVMOptions = templateSVM("KernelFunction", "linear");

% Training
tic;
bag = bagOfFeatures(trainingSet,"CustomExtractor", extractorFunction);
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag, "LearnerOptions", SVMOptions);
trainingTime = toc;

% Testing
tic;

% Run classifier on the testing set
confMatrix = evaluate(categoryClassifier, testingSet);

% Determine accuracy of classifier on testing set
testingAccuracy = mean(diag(confMatrix));

testingTime = toc;

save Detector\SVMFullImageDetectorModel categoryClassifier
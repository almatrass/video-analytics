% Load all positive and negative examples
images=imageDatastore('Resources/images','IncludeSubfolders',true,'LabelSource','foldernames');

% Define the feature extraction method to be used
extractorFunction = @extractMSERFeatures;

% Split the dataset randomly into training and testing sets, with 60% used
% for training and 40% for testing
[trainingSet, testingSet] = splitEachLabel(images, 0.6, 'randomize');

% Define options to be used by SVM classifier (if any)
%SVMOptions = templateSVM("KernelFunction", "linear");

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

save Detector\SVMMserDetectorModel categoryClassifier
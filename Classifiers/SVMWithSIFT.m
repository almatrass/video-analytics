% Load all positive and negative examples
images=imageDatastore('Resources/images','IncludeSubfolders',true,'LabelSource','foldernames');

% Define the feature extraction method to be used
extractorFunction = @extractSIFTFeatures;

frameCells=readall(images);
frameArray = zeros(486, length(frameCells));
for i = 1:length(frameCells)

frameArray(:, i) = reshape(frameCells{i}, 486, 1);
end

labels = images.Labels;

tbl = countEachLabel(images);
minimumSetCount = min(tbl{:,2});
ims = splitEachLabel(images,minimumSetCount, 'randomize');
indices = crossvalind('Kfold', labels , 3);
cp = classperf(string(labels));

for i = 1:3
    test = (indices == i); 
    train = ~test;
    class = classify(frameArray(test, :),frameArray(train, :), labels(train,:), 'diaglinear');
    classperf(cp,string(class),test);
end
cp.ErrorRate;

% Define options to be used by SVM classifier (if any)
SVMOptions = templateSVM("KernelFunction", "linear");

% Training
tic;
bag = bagOfFeatures(images.subset(train),"CustomExtractor", extractorFunction);
% categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
categoryClassifier = trainImageCategoryClassifier(images.subset(train), bag, "LearnerOptions", SVMOptions);
trainingTime = toc;

% Testing
tic;

% Run classifier on the testing set
confMatrix = evaluate(categoryClassifier, images.subset(test));

% Determine accuracy of classifier on testing set
testingAccuracy = mean(diag(confMatrix));

testingTime = toc;

% As this is the chosen method for the detector, the model is saved to be
% used by the detector
save Models\SVMSIFTDetectorModel categoryClassifier
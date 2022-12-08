% Load all positive and negative examples
images=imageDatastore('Resources/images','IncludeSubfolders',true,'LabelSource','foldernames');

% Define the feature extraction method to be used
extractorFunction = @extractMSERFeatures;

% Format data for cross validation
frameCells=readall(images);
frameArray = zeros(486, length(frameCells));
for i = 1:length(frameCells)

frameArray(:, i) = reshape(frameCells{i}, 486, 1);
end

labels = images.Labels;

% K fold cross validation, K=5
tbl = countEachLabel(images);
minimumSetCount = min(tbl{:,2});
ims = splitEachLabel(images,minimumSetCount, 'randomize');
indices = crossvalind('Kfold', labels , 5);
cp = classperf(string(labels));

for i = 1:5
    test = (indices == i); 
    train = ~test;
    class = classify(frameArray(test, :),frameArray(train, :), labels(train,:), 'diaglinear');
    classperf(cp,string(class),test);
end
cp.ErrorRate;

% Define options to be used by SVM classifier (if any)
%SVMOptions = templateSVM("KernelFunction", "linear");

% Training
tic;
bag = bagOfFeatures(images.subset(train),"CustomExtractor", extractorFunction);
categoryClassifier = trainImageCategoryClassifier(images.subset(train), bag, "LearnerOptions", SVMOptions);
trainingTime = toc;

% Testing
tic;

% Run classifier on the testing set
confMatrix = evaluate(categoryClassifier, images.subset(test));

% Determine accuracy of classifier on testing set
testingAccuracy = mean(diag(confMatrix));

testingTime = toc;

save Models\SVMMserDetectorModel categoryClassifier
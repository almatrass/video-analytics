function modelNN = NNtrainingHOG(images, labels)

numTrainingImages=size(images,1);

% Initialise matrix of features
modelFeatures=zeros(numTrainingImages);

% Add features from every training image
for index=1:numTrainingImages

    % Reshape images from 1D pixel array to correct dimensions
    currentImage=reshape(images(index,:),27,18);

    HOGFeatures=extractHOGFeatures(currentImage);

    % Add HOG features to model
    numFeatures=size(HOGFeatures,2);
    for featureIndex=1:numFeatures
        modelFeatures(index,featureIndex)=HOGFeatures(featureIndex);
    end
end

modelNN.neighbours=modelFeatures;
modelNN.labels=labels;

end
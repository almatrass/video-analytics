load Detector\SVMSIFTDetectorModel.mat

% Skip first two lines of test dataset to get data for frames
datasetOptions = delimitedTextImportOptions("DataLines",3);

% Load test dataset
testDataset = readtable("Resources\test.dataset",datasetOptions);
datasetCells=testDataset(:,:).Var1;

% Results of box detections
detectedFrameData=string([]);

% Load file paths from test dataset to be compared
testFrameFilePaths=string([]);
numTestFrames=10;
for testFrameNumber=1:numTestFrames
    frameData=string(datasetCells{testFrameNumber});
    parsedData=regexp(frameData, '[ ]+','split');
    testFrameFilePaths(testFrameNumber,1)=parsedData(1,1);
end

trainingImageHeight=160;
trainingImageWidth=96;

videoFrameHeight=480;
videoFrameWidth=640;

% Settings for detection
imageSampleHeights=[80,160,240,320,400];
imageSampleWidths=[48,96,144,192,240];
numSampleSizes=5;
stepIncrement=10;

for currentFrame=1:numTestFrames

    % Get frame filepath and add to results
    testImagePath=testFrameFilePaths(currentFrame,1);
    detectedFrameData(currentFrame,1)=testImagePath;
    testImagePath=strrep(testImagePath, 'images/', '\');

    % Load given frame and apply preprocessing
    testImage=imread("Resources\pedestrian" + testImagePath);
    testImage=im2gray(testImage);
    testImage=enhanceContrastALS(testImage);
    
    numBoxes=0;
    boxes=[];
    
    % Go over the frame once for each sample size
    for sampleNum=1:numSampleSizes
        currentSampleWidth=imageSampleWidths(sampleNum);
        currentSampleHeight=imageSampleHeights(sampleNum);

        % Move the sliding window across and down the image, moving the
        % given increment of pixels each time
        for xPos=1:stepIncrement:videoFrameWidth-currentSampleWidth
            for yPos=1:stepIncrement:videoFrameHeight-currentSampleHeight

                % Check to ensure sliding window does not scan beyond the
                % frame
                if (xPos < videoFrameWidth-currentSampleWidth) && (yPos < videoFrameHeight-currentSampleHeight)

                    % Get the current section of the frame to examine
                    currentSample=testImage(yPos:yPos+currentSampleHeight-1,xPos:xPos+currentSampleWidth-1);

                    % Resize sample to same size as training images
                    currentSample=imresize(currentSample,[trainingImageHeight,trainingImageWidth]);

                    % Predict the class of the sample
                    [labelIndex, score] = categoryClassifier.predict(currentSample);
                    label=categoryClassifier.Labels(labelIndex);

                    % Confidence is calculated based on combination of how
                    % likely the sample is to be a pedestrian (score(2))
                    % and how likely the sample is to not be a pedestrian
                    % (score(1))
                    sampleConfidence = score(2)-score(1);

                    % If person detected, add a box
                    if label=="pos"
                        numBoxes=numBoxes+1;
                        boxes(numBoxes,1:5)=[xPos,yPos,currentSampleWidth,currentSampleHeight,sampleConfidence];
                    end
                end      
            end
        end
    end
    
    % Non-Maxima Suppression
    boxes=NonMaximaSuppression(boxes, 0.25);
    numBoxes=size(boxes,1);

    % Store results in detectedFrameData
    detectedFrameData(currentFrame,2)=numBoxes;
    nextIndex=3;
    for i=1:numBoxes
        for j=1:5
            detectedFrameData(currentFrame, nextIndex)=boxes(i,j);
            nextIndex=nextIndex+1;
        end
    end

end

% Save results of detection
save Detector\testFrameBoxes detectedFrameData
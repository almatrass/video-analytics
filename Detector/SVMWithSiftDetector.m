% Load chosen model
load Models\SVMSIFTDetectorModel.mat

trainingImageHeight=27;
trainingImageWidth=18;

% Settings for detection
imageSampleHeights=[27, 18, 20];
imageSampleWidths=[18, 27, 20];
numSampleSizes=3;
stepIncrement=10;

% Load video frames
frameDatastore=imageDatastore("Resources/tests");
frameCells=readall(frameDatastore);
numFrames=size(frameCells,1);
i = 1;
for currentFrame=1:numFrames

    % Apply same preprocessing as was used in training
    testImage=im2gray(frameCells{currentFrame});
    testImage=enhanceContrastALS(testImage);
    
    imageFrame = size(frameCells{currentFrame});
    imageFrameHeight= imageFrame(1);
    imageFrameWidth= imageFrame(2);

    numBoxes=0;
    boxes=[];
    % Go over the frame once for each sample size
    for sampleNum=1:numSampleSizes
        currentSampleWidth=imageSampleWidths(sampleNum);
        currentSampleHeight=imageSampleHeights(sampleNum);

        % Move the sliding window across and down the image, moving the
        % given increment of pixels each time
        for xPos=1:stepIncrement:imageFrameWidth-currentSampleWidth
            for yPos=1:stepIncrement:imageFrameHeight-currentSampleHeight

                % Check to ensure sliding window does not scan beyond the
                % frame
                if (xPos < imageFrameWidth-currentSampleWidth) && (yPos < imageFrameHeight-currentSampleHeight)
                    
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
                    if label=="face"
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
    
    % Display the frame and draw boxes
    figure
    imshow(testImage), hold on;
    for index=1:numBoxes
        x1=boxes(index,1);
        x2=boxes(index,1)+boxes(index,3);
        y1=boxes(index,2);
        y2=boxes(index,2)+boxes(index,4);
        plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1],'b');
    end    
    
    boxedFigure=getframe(gca);
    imwrite(boxedFigure.cdata, strcat("Resources/boxedImages/boxedTestImage", string(i), ".png"));
    i = i+1;

    % Clear figure before next frame
    close all

end



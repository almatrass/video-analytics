% Skip first two lines of test dataset to get data for frames
datasetOptions = delimitedTextImportOptions("DataLines",3);

% Load test dataset
testDataset = readtable("Resources\test.dataset",datasetOptions);
datasetCells=testDataset(:,:).Var1;

% Create and open video writer
videoObject = VideoWriter("Test Dataset Boxes","MPEG-4");
open(videoObject);

numTestFrames=size(datasetCells,1);
for testFrameNumber=1:numTestFrames
    frameData=string(datasetCells{testFrameNumber});
    parsedData=regexp(frameData, '[ ]+','split');
    testImagePath=parsedData(1,1);
    testImagePath=strrep(testImagePath, 'images/', '\');

    % Load given frame and apply preprocessing
    testImage=imread("Resources\pedestrian" + testImagePath);
    testImage=im2gray(testImage);
    testImage=enhanceContrastALS(testImage);

    numBoxes=str2num(parsedData(1,2));
    startingX=3;
    startingY=4;
    startingWidth=5;
    startingHeight=6;

    boxes=zeros(numBoxes,5);
    for index=1:numBoxes
        boxes(index,1)=parsedData(1, startingX+((index-1)*5));
        boxes(index,2)=parsedData(1, startingY+((index-1)*5));
        boxes(index,3)=parsedData(1, startingWidth+((index-1)*5));
        boxes(index,4)=parsedData(1, startingHeight+((index-1)*5));
    end

    % Display the frame and draw boxes
    figure
    imshow(testImage), hold on;
    for index=1:numBoxes
        x1=boxes(index,1)-(boxes(index,3)/2);
        x2=boxes(index,1)+boxes(index,3)-(boxes(index,3)/2);
        y1=boxes(index,2)-(boxes(index,4)/2);
        y2=boxes(index,2)+boxes(index,4)-(boxes(index,4)/2);
        plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1],'g');
    end
    
    % Save boxed frame and write to video
    boxedFigure=getframe(gca);
    writeVideo(videoObject, boxedFigure);
    
    % Clear figure before next frame
    close all


end

% Close the video writer to complete the video
close(videoObject);

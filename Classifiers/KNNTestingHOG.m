function prediction = KNNTestingHOG(testImage, modelNN, k)

% Reshape image from 1D pixel array to correct dimensions
testImage=reshape(testImage,27,18);

HOGFeatures=extractHOGFeatures(testImage);

% Initialise a matrix that will store the nearest neighbour index, the
% distance to the neighbour, and that neighbour's label.
% The matrix is initialised with the first k possible neighbours.
nearestNeighbours=zeros(k,3);
for index=1:k
    nearestNeighbours(index,1)=index;
    nearestNeighbours(index,2)=EuclideanDistance(HOGFeatures, modelNN.neighbours(index,:));
    nearestNeighbours(index,3)=modelNN.labels(index);
end

% The neighbour matrix is sorted into distance from testImage, so the last
% neighbour will always be replaced if a closer neighbour is found.
nearestNeighbours=sortrows(nearestNeighbours,2);

% Total number of neighbours to compare
numNeighbours=size(modelNN.neighbours);

% Check every neighbour
for index=k+1:numNeighbours

    % Distance from test image to current neighbour
    currentDistance=EuclideanDistance(HOGFeatures, modelNN.neighbours(index,:));

    % If current neighbour is closer than the last nearest neighbour,
    % replace the last nearest neighbour and sort the matrix.
    if currentDistance<nearestNeighbours(k,2)
        nearestNeighbours(k,2)=currentDistance;
        nearestNeighbours(k,1)=index;
        nearestNeighbours(k,3)=modelNN.labels(index);
        nearestNeighbours=sortrows(nearestNeighbours,2);
    end
end

% Count labels of nearest neighbours
zeroVotes=0;
oneVotes=0;
for index=1:k
    if nearestNeighbours(index,3)==0
        zeroVotes=zeroVotes+1;
    else
        oneVotes=oneVotes+1;
    end
end

% Prediction is the most common label among nearest neighbours.
if zeroVotes>oneVotes
    prediction=0;
else
    prediction=1;
end

end


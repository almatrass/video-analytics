function Boxes = NonMaximaSuppression(Boxes,threshold)

numBoxes=size(Boxes,1);

for i=1:numBoxes
    currentBoxScore=Boxes(i,5);
    boxArea=Boxes(i,3)*Boxes(i,4);
    for j=1:numBoxes

        % Do not compare boxes with themselves
        if i~=j
            otherBoxScore=Boxes(j,5);
            intersectionArea=rectint(Boxes(i,1:4), Boxes(j,1:4));

            % If boxes intersect too much, remove the box with lower score
            if (intersectionArea / boxArea) > threshold
                if currentBoxScore < otherBoxScore
                    Boxes(i,1:5) = zeros(1,5);
                else
                    Boxes(j,1:5) = zeros(1,5);
                end
            end
        end
    end

end

% Remove lines in matrix set to 0 for boxes that were removed
Boxes = Boxes(~all(Boxes == 0, 2),:);

end


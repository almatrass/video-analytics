function [features, featureMetrics, location] = extractSURFFeatures(image)

% Images are converted to grayscale
image=im2gray(image);

% Perform preprocessing (if any)
% image=enhanceContrastALS(image);

% Extract SURF features
points = detectSURFFeatures(image);

% Return the data needed by classifier
features = extractFeatures(image, points, "Method", "SURF");
featureMetrics = points.Metric;
location = points.Location;

end


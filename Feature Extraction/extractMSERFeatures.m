function [features, featureMetrics, location] = extractMSERFeatures(image)

% Images are converted to grayscale
image=im2gray(image);

% Perform preprocessing (if any)
% image=enhanceContrastALS(image);

% Extract MSER features
regions = detectMSERFeatures(image);

% Return the data needed by classifier
[features, points] = extractFeatures(image, regions);
featureMetrics = points.Metric;
location = points.Location;

end


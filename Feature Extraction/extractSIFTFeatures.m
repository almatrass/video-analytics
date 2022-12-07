function [features, featureMetrics, location] = extractSIFTFeatures(image)

% Images are converted to grayscale
image=im2gray(image);

% Perform preprocessing (if any)
%image=enhanceContrastALS(image);

% Extract SIFT features
regions = detectSIFTFeatures(image);

% Return the data needed by classifier
[features, points] = extractFeatures(image, regions, "Method", "SIFT");
featureMetrics = points.Metric;
location = points.Location;

end


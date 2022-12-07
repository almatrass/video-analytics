function [features, featureMetrics, location] = extractBRISKFeatures(image)

% Images are converted to grayscale
image=im2gray(image);
disp(image);
% Perform preprecessing (if any)
%image=enhanceContrastALS(image);

% Extract SIFT features
regions = detectBRISKFeatures(image);

% Return the data needed by classifier
[features, points] = extractFeatures(image, regions);
featureMetrics = points.Metric;
location = points.Location;

end


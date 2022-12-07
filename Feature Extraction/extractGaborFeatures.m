function [features, featureMetrics] = extractGaborFeatures(image)

% Perform preprocessing (if any)
%image=enhanceContrastALS(image);

features=gabor_feature_vector(image);
featureMetrics = zeros(size(features(1),1));

end


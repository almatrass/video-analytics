function [features, featureMetrics] = extractHOGFeatures(image)

% Perform preprocessing (if any)
%image=enhanceContrastALS(image);

features=hog_feature_vector(image);
featureMetrics = zeros(size(features(1),1));

end


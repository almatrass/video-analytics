function [features, featureMetrics] = getImagePixelArray(image)

% This function converts a given image into a row vector
% of the pixel values of the image as doubles

image=im2gray(image);

% Perform preprocessing (if any)
%image=enhanceContrastALS(image);

% Convert matrix into a row vector
image=reshape(image, 1, []);

numPixels=size(image,2);
features=zeros(1,numPixels);

% Feature metrics is needed for the MATLAB implementation of SVM (category
% classifier). They represent how strong the given metric is. For full
% image, all will have the same value.
featureMetrics=(1);

% Convert the original image pixels to double and add them to the new
% vector
for i=1:numPixels
    features(i)=double(image(i));
    features(i)=features(i)/255.0;
end

end


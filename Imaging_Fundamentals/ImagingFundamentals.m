%% Alexander Faust
% ECE 435 - Imaging Fundamentals Homework 1
%
%   September 19, 2023
%
clc; clear; close all;

%% 1 - Slicing

% List number of JPEG images to store in 3-D array
num_images = 237;                       

% Preallocate 3-D Matrix of grayscale values:
stack = zeros(512, 512, num_images, 'uint8');

% Parse through images and convert rgb to grayscale values accordingly
for i = 1:num_images
    if i < 10
        imdata = imread("C:\Cooper Union\Junior Year\ECE 435 - Medical Imaging\Homeworks\HW 1 - Imaging Fundamentals\thoraxCT\axial0000" + i + ".jpg");
        grayscale = im2gray(imdata);
        stack(:, :, i) = grayscale;
    else
        if i < 100
            imdata = imread("C:\Cooper Union\Junior Year\ECE 435 - Medical Imaging\Homeworks\HW 1 - Imaging Fundamentals\thoraxCT\axial000" + i + ".jpg");
            grayscale = im2gray(imdata);
            stack(:, :, i) = grayscale;
        else
            imdata = imread("C:\Cooper Union\Junior Year\ECE 435 - Medical Imaging\Homeworks\HW 1 - Imaging Fundamentals\thoraxCT\axial00" + i + ".jpg");
            grayscale = im2gray(imdata);
            stack(:, :, i) = grayscale;
        end
    end

end
% Obtain 3 slices of the image stack:
slice1 = squeeze(stack(200, :, :));
slice2 = squeeze(stack(:, 200, :));          
slice3 = squeeze(stack(:, :, 120));          

figure;
hold on;
subplot(1,3,1);
imshow(slice1);
title("Lateral slice at x = 200");


subplot(1,3,2);
imshow(slice2);
title("Lateral slice at y = 200");

subplot(1,3,3);
imshow(slice3);
title("Axial slice at z = 120");
hold off;

%% 2 - Salt & Pepper
% Use slice 3 and obtain MEDIAN filtered image in 3x3 neighborhood:
median_filter_slice3x3 = medfilt2(slice3, [3,3]);
median_filter_slice5x5 = medfilt2(slice3, [5,5]);

figure;
hold on;
subplot(2,2,1);
imshow(median_filter_slice3x3);
title("z = 120 slice MEDIAN filtered 3x3");

subplot(2,2,2);
imshow(median_filter_slice5x5);
title("z = 120 slice MEDIAN filtered 5x5");

% Use slice 3 and obtain MEAN filtered image in 3x3 and 5x5 neighborhood:
% Create window to filter image with, should have equal weights
window3x3 = ones(3,3) / 9;
window5x5 = ones(5,5) / 25;

mean_filter_slice3x3 = imfilter(slice3, window3x3);
mean_filter_slice5x5 = imfilter(slice3, window5x5);

subplot(2,2,3);
imshow(mean_filter_slice3x3);
title("z = 120 slice MEAN filtered 3x3");

subplot(2,2,4);
imshow(mean_filter_slice5x5);
title("z = 120 slice MEAN filtered 5x5");
hold off;

% Corrupt slice3 with salt and pepper noise and do the appropriate mean and
% median filtering for 3x3 neighborhoods:

% List probabilities for replacement of each pixel for salt and pepper
% noise
P = [0.005 0.015 0.05]; 

% Obtain added salt and pepper noise to the same slice considered above
% (i.e. slice3):
SP1 = saltandpepper(slice3, P(1));
SP2 = saltandpepper(slice3, P(2));
SP3 = saltandpepper(slice3, P(3));

% Perform mean and median filtering in 3x3 neighborhood:
SP1_medfilt = medfilt2(SP1);
SP2_medfilt = medfilt2(SP2);
SP3_medfilt = medfilt2(SP3);

SP1_meanfilt = imfilter(SP1, window3x3);
SP2_meanfilt = imfilter(SP2, window3x3);
SP3_meanfilt = imfilter(SP3, window3x3);

% Display images of salt and pepper MEDIAN filtered:
figure;
hold on;
subplot(3, 1, 1);
imshow(SP1_medfilt, []);
title("z = 120 slice | S & P MEDIAN filtered with prob = " + P(1));

subplot(3, 1, 2);
imshow(SP2_medfilt, []);
title("z = 120 slice | S & P MEDIAN filtered with prob = " + P(2));

subplot(3, 1, 3);
imshow(SP3_medfilt, []);
title("z = 120 slice | S & P MEDIAN filtered with prob = " + P(3));
hold off;

% Display images of salt and pepper MEAN filtered:
figure;
hold on;
subplot(3, 1, 1);
imshow(SP1_meanfilt, []);
title("z = 120 slice | S & P MEAN filtered with prob = " + P(1));

subplot(3, 1, 2);
imshow(SP2_meanfilt, []);
title("z = 120 slice | S & P MEAN filtered with prob = " + P(2));

subplot(3, 1, 3);
imshow(SP3_meanfilt, []);
title("z = 120 slice | S & P MEAN filtered with prob = " + P(3));
hold off;

%% 3 - Gauss' Wrath
sigma = 0.3;

Gauss1 = double(slice3) + 20.*randn(size(slice3));
Gauss2 = double(slice3) + 60.*randn(size(slice3));

awgnIm1 = imgaussfilt(Gauss1, sigma);
awgnIm2 = imgaussfilt(Gauss2, sigma);

figure;
hold on;
subplot(1,3,1);
imshow(slice3);
title("Originalslice z = 120");
subplot(1,3,2);
imshow(Gauss1./255);
title("AWGN with \sigma 20");
subplot(1,3,3);
imshow(awgnIm1./255);
title("Gaussian filter \sigma = 0.3");
hold off;

figure;
hold on;
subplot(1,3,1);
imshow(slice3);
title("Original slice z = 120");
subplot(1,3,2);
imshow(Gauss2./255);
title("AWGN with \sigma = 60");
subplot(1,3,3);
imshow(awgnIm2./255);
title("Gaussian filter \sigma^2 = 0.3");
hold off;

%% Gaussian Blurring:
% Create the 3D Gaussian PSF:
% List full width half mean values in each direction [mm]:
FWHM_x = 2;
FWHM_y = 2;
FWHM_z = 1;

% Calculate sigma from these values:
sigma_x = FWHM_x / (2*sqrt(2*log(2)));
sigma_y = FWHM_y / (2*sqrt(2*log(2)));
sigma_z = FWHM_z / (2*sqrt(2*log(2)));

% Amplitude required to normalize the Gaussian volume:
A = 1/(((2*pi)^3/2) * sigma_x * sigma_y * sigma_z);

% Create meshgrid to evaluate Gaussian volume over space:
[x, y, z] = meshgrid(-10:0.1:10, -10:0.1:10, -10:0.1:10);
Gauss_PSF = A* exp(-(x.^2 / (2*sigma_x^2) + y.^2 / (2*sigma_y^2) + z.^2 / (2*sigma_z^2)));

% Normalize the Gaussian PSF:
Gauss_PSF = Gauss_PSF / sum(Gauss_PSF(:));
% Convolve Gaussian with image to blur:
blur_slice1 = convn(slice1, Gauss_PSF, 'same');
blur_slice2 = convn(slice2, Gauss_PSF, 'same');
blur_slice3 = convn(slice3, Gauss_PSF, 'same');

figure;
hold on;
subplot(1,2,1);
imshow(slice1);
title("Original image (x = 200)");
subplot(1,2,2);
imshow(blur_slice1, []);
title("Blurred slice caused by PSF");
hold off;

figure;
hold on;
subplot(1,2,1);
imshow(slice2);
title("Original image (y = 200)");
subplot(1,2,2);
imshow(blur_slice2, []);
title("Blurred slice caused by PSF");
hold off;

figure;
hold on;
subplot(1,2,1);
imshow(slice3);
title("Original image (z = 120)");
subplot(1,2,2);
imshow(blur_slice3, []);
title("Blurred slice caused by PSF");
hold off;

%% 4 - The Plural of Floramen is Floramina
% Examine slice y = 180:
slice_abdomen = squeeze(stack(:, 180, :));
figure;
imshow(slice_abdomen, []);
title("Abdomen Slice Unperturbed")
colormap("gray");

% a) Image filtered using 3x3 median filter:
abdomen_3x3median = medfilt2(slice_abdomen);
% b) Image distorted using Fancy 2D Gaussian filter:
abdomen_gauss_blur = convn(slice_abdomen, Gauss_PSF, 'same');
% c) Image distorted by Gauss noise:
abdomen_awgn = double(slice_abdomen) + 1.*randn(size(slice_abdomen));
% d) Image filtered by Gaussian filter with std deviation 3 pixels in each direction:
abdomen_3px = imgaussfilt(slice_abdomen, 3);

% Apply Canny edge detector to each processed image:
edge1 = edge(abdomen_3x3median, 'canny');
edge2 = edge(abdomen_gauss_blur, 'canny');
edge3 = edge(abdomen_awgn, 'canny');
edge4 = edge(abdomen_3px, 'canny');

% Create subplot of each image (original
figure;
subplot(2,2,1);
imshow(abdomen_3x3median, []);
title("Abdomen median Filtered");
subplot(2,2,2);
imshow(abdomen_gauss_blur, []);
title("Abdomen 2D Gaussian filtered");
subplot(2,2,3);
imshow(abdomen_awgn, []);
title("Abdomen Distorted with AWGN");
subplot(2,2,4);
imshow(abdomen_3px, []);
title("Abdomen 3pixel AWGN");

% Create subplot of Canny edge detector on previous images:
figure;
subplot(2,2,1);
imshow(edge1);
title("Median Filtered");
subplot(2,2,2);
imshow(edge2);
title("2D Gaussian filter");
subplot(2,2,3);
imshow(edge3);
title("AWGN Distortion");
subplot(2,2,4);
imshow(edge4);
title("3 Pixels Each Direction");

% Extra plot of axial, z slice:
figure;
imshow(edge(stack(:, :, 140), 'canny'));


%% Functions Created
% Function that takes in an image and determines whether or not to add salt
% & pepper noise to each pixel based on a probability p
function outputImage = saltandpepper(image, p)
    % Function takes input p - as the probability of choosing black or white pixel 
    q = 1 - 2*p;        % Probability of leaving pixel alone
    % Determine number of rows and columns of input image:
    [rows, columns] = size(image);

    % Preallocate output image:
    outputImage = zeros(rows, columns);
    for i = 1:rows
        for j = 1:columns
            pixel = randsample([0 255 image(i,j)], 1, true, [p p q]);
            % Replace pixel in image with the randomly chosen one:
            outputImage(i,j) = pixel;
        end
    end
end


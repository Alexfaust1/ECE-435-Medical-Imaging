%% Alexander Faust
%
% ECE 435 - CS/MRI Final Project
%
% December 5, 2023
clear; clc; close all;

%% Part 1 - Navigating the volumes

% Read the .nii files:
CSI1_T1_data = niftiread("MRI Images\sub-CSI1_ses-16_run-01_T1w.nii.gz");
CSI1_T2_data = niftiread("MRI Images\sub-CSI1_ses-16_T2w.nii.gz");

% Begin with plotting the Coronal slices along the constant y-direction:
figure;
subplot(2,3,1);
imagesc(squeeze(CSI1_T1_data(:,128,:)));
colormap("gray");
axis("square");
title("Coronal Slice 128, T1-weighted");

subplot(2,3,4);
imagesc(squeeze(CSI1_T2_data(:,128,:)));
colormap("gray");
axis("square");
title("Coronal Slice 128, T2-weighted");

% Next plot the Sagittal slices along constant x-direction:
subplot(2,3,2);
imagesc(squeeze(CSI1_T1_data(88,:,:)));
colormap("gray");
axis("square");
title("Sagittal Slice 88, T1-weighted");

subplot(2,3,5);
imagesc(squeeze(CSI1_T2_data(88,:,:)));
colormap("gray");
axis("square");
title("Sagittal Slice 88, T2-weighted");


% Lastly, plot Axial slices along constant z-direction:
subplot(2,3,3);
imagesc(squeeze(CSI1_T1_data(:,:,128)));
colormap("gray");
axis("square");
title("Axial Slice 128, T1-weighted");

subplot(2,3,6);
imagesc(squeeze(CSI1_T2_data(:,:,128)));
colormap("gray");
axis("square");
title("Axial Slice 128, T2-weighted");


%% Part 3 - Fourier Domain Compression
% In this section we will choose a guiding image to perform experiments on.
% The idea is to see the effects of compression on the data set images:

% As suggested in the project I will choose the x/2 image:
sample_image = cast(squeeze(CSI1_T1_data(88,:,:)),"double") + 32768;

% Take the 2D fourier transform and FFT shift:
sample_image_fft = fftshift(fft2(sample_image));

% Plot the guiding iamge
figure;
imagesc(sample_image);
title("Guiding Image");
colormap("gray");
axis("square");

% Plot 2D FFT spectrum in dB:
figure;
imagesc(20*log10(abs(sample_image_fft)));
title("2D Magnitude Spectrum [dB] of the Guiding Image fftshifted");
colormap("gray");
axis("square");
colorbar;

% Plot histogram with N = 400 bins:
figure;
histogram(abs(sample_image_fft(:)),400);
title("Histogram in F_{FT}(\Omega) domain");
xlabel("|F_{FT}(\Omega)|");
ylabel("Number of Occurrences");


% Plot the guiding image for different s percentages using the function
% created:
s_val = [30, 60, 90, 99];
plotS(sample_image, s_val(1));
plotS(sample_image, s_val(2));
plotS(sample_image, s_val(3));
plotS(sample_image, s_val(4));

%% Part 4 - Wavelet Domains

% Examine the experiments performed on the camera man image:
sample_image = cast(squeeze(CSI1_T1_data(88,:,:)),"double");
camera_man = imread("cameraman.tif");

% Compute and plot the Wavelet domain of the cameraman image:
waveletTransformDisplay(camera_man,"haar");
waveletTransformDisplay(camera_man,"db4");
waveletTransformDisplay(camera_man,"coif3");

% Construct the standard basis to use in magnitude spectrum plots for each
% wavelet basis:
standard_basis = zeros(256,256);
standard_basis(128,128) = 1;

% Wavelet transform using the function created:
plotStandardBasis(standard_basis, "haar")
plotStandardBasis(standard_basis, "db4")
plotStandardBasis(standard_basis, "coif3")

% Plot the wavelet domain information
waveletTransformDisplay(sample_image, "haar");
waveletTransformDisplay(sample_image, "db4");
waveletTransformDisplay(sample_image, "coif3");

% Plot histograms to determine sparsity - if at all
plotHist(sample_image, "haar");
plotHist(sample_image, "db4");
plotHist(sample_image, "coif3");

% Reconstruct the guiding image using the optimal wavelet basis determined
% from inspection of the detailed coefficient plots: (in this case I liked
% Haar)
plotCompressedImage(sample_image, "haar", 3, 10);

% Comments: The Haar wavelet gave the best clinical results in my opinion.
% Coiflet 3 wavelets had better sparsity (since more occurrances near 0)
% but I chose this as my approach, from a practical perspective.

%% Part 5 - Sparsity Checks

% Specify S range to plot over for the upcoming sparsity checks:
s = 1:99;

% Compute the MSE curve for each s: (kind of like a learning curve in terms
% of us being able to determine optimal s%:

figure;
hold on
semilogy(s, MSE_Reconstruction(sample_image, "haar", 2, s));
semilogy(s, MSE_Reconstruction(sample_image, "db4", 2, s));
semilogy(s, MSE_Reconstruction(sample_image, "coif3", 2, s));
semilogy(s, MSE_Reconstruction(sample_image, "db4", 3, s));
semilogy(s, MSE_Reconstruction(sample_image, "coif3", 3, s));
semilogy(s, MSE_Reconstruction(sample_image, "db4", 3, s));
hold off
xlabel("Percentage of coefficients thrown away - s%");
ylabel("Mean Squared Error");
legend(["2-Level Haar","2-Level Daubuchies 4","2-Level Coiflet 3", ...
        "3-Level Haar","3-Level Daubuchies 4","3-Level Coiflet 3"]);

% Plot the wavelet decompositions:
plotCompressedImage(sample_image, "haar", 2, s_val);
plotCompressedImage(sample_image, "db4", 2, s_val);
plotCompressedImage(sample_image, "coif3", 2, s_val);
plotCompressedImage(sample_image, "haar", 3, s_val);
plotCompressedImage(sample_image, "db4", 3, s_val);
plotCompressedImage(sample_image, "coif3", 3, s_val);

% Now perform tests on different images:
plotCompressedImage(cast(camera_man, "double"), "coif3", 2, s_val);
plotCompressedImage(cast(squeeze(CSI1_T2_data(88, :, :)), "double"), "coif3", 2, s_val);

% Finally, we want to test how many coefficients we can throw away while
% assuming we can accept 10% error:

% Choosing s = 93.822 produces a 10% error for the 2 level Coiflet 3 basis
NMSE = NMSE_Reconstruction(sample_image, "coif3", 2, 93.822);

NMSE_range = zeros(1,size(CSI1_T1_data,1));
for c = 1:size(CSI1_T1_data, 1)
    NMSE_range(c) = NMSE_Reconstruction(cast(squeeze(CSI1_T1_data(c, :, :)), "double"), "coif3", 2, 93.822);
end

maxNMSE = max(NMSE_range);

%% Part 6  - Compressed Sensing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TESTBENCH
p = 0;                % Probability of underasmpling
h = 1.4;                  % Step size
lambda = 25;        
iterations = 5000;      % Total iterations of the algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert image data to double before processing:
ImageToCompress = cast(squeeze(CSI1_T1_data(88, :, :)), "double");

% Pass the image to compress to the compressed sensing function:
compressedSensing(ImageToCompress, p, h, lambda, iterations/2, "coif3", 2, true);


% Obtain from sparsity using Fourier (did not end up working properly):
ImageToCompressFreq = cast(squeeze(CSI1_T1_data(88, :, :)), "double");
plotHist(abs(fftshift(fft2(ImageToCompressFreq))), "coif3");

% Should prove sparsity in frequency the above, so now push to algorithm:
[objF, zinf] = compressedSensingFourier(ImageToCompressFreq, p, h, lambda, iterations/2, "coif3", 2, true);



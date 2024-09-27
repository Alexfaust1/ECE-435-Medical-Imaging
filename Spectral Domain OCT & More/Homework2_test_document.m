% Alexander Faust
%
% Test Document
% Homework 2
%
clc; clear; close all;
warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
% List System parameters:
clc; clear; close all;

lamb_0 = 1310 * 1e-9;                   % Center wavelength [m]
delta_lamd = 100 * 1e-9;                % Bandwidth [m]
num_samples = 2048;
fs = 97656.25;                          % Sample frequency
                                        % Numerical aperture of lens: range of angles
NA = 0.055;                             % over which system can accept light
% List 'n' to imply data taken in air environment
n = 1;
                                        
dz = 2.7*10^(-6);                       % in [m]
%% Part I
% Compute lateral resolution: describes how well the beam can be focused

res_lateral = 0.37 * lamb_0 / NA;       % in [m]

% Compute axial resolution:
res_axial = 2 * log(2) * lamb_0^2 / (pi * delta_lamd);    % in [m]


% Compute B-scan aspect ratio without accounting for mirror effect:

width_b = 1*10^-3;          % in [m]
height_Bscan = num_samples * dz;

aspect_ratio_with_mirror = width_b / height_Bscan;
% Compute B-scan aspect ratio with accounting mirror image effect:
aspect_ratio_no_mirror = width_b / (height_Bscan/2);


%% Data Acquisition
% Retrieve A scan data from BScan_Layers.raw file:
FID_B = fopen("BScan_Layers.raw");
data_B = fread(FID_B, 'uint16');

% Retrieve A scan data from Mscan1.raw file:
FID_M1 = fopen("Mscan1.raw");
data_M1 = fread(FID_M1, 'uint16');

% Retrieve A scan data from Mscan1.raw file:
FID_M40 = fopen("Mscan40.raw");
data_M40 = fread(FID_M40, 'uint16');

% Load the lambda to k-domain information matrix:
load("L2K.mat");

% Pack the data into vectors of length 2048 corresponding to each pixel's
% corresponding wavelength:
num_background_Bscan = 175;
num_background_Mscan1 = 320;
num_background_Mscan40 = 320;

data_B_matrix = reshape(data_B, [num_samples, length(data_B) / num_samples]);
background_data_B_matrix = data_B_matrix(:, 1 : num_background_Bscan);
A_scan_matrix = data_B_matrix(:, num_background_Bscan + 1 : end);

data_M1_matrix = reshape(data_M1, [num_samples, length(data_M1) / num_samples]);
background_data_M1_matrix = data_M1_matrix(:, 1 : num_background_Mscan1);
M1_scan_matrix = data_M1_matrix(:, num_background_Mscan1 + 1 : end);

data_M40_matrix = reshape(data_M40, [num_samples, length(data_M40) / num_samples]);
background_data_M40_matrix = data_M40_matrix(:, 1 : num_background_Mscan40);
M40_scan_matrix = data_M40_matrix(:, num_background_Mscan40 + 1 : end);

%% Generate complex A-scans in the spatial domain using A scan function:

[Ascan, L, A_wo_deconvAndSub, A_wo_deconv] = generateAscan(num_samples, background_data_B_matrix, 2048, ...
                              A_scan_matrix, L2K, 'hann');
[M1scan, L_M1, M1_wo_deconvAndSub, M1_wo_deconv] = generateAscan(num_samples, background_data_M1_matrix, 2048, ...
                              M1_scan_matrix, L2K, 'hamming');
[M40scan, L_M40, M40_wo_deconvAndSub, M40_wo_deconv] = generateAscan(num_samples, background_data_M40_matrix, 2048, ...
                              M40_scan_matrix, L2K, 'hamming');

%% Plot 2 A-scans, and one M-scan from each raw file in Magnitude dB:
figure;
plot(L, 20*log10(abs(Ascan(:, 5000))));
xlim([L(1) L(end)]);
title("Magnitude Spectrum in dB (BScan Layers)");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");

figure;
plot(L, 20*log10(abs(Ascan(:, 4000))));
xlim([L(1) L(end)]);
title("Magnitude Spectrum in dB (BScan Layers)");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");

figure;
plot(L_M1, 20*log10(abs(M1scan(:, 50000))));
xlim([L_M1(1) L_M1(end)]);
title("Magnitude Spectrum in dB MScan1");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");

figure;
plot(L_M40, 20*log10(abs(M40scan(:, 50000))));
xlim([L_M40(1) L_M40(end)]);
title("Magnitude Spectrum in dB Mscan40");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");

% Plot A-scan image at index 5000 without deconvolution and without
% background subtraction; then one without deconvolution but with
% background subtraction

figure;
plot(L, 20*log10(abs(A_wo_deconvAndSub(:, 5000))));
xlim([L(1) L(end)]);
title("Magnitude Spectrum in dB - A scan w/o deconv & sub");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");
% NOTE FOR REPORT: can actually see spikes for this but they arent like
% filtered out
figure;
plot(L, 20*log10(abs(A_wo_deconv(:, 5000))));
xlim([L(1) L(end)]);
title("Magnitude Spectrum in dB - A scan w/o deconv but with sub");
xlabel("fftshift Index");
ylabel("Magnitude [dB]");


%% Part 3

% Image of A-Scan without additional processing:
figure;
imagesc(20*log10(abs(Ascan(1:1024, :))));
colormap(gray);
colorbar;
pbaspect([width_b (height_Bscan/2) 1]);

% Image of A-Scan with contrast enhancement and some cropping:
enhanceImage(Ascan, width_b, height_Bscan);

% Display image of A-scan without deconvolution but with background
% subtraction 
figure;
imagesc(20*log10(abs(A_wo_deconv(1:1024, :))));
colormap(gray);
colorbar;
pbaspect([width_b (height_Bscan/2) 1]);
clim([-5 110]);

%% Part 4
% Take mean of all A-scans in Mscans:
% Zero out DC component:
M1_Ascan = mean(20*log10(abs(M1scan(1:1024, :))), 2);
M40_Ascan = mean(20*log10(abs(M40scan(1:1024, :))), 2);

figure;
plot(M1_Ascan);
title("Averaged 1-tone M-Scan");
xlabel("Pixel");
ylabel("Magnitude [dB]");

figure;
plot(M40_Ascan);
title("Averaged M40 A-Scan");
xlabel("Pixel");
ylabel("Magnitude [dB]");

% Calculate time and frequency domain data from MScan1:
[mscan1_time1, mscan1_freq1, xaxis1_time1, xaxis1_freq1] = processMScan(M1scan, ...
    965, lamb_0, fs);
[mscan1_time2, mscan1_freq2, xaxis1_time2, xaxis1_freq2] = processMScan(M1scan, ...
    938, lamb_0, fs);

% Create plots of time and frequency domain for MScan1:
figure;
plot(xaxis1_time1, mscan1_time1);
xlabel("Time [seconds]");
ylabel("Movements [mm]");
title("Time Domain Plot MScan 1 Tone at Pixel 965");

figure;
plot(xaxis1_freq1, 20*log10(abs(mscan1_freq1)));
xlabel("Frequency [Hz]");
ylabel("Magnitude [dB]");
title("Frequency Domain Mscan 1 Tone at Pixel 965");

figure;
plot(xaxis1_time2, mscan1_time2);
xlabel("Time [seconds]");
ylabel("Movements [mm]");
title("Time Domain Plot Mscan 1 Tone at Pixel 938");

figure;
plot(xaxis1_freq2, 20*log10(abs(mscan1_freq2)));
xlabel("Frequency [Hz]");
ylabel("Magnitude [dB]");
title("Frequency Domain Mscan 1 Tone at Pixel 938");

% Calculate time and frequency domain for MScan40:
[mscan40_time1, mscan40_freq1, xaxis40_time1, xaxis40_freq1] = processMScan(M40scan, ...
    965, lamb_0, fs);
[mscan40_time2, mscan40_freq2, xaxis40_time2, xaxis40_freq2] = processMScan(M40scan, ...
    938, lamb_0, fs);

% Create Plots:
figure;
plot(xaxis40_time1, mscan40_time1);
xlabel("Time [seconds]");
ylabel("Movements [mm]");
title("Time Domain Plot MScan 40 Tones at Pixel 965");

figure;
plot(xaxis40_freq1, 20*log10(abs(mscan40_freq1)));
xlabel("Frequency [Hz]");
ylabel("Magnitude [dB]");
title("Frequency Domain Mscan 40 Tones at Pixel 965");

figure;
plot(xaxis40_time2, mscan40_time2);
xlabel("Time [seconds]");
ylabel("Movements [mm]");
title("Time Domain Mscan 40 Tones at Pixel 938");

figure;
plot(xaxis40_freq2, 20*log10(abs(mscan40_freq2)));
xlabel("Frequency [Hz]");
ylabel("Magnitude [dB]");
title("Frequency Domain Mscan 40 Tones at Pixel 938");

% Find frequencies in MScan40:
frequencies = xaxis40_freq1(20*log10(abs(mscan40_freq1)) > -80);

% Only choose positive frequencies:
frequencies = frequencies(frequencies > 4000);

% Only obtain frequencies separated by at least 5Hz:
index = diff(frequencies) > 5;

% Shift right and pad with true so that "diff" ignores first true:
index = [true, index];

% Adjust index for these conditions:
frequencies = frequencies(index);


%% Functions Created

function [A_full_pipeline, L, A_witho_deconv_and_sub, A_witho_deconv] = generateAscan( ...
    num_samples, background_scans, window_length, A_scans, L2K, window_type)
    
    total_time = tic;

    % 1) Convert background scans to K domain, THEN average the background scans:
    L2K_B_time = tic;

    background_scans_K = L2K * background_scans;
    disp("L2K matrix multiply for background scans: ");
    toc(L2K_B_time);
    % Average:
    
    avg_background_scans = mean(background_scans_K, 2);

    % Perform 2nd order polynomial fit of the averaged background scans
    coeff = polyfit(1 : num_samples, avg_background_scans, 9);
    avg_background_fitted = polyval(coeff, 1 : window_length)';

    % 2) Convert the given A scan to K domain:
    L2K_A_time = tic;
    A_witho_deconv_and_sub = L2K * A_scans;
    disp("L2K matrix multiply for A scans: ");
    toc(L2K_A_time);
    
    % 3) Subtract averaged background scans from A scan:
    background_sub_time = tic;
    A_witho_deconv = A_witho_deconv_and_sub - avg_background_scans;
    disp("Background subtraction: ");
    toc(background_sub_time);

    % 4) Multiply the subtracted background scans by a user specified window:
    window_time = tic;
    window = windowFunction(window_length, window_type);
    A_scan_K_windowed = A_witho_deconv .* window;    
    disp("Time for window: ");
    toc(window_time);

    % 5) Perform deconvolution: i.e. divide the windowed signal by the
    %    background scans:
    deconvolution_time = tic;
    deblur_scan = A_scan_K_windowed ./ avg_background_fitted;
    disp("Time for de-convolution: ");
    toc(deconvolution_time);

    % 6) Take an FFT and fftshift each column:
    fft_time = tic;
    A_full_pipeline = fft(deblur_scan, [], 1);
    A_full_pipeline = fftshift(A_full_pipeline, 1);
    % Return fftshifted lambda domain:
    L = -num_samples/2 + 1 : num_samples/2;
    disp("Time for fft: ");
    toc(fft_time);

    disp("Total function run time: ");
    toc(total_time);

    % Window the A scans without deconvolution and subtraction, as well as
    % A scan without just deconvolution then take FFT
    A_witho_deconv_and_sub_windowed = A_witho_deconv_and_sub .* window;
    A_witho_deconv_windowed = A_witho_deconv .* window;

    % Take FFT
    A_witho_deconv_and_sub = fft(A_witho_deconv_and_sub_windowed, [], 1);
    A_witho_deconv_and_sub = fftshift(A_witho_deconv_and_sub, 1);

    A_witho_deconv = fft(A_witho_deconv_windowed, [], 1);
    A_witho_deconv = fftshift(A_witho_deconv, 1);

end

function window = windowFunction(window_length, window_type)
    % Generate a window function of the specified type
    switch lower(window_type)
        case 'hamming'
            window = hamming(window_length);
        case 'hann'
            window = hann(window_length);
        case 'blackman'
            window = blackman(window_length);
        % Add more window types as needed
        otherwise
            error('Unsupported window type');
    end
end

%% TO DO:
% Function that takes the Bscan and performs some processing (If asked for)
% but also determines the proper dimensions based on aspect ratio. Also
% adjusts greyscale components
function enhanceImage(Scan, width_b, height_Bscan)
    % Pixel height 
    dz = 2.7*10^-6;
    % Remove mirror image effect:
    no_mirror_scan = Scan(1:1024, :);

    % Obtain magnitude in dB scale:
    final_image = 20*log10(abs(no_mirror_scan));

    % Apply a simple histogram equalization:
    final_image = histeq(final_image);

    % Create figure of image:
    figure;
    imagesc(final_image);
    colormap("gray");
    colorbar;
    clim([0.92, 1.02]);
    % Crop image to show primarily the location where reflective surfaces
    % begin:
    pbaspect([width_b (height_Bscan/2) 1]);
end

function [mscan_time, mscan_freq, xaxis_time, xaxis_freq] = processMScan(mscan, ...
    depth, lambda, fs)
    
    % Calculate data in time domain:
    mscan_time = (unwrap(angle(mscan(depth, :))) * lambda) / (2*pi);

    % Create time domain x-axis:
    T = 1/fs;               % Period
    xaxis_time = 0 : T : T*(length(mscan) - 1);

    % Calculate data in frequency domain:
    mscan_freq = fftshift(fft(mscan_time));

    % Create frequency domain x-axis based on sample rate:
    xaxis_freq = linspace(-fs/2, fs/2, length(mscan));

end



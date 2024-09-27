% Function to create the histogram from the wavelet decomposition
% coefficients. Level 1-3 wavelet transforms all taken
function plotHist(input, wavelet_name)

    % Compute the wavelet transform
    [C1, ~] = wavedec2(input,1,wavelet_name);
    [C2, ~] = wavedec2(input,2,wavelet_name);
    [C3, ~] = wavedec2(input,3,wavelet_name);
    
    % Decode wavelet_name
    titlename = WaveletName(wavelet_name);

    % Create the histogram plots from the coefficients generated:
    figure;
    subplot(1, 3, 1);
    histogram(C1, 50);
    title("Level 1 Decomposition");

    subplot(1, 3, 2);
    histogram(C2, 50);
    title("Level 2 Decomposition");

    subplot(1, 3, 3);
    histogram(C3, 50);
    title("Level 3 Decomposition");

    % Create title for plot:
    sgtitle("Coefficient Histogram with " + titlename + " Wavelet Transform");
end